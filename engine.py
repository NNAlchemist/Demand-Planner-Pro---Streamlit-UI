# engine.py
import pandas as pd
import numpy as np
from collections import namedtuple
from prophet import Prophet
from sklearn.preprocessing import LabelEncoder
from scipy.stats import linregress
from dateutil.relativedelta import relativedelta

TYPE_ORDER = [
    "In Market History",
    "In Market Sales Y-1",
    "In Market Forecast",
    "Rolling Average",
    "To-Market Forecast",
    "Projected SOH",
    "Target Stock",
    "Coverage",
]

# ---------- Utilities ----------
GroupKey = namedtuple("GroupKey", ["COUNTRY", "CHANNEL", "PRODUCT_ID"])

def _correct_outliers(series: pd.Series) -> pd.Series:
    q05, q95 = series.quantile(0.05), series.quantile(0.95)
    return series.clip(lower=q05, upper=q95)

def _pad_months(df: pd.DataFrame) -> pd.DataFrame:
    """Pad to monthly frequency for each COUNTRY-CHANNEL-PRODUCT_ID."""
    out = []
    for (c, ch, p), g in df.groupby(["COUNTRY", "CHANNEL", "PRODUCT_ID"]):
        g = g.sort_values("DATE").set_index("DATE")
        idx = pd.date_range(g.index.min(), g.index.max(), freq="MS")
        g = g.reindex(idx).fillna(0)
        g["DATE"] = g.index
        g["COUNTRY"], g["CHANNEL"], g["PRODUCT_ID"] = c, ch, p
        out.append(g.reset_index(drop=True))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()

def _classify_fmr(sales: pd.Series) -> str:
    active = (sales > 0).sum()
    return "F" if active >= 24 else ("M" if active >= 12 else "R")

def _classify_xyz(sales: pd.Series) -> str:
    if sales.mean() == 0: return "Z"
    cov = sales.std() / (sales.mean() + 1e-8)
    return "X" if cov < 0.5 else ("Y" if cov < 1.0 else "Z")

# ---------- 1) Prepare monthly + master ----------
def prepare_monthly_and_master(ims_df: pd.DataFrame):
    """
    Expect IMS raw columns (case-insensitive, spaces ok):
    ALT COUNTRY KEY, JP CUST TYPE, Product ID, PERIOD YEAR, PERIOD ID, CY QTY, CY BNS, NET
    """
    df_raw = ims_df.copy()
    df_raw.columns = df_raw.columns.str.strip()
    
    # Dates + quantity
    df_raw["PERIOD YEAR"] = pd.to_numeric(df_raw["PERIOD YEAR"], errors="coerce").astype("Int64")
    df_raw["PERIOD ID"]   = pd.to_numeric(df_raw["PERIOD ID"], errors="coerce").astype("Int64")
    df_raw["DATE"] = pd.to_datetime(df_raw["PERIOD YEAR"].astype(str) + "-" +
                                    df_raw["PERIOD ID"].astype(str).str.zfill(2) + "-01", errors="coerce")
    df_raw["CY_Total"] = df_raw.get("CY QTY", 0).fillna(0) + df_raw.get("CY BNS", 0).fillna(0)

    df = df_raw[['ALT COUNTRY KEY', 'JP CUST TYPE', 'Product ID', 'DATE', 'CY_Total', 'NET']].copy()
    df = df.rename(columns={
        "ALT COUNTRY KEY": "COUNTRY",
        "JP CUST TYPE": "CHANNEL",
        "Product ID": "PRODUCT_ID",
        "CY_Total": "CY_QTY",
        "NET": "ACT_VALUE"
    })[["COUNTRY","CHANNEL","PRODUCT_ID","DATE","CY_QTY","ACT_VALUE"]].dropna(subset=["DATE"])

    # GROUPING GUARD: make sure history is always aggregated at month level
    monthly = (df.groupby(["COUNTRY","CHANNEL","PRODUCT_ID","DATE"], as_index=False)
                 .agg({"CY_QTY":"sum","ACT_VALUE":"sum"}))

    monthly = _pad_months(monthly)

    # Master + forecastability
    master_cols = ["CLUSTER NAME","DIVISION","ALT COUNTRY KEY","CATEGORY ID","PRODUCT ID",
                   "PRODUCT","PROD GROUP","JP CUST TYPE","Product ID"]
    keep = [c for c in master_cols if c in df_raw.columns]
    master = df_raw[keep].drop_duplicates().rename(columns={
        "ALT COUNTRY KEY":"COUNTRY",
        "JP CUST TYPE":"CHANNEL",
        "Product ID":"PRODUCT_ID"
    })

    tags = []
    for key, g in monthly.groupby(["COUNTRY","CHANNEL","PRODUCT_ID"]):
        f, x = _classify_fmr(g["CY_QTY"]), _classify_xyz(g["CY_QTY"])
        level = "High" if (f=="F" and x=="X") else ("Medium" if (f=="F" or x=="X") else "Low")
        tags.append({"COUNTRY":key[0], "CHANNEL":key[1], "PRODUCT_ID":key[2], "FMR":f, "XYZ":x, "Forecastability_Level":level})
    forecastability = pd.DataFrame(tags) if tags else pd.DataFrame(columns=["COUNTRY","CHANNEL","PRODUCT_ID","FMR","XYZ","Forecastability_Level"])
    master_enriched = master.merge(forecastability, on=["COUNTRY","CHANNEL","PRODUCT_ID"], how="left")

    return monthly, master_enriched, df  # monthly (clean), master, actuals copy

# ---------- 2) In-Market: generate Prophet ensemble ----------
def _forecast_one(group_key: GroupKey, g: pd.DataFrame, horizon=24) -> pd.DataFrame | None:
    g = g.sort_values("DATE").copy()
    g["CY_QTY"] = _correct_outliers(g["CY_QTY"]).rolling(3, min_periods=1).mean()
    if g["CY_QTY"].sum() == 0 or len(g) < 6:
        return None

    df_p = g.rename(columns={"DATE":"ds","CY_QTY":"y"})[["ds","y"]]
    param_grid = [
        {"changepoint_prior_scale":0.001, "seasonality_prior_scale":0.1},
        {"changepoint_prior_scale":0.002, "seasonality_prior_scale":0.15},
        {"changepoint_prior_scale":0.0005, "seasonality_prior_scale":0.1},
    ]
    outs = []
    for i, p in enumerate(param_grid, 1):
        try:
            m = Prophet(seasonality_mode="additive", **p).fit(df_p)
            fut = m.make_future_dataframe(periods=horizon, freq="MS")
            fc = m.predict(fut)[["ds","yhat"]].rename(columns={"ds":"DATE","yhat":"Forecast"})
            fc["Forecast"] = fc["Forecast"].clip(lower=0)
            for col in GroupKey._fields: fc[col] = getattr(group_key, col)
            fc["Model"] = f"Prophet_V{i}"
            outs.append(fc)
        except Exception:
            continue
    return pd.concat(outs, ignore_index=True) if outs else None

def run_in_market_forecast(monthly_df: pd.DataFrame, actuals_df: pd.DataFrame, horizon_months=24):
    # 2.1 generate forecasts
    fc_parts = []
    for (c, ch, p), g in monthly_df.groupby(["COUNTRY","CHANNEL","PRODUCT_ID"]):
        out = _forecast_one(GroupKey(c,ch,p), g[["DATE","CY_QTY"]].copy(), horizon=horizon_months)
        if out is not None: fc_parts.append(out)
    all_model_forecasts = pd.concat(fc_parts, ignore_index=True) if fc_parts else pd.DataFrame()

    if all_model_forecasts.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # 2.2 evaluate models
    actuals = actuals_df[["COUNTRY","CHANNEL","PRODUCT_ID","DATE","CY_QTY"]].copy()
    actuals["DATE"] = pd.to_datetime(actuals["DATE"]).dt.to_period("M").dt.to_timestamp()
    forecasts = all_model_forecasts.copy()
    forecasts["DATE"] = pd.to_datetime(forecasts["DATE"]).dt.to_period("M").dt.to_timestamp()

    merged = forecasts.merge(
        actuals, on=["COUNTRY","CHANNEL","PRODUCT_ID","DATE"], how="left"
    ).dropna(subset=["CY_QTY"])

    def _metrics(d: pd.DataFrame):
        mape = np.mean(np.abs((d["CY_QTY"] - d["Forecast"]) / (d["CY_QTY"] + 1e-8)))
        bias = (d["Forecast"].sum() - d["CY_QTY"].sum()) / (d["CY_QTY"].sum() + 1e-8)
        return pd.Series({"MAPE":mape, "Bias":bias, "Combined_Score": mape + abs(bias)})

    all_model_metrics = (merged.groupby(["COUNTRY","CHANNEL","PRODUCT_ID","Model"])
                         .apply(_metrics).reset_index())

    # 2.3 winners
    winners = (all_model_metrics.sort_values("Combined_Score")
               .groupby(["COUNTRY","CHANNEL","PRODUCT_ID"], as_index=False).first())
    winners = winners.rename(columns={"Model":"Selected_Model"})

    # 2.4 24-month window
    last_actual = actuals["DATE"].max()
    start = (pd.to_datetime(f"{last_actual.year}-{last_actual.month}-01") + pd.offsets.MonthBegin(1))
    end   = (start + pd.DateOffset(months=horizon_months-1)).to_period("M").to_timestamp()

    final_list = []
    for _, row in winners.iterrows():
        mask = (
            (all_model_forecasts["COUNTRY"]==row["COUNTRY"]) &
            (all_model_forecasts["CHANNEL"]==row["CHANNEL"]) &
            (all_model_forecasts["PRODUCT_ID"]==row["PRODUCT_ID"]) &
            (all_model_forecasts["Model"]==row["Selected_Model"])
        )
        part = all_model_forecasts.loc[mask, ["COUNTRY","CHANNEL","PRODUCT_ID","DATE","Forecast"]].copy()
        part["DATE"] = pd.to_datetime(part["DATE"]).dt.to_period("M").dt.to_timestamp()
        part = part[(part["DATE"]>=start) & (part["DATE"]<=end)]
        final_list.append(part)

    final_inmarket = pd.concat(final_list, ignore_index=True) if final_list else pd.DataFrame()

    # 2.5 combined
    hist = actuals.rename(columns={"CY_QTY":"VALUE"}).assign(TYPE="In Market History")
    fcs  = final_inmarket.rename(columns={"Forecast":"VALUE"}).assign(TYPE="In Market Forecast")
    final_combined = pd.concat([hist, fcs], ignore_index=True).sort_values(
        ["COUNTRY","CHANNEL","PRODUCT_ID","DATE"]
    )
    
    return final_inmarket, all_model_metrics, final_combined

# ---------- 3) To-Market logic ----------
def run_to_market(ims_history_and_fc: pd.DataFrame,
                  soh_df: pd.DataFrame | None,
                  target_df: pd.DataFrame | None,
                  default_coverage: float = 3.0) -> pd.DataFrame:
    """
    Converts in market forecast to market based on available stock on hand at distributor using soh_df
    """
    if ims_history_and_fc is None or ims_history_and_fc.empty:
        return pd.DataFrame()

    src = ims_history_and_fc.copy()
    src["DATE"]  = pd.to_datetime(src["DATE"]).dt.to_period("M").dt.to_timestamp()
    src["VALUE"] = pd.to_numeric(src["VALUE"], errors="coerce").fillna(0)

    # --- BUILD DEMAND ONLY FROM IM HISTORY + IM FORECAST (prevents drift) ---
    demand_rows = (src[src["TYPE"].isin(["In Market History", "In Market Forecast"])]
                     .groupby(["COUNTRY","CHANNEL","PRODUCT_ID","DATE"], as_index=False)["VALUE"].sum())

    # --- normalize SOH (same as your version) ---
    soh = pd.DataFrame(columns=["COUNTRY", "CHANNEL", "PRODUCT_ID", "SOH"])
    if soh_df is not None and not soh_df.empty:
        try:
            _soh = soh_df[['ALT COUNTRY KEY', 'JP CUST TYPE', 'Product ID', 'Stock Quantity']].copy()
            _soh.rename(columns={
                "ALT COUNTRY KEY": "COUNTRY",
                "JP CUST TYPE": "CHANNEL",
                "Product ID": "PRODUCT_ID",
                "Stock Quantity": "SOH",
            }, inplace=True)
        except Exception:
            _soh = soh_df.copy()
        _soh["SOH"] = pd.to_numeric(_soh["SOH"], errors="coerce").fillna(0)
        soh = _soh.groupby(["COUNTRY", "CHANNEL", "PRODUCT_ID"], as_index=False)["SOH"].sum()

    # --- coverage map (unchanged) ---
    cov_map = {}
    if target_df is not None and not target_df.empty:
        cov_map = {(r["COUNTRY"], r["CHANNEL"], r["PRODUCT_ID"]): float(r["Target"])
                   for _, r in target_df.iterrows()}

    out = []

    # iterate per SKU
    for (c, ch, p), g_dem in demand_rows.groupby(["COUNTRY","CHANNEL","PRODUCT_ID"]):
        g_dem = g_dem.sort_values("DATE").reset_index(drop=True)
        demand = g_dem.set_index("DATE")["VALUE"].sort_index()

        # find forecast window START from original src (first IMF month)
        fc_dates = src[(src["COUNTRY"]==c) & (src["CHANNEL"]==ch) & (src["PRODUCT_ID"]==p) &
                       (src["TYPE"]=="In Market Forecast")]["DATE"]
        if fc_dates.empty:
            continue
        fc_start = fc_dates.min()
        fc_end   = demand.index.max()

        # opening stock at (fc_start - 1)
        soh_row = soh[(soh["COUNTRY"]==c) & (soh["CHANNEL"]==ch) & (soh["PRODUCT_ID"]==p)]
        projected_soh = float(soh_row["SOH"].iloc[0]) if not soh_row.empty else 0.0

        coverage = cov_map.get((c, ch, p), default_coverage)

        for d in pd.date_range(fc_start, fc_end, freq="MS"):
            ims_val = float(demand.get(d, 0.0))

            # rolling avg over d-3, d-2, d-1, d, d+1, d+2 from DEMAND ONLY
            win = [d - pd.DateOffset(months=i) for i in [3,2,1]] + [d] + \
                  [d + pd.DateOffset(months=i) for i in [1,2]]
            roll = demand.reindex(win).mean()

            target_stock = (roll * coverage) if not pd.isna(roll) else 0.0
            replenish    = max(target_stock + ims_val - projected_soh, 0.0)

            # keep your original dating: Projected SOH at (d - 1 month)
            out.append({
                "COUNTRY": c, "CHANNEL": ch, "PRODUCT_ID": p,
                "DATE": d - pd.offsets.MonthBegin(1),
                "TYPE": "Projected SOH", "VALUE": round(projected_soh, 0)
            })
            out.append({
                "COUNTRY": c, "CHANNEL": ch, "PRODUCT_ID": p,
                "DATE": d, "TYPE": "Rolling Average",
                "VALUE": round(0.0 if pd.isna(roll) else roll, 0)
            })
            out.append({
                "COUNTRY": c, "CHANNEL": ch, "PRODUCT_ID": p,
                "DATE": d, "TYPE": "Target Stock", "VALUE": round(target_stock, 0)
            })
            out.append({
                "COUNTRY": c, "CHANNEL": ch, "PRODUCT_ID": p,
                "DATE": d, "TYPE": "To-Market Forecast", "VALUE": round(replenish, 0)
            })
            # retain your original coverage math (rounded-months * 30)
            cov_val = 0.0 if (pd.isna(roll) or roll == 0) else round(projected_soh/roll, 0) * 30
            out.append({
                "COUNTRY": c, "CHANNEL": ch, "PRODUCT_ID": p,
                "DATE": d, "TYPE": "Coverage", "VALUE": cov_val
            })

            # next month opening stock
            projected_soh = projected_soh - ims_val + replenish

    return (pd.DataFrame(out)
              .sort_values(["COUNTRY","CHANNEL","PRODUCT_ID","DATE","TYPE"])
              .reset_index(drop=True))

# ---------- PIVOT helpers ----------
def _format_date_col(ts: pd.Timestamp) -> str:
    s = ts.strftime("%d-%b-%y")
    return s.lstrip("0")

def pivot_wide_by_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot the long master dataframe -> wide by TYPE with monthly date columns.
    - Ensures DATE normalized to month start
    - VALUE coerced numeric
    - Group keys coerced to str
    - Produces clean wide matrix with TYPE order enforced
    """

    if df is None or df.empty:
        return pd.DataFrame()

    _df = df.copy()

    # Normalize DATE and VALUE
    _df["DATE"] = pd.to_datetime(_df["DATE"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    _df["VALUE"] = pd.to_numeric(_df.get("VALUE", 0), errors="coerce").fillna(0)

    _df = _df.dropna(subset=["DATE"])
    if _df.empty:
        return pd.DataFrame()

    # Ensure grouping keys are strings
    for k in ["COUNTRY", "CHANNEL", "PRODUCT_ID", "TYPE"]:
        if k in _df.columns:
            _df[k] = _df[k].astype(str)

    # Pivot
    wide = _df.pivot_table(
        index=["COUNTRY", "CHANNEL", "PRODUCT_ID", "TYPE"],
        columns="DATE",
        values="VALUE",
        aggfunc="sum",
        fill_value=0
    )

    wide = wide.reset_index()

    # Rename date columns to "Mon-YYYY"
    date_cols = [c for c in wide.columns if isinstance(c, pd.Timestamp)]
    rename_map = {c: c.strftime("%b-%Y") for c in date_cols}
    wide = wide.rename(columns=rename_map)

    # Enforce TYPE ordering if available
    if "TYPE" in wide.columns and "TYPE_ORDER" in globals():
        wide["TYPE"] = pd.Categorical(wide["TYPE"], categories=TYPE_ORDER, ordered=True)

    wide = wide.sort_values(["COUNTRY", "CHANNEL", "PRODUCT_ID", "TYPE"]).reset_index(drop=True)
    return wide


def style_numeric_dataframe(df):
    formatter = {
        col: "{:,.0f}" for col in df.columns
        if pd.api.types.is_numeric_dtype(df[col])
    }
    styled_df = df.style.format(formatter).set_properties(**{'font-style': 'italic'})
    return styled_df


BASE_TYPES = {"In Market History", "In Market Forecast", "SOH"}
DERIVED_TYPES = {
    "Rolling Average",
    "Target Stock",
    "To-Market Forecast",
    "Projected SOH",
    "Coverage",
}

def _normalize_master(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["COUNTRY", "CHANNEL", "PRODUCT_ID", "DATE", "TYPE", "VALUE"])
    df = df.copy()
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    for c in ["COUNTRY", "CHANNEL", "PRODUCT_ID", "TYPE"]:
        df[c] = df[c].astype(str)
    df["VALUE"] = pd.to_numeric(df["VALUE"], errors="coerce").fillna(0)
    key = ["COUNTRY", "CHANNEL", "PRODUCT_ID", "DATE", "TYPE"]
    df = df.sort_values(key).drop_duplicates(key, keep="last").reset_index(drop=True)
    return df


def recompute_to_market_from_master(master_df, soh_df, target_df, default_coverage=3.0):
    # Normalize master_df only
    master_norm = _normalize_master(master_df)

    # Run To-Market forecast
    tm_all = run_to_market(
        ims_history_and_fc=master_norm[master_norm["TYPE"].isin(["In Market History", "In Market Forecast"])],
        soh_df=soh_df,        # raw
        target_df=target_df,  # raw
        default_coverage=default_coverage,
    )

    # Ensure output consistency
    for col in ["COUNTRY", "CHANNEL", "PRODUCT_ID", "TYPE"]:
        if col in tm_all.columns:
            tm_all[col] = tm_all[col].astype(str)

    tm_all["VALUE"] = pd.to_numeric(tm_all["VALUE"], errors="coerce").fillna(0)

    return tm_all
