# --- app.py top imports & session init ---
import io
import pandas as pd
import numpy as np
import streamlit as st
from copy import deepcopy

from engine import (
    pivot_wide_by_type,
    prepare_monthly_and_master,
    run_in_market_forecast,
    run_to_market,
    style_numeric_dataframe,
)



st.set_page_config(page_title="AI Demand & Logistics Agent", layout="wide")

# Ensure session state keys
def _ensure_state():
    ss = st.session_state
    ss.setdefault("ims_df", None)
    ss.setdefault("soh_df", None)
    ss.setdefault("target_df", None)
    ss.setdefault("monthly_df", None)
    ss.setdefault("master_enriched", None)
    ss.setdefault("final_inmarket", None)
    ss.setdefault("final_combined", None)
    ss.setdefault("to_market_df", None)
    ss.setdefault("model_metrics", None)
    ss.setdefault("base_master_df", pd.DataFrame())
    ss.setdefault("edited_master_df", pd.DataFrame())
    ss.setdefault("master_df", pd.DataFrame())
    if "audit_log" not in ss:
        ss.audit_log = pd.DataFrame(columns=[
            "TIMESTAMP", "USER", "COUNTRY", "CHANNEL", "PRODUCT_ID", "TYPE", "DATE", "OLD_VALUE", "NEW_VALUE"
        ])

_ensure_state()

# =========================
# Page + Sidebar
# =========================
st.set_page_config(page_title="AI Demand & Logistics Agent", layout="wide")
st.sidebar.image("https://www.jamjoompharma.com/wp-content/uploads/2024/11/jamjoom.png")
# --- minimal dark accents for sidebar + buttons ---
st.markdown("""
<style>
/* Sidebar */
[data-testid="stSidebar"] {
  background: #00249A; /* slate-900 */
  color: #ffffff;
}
[data-testid="stSidebar"] * { color: #ffffff !important; }
[data-testid="stSidebar"] input, [data-testid="stSidebar"] select, [data-testid="stSidebar"] textarea {
  color: #111827 !important; /* readable text inside inputs */
}

/* Buttons */
div.stButton>button {
  background: #00249A; /* slate-800 */
  color: #ffffff;
  border-radius: 8px;
  border: 1px solid #374151;
}
div.stButton>button:hover {
  background: #374151;
}
</style>
""", unsafe_allow_html=True)


with st.sidebar:
    st.markdown("### Demand Planner Pro")
    st.caption(
        "In-market forecasting → To-market plan → Editable → Excel export.\n"
        "Upload IMS/SOH, generate forecasts, adjust, and finalize."
    )
    editor_user = st.text_input("Editor", value="default_user")
    cycle_name = st.text_input("Cycle name", value="Aug-2025")
    default_cov = st.number_input("Default coverage (in months)", min_value=0.0, value=3.0, step=1.0)
    slider = st.slider("Months forecasted", 12, 48,24)

st.title("AI Demand & Logistics Agent — MVP")

# =========================
# Session State (single source + audit)
# =========================
def _ensure_state():
    ss = st.session_state
    # Raw inputs
    ss.setdefault("ims_df", None)
    ss.setdefault("soh_df", None)
    ss.setdefault("target_df", None)

    # Engine outputs
    ss.setdefault("monthly_df", None)
    ss.setdefault("master_enriched", None)
    ss.setdefault("final_inmarket", None)
    ss.setdefault("final_combined", None)  # history + in-market forecast
    ss.setdefault("to_market_df", None)
    ss.setdefault("model_metrics", None)
    ss.setdefault("base_master_df", pd.DataFrame())     # immutable reference after each run
    ss.setdefault("edited_master_df", pd.DataFrame())   # what planners edit


    # Single source of truth for the UI + export (ALWAYS use this everywhere else)
    ss.setdefault("master_df", pd.DataFrame())

    # Persistent audit log of edits
    if "audit_log" not in ss:
        ss.audit_log = pd.DataFrame(
            columns=[
                "TIMESTAMP", "USER",
                "COUNTRY", "CHANNEL", "PRODUCT_ID", "TYPE", "DATE",
                "OLD_VALUE", "NEW_VALUE"
            ]
        )

_ensure_state()

# =========================
# Utility: commit edits back to master_df + log changes
# =========================

# =========================
# Tabs
# =========================
tabs = st.tabs([
    "1) Upload Data",
    "2) In-Market Forecast",
    "3) To-Market Forecast",
    "4) Analytics",
    "5) Dashboard",
    "6) Finalize & Download",
    "7) Testing",
])

# ---------- Tab 1: Upload ----------
with tabs[0]:
    st.header("Upload Data")
    st.caption("Accepts .xlsx or .csv")

    def _read_any(file):
        if file.name.lower().endswith(".csv"):
            return pd.read_csv(file)
        return pd.read_excel(file)

    colA, colB, colC = st.columns(3)
    with colA:
        ims_file = st.file_uploader("IMS (Sales History, monthly)", type=["xlsx","csv"], key="ims_upl")
        if ims_file:
            st.session_state.ims_df = _read_any(ims_file)
            st.success(f"IMS loaded: {st.session_state.ims_df.shape}")
            st.dataframe(st.session_state.ims_df.head(20), use_container_width=True)

    with colB:
        soh_file = st.file_uploader("SOH (Stock on Hand)", type=["xlsx","csv"], key="soh_upl")
        if soh_file:
            st.session_state.soh_df = _read_any(soh_file)
            st.success(f"SOH loaded: {st.session_state.soh_df.shape}")
            st.dataframe(st.session_state.soh_df.head(20), use_container_width=True)

    with colC:
        target_file = st.file_uploader("Target stock (optional)", type=["xlsx","csv"], key="target_upl")
        if target_file:
            st.session_state.target_df = _read_any(target_file)
            st.success(f"Target loaded: {st.session_state.target_df.shape}")
            st.dataframe(st.session_state.target_df.head(20), use_container_width=True)

# ---------- Tab 2: In-Market ----------
with tabs[1]:
    st.header("In-Market Forecast")

    run_btn = st.button("Run In-Market Forecast")
    if run_btn:
        if st.session_state.ims_df is None:
            st.warning("Upload IMS first.")
        else:
            with st.spinner("Preparing data and running Prophet ensemble..."):
                monthly, master_enriched, actuals = prepare_monthly_and_master(st.session_state.ims_df)
                st.session_state.monthly_df = monthly
                st.session_state.master_enriched = master_enriched

                final_inmarket, model_metrics, final_combined = run_in_market_forecast(
                    monthly_df=monthly,
                    actuals_df=actuals,
                    horizon_months=slider
                )
                st.session_state.final_inmarket = final_inmarket
                st.session_state.final_combined = final_combined
                st.session_state.model_metrics = model_metrics

                # Initialize master_df (single source) with history + in-market forecast
                st.session_state.master_df = final_combined.copy()
                
                st.session_state.base_master_df = final_combined.copy()  # history + IMF (no TMS yet)
                # If user hasn't started editing, initialize Edited = Base
                if st.session_state.edited_master_df.empty:
                    st.session_state.edited_master_df = st.session_state.base_master_df.copy()


            if final_inmarket is None or final_inmarket.empty:
                st.error("No forecast generated. Check data coverage.")
            else:
                st.success("Forecast generated and Master dataset initialized.")
            

    if st.session_state.final_combined is not None and not st.session_state.final_combined.empty:
        wide = pivot_wide_by_type(st.session_state.final_combined)
        styled_wide = style_numeric_dataframe(wide)
        st.dataframe(styled_wide, use_container_width=True)

# ---------- Tab 3: To-Market ----------
with tabs[2]:
    st.header("To-Market Forecast")
    run_tm_btn = st.button("Run To-Market")
    if run_tm_btn:
        if st.session_state.master_df is None or st.session_state.master_df.empty or st.session_state.soh_df is None:
            st.warning("Need In-Market (master_df) and SOH uploads.")
        else:
            with st.spinner("Calculating rolling averages, targets, and replenishment..."):
                # Build drivers from BASE (no previous TMS calc noise)
                base = st.session_state.base_master_df.copy()
                drivers = base[base["TYPE"].isin(["In Market History","In Market Forecast"])].copy()

                tm = run_to_market(
                    ims_history_and_fc=drivers,
                    soh_df=st.session_state.soh_df,
                    target_df=st.session_state.target_df,
                    default_coverage=float(default_cov),
                )

                # Save To-Market output separately
                st.session_state.to_market_df = tm.copy()

                # Compose new BASE = history + IMF + (fresh) To-Market dependents
                keep = base[~base["TYPE"].isin(
                    ["Rolling Average","Target Stock","To-Market Forecast","Projected SOH","Coverage"]
                )]
                st.session_state.base_master_df = pd.concat([keep, tm], ignore_index=True).sort_values(
                    ["COUNTRY","CHANNEL","PRODUCT_ID","DATE","TYPE"]
                )

                # If Edited not started, set it now from Base
                if st.session_state.edited_master_df.empty:
                    st.session_state.edited_master_df = st.session_state.base_master_df.copy()

            st.success("To-Market computed and Base dataset refreshed.")


    if st.session_state.to_market_df is not None and not st.session_state.to_market_df.empty:
        wide_tm = pivot_wide_by_type(st.session_state.to_market_df)
        wide_tm = style_numeric_dataframe(wide_tm)
        st.dataframe(wide_tm, use_container_width=True)

# ---------- Tab 4: Analytics ----------
with tabs[3]:
    from engine import recompute_to_market_from_master

    st.header("Analytics (Editable Forecasts)")

    ss = st.session_state
    if ss.master_df is None or ss.master_df.empty:
        st.info("Run the In-Market forecast (and optionally To-Market) first.")
        st.stop()

    # --- Pivot full dataset wide (UI view) ---
    wide = pivot_wide_by_type(ss.master_df)
    id_cols = ["COUNTRY", "CHANNEL", "PRODUCT_ID", "TYPE"]
    date_cols = [c for c in wide.columns if c not in id_cols]

    # --- Only In-Market Forecast is editable ---
    EDITABLE_ROW = "In Market Forecast"
    disable_rows = wide["TYPE"].astype(str).apply(lambda t: t != EDITABLE_ROW)

    # Column configs
    col_cfg = {
        "COUNTRY": st.column_config.TextColumn("COUNTRY", disabled=True),
        "CHANNEL": st.column_config.TextColumn("CHANNEL", disabled=True),
        "PRODUCT_ID": st.column_config.TextColumn("PRODUCT_ID", disabled=True),
        "TYPE": st.column_config.TextColumn("TYPE", disabled=True),
    }
    for dc in date_cols:
        col_cfg[dc] = st.column_config.NumberColumn(dc, step=1.0, disabled=False)

    st.caption(
        "✏️ Edit only the *In Market Forecast* rows. "
        "Click **Apply & Recalculate To-Market** to refresh Rolling Average, Target Stock, "
        "Projected SOH, Coverage, and To-Market Forecast."
    )

    # --- Ensure only numeric fills on date columns (avoid categorical fillna crash) ---
    wide_clean = wide.copy()
    num_cols = [c for c in wide_clean.columns if c not in id_cols]
    wide_clean[num_cols] = (
        wide_clean[num_cols]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    # --- Editable grid ---
    edited_matrix = st.data_editor(
        wide_clean,
        num_rows="dynamic",
        column_config=col_cfg,
        hide_index=True,
        use_container_width=True,
        disabled=disable_rows,
    )

    # --- Melt IMF rows back to long ---
    def _melt_imf_only(wide_df: pd.DataFrame) -> pd.DataFrame:
        imf_only = wide_df[wide_df["TYPE"].astype(str) == EDITABLE_ROW].copy()
        long_imf = imf_only.melt(
            id_vars=id_cols,
            value_vars=date_cols,
            var_name="DATE",
            value_name="VALUE",
        )
        # date headers are "Mon-YYYY"
        long_imf["DATE"] = pd.to_datetime(long_imf["DATE"], format="%b-%Y")
        return long_imf

    edited_imf_long = _melt_imf_only(edited_matrix)

    # --- Apply button ---
    if st.button("✅ Apply & Recalculate To-Market", type="primary"):

        # Normalize edited IMF (keys as str, month-start dates, numeric values)
        edited_imf_long = edited_imf_long.copy()
        edited_imf_long["DATE"] = (
            pd.to_datetime(edited_imf_long["DATE"])
            .dt.to_period("M").dt.to_timestamp()
        )
        edited_imf_long["VALUE"] = pd.to_numeric(edited_imf_long["VALUE"], errors="coerce").fillna(0)
        for col in ["COUNTRY", "CHANNEL", "PRODUCT_ID", "TYPE"]:
            edited_imf_long[col] = edited_imf_long[col].astype(str)

        # Build base_clean = IMH (and any non-derived, non-IMF rows) from BASE
        # This keeps history stable and avoids stacking/duplication of derived rows
        base_clean = ss.base_master_df[
            ~ss.base_master_df["TYPE"].isin([
                "In Market Forecast",
                "Rolling Average",
                "Target Stock",
                "To-Market Forecast",
                "Projected SOH",
                "Coverage",
            ])
        ].copy()

        # Compose master_for_tm = base_clean (IMH, etc.) + edited IMF
        master_for_tm = pd.concat([base_clean, edited_imf_long], ignore_index=True)

        # Recompute To-Market (drift-proof) using the SAME engine function
        tm_all = recompute_to_market_from_master(
            master_df=master_for_tm,
            soh_df=ss.soh_df,               # raw; engine normalizes internally
            target_df=ss.target_df,         # raw
            default_coverage=float(default_cov),
        )

        # Final displayed master = base_clean + edited IMF + freshly computed TM
        ss.master_df = (
            pd.concat([base_clean, edited_imf_long, tm_all], ignore_index=True)
              .sort_values(["COUNTRY","CHANNEL","PRODUCT_ID","DATE","TYPE"])
              .reset_index(drop=True)
        )

        st.success("✅ IMF edits applied. To-Market and dependent figures recalculated (drift-proof).")


# ---------- Tab 5: Dashboard ----------
with tabs[4]:
    st.header("Dashboard")

    if st.session_state.master_df is None or st.session_state.master_df.empty:
        st.info("Run the In-Market forecast first.")
    else:
        df = st.session_state.master_df.copy()
        df["DATE"] = pd.to_datetime(df["DATE"]).dt.to_period("M").dt.to_timestamp()
        df["VALUE"] = pd.to_numeric(df["VALUE"], errors="coerce").fillna(0)

        # ---- Filters
        c1,c2,c3,c4 = st.columns(4)
        countries = sorted(df["COUNTRY"].dropna().astype(str).unique())
        channels  = sorted(df["CHANNEL"].dropna().astype(str).unique())
        prods     = sorted(df["PRODUCT_ID"].dropna().astype(str).unique())

        with c1:
            f_country = st.multiselect("Country", countries, default=countries[:1] if countries else [])
        with c2:
            f_channel = st.multiselect("Channel", channels, default=channels[:1] if channels else [])
        with c3:
            f_prod = st.multiselect("Product", prods, default=prods[:1] if prods else [])
        with c4:
            years = sorted(df["DATE"].dt.year.unique())
            view_year = st.selectbox("Year focus", years, index=0 if years else None)

        filt = df.copy()
        if f_country: filt = filt[filt["COUNTRY"].isin(f_country)]
        if f_channel: filt = filt[filt["CHANNEL"].isin(f_channel)]
        if f_prod:    filt = filt[filt["PRODUCT_ID"].isin(f_prod)]

        # ---- Main chart (History vs Forecast)
        base = filt[filt["TYPE"].isin(["In Market History","In Market Forecast"])].copy()
        if not base.empty:
            series = (base.groupby(["DATE","TYPE"])["VALUE"].sum().reset_index()
                           .pivot(index="DATE", columns="TYPE", values="VALUE").fillna(0))
            st.subheader("In-Market: History vs Forecast")
            st.line_chart(series, use_container_width=True)
        else:
            st.info("No In-Market data in current filter.")

        # ---- KPIs
        if base.empty:
            st.stop()

        y = int(view_year)
        prevy = y - 1
        fm = filt[filt["TYPE"].isin(["In Market History","In Market Forecast"])]
        FY_curr = fm.loc[fm["DATE"].dt.year==y, "VALUE"].sum()
        FY_prev = fm.loc[fm["DATE"].dt.year==prevy, "VALUE"].sum()

        max_in_year = fm.loc[fm["DATE"].dt.year==y, "DATE"].max()
        if pd.isna(max_in_year): max_in_year = pd.Timestamp(year=y, month=12, day=1)
        YTD = fm.loc[(fm["DATE"].dt.year==y) & (fm["DATE"] <= max_in_year), "VALUE"].sum()
        YTG = fm.loc[(fm["DATE"].dt.year==y) & (fm["DATE"] > max_in_year), "VALUE"].sum()

        tail3 = fm.sort_values("DATE")["VALUE"].tail(3).mean() if not fm.empty else 0
        RunRate = (tail3 or 0) * 12
        GrowthRate = (FY_curr / (FY_prev + 1e-8) - 1.0) if FY_prev > 0 else np.nan

        k1,k2,k3,k4,k5,k6,k7,k8 = st.columns(8)
        k1.metric("Full Year", f"{FY_curr:,.0f}")
        k2.metric("Run Rate", f"{RunRate:,.0f}")
        k3.metric("Growth Rate", f"{(GrowthRate*100 if not np.isnan(GrowthRate) else 0):,.1f}%")
        k4.metric("YTD", f"{YTD:,.0f}")
        k5.metric("YTG", f"{YTG:,.0f}")
        k6.metric("Lift vs PY", f"{(FY_curr-FY_prev):,.0f}")
        k7.metric("YTD RR", f"{(YTD/max((max_in_year.month/12)*FY_curr,1e-8)*100):.1f}%")
        k8.metric("YTG RR", f"{(YTG/max(((12-max_in_year.month)/12)*FY_curr,1e-8)*100):.1f}%")

        # Quarterly
        qtbl = (fm[fm["DATE"].dt.year.isin([prevy,y])]
                  .assign(
                      Q=lambda x: "Q" + ((x["DATE"].dt.month.sub(1)//3)+1).astype(str),
                      Y=lambda x: x["DATE"].dt.year
                  )
                  .groupby(["Y","Q"])["VALUE"].sum().unstack("Q").fillna(0).astype(int))
        st.caption("Quarterly totals (In-Market)")
        st.dataframe(qtbl.style.format("{:,.0f}"), use_container_width=True)

# ---------- Tab 6: Finalize & Download ----------
with tabs[5]:
    st.header("Finalize & Download")

    if st.session_state.master_df is not None and not st.session_state.master_df.empty:
        final_out = st.session_state.master_df.copy()
        final_out["Cycle"] = cycle_name

        to_market_out = st.session_state.to_market_df.copy() if st.session_state.to_market_df is not None else pd.DataFrame()
        if not to_market_out.empty:
            to_market_out["Cycle"] = cycle_name

        with io.BytesIO() as buffer:
            with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                # Optional reference sheets
                if st.session_state.master_enriched is not None:
                    st.session_state.master_enriched.to_excel(writer, index=False, sheet_name="Master_Data")
                if st.session_state.model_metrics is not None and not st.session_state.model_metrics.empty:
                    st.session_state.model_metrics.to_excel(writer, index=False, sheet_name="Model_Metrics")

                # Primary outputs (edited)
                final_out.to_excel(writer, index=False, sheet_name="In-Marketr")
                if not to_market_out.empty:
                    to_market_out.to_excel(writer, index=False, sheet_name="To-Market")

                # Audit trail
                if not st.session_state.audit_log.empty:
                    st.session_state.audit_log.sort_values("TIMESTAMP").to_excel(
                        writer, index=False, sheet_name="Audit_Log"
                    )
            data = buffer.getvalue()

        st.download_button(
            label=f"Download Final Excel ({cycle_name})",
            data=data,
            file_name=f"Forecast_{cycle_name.replace(' ','_')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    else:
        st.info("Run forecasts first.")

with tabs[6]:  # Assuming Testing is the 7th tab
    st.header("Testing: In-Market Forecast")

    test_run_btn = st.button("Run Test Forecast")
    st.write("Button clicked:", test_run_btn)

    if test_run_btn:
        if st.session_state.ims_df is None:
            st.warning("Upload IMS first.")
        else:
            with st.spinner("Running test forecast..."):
                monthly, master_enriched, actuals = prepare_monthly_and_master(st.session_state.ims_df)
                st.session_state.monthly_df = monthly
                st.session_state.master_enriched = master_enriched

                st.write("Monthly DF:", monthly)
                st.write("Actuals DF:", actuals)

                final_inmarket, model_metrics, final_combined = run_in_market_forecast(
                    monthly_df=monthly,
                    actuals_df=actuals,
                    horizon_months=slider  # Make sure slider is defined
                )

                st.session_state.final_inmarket = final_inmarket
                st.session_state.final_combined = final_combined
                st.session_state.model_metrics = model_metrics

                st.session_state.master_df = final_combined.copy()
                st.session_state.base_master_df = final_combined.copy()

                if st.session_state.edited_master_df.empty:
                    st.session_state.edited_master_df = st.session_state.base_master_df.copy()

            if final_inmarket is None or final_inmarket.empty:
                st.error("No forecast generated. Check data coverage.")
            else:
                st.success("Test forecast generated and Master dataset initialized.")

    if st.session_state.final_combined is not None and not st.session_state.final_combined.empty:
        st.write("Rendering final_combined...")
        try:
            wide = pivot_wide_by_type(st.session_state.final_combined)
            styled_wide = style_numeric_dataframe(wide)
            st.dataframe(styled_wide, use_container_width=True)
        except Exception as e:
            st.error(f"Error rendering dataframe: {e}")
