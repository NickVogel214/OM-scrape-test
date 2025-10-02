import streamlit as st
import pandas as pd
import pdfplumber
from homeharvest import scrape_property
from utils import clean_dataframe, format_currency, extract_unit_mix_from_pdf, _parse_om_meta

st.set_page_config(
    page_title="OM Benchmarking",
    page_icon="🏢",
    layout="wide"
)

st.title("🏢 Multifamily OM Benchmarking")
st.write("Upload an Offering Memorandum (PDF). The app will parse its data and benchmark against Realtor.com comps automatically.")

uploaded = st.file_uploader("Upload OM PDF", type=["pdf"])

if uploaded:
    tmp_path = f"/tmp/{uploaded.name}"
    with open(tmp_path, "wb") as f:
        f.write(uploaded.read())

    # -----------------------------
    # Extract text from PDF
    # -----------------------------
    text = ""
    with pdfplumber.open(tmp_path) as pdf:
        for page in pdf.pages:
            t = page.extract_text(x_tolerance=2, y_tolerance=2) or ""
            text += t + "\n"

    # -----------------------------
    # Parse OM metadata + unit mix
    # -----------------------------
    meta = _parse_om_meta(text)
    om_unit_mix = extract_unit_mix_from_pdf(tmp_path)

    st.subheader("📋 OM Metadata")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Location", meta.get("location") or "N/A")
    col2.metric("Value", format_currency(meta.get("value_usd")) if meta.get("value_usd") else "N/A")
    col3.metric("Year Built", str(meta.get("year_built")) if meta.get("year_built") else "N/A")
    col4.metric("Renovated", str(meta.get("year_renovated")) if meta.get("year_renovated") else "N/A")

    if not om_unit_mix.empty:
        st.write("**OM Unit Mix**")
        st.dataframe(om_unit_mix, use_container_width=True, hide_index=True)

    # -----------------------------
    # Scrape comps from Realtor.com
    # -----------------------------
    st.subheader("📊 Market Benchmarking")

    comps_location = meta.get("location") or st.text_input(
        "Detected no location in OM. Please enter one manually:", placeholder="e.g., Austin, TX"
    )

    if comps_location:
        with st.spinner("Scraping Realtor.com comps..."):
            comps = scrape_property(
                location=comps_location,
                listing_type="for_rent",
                return_type="pandas",
                radius=3,
                limit=400
            )
            comps = clean_dataframe(comps)

        if not comps.empty and "beds" in comps.columns and "list_price" in comps.columns:
            comps = comps[comps["list_price"].notna()].copy()
            comps["rent"] = comps["list_price"]

            # Market stats by cohort
            def bucket(beds):
                if beds == 0: return "Studio"
                if beds == 1: return "1BR"
                if beds == 2: return "2BR"
                if beds == 3: return "3BR"
                return "4BR+"

            comps["cohort"] = comps["beds"].apply(bucket)
            market_by = (
                comps.groupby("cohort")
                .agg(median_rent=("rent","median"), mean_rent=("rent","mean"), n=("rent","count"))
                .reset_index()
            )

            st.write("**Market Rent Benchmarks**")
            st.dataframe(market_by, use_container_width=True, hide_index=True)

            # -----------------------------
            # Compare OM vs Market
            # -----------------------------
            if not om_unit_mix.empty:
                merged = om_unit_mix.merge(market_by, on="cohort", how="left")
                merged["delta_vs_median"] = merged["avg_rent"] - merged["median_rent"]
                merged["pct_vs_median"] = merged["delta_vs_median"] / merged["median_rent"]

                merged["flag"] = merged["pct_vs_median"].apply(
                    lambda p: "↑ Above Market" if p > 0.08 else ("↓ Below Market" if p < -0.08 else "≈ Near Market")
                )

                # Pretty formatting
                display = merged.copy()
                display["om_rent"] = display["avg_rent"].map(lambda v: f"${v:,.0f}" if pd.notna(v) else "N/A")
                display["median_rent"] = display["median_rent"].map(lambda v: f"${v:,.0f}" if pd.notna(v) else "N/A")
                display["delta_vs_median"] = display["delta_vs_median"].map(lambda v: f"${v:,.0f}" if pd.notna(v) else "N/A")
                display["pct_vs_median"] = display["pct_vs_median"].map(lambda v: f"{v*100:+.1f}%" if pd.notna(v) else "N/A")

                st.write("**OM vs Market Rents**")
                st.dataframe(
                    display[["cohort","units","om_rent","median_rent","delta_vs_median","pct_vs_median","flag"]],
                    use_container_width=True, hide_index=True
                )

        else:
            st.warning("No comps pulled from Realtor.com for this location.")
