import streamlit as st
import pandas as pd
from homeharvest import scrape_property
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import json

from utils import format_currency, benchmark_om

st.set_page_config(
    page_title="HuntingParty Real Estate Scraper",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🏠 HuntingParty Real Estate Scraper & OM Benchmarking")

# ================================
# Sidebar – Search Parameters
# ================================
with st.sidebar:
    st.header("📍 Search Parameters")

    location = st.text_input(
        "Location*",
        placeholder="e.g., San Diego, CA or 92101",
        help="Enter ZIP code, city name, full address, neighborhood, or county"
    )

    listing_type = st.selectbox(
        "Listing Type*",
        options=["for_sale", "for_rent", "sold", "pending"],
        help="Select the type of property listing"
    )

    st.subheader("🔧 Advanced Filters")

    with st.expander("Property Filters", expanded=False):
        property_types = st.multiselect(
            "Property Types",
            options=[
                "single_family", "townhomes", "condos", "multi_family",
                "condo_townhome", "condo_townhome_rowhome_coop", "duplex_triplex",
                "farm", "land", "mobile"
            ],
            help="Select property types (leave empty for all)"
        )

        col1, col2 = st.columns(2)
        with col1:
            min_price = st.number_input("Min Price ($)", min_value=0, value=0, step=10000)
        with col2:
            max_price = st.number_input("Max Price ($)", min_value=0, value=0, step=10000)

        col1, col2 = st.columns(2)
        with col1:
            min_sqft = st.number_input("Min Sq Ft", min_value=0, value=0, step=100)
        with col2:
            max_sqft = st.number_input("Max Sq Ft", min_value=0, value=0, step=100)

        col1, col2 = st.columns(2)
        with col1:
            min_beds = st.number_input("Min Beds", min_value=0, max_value=10, value=0)
        with col2:
            max_beds = st.number_input("Max Beds", min_value=0, max_value=10, value=0)

        col1, col2 = st.columns(2)
        with col1:
            min_baths = st.number_input("Min Baths", min_value=0.0, max_value=10.0, value=0.0, step=0.5)
        with col2:
            max_baths = st.number_input("Max Baths", min_value=0.0, max_value=10.0, value=0.0, step=0.5)

    with st.expander("Search Options", expanded=False):
        radius = st.slider("Search Radius (miles)", 0, 50, 0)
        mls_only = st.checkbox("MLS Listings Only", value=False)
        foreclosure = st.checkbox("Foreclosure Properties Only", value=False)

        limit = st.number_input("Max Results", min_value=1, max_value=10000, value=100, step=10)

    with st.expander("Date Filters", expanded=False):
        use_date_filter = st.checkbox("Enable Date Filter")

        if use_date_filter:
            date_filter_type = st.radio("Filter Type", ["Past Days", "Date Range"])
            if date_filter_type == "Past Days":
                past_days = st.number_input("Past Days", min_value=1, max_value=365, value=30)
                date_from = None
                date_to = None
            else:
                col1, col2 = st.columns(2)
                with col1:
                    date_from = st.date_input("From Date")
                with col2:
                    date_to = st.date_input("To Date")
                past_days = None
        else:
            past_days = None
            date_from = None
            date_to = None

    with st.expander("Output Options", expanded=False):
        return_type = st.selectbox("Return Type", ["pandas", "pydantic", "raw"])
        show_raw_data = st.checkbox("Show Raw Data Table", value=True)
        show_visualizations = st.checkbox("Show Visualizations", value=True)
        show_statistics = st.checkbox("Show Statistics", value=True)

    scrape_button = st.button("🔍 Scrape Properties", type="primary", use_container_width=True)

# =====================================
# Realtor.com Scraper Section
# =====================================
if scrape_button:
    if not location:
        st.error("Please enter a location to search")
    else:
        try:
            with st.spinner(f"Scraping {listing_type} properties in {location}..."):
                kwargs = {"location": location, "listing_type": listing_type, "return_type": return_type, "limit": limit}
                if property_types: kwargs["property_type"] = property_types
                if radius > 0: kwargs["radius"] = radius
                if mls_only: kwargs["mls_only"] = True
                if foreclosure: kwargs["foreclosure"] = True
                if past_days: kwargs["past_days"] = past_days
                elif date_from and date_to:
                    kwargs["date_from"] = date_from.strftime("%Y-%m-%d")
                    kwargs["date_to"] = date_to.strftime("%Y-%m-%d")

                properties = scrape_property(**kwargs)
                if return_type == "pandas":
                    df = properties
                elif return_type == "pydantic":
                    df = pd.DataFrame([prop.dict() for prop in properties])
                else:
                    df = pd.DataFrame(properties)

                # Filters
                if max_price > 0 and 'list_price' in df.columns:
                    df = df[df['list_price'] <= max_price]
                if min_price > 0 and 'list_price' in df.columns:
                    df = df[df['list_price'] >= min_price]
                if max_sqft > 0 and 'sqft' in df.columns:
                    df = df[df['sqft'] <= max_sqft]
                if min_sqft > 0 and 'sqft' in df.columns:
                    df = df[df['sqft'] >= min_sqft]
                if max_beds > 0 and 'beds' in df.columns:
                    df = df[df['beds'] <= max_beds]
                if min_beds > 0 and 'beds' in df.columns:
                    df = df[df['beds'] >= min_beds]
                if max_baths > 0 and 'baths' in df.columns:
                    df = df[df['baths'] <= max_baths]
                if min_baths > 0 and 'baths' in df.columns:
                    df = df[df['baths'] >= min_baths]

                st.success(f"Successfully scraped {len(df)} properties!")

                # Downloads
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.download_button("📥 Download CSV", df.to_csv(index=False), file_name=f"properties_{location}.csv", mime="text/csv")
                with col2:
                    st.download_button("📥 Download Excel", df.to_excel("temp.xlsx"), file_name=f"properties_{location}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                with col3:
                    if return_type == "raw": json_str = json.dumps(properties, indent=2)
                    else: json_str = df.to_json(orient='records', indent=2)
                    st.download_button("📥 Download JSON", json_str, file_name=f"properties_{location}.json", mime="application/json")

                # Stats / Viz
                if show_statistics and not df.empty:
                    st.header("📊 Property Statistics")
                    col1, col2, col3, col4 = st.columns(4)
                    if 'list_price' in df.columns: st.metric("Average Price", f"${df['list_price'].mean():,.0f}")
                    if 'sqft' in df.columns: st.metric("Average Sq Ft", f"{df['sqft'].mean():,.0f}")
                    if 'beds' in df.columns: st.metric("Average Beds", f"{df['beds'].mean():.1f}")
                    if 'price_per_sqft' in df.columns: st.metric("Avg $/Sq Ft", f"${df['price_per_sqft'].mean():.0f}")

                if show_visualizations and not df.empty:
                    st.header("📈 Data Visualizations")
                    if 'list_price' in df.columns:
                        st.plotly_chart(px.histogram(df, x='list_price', nbins=20, title="Price Distribution"), use_container_width=True)
                    if 'property_style' in df.columns:
                        style_counts = df['property_style'].value_counts().head(10)
                        st.plotly_chart(px.pie(values=style_counts.values, names=style_counts.index, title="Property Types Distribution"), use_container_width=True)

                if show_raw_data:
                    st.header("📋 Raw Property Data")
                    st.dataframe(df, use_container_width=True, hide_index=True)

        except Exception as e:
            st.error(f"Error occurred: {str(e)}")

# =====================================
# OM vs Market Rents Benchmark Section
# =====================================
st.header("📑 OM vs Market Rents")
with st.container():
    colA, colB, colC = st.columns([2, 1, 1])
    with colA:
        uploaded = st.file_uploader("Upload Multifamily OM PDF(s)", type=["pdf"], accept_multiple_files=True)
    with colB:
        om_radius = st.slider("Comps radius (miles)", 1, 15, 3)
    with colC:
        om_limit = st.number_input("Max comps", min_value=50, max_value=2000, value=400, step=50)

    run_om = st.button("🔍 Analyze OMs", use_container_width=True)

    if run_om:
        if not uploaded:
            st.error("Please upload at least one PDF OM.")
        else:
            results = []
            for up in uploaded:
                tmp_path = f"/tmp/om_{up.name}"
                with open(tmp_path, "wb") as f:
                    f.write(up.read())
                with st.spinner(f"Parsing {up.name} and pulling comps..."):
                    res = benchmark_om(tmp_path, radius_miles=om_radius, limit=om_limit)
                    res["_file"] = up.name
                    results.append(res)

            for res in results:
                st.subheader(f"Report: {res['_file']}")
                meta = res["meta"]
                meta_cols = st.columns(4)
                meta_cols[0].metric("Location", meta.get("location") or "N/A")
                meta_cols[1].metric("Value", format_currency(meta.get("value_usd")) if meta.get("value_usd") else "N/A")
                meta_cols[2].metric("Year Built", str(meta.get("year_built")) if meta.get("year_built") else "N/A")
                meta_cols[3].metric("Year Reno", str(meta.get("year_renovated")) if meta.get("year_renovated") else "N/A")

                if isinstance(res["om_unit_mix"], pd.DataFrame) and not res["om_unit_mix"].empty:
                    st.write("**OM Unit Mix (parsed)**")
                    st.dataframe(res["om_unit_mix"], use_container_width=True, hide_index=True)

                if isinstance(res["benchmark"], pd.DataFrame) and not res["benchmark"].empty:
                    st.write("**Benchmark by Bedroom Cohort**")
                    display = res["benchmark"].copy()
                    display["om_rent"] = display["om_rent"].map(lambda v: f"${v:,.0f}" if pd.notna(v) else "N/A")
                    display["mkt_median"] = display["mkt_median"].map(lambda v: f"${v:,.0f}" if pd.notna(v) else "N/A")
                    display["delta_vs_median"] = display["delta_vs_median"].map(lambda v: f"${v:,.0f}" if pd.notna(v) else "N/A")
                    display["pct_vs_median"] = display["pct_vs_median"].map(lambda v: f"{v:+.1f}%" if pd.notna(v) else "N/A")
                    st.dataframe(display, use_container_width=True, hide_index=True)

                    csv = res["benchmark"].to_csv(index=False)
                    st.download_button("📥 Download Benchmark CSV", csv, file_name=f"benchmark_{res['_file']}.csv", mime="text/csv")
