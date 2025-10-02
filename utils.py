import pandas as pd
import streamlit as st
from typing import Dict, List, Any, Optional
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json

def format_currency(value: float) -> str:
    if pd.isna(value):
        return "N/A"
    if value >= 1_000_000:
        return f"${value/1_000_000:.1f}M"
    elif value >= 1_000:
        return f"${value/1_000:.0f}K"
    else:
        return f"${value:.0f}"

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    numeric_columns = ['list_price', 'sqft', 'beds', 'baths', 'stories', 'year_built',
                      'price_per_sqft', 'lot_sqft', 'latitude', 'longitude']

    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    date_columns = ['list_date', 'last_sold_date', 'new_open_house_date']
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    return df

def generate_summary_stats(df: pd.DataFrame) -> Dict[str, Any]:
    stats = {}

    if 'list_price' in df.columns:
        price_data = df['list_price'].dropna()
        if not price_data.empty:
            stats['price'] = {
                'mean': price_data.mean(),
                'median': price_data.median(),
                'min': price_data.min(),
                'max': price_data.max(),
                'std': price_data.std()
            }

    if 'sqft' in df.columns:
        sqft_data = df['sqft'].dropna()
        if not sqft_data.empty:
            stats['sqft'] = {
                'mean': sqft_data.mean(),
                'median': sqft_data.median(),
                'min': sqft_data.min(),
                'max': sqft_data.max()
            }

    if 'price_per_sqft' in df.columns:
        ppsf_data = df['price_per_sqft'].dropna()
        if not ppsf_data.empty:
            stats['price_per_sqft'] = {
                'mean': ppsf_data.mean(),
                'median': ppsf_data.median()
            }

    if 'beds' in df.columns:
        beds_data = df['beds'].dropna()
        if not beds_data.empty:
            stats['beds'] = {
                'mean': beds_data.mean(),
                'mode': beds_data.mode()[0] if not beds_data.mode().empty else None,
                'distribution': beds_data.value_counts().to_dict()
            }

    if 'property_style' in df.columns:
        stats['property_types'] = df['property_style'].value_counts().to_dict()

    stats['total_properties'] = len(df)

    return stats

def create_price_heatmap(df: pd.DataFrame) -> go.Figure:
    if 'beds' not in df.columns or 'baths' not in df.columns or 'list_price' not in df.columns:
        return None

    pivot_data = df.pivot_table(
        values='list_price',
        index='beds',
        columns='baths',
        aggfunc='median'
    )

    fig = go.Figure(data=go.Heatmap(
        z=pivot_data.values,
        x=[f"{b} baths" for b in pivot_data.columns],
        y=[f"{b} beds" for b in pivot_data.index],
        colorscale='Viridis',
        text=[[format_currency(val) for val in row] for row in pivot_data.values],
        texttemplate="%{text}",
        textfont={"size": 10},
        hovertemplate="Beds: %{y}<br>Baths: %{x}<br>Median Price: %{text}<extra></extra>"
    ))

    fig.update_layout(
        title="Median Price by Beds and Baths",
        xaxis_title="Bathrooms",
        yaxis_title="Bedrooms",
        height=400
    )

    return fig

def create_time_series_plot(df: pd.DataFrame, date_col: str = 'list_date') -> go.Figure:
    if date_col not in df.columns or 'list_price' not in df.columns:
        return None

    df_copy = df.copy()
    df_copy[date_col] = pd.to_datetime(df_copy[date_col])
    df_copy = df_copy.dropna(subset=[date_col, 'list_price'])

    if df_copy.empty:
        return None

    daily_avg = df_copy.groupby(df_copy[date_col].dt.date)['list_price'].agg(['mean', 'count']).reset_index()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=daily_avg[date_col],
        y=daily_avg['mean'],
        mode='lines+markers',
        name='Average Price',
        line=dict(width=2),
        hovertemplate="Date: %{x}<br>Avg Price: $%{y:,.0f}<br>Count: %{customdata}<extra></extra>",
        customdata=daily_avg['count']
    ))

    fig.update_layout(
        title=f"Average Price Over Time",
        xaxis_title="Date",
        yaxis_title="Average Price ($)",
        height=400,
        hovermode='x unified'
    )

    return fig

def create_property_comparison_table(df: pd.DataFrame, columns: List[str] = None) -> pd.DataFrame:
    if columns is None:
        columns = ['street', 'city', 'state', 'list_price', 'sqft', 'beds', 'baths',
                  'price_per_sqft', 'property_style', 'year_built']

    available_cols = [col for col in columns if col in df.columns]

    if not available_cols:
        return df.head(10)

    comparison_df = df[available_cols].head(20)

    if 'list_price' in comparison_df.columns:
        comparison_df['list_price'] = comparison_df['list_price'].apply(lambda x: format_currency(x) if pd.notna(x) else 'N/A')

    if 'price_per_sqft' in comparison_df.columns:
        comparison_df['price_per_sqft'] = comparison_df['price_per_sqft'].apply(lambda x: f"${x:.0f}" if pd.notna(x) else 'N/A')

    if 'sqft' in comparison_df.columns:
        comparison_df['sqft'] = comparison_df['sqft'].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else 'N/A')

    return comparison_df

def export_to_excel_with_styling(df: pd.DataFrame, filename: str) -> bytes:
    from io import BytesIO

    output = BytesIO()

    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Properties', index=False)

        workbook = writer.book
        worksheet = writer.sheets['Properties']

        header_format = workbook.add_format({
            'bold': True,
            'text_wrap': True,
            'valign': 'top',
            'fg_color': '#D7E4BD',
            'border': 1
        })

        for col_num, value in enumerate(df.columns.values):
            worksheet.write(0, col_num, value, header_format)

        for i, col in enumerate(df.columns):
            column_len = df[col].astype(str).str.len().max()
            column_len = max(column_len, len(col)) + 2
            worksheet.set_column(i, i, min(column_len, 50))

    output.seek(0)
    return output.getvalue()

def calculate_affordability_metrics(df: pd.DataFrame, monthly_income: float = None) -> Dict[str, Any]:
    if 'list_price' not in df.columns:
        return {}

    metrics = {}

    avg_price = df['list_price'].mean()
    median_price = df['list_price'].median()

    down_payment_20 = median_price * 0.20
    loan_amount = median_price * 0.80

    monthly_payment_30yr_5pct = (loan_amount * (0.05/12) * (1 + 0.05/12)**360) / ((1 + 0.05/12)**360 - 1)

    metrics['median_price'] = median_price
    metrics['down_payment_20'] = down_payment_20
    metrics['estimated_monthly_payment'] = monthly_payment_30yr_5pct

    if monthly_income:
        metrics['affordable_price'] = monthly_income * 0.28 * 360
        metrics['can_afford_median'] = monthly_income * 0.28 >= monthly_payment_30yr_5pct

    return metrics

def generate_market_insights(df: pd.DataFrame) -> List[str]:
    insights = []

    if 'list_price' in df.columns:
        avg_price = df['list_price'].mean()
        median_price = df['list_price'].median()

        if avg_price > median_price * 1.2:
            insights.append(f"⚠️ High-priced outliers detected: Average (${avg_price:,.0f}) is significantly higher than median (${median_price:,.0f})")

    if 'price_per_sqft' in df.columns:
        ppsf = df['price_per_sqft'].dropna()
        if not ppsf.empty:
            avg_ppsf = ppsf.mean()
            insights.append(f"💰 Average price per square foot: ${avg_ppsf:.0f}")

    if 'days_on_market' in df.columns:
        dom = df['days_on_market'].dropna()
        if not dom.empty:
            avg_dom = dom.mean()
            if avg_dom < 30:
                insights.append(f"🔥 Hot market: Properties average {avg_dom:.0f} days on market")
            elif avg_dom > 90:
                insights.append(f"❄️ Slow market: Properties average {avg_dom:.0f} days on market")

    if 'property_style' in df.columns:
        top_style = df['property_style'].value_counts().head(1)
        if not top_style.empty:
            insights.append(f"🏠 Most common property type: {top_style.index[0]} ({top_style.values[0]} properties)")

    if 'year_built' in df.columns:
        year_built = df['year_built'].dropna()
        if not year_built.empty:
            newest = year_built.max()
            oldest = year_built.min()
            insights.append(f"📅 Properties built between {oldest:.0f} and {newest:.0f}")

    return insights

# === OM PDF + COMPS COMPARISON UTILITIES (append to bottom of utils.py) ===
from typing import Tuple
from pdf_om_parser import parse_om_pdf
from homeharvest_comps import get_rent_comps, average_rent

def compare_om_to_market(pdf_path: str, radius_miles: float = 3.0, limit: int = 200) -> Dict[str, Any]:
    """
    Parse an OM PDF and compare OM avg rent to nearby rental comps (via homeharvest).
    Returns a dict suitable for display in Streamlit.
    """
    parsed = parse_om_pdf(pdf_path)

    comps_avg = None
    comps = []
    if parsed.location_str:
        comps = get_rent_comps(location=parsed.location_str, radius_miles=radius_miles, limit=limit)
        comps_avg = average_rent(comps)

    rent_delta = None
    if parsed.om_avg_rent is not None and comps_avg is not None:
        rent_delta = parsed.om_avg_rent - comps_avg

    return {
        "pdf_path": parsed.pdf_path,
        "location": parsed.location_str,
        "city": parsed.city,
        "state": parsed.state,
        "zip": parsed.zip,
        "value_usd": parsed.value_usd,
        "year_built": parsed.year_built,
        "year_renovated": parsed.year_renovated,
        "om_avg_rent": parsed.om_avg_rent,
        "comps_avg_rent": comps_avg,
        "rent_delta": rent_delta,
        "raw_comps": comps,  # for optional tables/plots
    }
# === OM Benchmarking helpers (append) ===
"1BR": "1BR",
"2BR": "2BR",
"2BR/2BA": "2BR",
"3BR": "3BR",
}


def _cohort_from_row(row: Dict) -> str:
label = (row.get("label") or "").strip()
return COHORT_MAP.get(label, f"{int(row.get('beds') or 0)}BR")




def unit_weighted_avg(unit_mix: list[dict], field: str = "rent") -> float | None:
df = pd.DataFrame(unit_mix)
if df.empty or field not in df.columns:
return None
df = df.dropna(subset=[field])
if df.empty:
return None
w = df["units"].fillna(1)
return float((df[field] * w).sum() / w.sum())




def benchmark_om(pdf_path: str, radius_miles: float = 3.0, limit: int = 400) -> Dict[str, Any]:
om: OMParsed = parse_om_pdf(pdf_path)
comps = get_rent_comps(location=om.location_str or "", radius_miles=radius_miles, limit=limit)
by_br = comps.get("by_br", pd.DataFrame())


# Build OM cohorts
om_rows = om.unit_mix or []
for r in om_rows:
r["cohort"] = _cohort_from_row(r)
om_df = pd.DataFrame(om_rows)


# Aggregate OM by cohort (unit-weighted)
om_by = pd.DataFrame(columns=["cohort","units","om_rent","om_rent_psf"])
if not om_df.empty:
grp = om_df.groupby("cohort")
om_by = pd.DataFrame({
"cohort": grp.size().index,
"units": grp["units"].sum().values,
"om_rent": grp.apply(lambda g: (g["rent"]*g["units"].fillna(1)).sum()/g["units"].fillna(1).sum()).values,
"om_rent_psf": grp.apply(lambda g: ((g["rent"].fillna(0))/g["avg_sf"].replace(0, pd.NA)).mean(skipna=True)).values,
})


# Join with market stats
bench = om_by.merge(by_br.rename(columns={
"median_rent":"mkt_median",
"mean_rent":"mkt_mean",
"mean_rent_psf":"mkt_rent_psf"
}), on="cohort", how="left")


# Deltas & flags
if not bench.empty:
bench["delta_vs_median"] = bench["om_rent"] - bench["mkt_median"]
bench["pct_vs_median"] = bench["delta_vs_median"] / bench["mkt_median"]
bench["flag"] = bench.apply(lambda r: "↑ above market" if r["pct_vs_median"]>0.08 else ("↓ below market" if r["pct_vs_median"]<-0.08 else "≈ near market"), axis=1)


portfolio_avg_om = unit_weighted_avg(om.unit_mix, "rent")
portfolio_avg_psf = unit_weighted_avg([{**r, "rent_psf": r.get("rent")/(r.get("avg_sf") or pd.NA)} for r in om.unit_mix], "rent_psf") if (om.unit_mix) else None


return {
"meta": {
"location": om.location_str,
"value_usd": om.value_usd,
"year_built": om.year_built,
"year_renovated": om.year_renovated,
"om_avg_rent_excerpt": om.om_avg_rent,
},
"om_unit_mix": om_df,
"market_raw": comps.get("raw"),
"market_by_bed": by_br,
"benchmark": bench,
"portfolio_avg_om": portfolio_avg_om,
"portfolio_rent_psf": portfolio_avg_psf,
}
