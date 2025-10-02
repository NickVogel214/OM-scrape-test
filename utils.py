# utils.py
import io
import re
from typing import Dict, List, Any, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pdfplumber
import usaddress
from homeharvest import scrape_property


# ============================================================
# Formatting helpers
# ============================================================

def format_currency(value: Optional[float]) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "N/A"
    try:
        v = float(value)
    except Exception:
        return str(value)
    if v >= 1_000_000:
        return f"${v/1_000_000:.1f}M"
    if v >= 1_000:
        return f"${v/1_000:.0f}K"
    return f"${v:.0f}"


# ============================================================
# Data cleaning & stats
# ============================================================

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    numeric_columns = [
        "list_price", "sqft", "beds", "baths", "stories", "year_built",
        "price_per_sqft", "lot_sqft", "latitude", "longitude"
    ]
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    date_columns = ["list_date", "last_sold_date", "new_open_house_date"]
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def generate_summary_stats(df: pd.DataFrame) -> Dict[str, Any]:
    stats: Dict[str, Any] = {}

    if "list_price" in df.columns:
        price_data = df["list_price"].dropna()
        if not price_data.empty:
            stats["price"] = {
                "mean": price_data.mean(),
                "median": price_data.median(),
                "min": price_data.min(),
                "max": price_data.max(),
                "std": price_data.std(),
            }

    if "sqft" in df.columns:
        sqft_data = df["sqft"].dropna()
        if not sqft_data.empty:
            stats["sqft"] = {
                "mean": sqft_data.mean(),
                "median": sqft_data.median(),
                "min": sqft_data.min(),
                "max": sqft_data.max(),
            }

    if "price_per_sqft" in df.columns:
        ppsf_data = df["price_per_sqft"].dropna()
        if not ppsf_data.empty:
            stats["price_per_sqft"] = {
                "mean": ppsf_data.mean(),
                "median": ppsf_data.median(),
            }

    if "beds" in df.columns:
        beds_data = df["beds"].dropna()
        if not beds_data.empty:
            mode = beds_data.mode()
            stats["beds"] = {
                "mean": beds_data.mean(),
                "mode": mode.iloc[0] if not mode.empty else None,
                "distribution": beds_data.value_counts().to_dict(),
            }

    if "property_style" in df.columns:
        stats["property_types"] = df["property_style"].value_counts().to_dict()

    stats["total_properties"] = len(df)
    return stats


# ============================================================
# Visualization helpers
# ============================================================

def create_price_heatmap(df: pd.DataFrame) -> Optional[go.Figure]:
    if not {"beds", "baths", "list_price"}.issubset(df.columns):
        return None

    pivot_data = df.pivot_table(
        values="list_price",
        index="beds",
        columns="baths",
        aggfunc="median"
    )

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot_data.values,
            x=[f"{b} baths" for b in pivot_data.columns],
            y=[f"{b} beds" for b in pivot_data.index],
            colorscale="Viridis",
            text=[[format_currency(val) for val in row] for row in pivot_data.values],
            texttemplate="%{text}",
            textfont={"size": 10},
            hovertemplate="Beds: %{y}<br>Baths: %{x}<br>Median Price: %{text}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Median Price by Beds and Baths",
        xaxis_title="Bathrooms",
        yaxis_title="Bedrooms",
        height=400,
    )
    return fig


def create_time_series_plot(df: pd.DataFrame, date_col: str = "list_date") -> Optional[go.Figure]:
    if date_col not in df.columns or "list_price" not in df.columns:
        return None

    df_copy = df.copy()
    df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors="coerce")
    df_copy = df_copy.dropna(subset=[date_col, "list_price"])
    if df_copy.empty:
        return None

    daily_avg = (
        df_copy
        .groupby(df_copy[date_col].dt.date)["list_price"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={date_col: "date"})
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=daily_avg["date"],
            y=daily_avg["mean"],
            mode="lines+markers",
            name="Average Price",
            line=dict(width=2),
            hovertemplate="Date: %{x}<br>Avg Price: $%{y:,.0f}<br>Count: %{customdata}<extra></extra>",
            customdata=daily_avg["count"],
        )
    )
    fig.update_layout(
        title="Average Price Over Time",
        xaxis_title="Date",
        yaxis_title="Average Price ($)",
        height=400,
        hovermode="x unified",
    )
    return fig


def create_property_comparison_table(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    if columns is None:
        columns = [
            "street", "city", "state", "list_price", "sqft", "beds", "baths",
            "price_per_sqft", "property_style", "year_built",
        ]

    available_cols = [c for c in columns if c in df.columns]
    if not available_cols:
        return df.head(10)

    comparison_df = df[available_cols].head(20).copy()

    if "list_price" in comparison_df.columns:
        comparison_df["list_price"] = comparison_df["list_price"].apply(
            lambda x: format_currency(x) if pd.notna(x) else "N/A"
        )

    if "price_per_sqft" in comparison_df.columns:
        comparison_df["price_per_sqft"] = comparison_df["price_per_sqft"].apply(
            lambda x: f"${x:.0f}" if pd.notna(x) else "N/A"
        )

    if "sqft" in comparison_df.columns:
        comparison_df["sqft"] = comparison_df["sqft"].apply(
            lambda x: f"{x:,.0f}" if pd.notna(x) else "N/A"
        )

    return comparison_df


def export_to_excel_with_styling(df: pd.DataFrame, filename: str) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name="Properties", index=False)

        workbook = writer.book
        worksheet = writer.sheets["Properties"]

        header_format = workbook.add_format({
            "bold": True,
            "text_wrap": True,
            "valign": "top",
            "fg_color": "#D7E4BD",
            "border": 1,
        })

        for col_num, value in enumerate(df.columns.values):
            worksheet.write(0, col_num, value, header_format)

        for i, col in enumerate(df.columns):
            column_len = df[col].astype(str).str.len().max()
            column_len = max(column_len, len(col)) + 2
            worksheet.set_column(i, i, min(column_len, 50))

    output.seek(0)
    return output.getvalue()


# ============================================================
# Affordability & insights
# ============================================================

def calculate_affordability_metrics(df: pd.DataFrame, monthly_income: Optional[float] = None) -> Dict[str, Any]:
    if "list_price" not in df.columns or df["list_price"].dropna().empty:
        return {}

    metrics: Dict[str, Any] = {}
    median_price = df["list_price"].median()

    down_payment_20 = median_price * 0.20
    loan_amount = median_price * 0.80
    monthly_payment_30yr_5pct = (
        loan_amount * (0.05 / 12) * (1 + 0.05 / 12) ** 360
    ) / ((1 + 0.05 / 12) ** 360 - 1)

    metrics["median_price"] = median_price
    metrics["down_payment_20"] = down_payment_20
    metrics["estimated_monthly_payment"] = monthly_payment_30yr_5pct

    if monthly_income:
        metrics["affordable_price"] = monthly_income * 0.28 * 360
        metrics["can_afford_median"] = monthly_income * 0.28 >= monthly_payment_30yr_5pct

    return metrics


def generate_market_insights(df: pd.DataFrame) -> List[str]:
    insights: List[str] = []

    if "list_price" in df.columns and not df["list_price"].dropna().empty:
        avg_price = df["list_price"].mean()
        median_price = df["list_price"].median()
        if avg_price > median_price * 1.2:
            insights.append(
                f"⚠️ High-priced outliers detected: Average (${avg_price:,.0f}) "
                f"is significantly higher than median (${median_price:,.0f})"
            )

    if "price_per_sqft" in df.columns and not df["price_per_sqft"].dropna().empty:
        avg_ppsf = df["price_per_sqft"].mean()
        insights.append(f"💰 Average price per square foot: ${avg_ppsf:.0f}")

    if "days_on_market" in df.columns and not df["days_on_market"].dropna().empty:
        avg_dom = df["days_on_market"].mean()
        if avg_dom < 30:
            insights.append(f"🔥 Hot market: Properties average {avg_dom:.0f} days on market")
        elif avg_dom > 90:
            insights.append(f"❄️ Slow market: Properties average {avg_dom:.0f} days on market")

    if "property_style" in df.columns and not df["property_style"].dropna().empty:
        top_style = df["property_style"].value_counts().head(1)
        if not top_style.empty:
            insights.append(f"🏠 Most common property type: {top_style.index[0]} ({top_style.values[0]} properties)")

    if "year_built" in df.columns and not df["year_built"].dropna().empty:
        newest = df["year_built"].max()
        oldest = df["year_built"].min()
        insights.append(f"📅 Properties built between {oldest:.0f} and {newest:.0f}")

    return insights


# ============================================================
# OM PDF parsing + Realtor.com benchmarking
# ============================================================

# Map bedroom counts to cohorts we’ll compare on
_BR_LABEL = {0: "Studio", 1: "1BR", 2: "2BR", 3: "3BR", 4: "4BR+"}


def _bucket_beds_to_cohort(val: Any) -> Optional[str]:
    try:
        b = int(float(val))
    except Exception:
        return None
    if b <= 0:
        return _BR_LABEL[0]
    if b == 1:
        return _BR_LABEL[1]
    if b == 2:
        return _BR_LABEL[2]
    if b == 3:
        return _BR_LABEL[3]
    return _BR_LABEL[4]


def _safe_float(s: Any) -> Optional[float]:
    if s is None:
        return None
    try:
        # strip $ and commas etc.
        x = re.sub(r"[^0-9.\-]", "", str(s))
        if x == "":
            return None
        return float(x)
    except Exception:
        return None


def _parse_om_meta(text: str) -> Dict[str, Any]:
    """Parse basic fields: value, years, and a best-effort location string."""
    meta: Dict[str, Any] = {
        "location": None,
        "value_usd": None,
        "year_built": None,
        "year_renovated": None,
        "om_avg_rent_excerpt": None,
    }

    # Value (purchase/offering)
    m_val = re.search(
        r"(?:asking|offer(?:ing)?|purchase|valuation|value)\s*(?:price)?\s*[:\-]?\s*\$?([\d,]+(?:\.\d{1,2})?)\s*(million|mm|m)?",
        text,
        re.I,
    )
    if m_val:
        raw, suffix = m_val.group(1), (m_val.group(2) or "").lower()
        num = _safe_float(raw)
        if num is not None:
            if suffix in {"million", "mm", "m"}:
                num *= 1_000_000
            meta["value_usd"] = num

    # Years
    m_built = re.search(r"(?:year\s*(?:built|completed))\s*[:\-]?\s*(19\d{2}|20\d{2})", text, re.I)
    m_reno = re.search(r"(?:year\s*(?:renovated|remodeled|rehab(?:bed)?|upgraded))\s*[:\-]?\s*(19\d{2}|20\d{2})", text, re.I)
    if m_built:
        meta["year_built"] = int(m_built.group(1))
    if m_reno:
        meta["year_renovated"] = int(m_reno.group(1))

    # OM avg rent excerpt (if present in text)
    m_rent = re.search(
        r"(?:avg|average|in\s*place|asking)\s*(?:\w+\s*)*rent(?:\s*\(per\s*unit\))?\s*[:\-]?\s*\$?([\d,]+(?:\.\d{1,2})?)",
        text,
        re.I,
    )
    if m_rent:
        meta["om_avg_rent_excerpt"] = _safe_float(m_rent.group(1))

    # Location heuristic (use first chunk of text; try usaddress)
    try:
        tagged, _ = usaddress.tag(text[:1000])
        # Compose a readable location if possible
        parts = [tagged.get(k) for k in ("AddressNumber", "StreetName", "StreetNamePostType", "PlaceName", "StateName", "ZipCode")]
        parts = [p for p in parts if p]
        meta["location"] = " ".join(parts) if parts else None
    except Exception:
        pass

    # If that failed, try looser City, ST ZIP
    if not meta["location"]:
        m_loc = re.search(r"\b([A-Za-z .'-]+),\s*([A-Z]{2})\s*(\d{5})(?:-\d{4})?\b", text)
        if m_loc:
            meta["location"] = f"{m_loc.group(1)}, {m_loc.group(2)} {m_loc.group(3)}"

    return meta


def _parse_om_unit_mix_lines(text: str) -> pd.DataFrame:
    """
    Best-effort parser for simple unit-mix lines, e.g.:
    "1BR - 24 units - $1,950"
    "2 BR / 2 BA ... Units: 12 ... Avg Rent: $2,450"
    "Studio ... Units 10 ... Rent $1,650"
    """
    rows: List[Dict[str, Any]] = []

    for line in text.splitlines():
        s = line.strip().lower()
        if not s:
            continue

        # Try to detect cohort first
        cohort = None
        if re.search(r"\bstudio\b|^std\b|efficiency", s):
            cohort = "Studio"
        elif re.search(r"\b1\s*br\b|\b1\s*bed\b|\bone bedroom\b", s):
            cohort = "1BR"
        elif re.search(r"\b2\s*br\b|\b2\s*bed\b|\btwo bedroom\b", s):
            cohort = "2BR"
        elif re.search(r"\b3\s*br\b|\b3\s*bed\b|\bthree bedroom\b", s):
            cohort = "3BR"
        elif re.search(r"\b4\s*br\b|\b4\s*bed\b|\bfour bedroom\b", s):
            cohort = "4BR+"

        if cohort is None:
            continue

        # Units
        m_units = re.search(r"(\d+)\s*units?", s)
        units = int(m_units.group(1)) if m_units else None

        # Avg rent
        m_rent = re.search(r"\$\s*([\d,]+(?:\.\d{1,2})?)", s)
        rent = _safe_float(m_rent.group(1)) if m_rent else None
        if rent and rent > 10_000:  # annual → monthly heuristic
            rent = rent / 12.0

        if units or rent:
            rows.append({"cohort": cohort, "units": units, "avg_rent": rent})

    if not rows:
        return pd.DataFrame(columns=["cohort", "units", "avg_rent"])

    # Aggregate duplicates (sum units, average rent weighted by units if present)
    df = pd.DataFrame(rows)
    if "units" not in df.columns:
        df["units"] = 1

    def _weighted_avg(g):
        g2 = g.dropna(subset=["avg_rent"])
        if g2.empty:
            return None
        w = g2["units"].fillna(1)
        return float((g2["avg_rent"] * w).sum() / w.sum())

    out = (
        df.groupby("cohort")
        .agg(units=("units", "sum"), avg_rent=("avg_rent", _weighted_avg))
        .reset_index()
    )
    return out


def benchmark_om(pdf_path: str, radius_miles: float = 3.0, limit: int = 400) -> Dict[str, Any]:
    """
    Parse an Offering Memorandum (PDF), extract:
      - location, value, year built/renovated,
      - unit-mix (cohort → units, avg_rent)

    Fetch nearby for-rent listings via homeharvest (Realtor.com),
    compute median rent per bedroom cohort, and compare OM vs market.
    """
    # --- Extract text from PDF ---
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            t = page.extract_text(x_tolerance=2, y_tolerance=2) or ""
            text += t + "\n"

    # --- Parse OM meta & unit mix ---
    meta = _parse_om_meta(text)
    om_unit_mix = _parse_om_unit_mix_lines(text)

    # --- Fetch comps (for_rent) near OM location ---
    comps_df = pd.DataFrame()
    if meta.get("location"):
        try:
            comps_df = scrape_property(
                location=meta["location"],
                listing_type="for_rent",
                return_type="pandas",
                radius=radius_miles,
                limit=limit
            )
        except Exception:
            comps_df = pd.DataFrame()

    if not isinstance(comps_df, pd.DataFrame):
        comps_df = pd.DataFrame()
    comps_df = clean_dataframe(comps_df)

    # Normalize comps: rent and cohort
    if "list_price" in comps_df.columns:
        comps_df = comps_df[comps_df["list_price"].notna()].copy()
        comps_df["rent"] = comps_df["list_price"]
    else:
        comps_df["rent"] = pd.NA

    if "beds" in comps_df.columns:
        comps_df["cohort"] = comps_df["beds"].apply(_bucket_beds_to_cohort)
    else:
        comps_df["cohort"] = None

    # Aggregate market by cohort (median is more robust vs outliers)
    market_by = (
        comps_df.dropna(subset=["cohort", "rent"])
        .groupby("cohort")
        .agg(
            n=("rent", "count"),
            mean_rent=("rent", "mean"),
            median_rent=("rent", "median"),
        )
        .reset_index()
    )

    # --- Join OM vs Market by cohort ---
    bench = pd.DataFrame()
    if not om_unit_mix.empty and not market_by.empty:
        bench = om_unit_mix.merge(
            market_by.rename(
                columns={
                    "median_rent": "mkt_median",
                    "mean_rent": "mkt_mean",
                }
            ),
            on="cohort",
            how="left",
        )

        # deltas
        if "avg_rent" in bench.columns and "mkt_median" in bench.columns:
            bench["delta_vs_median"] = bench["avg_rent"] - bench["mkt_median"]
            bench["pct_vs_median"] = bench["delta_vs_median"] / bench["mkt_median"]
            # simple flags
            def _flag(row):
                p = row.get("pct_vs_median")
                if p is None or pd.isna(p):
                    return "n/a"
                if p > 0.08:
                    return "↑ above market"
                if p < -0.08:
                    return "↓ below market"
                return "≈ near market"
            bench["flag"] = bench.apply(_flag, axis=1)

    return {
        "meta": meta,                   # dict
        "om_unit_mix": om_unit_mix,     # DataFrame: cohort, units, avg_rent
        "market_raw": comps_df,         # DataFrame
        "market_by_bed": market_by,     # DataFrame: cohort stats
        "benchmark": bench,             # DataFrame: join + deltas/flags
    }
