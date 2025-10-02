import os
import re
from typing import Dict, Any, List, Optional
import pandas as pd
import pdfplumber
import cohere

# Initialize Cohere client
co = cohere.Client(os.getenv("COHERE_API_KEY"))

# =============== helpers ===============

def format_currency(val: Optional[float]) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "N/A"
    try:
        v = float(val)
    except Exception:
        return str(val)
    return f"${v:,.0f}"


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce numeric-like object columns to numeric, when possible."""
    for col in df.columns:
        if df[col].dtype == "object":
            s = df[col].astype(str)
            s = s.str.replace(",", "", regex=False).str.replace("$", "", regex=False)
            df[col] = pd.to_numeric(s, errors="ignore")
    return df


# =============== metadata parse ===============

def _parse_om_meta(text: str) -> Dict[str, Any]:
    """
    Crude metadata parser for value and years.
    Location varies across OMs; the main app allows manual override if missing.
    """
    meta = {"location": None, "value_usd": None, "year_built": None, "year_renovated": None}

    m_built = re.search(r"(?:Year\s*Built|Built)\s*[:\-]?\s*(19\d{2}|20\d{2})", text, re.I)
    if m_built:
        meta["year_built"] = int(m_built.group(1))

    m_reno = re.search(r"(?:Year\s*(?:Renovated|Remodeled|Upgraded))\s*[:\-]?\s*(19\d{2}|20\d{2})", text, re.I)
    if m_reno:
        meta["year_renovated"] = int(m_reno.group(1))

    m_val = re.search(
        r"(?:Asking|Offering|Purchase|Valuation|Value)\s*(?:Price)?\s*[:\-]?\s*\$?\s*([\d,]+(?:\.\d{1,2})?)\s*(million|mm|m)?",
        text, re.I
    )
    if m_val:
        raw, suffix = m_val.group(1), (m_val.group(2) or "").lower()
        try:
            num = float(raw.replace(",", ""))
            if suffix in {"million", "mm", "m"}:
                num *= 1_000_000
            meta["value_usd"] = num
        except Exception:
            pass

    # best-effort loose "City, ST ZIP" fallback for location
    m_loc = re.search(r"\b([A-Za-z .'-]+),\s*([A-Z]{2})\s*(\d{5})(?:-\d{4})?\b", text)
    if m_loc:
        meta["location"] = f"{m_loc.group(1)}, {m_loc.group(2)} {m_loc.group(3)}"

    return meta


# =============== unit-mix extraction ===============

def _parse_unit_mix_llm_cohere(tables: List[List[str]]) -> pd.DataFrame:
    """
    Use Cohere LLM to identify the UNIT MIX table and return structured cohort/unit/rent info.
    """
    prompt = f"""
    You are given tables extracted from a multifamily Offering Memorandum PDF.
    Identify the UNIT MIX table and extract average rents by unit type.

    Return JSON ONLY in the following form:
    [
      {{"cohort": "Studio", "units": 10, "avg_rent": 1200}},
      {{"cohort": "1BR", "units": 20, "avg_rent": 1450}},
      {{"cohort": "2BR", "units": 15, "avg_rent": 1850}}
    ]

    If you cannot find rent data, return [].

    Tables:
    {tables}
    """

    resp = co.generate(
        model="command-xlarge",
        prompt=prompt,
        max_tokens=500,
        temperature=0
    )

    import json
    try:
        rows = json.loads(resp.generations[0].text.strip())
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame(columns=["cohort", "units", "avg_rent"])


def extract_unit_mix_from_pdf(pdf_path: str) -> pd.DataFrame:
    """
    Try extracting unit mix (cohort/units/avg_rent) using tables + Cohere LLM fallback.
    """
    tables = []
    text_blocks = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            # collect tables
            for t in page.extract_tables() or []:
                clean_t = [[c for c in row if c is not None] for row in t if any(row)]
                if clean_t:
                    tables.append(clean_t)
            # collect text for crude regex fallback
            txt = page.extract_text() or ""
            text_blocks.append(txt)

    # LLM attempt
    if tables:
        df = _parse_unit_mix_llm_cohere(tables)
        if not df.empty:
            return df

    # fallback regex if LLM fails
    rows = []
    for txt in text_blocks:
        for line in txt.splitlines():
            m = re.match(r"(Studio|[1234])\s*BR.*?(\d+)\s*units?.*?\$([0-9,]+)", line, re.I)
            if m:
                cohort = m.group(1)
                units  = int(m.group(2))
                rent   = float(m.group(3).replace(",", ""))
                rows.append({"cohort": cohort, "units": units, "avg_rent": rent})
    return pd.DataFrame(rows)
