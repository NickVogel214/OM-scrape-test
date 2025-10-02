import pdfplumber
import re
import pandas as pd
from typing import Dict, Any, List, Optional
from openai import OpenAI
import os

# Initialize LLM client (requires OPENAI_API_KEY in env)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --------------------------
# Helpers
# --------------------------

def format_currency(value: float) -> str:
    if pd.isna(value):
        return "N/A"
    return f"${value:,.0f}"

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce numeric-looking strings into numbers where possible."""
    for col in df.columns:
        if df[col].dtype == "object":
            try:
                df[col] = (
                    df[col]
                    .astype(str)
                    .str.replace(",", "")
                    .str.replace("$", "")
                )
                df[col] = pd.to_numeric(df[col], errors="ignore")
            except Exception:
                pass
    return df

# --------------------------
# Meta info extraction
# --------------------------

def _parse_om_meta(text: str) -> Dict[str, Any]:
    """Pulls location, value, year built/reno from OM text crudely."""
    meta = {"location": None, "value_usd": None, "year_built": None, "year_renovated": None}

    year = re.search(r"Year Built[: ]+(\d{4})", text, re.I)
    reno = re.search(r"(Renovated|Remodeled)[: ]+(\d{4})", text, re.I)
    val  = re.search(r"\$([0-9,.]+)", text)

    if year:
        meta["year_built"] = int(year.group(1))
    if reno:
        meta["year_renovated"] = int(reno.group(2))
    if val:
        try:
            meta["value_usd"] = float(val.group(1).replace(",", ""))
        except:
            pass
    return meta

# --------------------------
# Unit Mix Extraction
# --------------------------

def _parse_unit_mix_llm(tables: List[List[str]]) -> pd.DataFrame:
    """
    Send candidate tables to LLM to decide which one is the unit mix,
    and return structured cohort/unit/rent info.
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

    If you cannot find rent data, return an empty list [].

    Tables:
    {tables}
    """

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "You are a data parser."},
                  {"role": "user", "content": prompt}],
        temperature=0
    )

    import json
    try:
        rows = json.loads(resp.choices[0].message.content.strip())
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame(columns=["cohort", "units", "avg_rent"])

def extract_unit_mix_from_pdf(pdf_path: str) -> pd.DataFrame:
    """
    Try extracting unit mix (cohort/units/avg_rent) using tables + LLM fallback.
    """
    tables = []
    text_blocks = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            # collect tables
            for t in page.extract_tables():
                clean_t = [[c for c in row if c is not None] for row in t if any(row)]
                if clean_t:
                    tables.append(clean_t)
            # collect text for crude regex fallback
            txt = page.extract_text() or ""
            text_blocks.append(txt)

    if tables:
        df = _parse_unit_mix_llm(tables)
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
