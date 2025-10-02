# pdf_om_parser.py
from __future__ import annotations

import re, math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple
import pdfplumber, usaddress

PATTERNS = {
    "year_built": re.compile(r"(?:year\s*(?:built|completed))\s*[:\-]?\s*(19\d{2}|20\d{2})", re.I),
    "year_renovated": re.compile(r"(?:year\s*(?:renovated|remodeled|rehab(?:bed)?|upgraded))\s*[:\-]?\s*(19\d{2}|20\d{2})", re.I),
    "value": re.compile(r"(?:asking|offer(?:ing)?|purchase|valuation|value)\s*(?:price)?\s*[:\-]?\s*\$?([\d,]+(?:\.\d{1,2})?)\s*(million|mm|m)?", re.I),
    "avg_rent": re.compile(r"(?:avg|average|in\s*place|asking)\s*(?:\w+\s*)*rent(?:\s*\(per\s*unit\))?\s*[:\-]?\s*\$?([\d,]+(?:\.\d{1,2})?)", re.I),
    "address_line": re.compile(r"(\d{1,6}[^\n,]*?\b(?:St|Ave|Blvd|Rd|Road|Drive|Dr|Lane|Ln|Court|Ct|Way|Terrace|Ter|Place|Pl|Highway|Hwy)\b[^\n,]*,\s*[A-Za-z .'-]+,\s*[A-Z]{2}\s*\d{5})(?:-\d{4})?", re.I),
    "city_state_zip": re.compile(r"\b([A-Za-z .'-]+),\s*([A-Z]{2})\s*(\d{5})(?:-\d{4})?\b"),
}

@dataclass
class OMParsed:
    pdf_path: str
    location_str: Optional[str]
    city: Optional[str]
    state: Optional[str]
    zip: Optional[str]
    value_usd: Optional[float]
    year_built: Optional[int]
    year_renovated: Optional[int]
    om_avg_rent: Optional[float]
    comps_avg_rent: Optional[float] = None
    rent_delta: Optional[float] = None

def _extract_text(pdf_path: str) -> str:
    out: List[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            try:
                t = page.extract_text(x_tolerance=2, y_tolerance=2) or ""
            except Exception:
                t = ""
            out.append(t)
    return "\n".join(out)

def _parse_year(pat: re.Pattern, text: str) -> Optional[int]:
    m = pat.search(text)
    return int(m.group(1)) if m else None

def _parse_value(text: str) -> Optional[float]:
    m = PATTERNS["value"].search(text)
    if not m:
        return None
    raw, suffix = m.group(1), (m.group(2) or "").lower()
    num = float(raw.replace(",", ""))
    if suffix in {"million", "mm", "m"}:
        num *= 1_000_000
    return num

def _parse_avg_rent(text: str) -> Optional[float]:
    m = PATTERNS["avg_rent"].search(text)
    if not m:
        return None
    val = float(m.group(1).replace(",", ""))
    # heuristic: >$10k likely annual → monthly
    if val > 10000:
        val /= 12.0
    return val

def _parse_location(text: str):
    m = PATTERNS["address_line"].search(text)
    if m:
        addr = m.group(1).strip()
        try:
            tagged, _ = usaddress.tag(addr)
            return addr, tagged.get("PlaceName"), tagged.get("StateName"), tagged.get("ZipCode")
        except Exception:
            pass
    m2 = PATTERNS["city_state_zip"].search(text)
    if m2:
        city, state, zipcode = m2.group(1).strip(), m2.group(2), m2.group(3)
        return f"{city}, {state} {zipcode}", city, state, zipcode
    return None, None, None, None

def parse_om_pdf(pdf_path: str) -> OMParsed:
    text = _extract_text(pdf_path)
    location_str, city, state, zipcode = _parse_location(text)
    year_built = _parse_year(PATTERNS["year_built"], text)
    year_renovated = _parse_year(PATTERNS["year_renovated"], text)
    value_usd = _parse_value(text)
    om_avg_rent = _parse_avg_rent(text)
    return OMParsed(
        pdf_path=pdf_path,
        location_str=location_str,
        city=city,
        state=state,
        zip=zipcode,
        value_usd=value_usd,
        year_built=year_built,
        year_renovated=year_renovated,
        om_avg_rent=om_avg_rent,
    )
