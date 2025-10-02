from __future__ import annotations
import re, math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple
from pathlib import Path
import pdfplumber, usaddress


PAT = {
rent_psf = rent / avg_sf if avg_sf > 0 else None
rows.append({
"label": label,
"beds": beds,
"baths": baths,
"units": units,
"avg_sf": avg_sf,
"rent": rent,
"rent_psf": rent_psf,
})
# Merge duplicates of same label by summing units and averaging weighted by units
if not rows:
return rows
agg: Dict[str, Dict] = {}
for r in rows:
key = (r["label"], r["beds"], r["baths"])
a = agg.get(key)
if not a:
agg[key] = r.copy()
continue
u1, u2 = (a.get("units") or 0), (r.get("units") or 0)
a["units"] = (u1 or 0) + (u2 or 0)
# weighted avg for rent and sf
for k in ("rent","avg_sf","rent_psf"):
v1, v2 = a.get(k), r.get(k)
if v1 is None:
a[k] = v2
elif v2 is None:
continue
else:
w1 = u1 if u1 else 1
w2 = u2 if u2 else 1
a[k] = (v1*w1 + v2*w2) / (w1 + w2)
return [
{"label": k[0], "beds": k[1], "baths": k[2], **v}
for k,v in [(k, agg[k]) for k in agg]
]


# ---------------
# Public API
# ---------------


def parse_om_pdf(pdf_path: str) -> OMParsed:
text = _extract_text(pdf_path)
location_str, city, state, zipcode = _parse_location(text)
year_built = _parse_year(PAT["year_built"], text)
year_reno = _parse_year(PAT["year_reno"], text)
value_usd = _parse_value(text)
om_avg_rent = _parse_avg_rent(text)
unit_mix = parse_unit_mix(pdf_path)
return OMParsed(
pdf_path=str(pdf_path),
location_str=location_str,
city=city,
state=state,
zip=zipcode,
value_usd=value_usd,
year_built=year_built,
year_renovated=year_reno,
om_avg_rent=om_avg_rent,
unit_mix=unit_mix,
)
