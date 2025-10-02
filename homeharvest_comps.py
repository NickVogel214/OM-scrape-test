# homeharvest_comps.py
from __future__ import annotations
from typing import List, Dict, Optional
import pandas as pd
from homeharvest import scrape_property

def get_rent_comps(location: str, radius_miles: float = 3.0, limit: int = 200) -> List[Dict]:
    """
    Pull nearby rental comps via homeharvest (Realtor.com).
    Treats list_price as monthly rent for 'for_rent' listings.
    Returns a list of dicts with a normalized rent field 'avg_rent'.
    """
    df = scrape_property(
        location=location,
        listing_type="for_rent",
        return_type="pandas",
        radius=radius_miles,
        limit=limit
    )
    if not isinstance(df, pd.DataFrame) or df.empty:
        return []

    # Normalize obvious rent column → 'avg_rent'
    if "list_price" in df.columns:
        df = df[df["list_price"].notna()].copy()
        df["avg_rent"] = df["list_price"]
    else:
        return []

    cols = ["street","city","state","zip_code","avg_rent","sqft","beds","baths","latitude","longitude"]
    present = [c for c in cols if c in df.columns]
    out = df[present].to_dict(orient="records")
    return out

def average_rent(comps: List[Dict]) -> Optional[float]:
    vals = [float(c["avg_rent"]) for c in comps if c.get("avg_rent") is not None]
    return sum(vals)/len(vals) if vals else None
