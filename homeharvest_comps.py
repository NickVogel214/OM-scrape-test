# homeharvest_comps.py (v2)
BR_BINS = {
0: "Studio",
1: "1BR",
2: "2BR",
3: "3BR",
4: "4BR+",
}


def _bucket_beds(val) -> Optional[str]:
try:
b = int(float(val))
except Exception:
return None
if b <= 0: return BR_BINS[0]
if b == 1: return BR_BINS[1]
if b == 2: return BR_BINS[2]
if b == 3: return BR_BINS[3]
return BR_BINS[4]


def get_rent_comps(location: str, radius_miles: float = 3.0, limit: int = 400) -> Dict[str, pd.DataFrame]:
"""Fetch for_rent comps near location (Realtor via homeharvest).
Returns a dict with raw df and aggregates per bedroom cohort.
"""
df = scrape_property(
location=location,
listing_type="for_rent",
return_type="pandas",
radius=radius_miles,
limit=limit,
)
if not isinstance(df, pd.DataFrame) or df.empty:
return {"raw": pd.DataFrame(), "by_br": pd.DataFrame()}


# normalize
if "list_price" in df.columns:
df = df[df["list_price"].notna()].copy()
df["rent"] = df["list_price"].astype(float)
else:
df["rent"] = pd.NA


if "beds" in df.columns:
df["cohort"] = df["beds"].apply(_bucket_beds)
else:
df["cohort"] = None


# rent per sf when sqft present
if "sqft" in df.columns:
df["rent_psf"] = df.apply(lambda r: r["rent"] / r["sqft"] if pd.notna(r["rent"]) and pd.notna(r["sqft"]) and r["sqft"]>0 else pd.NA, axis=1)


# aggregate by cohort
by_br = (
df.dropna(subset=["cohort", "rent"])
.groupby("cohort")
.agg(
n=("rent","count"),
mean_rent=("rent","mean"),
median_rent=("rent","median"),
mean_rent_psf=("rent_psf","mean")
)
.reset_index()
)


return {"raw": df, "by_br": by_br}
