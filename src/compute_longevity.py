# src/01_compute_longevity.py
import pandas as pd
from pathlib import Path


# raw = pd.read_csv("data/itrdb/sample_series.csv")
raw = pd.read_csv("data/itrdb/series_long_extended.csv")
print('Data loaded!')

# species-level longevity proxy
species_longevity = (raw.groupby("species")["n_years"]
                       .agg(max_series_years="max", q95_series_years=lambda s: s.quantile(0.95))
                       .reset_index())



species_longevity["n_series"] = raw.groupby("species")["series_id"].nunique().values

print("Species-level longevity")

# site-level maxima 
site_max = (raw.groupby(["site_id","species"])["n_years"].max().reset_index(name="site_max_years"))

print("site-level")

Path("data/derived").mkdir(parents=True, exist_ok=True)
species_longevity.to_csv("data/derived/species_longevity.csv", index=False)
site_max.to_csv("data/derived/site_max.csv", index=False)

print("Wrote: data/derived/species_longevity.csv, data/derived/site_max.csv")
