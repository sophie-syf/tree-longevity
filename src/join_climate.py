# src/02_join_climate.py
import pandas as pd
from pathlib import Path

series = pd.read_csv("data/itrdb/sample_series.csv")
clim = pd.read_csv("data/climate/worldclim/sample_bio.csv")
biomes = pd.read_csv("data/biomes/sample_biomes.csv")

# site climate
site = (series[["site_id","species"]].drop_duplicates()
        .merge(clim, on="site_id", how="left")
        .merge(biomes, on="site_id", how="left"))

# species-level mean climate across its sites
sp_clim = (site.groupby("species")[["bio1","bio12","bio4"]].mean().reset_index())
sp_clim["n_sites"] = site.groupby("species")["site_id"].nunique().values

sp_clim.to_csv("data/derived/species_climate.csv", index=False)
site.to_csv("data/derived/site_table.csv", index=False)

print("Wrote: data/derived/species_climate.csv, data/derived/site_table.csv")
