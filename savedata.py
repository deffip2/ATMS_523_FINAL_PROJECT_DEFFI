import os
import requests
import xarray as xr
import pandas as pd
from datetime import datetime

event_path = "/data/keeling/a/deffip2/ATMS_523/FINAL_PROJECT/event_days_2000_2020.xlsx"
output_dir = "/data/keeling/a/deffip2/a/ATMS523/MOIST_PER_YEAR"
os.makedirs(output_dir, exist_ok=True)

variable_dict = {
    "850": ["specific_humidity", "u_component_of_wind", "v_component_of_wind"],
    "500": ["specific_humidity", "u_component_of_wind", "v_component_of_wind"],
}

varname_map = {
    "u_component_of_wind": "u",
    "v_component_of_wind": "v",
    "temperature": "t",
    "specific_humidity": "q",
    "geopotential": "z",
}

# Domain 2
lat_min, lat_max = 37, 52
lon_min, lon_max = 255, 290

print("Reading event days...")
df_ev = pd.read_excel(event_path)
df_ev["date"] = pd.to_datetime(df_ev["date"])
df_ev["year"] = df_ev["date"].dt.year
df_ev["day"] = df_ev["date"].dt.date

# Download
def download_nc_file(var, level, year, month, day):
    url = (
        "https://storage.googleapis.com/gcp-public-data-arco-era5/"
        f"raw/date-variable-pressure_level/{year}/{month}/{day}/{var}/{level}.nc"
    )

    local_filename = os.path.join(
        output_dir,
        f"tmp_{year}{month}{day}_{var}_{level}.nc"
    )

    if not os.path.exists(local_filename):
        print(f"    Downloading: {url}")
        r = requests.get(url)
        if r.status_code == 200:
            with open(local_filename, "wb") as f:
                f.write(r.content)
        else:
            print(f"    FAILED {url} (HTTP {r.status_code})")
            return None

    return local_filename

for year in range(2000, 2020):

    df_y = df_ev[df_ev["year"] == year]
    event_days_year = sorted(df_y["day"].unique())

    if len(event_days_year) == 0:
        print(f"\n=== {year}: NO EVENTS, SKIPPING ===")
        continue

    print(f"\n==============================")
    print(f"YEAR {year} → {len(event_days_year)} EVENT DAYS")
    print(f"==============================")

    for level, variables in variable_dict.items():

        print(f"  Level {level} hPa")

        yearly_event_datasets = []

        for d in event_days_year:
            print(f"  - {d}")

            y = f"{d.year:04d}"
            m = f"{d.month:02d}"
            dd = f"{d.day:02d}"

            var_fields = {}
            lat_da, lon_da = None, None

            for var in variables:
                real_var = varname_map[var]
                nc_path = download_nc_file(var, level, y, m, dd)

                if nc_path is None:
                    continue

                try:
                    ds = xr.open_dataset(nc_path)
                except Exception as e:
                    print("    Open failed:", e)
                    continue

                if real_var not in ds:
                    print("    Variable missing:", real_var)
                    continue

                ds_sub = ds.sel(
                    latitude=slice(lat_max, lat_min),
                    longitude=slice(lon_min, lon_max),
                )

                da = ds_sub[real_var].mean(dim="time")
                da = da.expand_dims(time=[pd.Timestamp(d)])

                var_fields[real_var] = da

                if lat_da is None:
                    lat_da = ds_sub["latitude"]
                    lon_da = ds_sub["longitude"]

            if not var_fields:
                continue

            ds_event = xr.Dataset(var_fields)
            ds_event = ds_event.assign_coords(latitude=lat_da, longitude=lon_da)

            yearly_event_datasets.append(ds_event)

        if not yearly_event_datasets:
            print(f"  >> No valid data for {year} @ {level} hPa")
            continue

        ds_year = xr.concat(yearly_event_datasets, dim="time")
        ds_year = ds_year.sortby("time")

        out_nc = os.path.join(
            output_dir,
            f"era5_moist_events_{year}_{level}hPa.nc"
        )

        ds_year.to_netcdf(out_nc)
        print(f"  SAVED → {out_nc}")

        deleted = 0
        for f in os.listdir(output_dir):
            full_path = os.path.join(output_dir, f)

            if (
                f.startswith(f"tmp_{year}")
                and f"_{level}.nc" in f
                and os.path.isfile(full_path)
            ):
                try:
                    os.remove(full_path)
                    deleted += 1
                    print("    Deleted:", f)
                except Exception as e:
                    print("    Failed to delete:", f, "|", e)
