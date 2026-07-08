# ERA5 pressure-level ETL pipeline

This directory contains a deterministic, inspectable ETL pipeline for the dataset contract:

- Dataset slug: `reanalysis_era5_pressure_levels`
- Source: Copernicus Climate Data Store ERA5 pressure levels
- Dataset ID: `reanalysis-era5-pressure-levels`
- Scope: January 2024, UTC times 00:00 through 21:00 every 3 hours, global area
- Selected field/pressure-level combinations:
  - `temperature` at `500` hPa
  - `temperature` at `850` hPa
  - `geopotential` at `500` hPa

The CDS requests are intentionally split into pressure-level groups so the pipeline does **not** silently widen the request to `geopotential` at 850 hPa.

## Credentials

The pipeline follows the supplied AccessContext. It loads `.env` files from this pipeline directory and ancestor directories without overriding variables that were already present in the shell environment.

Required credential variable names:

- `CDSAPI_URL`
- `CDSAPI_KEY`

Accepted key aliases, if `CDSAPI_KEY` is not set:

- `CDS_PERSONAL_ACCESS_TOKEN`
- `CDS_API_TOKEN`

Do not commit credential values. A local `.env` file may contain variable names such as:

```bash
CDSAPI_URL=https://cds.climate.copernicus.eu/api
CDSAPI_KEY=your-local-secret-value
```

## Install

From this pipeline directory:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

## Run

Inspect the deterministic requests without network access:

```bash
python pipeline.py dry-run
```

Check that required credential variables are available without printing secret values:

```bash
python pipeline.py check-credentials
```

Retrieve raw NetCDF files from CDS:

```bash
python pipeline.py retrieve
```

Transform raw NetCDF files into an analysis-ready Zarr store:

```bash
python pipeline.py transform
```

Run retrieve and transform:

```bash
python pipeline.py all
```

## Outputs

Default output locations are under this pipeline directory:

- Raw CDS NetCDF files: `data/raw/*.nc`
- Analysis-ready Zarr store: `data/processed/era5_pressure_levels_jan2024.zarr`

The processed Zarr variables are named by exact selected field and pressure level:

- `temperature_pl500`
- `temperature_pl850`
- `geopotential_pl500`

Each processed variable has dimensions such as `time`, `lat`, and `lon` when available in the source files. The pressure level is stored as variable metadata after exact subsetting/squeezing.

## Tests

Run standard-library tests from this directory:

```bash
python -m unittest discover -s tests
```
