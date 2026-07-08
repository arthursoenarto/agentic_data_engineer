#!/usr/bin/env python3
"""Deterministic ETL pipeline for ERA5 pressure-level January 2024 fields.

This pipeline is generated from a DatasetContract and runs without an LLM.
It follows the provided AccessContext for CDS client setup and credential names.
Credential values are read only from environment variables or local .env files and
are never printed, logged, embedded in config, or persisted by this code.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

PIPELINE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = PIPELINE_DIR / "config" / "request_config.json"

CANONICAL_URL_ENV = "CDSAPI_URL"
CANONICAL_KEY_ENV = "CDSAPI_KEY"
KEY_ALIASES = ("CDS_PERSONAL_ACCESS_TOKEN", "CDS_API_TOKEN")


def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def _strip_dotenv_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _parse_dotenv_line(line: str) -> Optional[Tuple[str, str]]:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    if stripped.startswith("export "):
        stripped = stripped[len("export ") :].strip()
    if "=" not in stripped:
        return None
    key, value = stripped.split("=", 1)
    key = key.strip()
    if not key:
        return None
    value = value.strip()
    if value and value[0] not in {"'", '"'}:
        hash_index = value.find(" #")
        if hash_index != -1:
            value = value[:hash_index].rstrip()
    return key, _strip_dotenv_quotes(value)


def load_dotenv_from_ancestors(start_dir: Path = PIPELINE_DIR) -> List[Path]:
    """Load .env files from ancestors without overriding explicit shell env.

    AccessContext requires loading from the pipeline directory and ancestors, with
    shell environment variables overriding dotenv values. To make precedence
    deterministic, ancestor .env files are loaded from repository/rootward parents
    toward the pipeline directory. Values loaded from a farther ancestor may be
    overridden by a nearer .env file, but variables present in the original shell
    environment are never overridden.
    """
    original_env_keys = set(os.environ.keys())
    chain = [start_dir.resolve(), *start_dir.resolve().parents]
    dotenv_paths = [p / ".env" for p in reversed(chain) if (p / ".env").is_file()]
    loaded: List[Path] = []
    for dotenv_path in dotenv_paths:
        try:
            with dotenv_path.open("r", encoding="utf-8") as f:
                for line in f:
                    parsed = _parse_dotenv_line(line)
                    if parsed is None:
                        continue
                    key, value = parsed
                    if key in original_env_keys:
                        continue
                    os.environ[key] = value
            loaded.append(dotenv_path)
        except OSError as exc:
            raise RuntimeError(f"Unable to read dotenv file at {dotenv_path}: {exc}") from exc
    return loaded


def ensure_cds_environment(require_values: bool = True) -> Dict[str, str]:
    """Prepare CDS environment variables according to AccessContext.

    Returns variable-name metadata only. It never returns or prints secret values.
    If CDSAPI_KEY is absent but an accepted alias is present, this function copies
    the alias value into CDSAPI_KEY in-process so cdsapi.Client() can use the
    canonical variable expected by the AccessContext.
    """
    load_dotenv_from_ancestors(PIPELINE_DIR)

    if not os.environ.get(CANONICAL_KEY_ENV):
        for alias in KEY_ALIASES:
            alias_value = os.environ.get(alias)
            if alias_value:
                os.environ[CANONICAL_KEY_ENV] = alias_value
                break

    missing = []
    if not os.environ.get(CANONICAL_URL_ENV):
        missing.append(CANONICAL_URL_ENV)
    if not os.environ.get(CANONICAL_KEY_ENV):
        missing.append(f"{CANONICAL_KEY_ENV} or one of {', '.join(KEY_ALIASES)}")

    if missing and require_values:
        raise RuntimeError(
            "Missing required CDS credential environment variable(s): " + "; ".join(missing)
        )

    return {
        "url_env_var": CANONICAL_URL_ENV,
        "key_env_var": CANONICAL_KEY_ENV,
        "key_aliases": ",".join(KEY_ALIASES),
    }


def dry_run() -> None:
    config = load_config()
    print("Dataset:", config["dataset_slug"])
    print("CDS dataset id:", config["cds_dataset_id"])
    print("Retrieve groups:", len(config["retrieve_groups"]))
    for group in config["retrieve_groups"]:
        request = group["request"]
        combos = group["field_selector_combinations"]
        print("-")
        print("  group_id:", group["group_id"])
        print("  target_filename:", group["target_filename"])
        print("  variable:", ",".join(request["variable"]))
        print("  pressure_level:", ",".join(request["pressure_level"]))
        print("  exact_combinations:", ", ".join(f"{c['field']}@{c['pressure_level']}hPa" for c in combos))
    print("Processed Zarr:", config["output"]["zarr_store"])


def check_credentials() -> None:
    ensure_cds_environment(require_values=True)
    print("CDS credential environment is configured using approved variable names; values were not displayed.")


def retrieve() -> None:
    config = load_config()
    ensure_cds_environment(require_values=True)
    try:
        import cdsapi
    except ImportError as exc:
        raise RuntimeError("Missing dependency 'cdsapi'. Install requirements.txt first.") from exc

    dataset_id = config["cds_dataset_id"]
    raw_dir = PIPELINE_DIR / config["output"]["raw_dir"]
    raw_dir.mkdir(parents=True, exist_ok=True)

    client = cdsapi.Client()
    for group in config["retrieve_groups"]:
        target_path = raw_dir / group["target_filename"]
        if target_path.exists() and target_path.stat().st_size > 0:
            print(f"Skipping existing raw file: {target_path.relative_to(PIPELINE_DIR)}")
            continue
        print(f"Retrieving {group['group_id']} to {target_path.relative_to(PIPELINE_DIR)}")
        client.retrieve(dataset_id, group["request"]).download(str(target_path))


def _find_data_var(ds, field_name: str, candidates: Sequence[str]) -> str:
    for candidate in candidates:
        if candidate in ds.data_vars:
            return candidate
    normalized = {name.lower(): name for name in ds.data_vars}
    for candidate in candidates:
        lowered = candidate.lower()
        if lowered in normalized:
            return normalized[lowered]
    available = ", ".join(ds.data_vars)
    raise KeyError(f"Could not find data variable for field '{field_name}'. Available variables: {available}")


def _find_pressure_coord_name(da) -> Optional[str]:
    candidates = (
        "pressure_level",
        "level",
        "isobaricInhPa",
        "isobaricInPa",
        "plev",
    )
    for name in candidates:
        if name in da.dims or name in da.coords:
            return name
    return None


def _select_pressure_level(da, pressure_level: str):
    coord_name = _find_pressure_coord_name(da)
    if coord_name is None:
        return da

    if coord_name in da.coords:
        coord = da.coords[coord_name]
        for value in (pressure_level, int(pressure_level), float(pressure_level)):
            try:
                selected = da.sel({coord_name: value})
                return selected
            except Exception:
                pass
        # Some backends expose Pa instead of hPa.
        try:
            selected = da.sel({coord_name: float(pressure_level) * 100.0})
            return selected
        except Exception:
            pass
    if coord_name in da.dims and da.sizes.get(coord_name) == 1:
        return da.isel({coord_name: 0})
    raise KeyError(f"Could not select pressure level {pressure_level} using coordinate {coord_name}")


def _standardize_coord_names(obj):
    rename = {}
    for old, new in (
        ("valid_time", "time"),
        ("latitude", "lat"),
        ("longitude", "lon"),
    ):
        if old in obj.dims or old in obj.coords:
            rename[old] = new
    if rename:
        return obj.rename(rename)
    return obj


def transform() -> None:
    config = load_config()
    try:
        import xarray as xr
    except ImportError as exc:
        raise RuntimeError("Missing dependency 'xarray'. Install requirements.txt first.") from exc

    raw_dir = PIPELINE_DIR / config["output"]["raw_dir"]
    zarr_store = PIPELINE_DIR / config["output"]["zarr_store"]
    zarr_store.parent.mkdir(parents=True, exist_ok=True)

    arrays = []
    variable_candidates: Mapping[str, Sequence[str]] = config["variable_name_candidates"]

    for group in config["retrieve_groups"]:
        raw_path = raw_dir / group["target_filename"]
        if not raw_path.is_file():
            raise FileNotFoundError(
                f"Missing raw file {raw_path}. Run 'python pipeline.py retrieve' first."
            )
        ds = xr.open_dataset(raw_path, chunks={})
        ds = _standardize_coord_names(ds)
        for combo in group["field_selector_combinations"]:
            field_name = combo["field"]
            pressure_level = combo["pressure_level"]
            output_variable = combo["output_variable"]
            source_var = _find_data_var(ds, field_name, variable_candidates[field_name])
            da = ds[source_var]
            da = _standardize_coord_names(da)
            da = _select_pressure_level(da, pressure_level)
            squeeze_dims = [dim for dim in da.dims if dim in {"pressure_level", "level", "isobaricInhPa", "isobaricInPa", "plev"}]
            if squeeze_dims:
                da = da.squeeze(dim=squeeze_dims, drop=True)
            da = da.rename(output_variable)
            da.attrs = dict(da.attrs)
            da.attrs["source_cds_variable"] = field_name
            da.attrs["source_netcdf_variable"] = source_var
            da.attrs["pressure_level"] = pressure_level
            da.attrs["pressure_level_unit"] = "hPa"
            da.attrs["selector_preserved_exactly"] = "true"
            arrays.append(da)

    if not arrays:
        raise RuntimeError("No arrays were produced during transform.")

    out = xr.merge(arrays, compat="override", join="outer")
    out.attrs["dataset_slug"] = config["dataset_slug"]
    out.attrs["cds_dataset_id"] = config["cds_dataset_id"]
    out.attrs["date_range"] = "2024-01-01/2024-01-31 inclusive"
    out.attrs["times_utc"] = ",".join(config["scope"]["time"])
    out.attrs["note"] = "Variables represent exact DatasetContract field/pressure-level combinations."

    if zarr_store.exists():
        import shutil

        shutil.rmtree(zarr_store)
    out.to_zarr(zarr_store, mode="w", consolidated=True)
    print(f"Wrote processed Zarr store: {zarr_store.relative_to(PIPELINE_DIR)}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="ERA5 pressure-level ETL pipeline")
    parser.add_argument(
        "command",
        choices=("dry-run", "check-credentials", "retrieve", "transform", "all"),
        help="Pipeline command to execute.",
    )
    args = parser.parse_args(argv)

    try:
        if args.command == "dry-run":
            dry_run()
        elif args.command == "check-credentials":
            check_credentials()
        elif args.command == "retrieve":
            retrieve()
        elif args.command == "transform":
            transform()
        elif args.command == "all":
            retrieve()
            transform()
        else:
            raise AssertionError(args.command)
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
