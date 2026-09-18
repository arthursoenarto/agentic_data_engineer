Supported evaluator targets:

1. `regular_rectilinear_grid` -> `zarr` v3 -> `full_field_tensor`
2. `station_time_series` -> `parquet` v1 -> `filtered_station_scan`

Both use engineering profile `general_pipeline`.

Trusted inventory summary:

{
  "availability": {
    "constraint_fields": [
      "day",
      "month",
      "pressure_level",
      "product_type",
      "time",
      "variable",
      "year"
    ],
    "constraint_rule_count": 126,
    "constraints_url": "https://object-store.os-api.cci2.ecmwf.int:443/cci2-prod-catalogue/resources/reanalysis-era5-pressure-levels/constraints_260b0265b2126a5264d333c4f201720032e588c038f45ba2db0d50456f2f6b99.json"
  },
  "catalogue_metadata": {
    "assets": {
      "thumbnail": {
        "href": "https://object-store.os-api.cci2.ecmwf.int:443/cci2-prod-catalogue/resources/reanalysis-era5-pressure-levels/overview_652fd83a7b2ed724ce541e563beff9c4484c3482bc08334a638a4bc47ae4cf0f.png",
        "roles": [
          "thumbnail"
        ],
        "type": "image/jpg"
      }
    },
    "data_description": {
      "data_type": "Gridded",
      "file_format": "GRIB",
      "horizontal_coverage": "Global",
      "horizontal_resolution": "\nReanalysis: 0.25\u00b0 x 0.25\u00b0\n\nMean, spread and members: 0.5\u00b0 x 0.5\u00b0",
      "projection": "Regular latitude-longitude grid.",
      "temporal_coverage": "1940 to present",
      "temporal_resolution": "Hourly",
      "update_frequency": "Daily",
      "vertical_coverage": "1000 hPa to 1 hPa",
      "vertical_resolution": "37 pressure levels"
    },
    "doi": "10.24381/cds.bd0915c6",
    "extent": {
      "spatial": {
        "bbox": [
          [
            0.0,
            -89.0,
            360.0,
            89.0
          ]
        ]
      },
      "temporal": {
        "interval": [
          [
            "1940-01-01T00:00:00+00:00",
            "2026-06-30T00:00:00+00:00"
          ]
        ]
      }
    },
    "fair_score": 92,
    "keywords": [
      "Product type: Reanalysis",
      "Temporal coverage: Past",
      "Spatial coverage: Global",
      "Variable domain: Atmosphere (surface)",
      "Variable domain: Atmosphere (upper air)",
      "Provider: Copernicus C3S"
    ],
    "license": "CC-BY-4.0",
    "metadata_links": {
      "catalogue": "https://cds.climate.copernicus.eu/api/catalogue/v1/collections/reanalysis-era5-pressure-levels",
      "constraints": "https://object-store.os-api.cci2.ecmwf.int:443/cci2-prod-catalogue/resources/reanalysis-era5-pressure-levels/constraints_260b0265b2126a5264d333c4f201720032e588c038f45ba2db0d50456f2f6b99.json",
      "form": "https://object-store.os-api.cci2.ecmwf.int:443/cci2-prod-catalogue/resources/reanalysis-era5-pressure-levels/form_8fdaea953376acee033279ea20b65b91d51bf4c2d30a3fb036ebd48909254a6f.json",
      "layout": "https://object-store.os-api.cci2.ecmwf.int:443/cci2-prod-catalogue/resources/reanalysis-era5-pressure-levels/layout_1a07f2404b0076781b7b0ecdb75e8f2228b2655f3712282905df89c255bd207c.json"
    },
    "providers": [
      {
        "name": "ECMWF"
      }
    ],
    "published": "2018-06-14T00:00:00Z",
    "stac_version": "1.1.0",
    "update_frequency": "Daily",
    "updated": "2026-07-06T00:00:00Z"
  },
  "constraints": {
    "area": {
      "items": {
        "type": "number"
      },
      "maxItems": 4,
      "minItems": 4,
      "type": "array"
    },
    "data_format": {
      "type": "string"
    },
    "day": {
      "items": {
        "type": "string"
      },
      "type": "array"
    },
    "download_format": {
      "type": "string"
    },
    "month": {
      "items": {
        "type": "string"
      },
      "type": "array"
    },
    "pressure_level": {
      "items": {
        "type": "string"
      },
      "type": "array"
    },
    "product_type": {
      "items": {
        "type": "string"
      },
      "type": "array"
    },
    "time": {
      "items": {
        "type": "string"
      },
      "type": "array"
    },
    "variable": {
      "items": {
        "type": "string"
      },
      "type": "array"
    },
    "year": {
      "items": {
        "type": "string"
      },
      "type": "array"
    }
  },
  "dataset_id": "reanalysis-era5-pressure-levels",
  "dataset_slug": "reanalysis_era5_pressure_levels",
  "defaults": {
    "area": [
      90,
      -180,
      -90,
      180
    ],
    "data_format": "grib",
    "download_format": "unarchived",
    "product_type": [
      "reanalysis"
    ]
  },
  "description": "ERA5 is the fifth generation ECMWF reanalysis for the global climate and weather for the past 8 decades.\nData is available from 1940 onwards.\nERA5 replaces the ERA-Interim reanalysis.\nReanalysis combines model data with observations from across the world into a globally complete and consistent dataset using the laws of physics. This principle, called data assimilation, is based on the method used by numerical weather prediction centres, where every so many hours (12 hours at ECMWF) a previous forecast is combined with newly available observations in an optimal way to produce a new best estimate of the state of the atmosphere, called analysis, from which an updated, improved forecast is issued. Reanalysis works in the same way, but at reduced resolution to allow for the provision of a dataset spanning back several decades. Reanalysis does not have the constraint of issuing timely forecasts, so there is more time to collect observations, and when going further back in time, to allow for the ingestion of improved versions of the original observations, which all benefit the quality of the reanalysis product.\nERA5 provides hourly estimates for a large number of atmospheric, ocean-wave and land-surface quantities.\nAn uncertainty estimate is sampled by an underlying 10-member ensemble\nat three-hourly intervals. Ensemble mean and spread have been pre-computed for convenience.\nSuch uncertainty estimates are closely related to the information content of the available observing system which\nhas evolved considerably over time. They also indicate flow-dependent sensitive areas.\nTo facilitate many climate applications, monthly-mean averages have been pre-calculated too,\nthough monthly means are not available for the ensemble mean and spread.\nERA5 is updated daily with a latency of about 5 days. In case that serious flaws are detected in this early release (called ERA5T), this data could be different from the final release 2 to 3 months later. In case that this occurs users are notified.\nThe data set presented here is a regridded subset of the full ERA5 data set on native resolution.\nIt is online on spinning disk, which should ensure fast and easy access.\nIt should satisfy the requirements for most common applications.\nAn overview of all ERA5 datasets can be found in this article.\nInformation on access to ERA5 data on native resolution is provided in these guidelines.\nData has been regridded to a regular lat-lon grid of 0.25 degrees for the reanalysis and 0.5 degrees for\nthe uncertainty estimate (0.5 and 1 degree respectively for ocean waves).\nThere are four main sub sets: hourly and monthly products, both on pressure levels (upper air fields) and single levels (atmospheric, ocean-wave and land surface quantities).\nThe present entry is \"ERA5 hourly data on pressure levels from 1940 to present\".",
  "option_summary": {
    "data_format": {
      "count": 2,
      "examples": [
        "grib",
        "netcdf"
      ]
    },
    "day": {
      "count": 31,
      "examples": [
        "01",
        "02",
        "03",
        "04",
        "05",
        "06",
        "07",
        "08",
        "09",
        "10",
        "11",
        "12"
      ]
    },
    "download_format": {
      "count": 2,
      "examples": [
        "zip",
        "unarchived"
      ]
    },
    "month": {
      "count": 12,
      "examples": [
        "01",
        "02",
        "03",
        "04",
        "05",
        "06",
        "07",
        "08",
        "09",
        "10",
        "11",
        "12"
      ]
    },
    "pressure_level": {
      "count": 37,
      "examples": [
        "1",
        "10",
        "100",
        "1000",
        "125",
        "150",
        "175",
        "2",
        "20",
        "200",
        "225",
        "250"
      ]
    },
    "product_type": {
      "count": 4,
      "examples": [
        "ensemble_mean",
        "ensemble_members",
        "ensemble_spread",
        "reanalysis"
      ]
    },
    "time": {
      "count": 24,
      "examples": [
        "00:00",
        "01:00",
        "02:00",
        "03:00",
        "04:00",
        "05:00",
        "06:00",
        "07:00",
        "08:00",
        "09:00",
        "10:00",
        "11:00"
      ]
    },
    "variable": {
      "count": 16,
      "examples": [
        "divergence",
        "fraction_of_cloud_cover",
        "geopotential",
        "ozone_mass_mixing_ratio",
        "potential_vorticity",
        "relative_humidity",
        "specific_cloud_ice_water_content",
        "specific_cloud_liquid_water_content",
        "specific_humidity",
        "specific_rain_water_content",
        "specific_snow_water_content",
        "temperature"
      ]
    },
    "year": {
      "count": 87,
      "examples": [
        "1940",
        "1941",
        "1942",
        "1943",
        "1944",
        "1945",
        "1946",
        "1947",
        "1948",
        "1949",
        "1950",
        "1951"
      ]
    }
  },
  "option_units": {
    "area": "north/west/south/east degrees",
    "pressure_level": "hPa",
    "time": "UTC"
  },
  "output_transmission": [
    "reference"
  ],
  "outputs": {
    "asset": {
      "description": "Downloadable asset description",
      "schema": {
        "properties": {
          "value": {
            "properties": {
              "file:checksum": {
                "title": "File:Checksum",
                "type": "string"
              },
              "file:local_path": {
                "title": "File:Local Path",
                "type": "string"
              },
              "file:size": {
                "title": "File:Size",
                "type": "integer"
              },
              "href": {
                "title": "Href",
                "type": "string"
              },
              "type": {
                "title": "Type",
                "type": "string"
              }
            },
            "required": [
              "type",
              "href",
              "file:checksum",
              "file:size",
              "file:local_path"
            ],
            "title": "FileInfoModel",
            "type": "object"
          }
        },
        "type": "object"
      },
      "title": "Asset"
    }
  },
  "process_version": "1.0.0",
  "provider": "ECMWF",
  "request_fields": [
    "product_type",
    "variable",
    "year",
    "month",
    "day",
    "time",
    "pressure_level",
    "area",
    "data_format",
    "download_format"
  ],
  "schema_version": "dataset_inventory.v1",
  "source_url": "https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels?tab=download",
  "title": "ERA5 hourly data on pressure levels from 1940 to present",
  "warnings": []
}

Trusted frozen contract:

{
  "contract": {
    "access_methods": [
      "cdsapi"
    ],
    "advanced_options": {
      "data_format": "grib",
      "dataset_id": "reanalysis-era5-pressure-levels",
      "download_format": "unarchived",
      "product_type": [
        "reanalysis"
      ]
    },
    "assumptions": [
      "Credentials are supplied through CDSAPI_KEY and CDSAPI_URL environment-variable references.",
      "No aggregation, unit conversion, feature engineering, or physical storage layout has been selected."
    ],
    "credential_requirements": [
      "CDSAPI_KEY",
      "CDSAPI_URL"
    ],
    "dataset_family": "reanalysis",
    "dataset_slug": "reanalysis_era5_pressure_levels",
    "defaults_used": [],
    "evidence": [
      {
        "note": "Trusted dataset inventory and user-selected scope.",
        "title": "ERA5 hourly data on pressure levels from 1940 to present",
        "url": "https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels?tab=download"
      }
    ],
    "fields": [
      {
        "description": null,
        "display_name": "Temperature",
        "name": "temperature",
        "selectors": [
          {
            "dimension": "pressure_level",
            "label": null,
            "unit": "hPa",
            "value": "500"
          }
        ],
        "units": null
      },
      {
        "description": null,
        "display_name": "Temperature",
        "name": "temperature",
        "selectors": [
          {
            "dimension": "pressure_level",
            "label": null,
            "unit": "hPa",
            "value": "850"
          }
        ],
        "units": null
      },
      {
        "description": null,
        "display_name": "Geopotential",
        "name": "geopotential",
        "selectors": [
          {
            "dimension": "pressure_level",
            "label": null,
            "unit": "hPa",
            "value": "500"
          }
        ],
        "units": null
      }
    ],
    "human_confirmed": true,
    "human_editable_fields": [
      "fields",
      "scope",
      "advanced_options",
      "pipeline_requirements"
    ],
    "intent": "Prepare an ML-ready pipeline for ERA5 pressure levels using exactly the selected data and scope.",
    "pipeline_requirements": {
      "downstream_use": {
        "description": "The data will be read repeatedly during ML training.",
        "kind": "ml_training",
        "workload_kind": "full_field_tensor"
      },
      "optimization": {
        "descriptive_measurements": [
          "materialization_seconds",
          "q_engineering"
        ],
        "ordered_objectives": [
          {
            "direction": "maximize",
            "objective": "consumer_samples_per_second",
            "priority": 1
          },
          {
            "direction": "minimize",
            "objective": "output_bytes",
            "priority": 2
          }
        ]
      }
    },
    "provider": "ECMWF",
    "recommended_next_step": "Generate and evaluate candidate Zarr v3 pipelines against the declared workload and ordered objectives.",
    "risks_or_unknowns": [],
    "schema_version": "dataset_contract.v2",
    "scope": {
      "date_range": {
        "end_date": "2024-01-07",
        "inclusive": true,
        "start_date": "2024-01-01"
      },
      "geography": {
        "area": "global",
        "cds_area": [
          90,
          -180,
          -90,
          180
        ],
        "cds_area_order": [
          "north",
          "west",
          "south",
          "east"
        ]
      },
      "product_type": "reanalysis",
      "time": {
        "selected_times": [
          "00:00",
          "06:00",
          "12:00",
          "18:00"
        ],
        "timestep": "6 hours",
        "timezone": "UTC"
      }
    },
    "source_url": "https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels?tab=download",
    "summary": "ERA5 pressure-level reanalysis subset for repeated ML training reads, output as Zarr version 3.",
    "title": "ERA5 workload-aware expert prompt pilot"
  },
  "contract_schema_version": "dataset_contract.v2",
  "created_at": "2026-09-07T01:32:00.990669+00:00",
  "intention_sha256": "6890676782f73c26baa0ee647983f5da1599a33238232544798e918a8c6027c3",
  "interaction_sha256": "be298fe7bdf814e9f4ed36819f6e2bae644233b092df325a6fdef148242f520f",
  "inventory_schema_version": "dataset_inventory.v1",
  "inventory_sha256": "7cc09995d907e537e117989c85665c5d58a8c6ae901a7cd0fc1223a64d2e15ca",
  "lock_version": 1,
  "schema_version": "dataset_contract_lock.v1",
  "source_yaml_sha256": "aab1152d55bc9eb7ce2c815e30b6f71e894fa98a334e309ee417663cf859ea18"
}

Canonical contract field IDs that the mapping must contain exactly:

[
  "geopotential[pressure_level=\"500\"]",
  "temperature[pressure_level=\"500\"]",
  "temperature[pressure_level=\"850\"]"
]

Existing trusted catalog summary:

[
  {
    "check_id": "contract.declared_metadata",
    "description": "Checks only dtype, fill-value, and metadata assertions declared by the suite.",
    "layer": "data_class",
    "required_profile_tags": [
      "regular-grid"
    ],
    "role": "hard_gate",
    "title": "Declared grid metadata"
  },
  {
    "check_id": "contract.filtered_station_scan",
    "description": "Loads and filters one declared station-field workload from the published table.",
    "layer": "workload",
    "required_profile_tags": [
      "workload:filtered-station-scan"
    ],
    "role": "hard_gate",
    "title": "Filtered station scan"
  },
  {
    "check_id": "contract.grounding",
    "description": "Validates the frozen contract, inventory, source, command, and mappings.",
    "layer": "core",
    "required_profile_tags": [
      "core"
    ],
    "role": "hard_gate",
    "title": "Suite grounding"
  },
  {
    "check_id": "contract.initial_execution",
    "description": "Requires the candidate command to finish successfully in evaluator control.",
    "layer": "core",
    "required_profile_tags": [
      "core"
    ],
    "role": "hard_gate",
    "title": "Initial controlled execution"
  },
  {
    "check_id": "contract.initial_materialization",
    "description": "Records whether trusted inputs allow the initial controlled run.",
    "layer": "core",
    "required_profile_tags": [
      "core"
    ],
    "role": "hard_gate",
    "title": "Initial materialization availability"
  },
  {
    "check_id": "contract.parquet_integrity",
    "description": "Requires readable Parquet metadata, row groups, schema, and Zstandard encoding.",
    "layer": "output_format",
    "required_profile_tags": [
      "parquet"
    ],
    "role": "hard_gate",
    "title": "Parquet integrity"
  },
  {
    "check_id": "contract.pytorch_consumer",
    "description": "Loads shuffled warm float32 C,H,W samples through the shared consumer.",
    "layer": "workload",
    "required_profile_tags": [
      "workload:full-field-tensor"
    ],
    "role": "hard_gate",
    "title": "Full-field tensor consumption"
  },
  {
    "check_id": "contract.reference_integrity",
    "description": "Requires the independent regular-grid oracle to be readable and complete.",
    "layer": "data_class",
    "required_profile_tags": [
      "regular-grid"
    ],
    "role": "hard_gate",
    "title": "Reference integrity"
  },
  {
    "check_id": "contract.scope_coordinates",
    "description": "Compares sample and spatial coordinates with declared normalization.",
    "layer": "data_class",
    "required_profile_tags": [
      "regular-grid"
    ],
    "role": "hard_gate",
    "title": "Grid scope and coordinates"
  },
  {
    "check_id": "contract.station_reference_integrity",
    "description": "Requires the independent canonical station-table oracle to be readable.",
    "layer": "data_class",
    "required_profile_tags": [
      "data-class:station-time-series"
    ],
    "role": "hard_gate",
    "title": "Station reference integrity"
  },
  {
    "check_id": "contract.station_scope_and_keys",
    "description": "Checks field/station identities, UTC time bounds, required columns, and unique keys.",
    "layer": "data_class",
    "required_profile_tags": [
      "data-class:station-time-series"
    ],
    "role": "hard_gate",
    "title": "Station scope and primary keys"
  },
  {
    "check_id": "contract.zarr_integrity",
    "description": "Requires every mapped Zarr array and chunk to be readable.",
    "layer": "output_format",
    "required_profile_tags": [
      "zarr"
    ],
    "role": "hard_gate",
    "title": "Zarr integrity"
  },
  {
    "check_id": "contract.zarr_output_policy",
    "description": "Checks Zarr v3 and complete consolidated metadata.",
    "layer": "output_format",
    "required_profile_tags": [
      "zarr",
      "zarr-v3"
    ],
    "role": "hard_gate",
    "title": "Zarr v3 publication policy"
  },
  {
    "check_id": "engineering.meroda",
    "description": "Assesses applicable engineering dimensions from blinded bounded evidence.",
    "layer": "engineering",
    "required_profile_tags": [
      "engineering:meroda"
    ],
    "role": "assessment",
    "title": "MERODA engineering quality"
  },
  {
    "check_id": "objective.consumer_samples_per_second",
    "description": "Measures median samples per second for the declared workload.",
    "layer": "objective",
    "required_profile_tags": [
      "core"
    ],
    "role": "objective",
    "title": "Consumer throughput"
  },
  {
    "check_id": "objective.materialization_seconds",
    "description": "Measures frozen local input to published output wall time.",
    "layer": "objective",
    "required_profile_tags": [
      "core"
    ],
    "role": "objective",
    "title": "Materialization time"
  },
  {
    "check_id": "objective.output_bytes",
    "description": "Counts bytes across unique mapped native output stores.",
    "layer": "objective",
    "required_profile_tags": [
      "core"
    ],
    "role": "objective",
    "title": "Native output footprint"
  },
  {
    "check_id": "provenance.candidate_claims",
    "description": "Checks manifest and pipeline-contract claims against trusted observations.",
    "layer": "core",
    "required_profile_tags": [
      "core"
    ],
    "role": "hard_gate",
    "title": "Candidate claim consistency"
  },
  {
    "check_id": "provenance.identities",
    "description": "Requires hashes for trusted inputs, source, oracle, cache, and outputs.",
    "layer": "core",
    "required_profile_tags": [
      "core"
    ],
    "role": "hard_gate",
    "title": "Provenance identities"
  },
  {
    "check_id": "provenance.receipt.initial",
    "description": "Validates linked identities and cache evidence in the initial receipt.",
    "layer": "core",
    "required_profile_tags": [
      "core"
    ],
    "role": "hard_gate",
    "title": "Initial receipt validity"
  },
  {
    "check_id": "provenance.receipt.rerun",
    "description": "Validates linked identities and cache evidence in the rerun receipt.",
    "layer": "core",
    "required_profile_tags": [
      "core"
    ],
    "role": "hard_gate",
    "title": "Rerun receipt validity"
  },
  {
    "check_id": "provenance.receipts",
    "description": "Records that receipt validation was blocked before execution.",
    "layer": "core",
    "required_profile_tags": [
      "core"
    ],
    "role": "hard_gate",
    "title": "Receipt availability"
  },
  {
    "check_id": "provenance.trusted_inputs_unchanged",
    "description": "Rehashes evaluator-owned inputs after candidate execution.",
    "layer": "core",
    "required_profile_tags": [
      "core"
    ],
    "role": "hard_gate",
    "title": "Trusted input immutability"
  },
  {
    "check_id": "rerun.declared_metadata",
    "description": "Checks declared metadata again on the independently published rerun.",
    "layer": "data_class",
    "required_profile_tags": [
      "regular-grid"
    ],
    "role": "hard_gate",
    "title": "Rerun metadata stability"
  },
  {
    "check_id": "rerun.execution",
    "description": "Executes the same frozen request again in a fresh output location.",
    "layer": "core",
    "required_profile_tags": [
      "core"
    ],
    "role": "hard_gate",
    "title": "Exact rerun execution"
  },
  {
    "check_id": "rerun.logical_equivalence",
    "description": "Compares evaluator-owned logical fingerprints from both executions.",
    "layer": "core",
    "required_profile_tags": [
      "core"
    ],
    "role": "hard_gate",
    "title": "Exact rerun equivalence"
  },
  {
    "check_id": "rerun.public_output_policy",
    "description": "Records a generation-visible output-policy failure before comparison.",
    "layer": "output_format",
    "required_profile_tags": [
      "zarr",
      "zarr-v3"
    ],
    "role": "hard_gate",
    "title": "Blocked rerun output policy"
  },
  {
    "check_id": "rerun.station_logical_equivalence",
    "description": "Requires independently materialized canonical station tables to match logically.",
    "layer": "data_class",
    "required_profile_tags": [
      "data-class:station-time-series"
    ],
    "role": "hard_gate",
    "title": "Station exact-rerun equivalence"
  },
  {
    "check_id": "rerun.zarr_output_policy",
    "description": "Checks that the rerun preserves the publication policy.",
    "layer": "output_format",
    "required_profile_tags": [
      "zarr",
      "zarr-v3"
    ],
    "role": "hard_gate",
    "title": "Rerun Zarr v3 policy"
  },
  {
    "check_id": "robustness.alternate_contract",
    "description": "Runs the same candidate against another valid frozen contract.",
    "layer": "robustness",
    "required_profile_tags": [
      "robustness:alternate-contract"
    ],
    "role": "assessment",
    "title": "Unchanged-source alternate contract"
  },
  {
    "check_id": "security.execution_isolation",
    "description": "Checks network denial, write confinement, and secret-free execution.",
    "layer": "core",
    "required_profile_tags": [
      "core"
    ],
    "role": "hard_gate",
    "title": "Execution isolation"
  },
  {
    "check_id": "security.secret_scan",
    "description": "Scans candidate evidence and outputs for configured credential material.",
    "layer": "core",
    "required_profile_tags": [
      "core"
    ],
    "role": "hard_gate",
    "title": "Secret scan"
  },
  {
    "check_id": "semantic.native_values_and_missingness",
    "description": "Compares all mapped logical values and missingness against the oracle.",
    "layer": "data_class",
    "required_profile_tags": [
      "regular-grid"
    ],
    "role": "hard_gate",
    "title": "Native value equivalence"
  },
  {
    "check_id": "semantic.station_values_and_missingness",
    "description": "Compares all canonical keys, native values, and missingness with the oracle.",
    "layer": "data_class",
    "required_profile_tags": [
      "data-class:station-time-series"
    ],
    "role": "hard_gate",
    "title": "Station value equivalence"
  }
]

Evaluator-owned planning constraints and frozen source context:

{
  "oracle_requirement": "independent_trusted_reference",
  "provider_network_enabled": false,
  "source_mode": "immutable_local_fixture_replay",
  "target": {
    "data_class": "regular_rectilinear_grid",
    "engineering_profile": "general_pipeline",
    "output_format": "zarr",
    "output_format_version": 3,
    "workload_kind": "full_field_tensor"
  }
}

Return one concise planning draft. `domain_tags` should contain only broad
domain labels such as `earth-science`, not dataset names, providers, formats,
cadences, or variables. Do not repeat existing checks as proposals.
