Replayed interaction:

[
  {
    "role": "user",
    "content": "Prepare an ML-ready pipeline for ERA5 pressure levels. Use exactly this selected data and scope:\n{\n  \"advanced_options\": {\n    \"data_format\": \"grib\",\n    \"dataset_id\": \"reanalysis-era5-pressure-levels\",\n    \"download_format\": \"unarchived\",\n    \"product_type\": [\n      \"reanalysis\"\n    ]\n  },\n  \"credential_references\": [\n    \"CDSAPI_KEY\",\n    \"CDSAPI_URL\"\n  ],\n  \"dataset_family\": \"reanalysis\",\n  \"dataset_slug\": \"reanalysis_era5_pressure_levels\",\n  \"fields\": [\n    {\n      \"description\": null,\n      \"display_name\": \"Temperature\",\n      \"name\": \"temperature\",\n      \"selectors\": [\n        {\n          \"dimension\": \"pressure_level\",\n          \"label\": null,\n          \"unit\": \"hPa\",\n          \"value\": \"500\"\n        }\n      ],\n      \"units\": null\n    },\n    {\n      \"description\": null,\n      \"display_name\": \"Temperature\",\n      \"name\": \"temperature\",\n      \"selectors\": [\n        {\n          \"dimension\": \"pressure_level\",\n          \"label\": null,\n          \"unit\": \"hPa\",\n          \"value\": \"850\"\n        }\n      ],\n      \"units\": null\n    },\n    {\n      \"description\": null,\n      \"display_name\": \"Geopotential\",\n      \"name\": \"geopotential\",\n      \"selectors\": [\n        {\n          \"dimension\": \"pressure_level\",\n          \"label\": null,\n          \"unit\": \"hPa\",\n          \"value\": \"500\"\n        }\n      ],\n      \"units\": null\n    }\n  ],\n  \"provider\": \"Copernicus Climate Data Store\",\n  \"required_output\": {\n    \"format\": \"zarr\",\n    \"version\": 3\n  },\n  \"scope\": {\n    \"date_range\": {\n      \"end_date\": \"2024-01-07\",\n      \"inclusive\": true,\n      \"start_date\": \"2024-01-01\"\n    },\n    \"geography\": {\n      \"area\": \"global\",\n      \"cds_area\": [\n        90,\n        -180,\n        -90,\n        180\n      ],\n      \"cds_area_order\": [\n        \"north\",\n        \"west\",\n        \"south\",\n        \"east\"\n      ]\n    },\n    \"product_type\": \"reanalysis\",\n    \"time\": {\n      \"selected_times\": [\n        \"00:00\",\n        \"06:00\",\n        \"12:00\",\n        \"18:00\"\n      ],\n      \"timestep\": \"6 hours\",\n      \"timezone\": \"UTC\"\n    }\n  },\n  \"source_url\": \"https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels?tab=download\",\n  \"title\": \"ERA5 workload-aware expert prompt pilot\"\n}"
  },
  {
    "role": "system",
    "content": "Confirm the downstream access pattern and the relative importance of runtime, storage, and engineering objectives."
  },
  {
    "role": "user",
    "content": "The data will be read repeatedly during ML training. Correctness is mandatory. First maximise consumer throughput; second minimise native output size. Still report materialisation time and engineering quality for analysis, but do not use them to override those priorities."
  }
]

Trusted dataset inventory:

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
  "extraction_method": "deterministic",
  "extractor_name": "cds_process_metadata",
  "note": "Extracted from official CDS process, catalogue, form, and layout metadata without LLM calls.",
  "option_units": {
    "area": "north/west/south/east degrees",
    "pressure_level": "hPa",
    "time": "UTC"
  },
  "options": {
    "data_format": [
      "grib",
      "netcdf"
    ],
    "day": [
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
      "12",
      "13",
      "14",
      "15",
      "16",
      "17",
      "18",
      "19",
      "20",
      "21",
      "22",
      "23",
      "24",
      "25",
      "26",
      "27",
      "28",
      "29",
      "30",
      "31"
    ],
    "download_format": [
      "zip",
      "unarchived"
    ],
    "month": [
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
    ],
    "pressure_level": [
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
      "250",
      "3",
      "30",
      "300",
      "350",
      "400",
      "450",
      "5",
      "50",
      "500",
      "550",
      "600",
      "650",
      "7",
      "70",
      "700",
      "750",
      "775",
      "800",
      "825",
      "850",
      "875",
      "900",
      "925",
      "950",
      "975"
    ],
    "product_type": [
      "ensemble_mean",
      "ensemble_members",
      "ensemble_spread",
      "reanalysis"
    ],
    "time": [
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
      "11:00",
      "12:00",
      "13:00",
      "14:00",
      "15:00",
      "16:00",
      "17:00",
      "18:00",
      "19:00",
      "20:00",
      "21:00",
      "22:00",
      "23:00"
    ],
    "variable": [
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
      "temperature",
      "u_component_of_wind",
      "v_component_of_wind",
      "vertical_velocity",
      "vorticity"
    ],
    "year": [
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
      "1951",
      "1952",
      "1953",
      "1954",
      "1955",
      "1956",
      "1957",
      "1958",
      "1959",
      "1960",
      "1961",
      "1962",
      "1963",
      "1964",
      "1965",
      "1966",
      "1967",
      "1968",
      "1969",
      "1970",
      "1971",
      "1972",
      "1973",
      "1974",
      "1975",
      "1976",
      "1977",
      "1978",
      "1979",
      "1980",
      "1981",
      "1982",
      "1983",
      "1984",
      "1985",
      "1986",
      "1987",
      "1988",
      "1989",
      "1990",
      "1991",
      "1992",
      "1993",
      "1994",
      "1995",
      "1996",
      "1997",
      "1998",
      "1999",
      "2000",
      "2001",
      "2002",
      "2003",
      "2004",
      "2005",
      "2006",
      "2007",
      "2008",
      "2009",
      "2010",
      "2011",
      "2012",
      "2013",
      "2014",
      "2015",
      "2016",
      "2017",
      "2018",
      "2019",
      "2020",
      "2021",
      "2022",
      "2023",
      "2024",
      "2025",
      "2026"
    ]
  },
  "output_transmission": [
    "reference"
  ],
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

Supported capabilities and access references:

{
  "credential_references": [
    "CDSAPI_KEY",
    "CDSAPI_URL"
  ],
  "evaluation_targets": [
    {
      "data_class": "regular_rectilinear_grid",
      "engineering_profile": "general_pipeline",
      "output_format": "zarr",
      "output_format_version": 3,
      "workload_kind": "full_field_tensor"
    }
  ],
  "provider_network_during_evaluation": false,
  "required_output_format": "zarr"
}

Existing intention-space state:

{}

Produce one concise structured proposal. Copy the authoritative selection block exactly into the contract. Do not broaden, normalize, or duplicate its scope. Record only decisions and assumptions that materially affect execution or evaluation.
