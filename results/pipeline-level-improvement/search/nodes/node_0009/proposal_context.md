## Current F(p) and evaluator feedback
{
  "evaluation_evidence": {
    "detailed_feedback": {
      "deterministic_feedback": [],
      "engineering_components": [],
      "extensibility_probe": {
        "error": "No alternate-contract behavioral probe is configured.",
        "feedback_code": "EXTENSIBILITY_PROBE_NOT_CONFIGURED",
        "status": "not_assessed"
      },
      "operational_evidence": {
        "actionable_observations": [
          "Zarr metadata is 0.0% of measured output bytes.",
          "Consumer timing uses 28 samples and spans 204.8-213.9 samples/s within one run.",
          "The measured materialization path is cache-hit only; provider download changes cannot improve this objective.",
          "Candidate execution is 87.9% of initial end-to-end materialization time."
        ],
        "cache_evidence": {
          "initial": {
            "acquired": 0,
            "acquired_keys": [],
            "hits": 2,
            "misses": 0,
            "reused_keys": [
              "temperature",
              "geopotential"
            ]
          },
          "rerun": {
            "acquired": 0,
            "acquired_keys": [],
            "hits": 2,
            "misses": 0,
            "reused_keys": [
              "temperature",
              "geopotential"
            ]
          }
        },
        "consumer_workload": {
          "access_pattern": "shuffled",
          "dtype": "float32",
          "durations_seconds": [
            0.13092691699421266,
            0.13351191699621268,
            0.13460483399830991,
            0.13232387499738252,
            0.13671449999674223
          ],
          "effective_cache_mode": "warm",
          "kind": "full_field_tensor",
          "max_samples_per_second": 213.8597672871017,
          "median_samples_per_second": 209.71910695278441,
          "min_samples_per_second": 204.80636655707485,
          "repetitions": 5,
          "sample_shape": [
            3,
            721,
            1440
          ],
          "samples": 28,
          "samples_per_second": [
            213.8597672871017,
            209.71910695278441,
            208.01630348841383,
            211.6020257157211,
            204.80636655707485
          ]
        },
        "objective_repetitions": [
          {
            "consumer_samples_per_second": 209.71910695278441,
            "materialization_seconds": 2.9956199579974054,
            "output_bytes": 348881582,
            "repetition": 1
          }
        ],
        "output_storage": {
          "chunk_bytes": 348850906,
          "chunk_object_count": 91,
          "consolidated_metadata": true,
          "dataset_open_latency_seconds": {
            "maximum": 0.0016158750004251488,
            "median": 0.0012322920010774396,
            "minimum": 0.0012067499992554076,
            "repetitions": 5,
            "values": [
              0.0016158750004251488,
              0.0012067499992554076,
              0.0012322920010774396,
              0.0013304160020197742,
              0.0012148329988121986
            ]
          },
          "metadata_bytes": 30676,
          "metadata_object_count": 13,
          "object_count": 104,
          "output_bytes": 348881582,
          "zarr_format": 3
        },
        "resources": {
          "initial": {
            "peak_rss_bytes": 1376436224,
            "read_bytes": null,
            "sample_interval_seconds": 0.05,
            "system_cpu_seconds": 0.757116608,
            "user_cpu_seconds": 1.581123456,
            "write_bytes": null
          },
          "rerun": {
            "peak_rss_bytes": 1352794112,
            "read_bytes": null,
            "sample_interval_seconds": 0.05,
            "system_cpu_seconds": 0.672084736,
            "user_cpu_seconds": 1.548656384,
            "write_bytes": null
          }
        },
        "timings": {
          "initial": {
            "execution_seconds": 2.9956199579974054,
            "setup_seconds": 0.15050287500343984,
            "total_seconds": 3.4063520830022753,
            "validation_seconds": 0.2542310839999118
          },
          "rerun": {
            "execution_seconds": 2.1018246250023367,
            "setup_seconds": 0.13978262500313576,
            "total_seconds": 2.505448500000057,
            "validation_seconds": 0.2577275420044316
          }
        },
        "variation": {
          "consumer_samples_per_second": {
            "maximum": 209.71910695278441,
            "median": 209.71910695278441,
            "minimum": 209.71910695278441,
            "relative_span": 0.0
          },
          "materialization_seconds": {
            "maximum": 2.9956199579974054,
            "median": 2.9956199579974054,
            "minimum": 2.9956199579974054,
            "relative_span": 0.0
          },
          "output_bytes": {
            "maximum": 348881582.0,
            "median": 348881582.0,
            "minimum": 348881582.0,
            "relative_span": 0.0
          }
        }
      }
    },
    "evaluation_report": "<REPOSITORY_ROOT>/project/datasets/reanalysis_era5_pressure_levels/benchmarks/experiments/pipeline_pareto_tensor_20260907_v3/workspace/runs/self_improvement/si-era5-tensor-pareto-20-v3-20260907/nodes/node_0003/search_measurement.json",
    "summary": {
      "constraints": {
        "contract_correctness": "pass",
        "provenance_security": "pass",
        "rerun_safety": "pass",
        "semantic_equivalence": "pass"
      },
      "diagnostic_engineering_quality": {},
      "diagnostic_operational_metrics": {
        "consumer_samples_per_second": 209.71910695278441,
        "materialization_seconds": 2.9956199579974054,
        "output_bytes": 348881582
      },
      "engineering_assessed": false,
      "engineering_objective": {},
      "engineering_profile": "general_pipeline",
      "feasible": true,
      "feedback_codes": [
        "EXTENSIBILITY_PROBE_NOT_CONFIGURED"
      ],
      "objective_directions": {
        "consumer_samples_per_second": "maximize",
        "materialization_seconds": "minimize",
        "output_bytes": "minimize",
        "q_engineering": "maximize"
      },
      "objective_groups": {
        "engineering": {
          "objectives": {},
          "priority": "secondary",
          "profile": "general_pipeline"
        },
        "operational": {
          "objectives": {
            "consumer_samples_per_second": {
              "direction": "maximize",
              "unit": "samples_per_second",
              "value": 209.71910695278441
            },
            "materialization_seconds": {
              "direction": "minimize",
              "unit": "seconds",
              "value": 2.9956199579974054
            },
            "output_bytes": {
              "direction": "minimize",
              "unit": "bytes",
              "value": 348881582
            }
          },
          "priority": "primary"
        }
      },
      "objective_vector_complete": false,
      "operational_objectives": {
        "consumer_samples_per_second": 209.71910695278441,
        "materialization_seconds": 2.9956199579974054,
        "output_bytes": 348881582
      },
      "optimization_ready": false,
      "target": "pipeline-era5-pareto-baseline-r1-20260907",
      "thesis_evidence_ready": false
    }
  },
  "frozen_search_invariants": {
    "candidate_channel_mappings": [
      {
        "candidate": {
          "array_path": "temperature_pressure_level_500",
          "indices": {},
          "selector_coordinate_paths": {
            "pressure_level": "pressure_level__temperature_pressure_level_500"
          },
          "selectors": {
            "pressure_level": "500"
          },
          "store_path": "{output_dir}/dataset.zarr"
        },
        "field_id": "temperature[pressure_level=\"500\"]"
      },
      {
        "candidate": {
          "array_path": "temperature_pressure_level_850",
          "indices": {},
          "selector_coordinate_paths": {
            "pressure_level": "pressure_level__temperature_pressure_level_850"
          },
          "selectors": {
            "pressure_level": "850"
          },
          "store_path": "{output_dir}/dataset.zarr"
        },
        "field_id": "temperature[pressure_level=\"850\"]"
      },
      {
        "candidate": {
          "array_path": "geopotential_pressure_level_500",
          "indices": {},
          "selector_coordinate_paths": {
            "pressure_level": "pressure_level__geopotential_pressure_level_500"
          },
          "selectors": {
            "pressure_level": "500"
          },
          "store_path": "{output_dir}/dataset.zarr"
        },
        "field_id": "geopotential[pressure_level=\"500\"]"
      }
    ],
    "candidate_grid": {
      "dimensions": {
        "sample": "time",
        "x": "longitude",
        "y": "latitude"
      },
      "sample_coordinate": {
        "array_path": "time",
        "expected_dtype": null,
        "expected_fill_value": null,
        "expected_step": null,
        "expected_step_seconds": 21600.0,
        "metadata": {},
        "monotonic": "increasing",
        "normalization": "cf_datetime_ascending",
        "store_path": "{output_dir}/dataset.zarr"
      },
      "x_coordinate": {
        "array_path": "longitude",
        "expected_dtype": null,
        "expected_fill_value": null,
        "expected_step": 0.25,
        "expected_step_seconds": null,
        "metadata": {},
        "monotonic": "increasing",
        "normalization": "longitude_modulo_360",
        "store_path": "{output_dir}/dataset.zarr"
      },
      "y_coordinate": {
        "array_path": "latitude",
        "expected_dtype": null,
        "expected_fill_value": null,
        "expected_step": 0.25,
        "expected_step_seconds": null,
        "metadata": {},
        "monotonic": "increasing",
        "normalization": "ascending",
        "store_path": "{output_dir}/dataset.zarr"
      }
    },
    "candidate_id": "pipeline-era5-pareto-baseline-r1-20260907",
    "command_template": [
      "python",
      "run_pipeline.py",
      "--contract",
      "{contract_lock_json}",
      "--inventory",
      "{dataset_inventory_json}",
      "--cache-dir",
      "{cache_dir}",
      "--output-dir",
      "{output_dir}",
      "--run-receipt",
      "{pipeline_run_json}"
    ],
    "output_policy": {
      "consolidated_metadata": true,
      "coordinate_metadata": {
        "sample": {},
        "x": {},
        "y": {}
      },
      "dataset_class": "regular_rectilinear_grid",
      "format_version": 3,
      "open_latency_repetitions": 5,
      "schema_version": "regular_grid_zarr_output_policy.v1"
    },
    "protected_source_paths": [
      "manifest.json",
      "pipeline_contract.json",
      "requirements.txt",
      "run_pipeline.py"
    ],
    "rule": "These names, paths, mappings, policies, identities, and command semantics are frozen evaluator interfaces. Preserve them exactly."
  },
  "search_directive": {
    "documentation_edits_allowed": false,
    "documentation_proposal_limit": 0,
    "documentation_proposals_used": 0,
    "iteration": 9,
    "near_duplicate_similarity_threshold": 0.72,
    "proposal_focus": "throughput",
    "required_target_objectives": [
      "consumer_samples_per_second"
    ],
    "rules": [
      "target_objectives must exactly match required_target_objectives",
      "do not edit Markdown unless documentation_edits_allowed is true",
      "propose a causal mechanism distinct from all prior proposals",
      "use operational_evidence when targeting operational objectives"
    ]
  }
}

## Prior experiments
[
  {
    "admitted_to_archive": true,
    "branch_id": null,
    "failure": null,
    "node_id": "node_0000_root",
    "objective": {
      "consumer_samples_per_second": 44.985339599048366,
      "materialization_seconds": 3.169357165999827,
      "output_bytes": 182307091,
      "q_engineering": null
    },
    "objective_changes": {},
    "parent_id": null,
    "parent_relation": "not_comparable",
    "phase": "root",
    "policy_reason": "Initialized the Pareto archive.",
    "status": "optimization_ready"
  },
  {
    "admitted_to_archive": true,
    "branch_id": null,
    "failure": null,
    "node_id": "node_0001",
    "objective": {
      "consumer_samples_per_second": 54.33852261922518,
      "materialization_seconds": 2.350469040997268,
      "output_bytes": 175196006,
      "q_engineering": null
    },
    "objective_changes": {
      "consumer_samples_per_second": 0.20791624790523255,
      "output_bytes": 0.039006080131024634
    },
    "parent_id": "node_0000_root",
    "parent_relation": "dominates",
    "phase": "pareto_archive",
    "policy_reason": "Admitted and dominated ['node_0000_root'].",
    "proposal": {
      "edit_paths": [
        "pipeline_impl.py"
      ],
      "has_documentation_edits": false,
      "hypothesis": "Operational evidence shows the measured consumer workload is warm-cache, shuffled full-field tensor access over 28 samples with sample shape [3, 721, 1440], while provider/cache changes cannot improve it. If each published data variable is written with chunks aligned to one time sample, the consumer should read each shuffled full-field sample using one spatially contiguous chunk per channel instead of assembling it from several spatial chunks, increasing consumer_samples_per_second without changing names, selectors, coordinates, metadata validation, or public interface.",
      "target_objectives": [
        "consumer_samples_per_second"
      ],
      "title": "Chunk published fields by individual full-field samples"
    },
    "status": "optimization_ready"
  },
  {
    "admitted_to_archive": true,
    "branch_id": null,
    "failure": null,
    "node_id": "node_0002",
    "objective": {
      "consumer_samples_per_second": 55.99926271340618,
      "materialization_seconds": 2.602119416995265,
      "output_bytes": 161615129,
      "q_engineering": null
    },
    "objective_changes": {
      "consumer_samples_per_second": 0.030562849597854637,
      "output_bytes": 0.0775181883997972
    },
    "parent_id": "node_0001",
    "parent_relation": "dominates",
    "phase": "pareto_archive",
    "policy_reason": "Admitted and dominated ['node_0001'].",
    "proposal": {
      "edit_paths": [
        "pipeline_impl.py",
        "pipeline_impl.py"
      ],
      "has_documentation_edits": false,
      "hypothesis": "Operational evidence shows Zarr metadata is 0.0% of measured output bytes and chunk payloads account for nearly all 175196006 output bytes, while the materialization path is cache-hit only. If the published data arrays keep the frozen one-sample chunk layout but explicitly use a higher Zstd compression level for data chunks, the evaluator should measure lower output_bytes without changing public array paths, coordinates, selectors, metadata validation, cache behavior, or semantic values.",
      "target_objectives": [
        "output_bytes"
      ],
      "title": "Use stronger Zstd compression for published data chunks"
    },
    "status": "optimization_ready"
  },
  {
    "admitted_to_archive": true,
    "branch_id": null,
    "failure": null,
    "node_id": "node_0003",
    "objective": {
      "consumer_samples_per_second": 209.71910695278441,
      "materialization_seconds": 2.9956199579974054,
      "output_bytes": 348881582,
      "q_engineering": null
    },
    "objective_changes": {
      "consumer_samples_per_second": 2.7450333592084566,
      "output_bytes": -1.1587185813526157
    },
    "parent_id": "node_0002",
    "parent_relation": "tradeoff",
    "phase": "pareto_archive",
    "policy_reason": "Admitted as a non-dominated tradeoff.",
    "proposal": {
      "edit_paths": [
        "pipeline_impl.py",
        "pipeline_impl.py"
      ],
      "has_documentation_edits": false,
      "hypothesis": "Operational evidence shows the measured consumer workload is warm-cache, shuffled full-field tensor access over 28 samples, and provider/cache download changes cannot improve it. The current one-sample full-field chunks require Zstd decompression for every channel chunk read. If published data chunks are stored uncompressed while preserving the same array paths, dimensions, coordinates, selectors, validation, and Zarr v3 consolidated store policy, the consumer should spend less CPU per full-field sample and increase consumer_samples_per_second.",
      "target_objectives": [
        "consumer_samples_per_second"
      ],
      "title": "Bypass data-chunk compression for warm full-field reads"
    },
    "status": "optimization_ready"
  },
  {
    "admitted_to_archive": true,
    "branch_id": null,
    "failure": null,
    "node_id": "node_0004",
    "objective": {
      "consumer_samples_per_second": 61.58475161517731,
      "materialization_seconds": 2.97830729099951,
      "output_bytes": 109455072,
      "q_engineering": null
    },
    "objective_changes": {
      "consumer_samples_per_second": 0.09974218643478616,
      "output_bytes": 0.3227424147896451
    },
    "parent_id": "node_0002",
    "parent_relation": "dominates",
    "phase": "pareto_archive",
    "policy_reason": "Admitted and dominated ['node_0002'].",
    "proposal": {
      "edit_paths": [
        "pipeline_impl.py",
        "pipeline_impl.py"
      ],
      "has_documentation_edits": false,
      "hypothesis": "Operational evidence shows the consumer benchmark is warm-cache shuffled full-field tensor access, and output bytes are almost entirely data chunks rather than metadata. If the existing one-sample full-field chunks are encoded with Blosc's lossless bitshuffle preconditioner plus Zstd instead of plain Zstd, smooth float32 ERA5 fields should compress to fewer chunk bytes while Blosc's blocked decode path and reduced compressed payload should improve per-sample read throughput. This preserves all frozen store paths, array names, dimensions, selector coordinates, Zarr v3 consolidated publication, validation, and semantic values.",
      "target_objectives": [
        "consumer_samples_per_second",
        "output_bytes"
      ],
      "title": "Use byte-efficient Blosc bitshuffle with Zstd for data chunks"
    },
    "status": "optimization_ready"
  },
  {
    "admitted_to_archive": false,
    "branch_id": null,
    "failure": null,
    "node_id": "node_0005",
    "objective": {
      "consumer_samples_per_second": 91.55756160748207,
      "materialization_seconds": 2.2601829999985057,
      "output_bytes": 348886346,
      "q_engineering": null
    },
    "objective_changes": {
      "consumer_samples_per_second": -0.5634276583673652,
      "output_bytes": -1.365506305231097e-05
    },
    "parent_id": "node_0003",
    "parent_relation": "dominated",
    "phase": "pareto_archive",
    "policy_reason": "Pareto-dominated by ['node_0003'].",
    "proposal": {
      "edit_paths": [
        "pipeline_impl.py"
      ],
      "has_documentation_edits": false,
      "hypothesis": "Operational evidence shows the consumer benchmark is warm-cache shuffled full-field tensor access over 28 samples, while the current uncompressed one-sample chunks still create 91 chunk objects and candidate execution dominates initial materialization time. If each published channel keeps the same logical one-sample Zarr chunks but stores all time chunks for that channel inside a single Zarr v3 shard, shuffled full-field reads should avoid repeated traversal/open of many per-time chunk objects while preserving the exact array paths, dimensions, selector coordinates, values, validation, Zarr v3 consolidated policy, and public interface. This should increase consumer_samples_per_second without relying on provider/cache changes.",
      "target_objectives": [
        "consumer_samples_per_second"
      ],
      "title": "Shard per-sample chunks by channel for shuffled warm reads"
    },
    "status": "optimization_ready"
  },
  {
    "admitted_to_archive": false,
    "branch_id": null,
    "failure": null,
    "node_id": "node_0006",
    "objective": {
      "consumer_samples_per_second": 61.613328502656316,
      "materialization_seconds": 3.3259435829968425,
      "output_bytes": 115493408,
      "q_engineering": null
    },
    "objective_changes": {
      "consumer_samples_per_second": 0.00046402537526777087,
      "output_bytes": -0.05516725620535885
    },
    "parent_id": "node_0004",
    "parent_relation": "dominated",
    "phase": "pareto_archive",
    "policy_reason": "Pareto-dominated by ['node_0004'].",
    "proposal": {
      "edit_paths": [
        "pipeline_impl.py"
      ],
      "has_documentation_edits": false,
      "hypothesis": "Operational evidence shows Zarr metadata is 0.0% of measured output bytes and chunk payloads account for essentially all 109455072 output bytes, so footprint changes must target data-chunk encoding rather than metadata or provider/cache behavior. If the existing lossless Blosc+Zstd compressor uses byte-shuffle instead of bitshuffle for the published float32 ERA5 fields, stable high-order float bytes across smooth spatial grids may form longer byte runs for Zstd than bit-plane rearrangement of noisy mantissa bits, reducing output_bytes while preserving all frozen array paths, dimensions, coordinates, selector paths, Zarr v3 consolidated publication, validation, and semantic values.",
      "target_objectives": [
        "output_bytes"
      ],
      "title": "Use Blosc byte-shuffle for float32 chunk compression"
    },
    "status": "optimization_ready"
  },
  {
    "admitted_to_archive": true,
    "branch_id": null,
    "failure": null,
    "node_id": "node_0007",
    "objective": {
      "consumer_samples_per_second": 81.51117859147547,
      "materialization_seconds": 2.163911415998882,
      "output_bytes": 124939790,
      "q_engineering": null
    },
    "objective_changes": {
      "consumer_samples_per_second": 0.3235610512941871,
      "output_bytes": -0.14147099551494516
    },
    "parent_id": "node_0004",
    "parent_relation": "tradeoff",
    "phase": "pareto_archive",
    "policy_reason": "Admitted as a non-dominated tradeoff.",
    "proposal": {
      "edit_paths": [
        "pipeline_impl.py"
      ],
      "has_documentation_edits": false,
      "hypothesis": "Operational evidence shows the consumer benchmark is warm-cache shuffled full-field tensor access over 28 samples, while provider/cache download changes cannot improve this objective. With the frozen one-sample full-field chunk layout, each sample read still pays decompression CPU for three large float32 channel chunks. If the data chunks keep the same Blosc bitshuffle preconditioner but switch from Zstd to the lower-CPU lossless LZ4 codec, warm full-field reads should decode faster and increase consumer_samples_per_second without changing array paths, dimensions, coordinates, selector paths, validation, provenance, rerun behavior, or the public pipeline interface.",
      "target_objectives": [
        "consumer_samples_per_second"
      ],
      "title": "Use Blosc LZ4 for lower-CPU full-field chunk reads"
    },
    "status": "optimization_ready"
  },
  {
    "admitted_to_archive": true,
    "branch_id": null,
    "failure": null,
    "node_id": "node_0008",
    "objective": {
      "consumer_samples_per_second": 75.30366883429723,
      "materialization_seconds": 3.102549832998193,
      "output_bytes": 114727843,
      "q_engineering": null
    },
    "objective_changes": {
      "consumer_samples_per_second": -0.0761553183801888,
      "output_bytes": 0.08173494608883207
    },
    "parent_id": "node_0007",
    "parent_relation": "tradeoff",
    "phase": "pareto_archive",
    "policy_reason": "Admitted as a non-dominated tradeoff.",
    "proposal": {
      "edit_paths": [
        "pipeline_impl.py"
      ],
      "has_documentation_edits": false,
      "hypothesis": "Operational evidence shows the consumer benchmark is warm-cache shuffled full-field tensor access, provider/cache changes cannot improve it, and Zarr metadata is 0.0% of measured output bytes, so the joint lever is data-chunk encoding. If the existing one-sample full-field chunks keep Blosc bitshuffle but switch from normal LZ4 to LZ4HC, chunk writes should use a higher-compression LZ4-family encoder while reads retain the same fast LZ4 block decoder path. This should reduce output_bytes and may improve consumer_samples_per_second by moving fewer compressed bytes per sample without changing array paths, dimensions, coordinates, selector paths, validation, provenance, rerun behavior, dependencies, or the public interface.",
      "target_objectives": [
        "consumer_samples_per_second",
        "output_bytes"
      ],
      "title": "Use LZ4HC blocks for smaller chunks with LZ4-speed reads"
    },
    "status": "optimization_ready"
  }
]

## Editable files
[
  "README.md",
  "pipeline_impl.py",
  "pyproject.toml"
]

At most 3 exact replacements may be proposed across the editable
files. Target no more than two objectives and state expected tradeoffs. Prior
experiments that failed or were Pareto-rejected should not be repeated.

## Current candidate source
FILE: README.md
```
# ERA5 pressure-level family adapter

This directory contains a standalone deterministic implementation of `pipeline_impl.run_pipeline(contract_lock, inventory, cache_dir, output_dir)` for the frozen ERA5 pressure-level inventory and the fixed regular-grid Zarr publication policy.

The adapter validates the selected `dataset_contract.v1` from a full lock envelope, checks requested variables, pressure-level selectors, product type, date/time selections, area bounds, and inventory enumerations, then materializes a consolidated Zarr v3 store at `dataset.zarr` under the supplied output directory.

`cache_dir` is treated as an external framework-owned cache/input root. If `cache_dir/source_fixture_manifest.json` exists, every listed file is verified by size and SHA-256 before source decoding. A complete fixture runs read-only and credential-free; this implementation deliberately performs no provider construction or network acquisition when no verified fixture is present.

Source fixtures may be GRIB, CF NetCDF, or Zarr. CF decoding uses xarray with mask-and-scale enabled; source encodings are cleared before publication so decoded values are not silently repacked in the public Zarr. Each requested field-selector pair is published as its own data array, and the selected `pressure_level` is preserved as a real length-one selector dimension rather than converted to a scalar coordinate.

```

FILE: pipeline_impl.py
```
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

PIPELINE_ID = 'pipeline-era5-pareto-baseline-r1-20260907'
DATASET_SLUG = 'reanalysis_era5_pressure_levels'
DATASET_ID = 'reanalysis-era5-pressure-levels'
ZARR_STORE_NAME = 'dataset.zarr'

FIELD_ALIASES = {
    'temperature': ('temperature', 't'),
    'geopotential': ('geopotential', 'z'),
    'u_component_of_wind': ('u_component_of_wind', 'u'),
    'v_component_of_wind': ('v_component_of_wind', 'v'),
    'relative_humidity': ('relative_humidity', 'r'),
    'specific_humidity': ('specific_humidity', 'q'),
    'vertical_velocity': ('vertical_velocity', 'w'),
    'vorticity': ('vorticity', 'vo'),
    'divergence': ('divergence', 'd'),
}

TIME_NAMES = ('time', 'valid_time', 'timestamp')
LAT_NAMES = ('latitude', 'lat', 'y')
LON_NAMES = ('longitude', 'lon', 'x')
PLEV_NAMES = ('pressure_level', 'level', 'isobaricInhPa', 'plev')


class ContractError(ValueError):
    pass


def run_pipeline(contract_lock: dict[str, Any], inventory: dict[str, Any], cache_dir: str | os.PathLike[str], output_dir: str | os.PathLike[str]) -> dict[str, Any]:
    '''Materialize a validated ERA5 pressure-level regular-grid contract to consolidated Zarr v3.

    The function is intentionally framework-independent.  If cache_dir contains
    source_fixture_manifest.json, every listed raw source object is verified and
    consumed before any provider or credential path.  This adapter does not use
    credentials and deliberately performs no network access when no complete
    fixture is supplied.
    '''
    contract = _extract_contract(contract_lock)
    request = _validate_contract(contract, inventory)

    cache_root = Path(cache_dir)
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    fixture = _verify_fixture(cache_root)
    if not fixture:
        raise RuntimeError('No verified source_fixture_manifest.json was found in cache_dir; network/provider acquisition is intentionally disabled for this fixture-authoritative adapter.')

    ds = _open_fixture_dataset([entry['path'] for entry in fixture['entries']])
    ds = _standardize_grid_names(ds)
    ds = _filter_dataset(ds, request)
    public_ds, channels = _build_public_dataset(ds, contract, inventory, request)

    final_store = out_root / ZARR_STORE_NAME
    _publish_zarr_atomically(public_ds, final_store, request, channels)
    _validate_published_zarr(final_store, request, channels)

    return {
        'cache': {
            'hits': len(fixture['entries']),
            'misses': 0,
            'acquired': 0,
            'reused_keys': [entry['entry_id'] for entry in fixture['entries']],
            'acquired_keys': [],
        },
        'dataset_artifact': {
            'schema_version': 'dataset_artifact_layout.v1',
            'storage_format': 'zarr',
            'store_path': ZARR_STORE_NAME,
            'dimensions': {
                'sample': 'time',
                'y': 'latitude',
                'x': 'longitude',
            },
            'coordinates': {
                'sample': 'time',
                'y': 'latitude',
                'x': 'longitude',
            },
            'channels': channels,
        },
        'warnings': fixture['warnings'],
    }


def _extract_contract(lock: dict[str, Any]) -> dict[str, Any]:
    if isinstance(lock, dict) and lock.get('schema_version') == 'dataset_contract.v1':
        return lock
    preferred_keys = ('contract', 'dataset_contract', 'selected_contract', 'runtime_contract')
    for key in preferred_keys:
        value = lock.get(key) if isinstance(lock, dict) else None
        if isinstance(value, dict) and value.get('schema_version') == 'dataset_contract.v1':
            return value
    found: list[dict[str, Any]] = []

    def walk(obj: Any) -> None:
        if isinstance(obj, dict):
            if obj.get('schema_version') == 'dataset_contract.v1':
                found.append(obj)
                return
            for value in obj.values():
                walk(value)
        elif isinstance(obj, list):
            for value in obj:
                walk(value)

    walk(lock)
    matching = [c for c in found if c.get('dataset_slug') == DATASET_SLUG]
    if len(matching) == 1:
        return matching[0]
    if not matching:
        raise ContractError('No dataset_contract.v1 for reanalysis_era5_pressure_levels was found in the supplied lock envelope.')
    raise ContractError('Multiple matching contracts were found; the lock envelope must identify one selected contract.')


def _validate_contract(contract: dict[str, Any], inventory: dict[str, Any]) -> dict[str, Any]:
    if inventory.get('schema_version') != 'dataset_inventory.v1':
        raise ContractError('inventory.schema_version must be dataset_inventory.v1')
    if contract.get('schema_version') != 'dataset_contract.v1':
        raise ContractError('contract.schema_version must be dataset_contract.v1')
    if contract.get('dataset_slug') != inventory.get('dataset_slug') or contract.get('dataset_slug') != DATASET_SLUG:
        raise ContractError('contract dataset_slug must match the frozen inventory')
    if inventory.get('dataset_id', DATASET_ID) != DATASET_ID:
        raise ContractError('inventory dataset_id does not match the fixed policy')
    if contract.get('human_confirmed') is not True:
        raise ContractError('contract.human_confirmed must be true')

    options = inventory.get('options') or {}
    fields = contract.get('fields') or []
    if not fields:
        raise ContractError('At least one field is required')

    selected_levels: list[str] = []
    field_specs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for field in fields:
        name = field.get('name')
        if name not in options.get('variable', []):
            raise ContractError(f'Unsupported variable: {name!r}')
        selectors = field.get('selectors') or []
        plev = None
        for selector in selectors:
            dim = selector.get('dimension')
            if dim != 'pressure_level':
                raise ContractError(f'Unsupported selector dimension for ERA5 pressure-level data: {dim!r}')
            value = str(selector.get('value'))
            if value not in options.get('pressure_level', []):
                raise ContractError(f'Unsupported pressure_level: {value!r}')
            plev = value
        if plev is None:
            raise ContractError(f'Field {name!r} must include a pressure_level selector')
        fid = _canonical_field_id(field)
        if fid in seen:
            raise ContractError(f'Duplicate field-selector request: {fid}')
        seen.add(fid)
        selected_levels.append(plev)
        field_specs.append({'field': field, 'name': name, 'pressure_level': plev, 'field_id': fid})

    scope = contract.get('scope') or {}
    product_type = scope.get('product_type') or (contract.get('advanced_options') or {}).get('product_type') or (inventory.get('defaults') or {}).get('product_type')
    if isinstance(product_type, str):
        product_types = [product_type]
    else:
        product_types = list(product_type or [])
    if not product_types:
        raise ContractError('product_type is required')
    invalid_pt = [pt for pt in product_types if pt not in options.get('product_type', [])]
    if invalid_pt:
        raise ContractError(f'Unsupported product_type values: {invalid_pt!r}')
    if product_types != ['reanalysis']:
        raise ContractError('The fixed publication policy for this adapter supports product_type=[reanalysis] only.')

    adv = contract.get('advanced_options') or {}
    data_format = adv.get('data_format', (inventory.get('defaults') or {}).get('data_format'))
    download_format = adv.get('download_format', (inventory.get('defaults') or {}).get('download_format'))
    if data_format not in options.get('data_format', []):
        raise ContractError(f'Unsupported data_format: {data_format!r}')
    if data_format != 'grib':
        raise ContractError('The fixed acquisition policy requires data_format=grib in the runtime lock.')
    if download_format not in options.get('download_format', []):
        raise ContractError(f'Unsupported download_format: {download_format!r}')

    date_range = (scope.get('date_range') or {})
    start_date = _parse_date(date_range.get('start_date'), 'start_date')
    end_date = _parse_date(date_range.get('end_date'), 'end_date')
    if end_date < start_date:
        raise ContractError('date_range.end_date must be on or after start_date')
    inclusive = bool(date_range.get('inclusive', True))

    time_scope = scope.get('time') or {}
    if time_scope.get('timezone', 'UTC') != 'UTC':
        raise ContractError('Only UTC selected_times are supported')
    selected_times = list(time_scope.get('selected_times') or [])
    if not selected_times:
        raise ContractError('scope.time.selected_times is required')
    invalid_times = [t for t in selected_times if t not in options.get('time', [])]
    if invalid_times:
        raise ContractError(f'Unsupported selected_times: {invalid_times!r}')

    requested_timestamps = _requested_timestamps(start_date, end_date, inclusive, selected_times)
    _validate_temporal_options(requested_timestamps, options)
    _validate_temporal_extent(requested_timestamps, inventory)

    geography = scope.get('geography') or {}
    area = geography.get('cds_area', (inventory.get('defaults') or {}).get('area'))
    if not isinstance(area, list) or len(area) != 4:
        raise ContractError('scope.geography.cds_area must be [north, west, south, east]')
    north, west, south, east = [float(v) for v in area]
    if not (-90 <= south <= north <= 90):
        raise ContractError('Latitude bounds must satisfy -90 <= south <= north <= 90')
    if not (-360 <= west <= 360 and -360 <= east <= 360):
        raise ContractError('Longitude bounds must be within [-360, 360]')

    return {
        'field_specs': field_specs,
        'variables': sorted({spec['name'] for spec in field_specs}),
        'pressure_levels': _unique_preserve_order(selected_levels),
        'timestamps': requested_timestamps,
        'area': [north, west, south, east],
        'product_type': product_types,
        'data_format': data_format,
        'download_format': download_format,
    }


def _parse_date(value: Any, name: str) -> pd.Timestamp:
    if not isinstance(value, str):
        raise ContractError(f'date_range.{name} must be an ISO date string')
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert('UTC').tz_localize(None)
    return ts.normalize()


def _requested_timestamps(start: pd.Timestamp, end: pd.Timestamp, inclusive: bool, selected_times: list[str]) -> pd.DatetimeIndex:
    final_day = end if inclusive else end - pd.Timedelta(days=1)
    if final_day < start:
        return pd.DatetimeIndex([], dtype='datetime64[ns]')
    days = pd.date_range(start, final_day, freq='D')
    values: list[pd.Timestamp] = []
    for day in days:
        for time_text in selected_times:
            hour, minute = [int(part) for part in time_text.split(':')]
            values.append(day + pd.Timedelta(hours=hour, minutes=minute))
    return pd.DatetimeIndex(values).astype('datetime64[ns]')


def _validate_temporal_options(timestamps: pd.DatetimeIndex, options: dict[str, Any]) -> None:
    years = set(options.get('year', []))
    months = set(options.get('month', []))
    days = set(options.get('day', []))
    for ts in timestamps:
        if f'{ts.year:04d}' not in years or f'{ts.month:02d}' not in months or f'{ts.day:02d}' not in days:
            raise ContractError(f'Requested date is outside inventory enumerations: {ts.date()}')


def _validate_temporal_extent(timestamps: pd.DatetimeIndex, inventory: dict[str, Any]) -> None:
    intervals = (((inventory.get('catalogue_metadata') or {}).get('extent') or {}).get('temporal') or {}).get('interval') or []
    if not intervals or not timestamps.size:
        return
    start_raw, end_raw = intervals[0]
    start = pd.Timestamp(start_raw).tz_convert('UTC').tz_localize(None) if pd.Timestamp(start_raw).tzinfo else pd.Timestamp(start_raw)
    end = pd.Timestamp(end_raw).tz_convert('UTC').tz_localize(None) if pd.Timestamp(end_raw).tzinfo else pd.Timestamp(end_raw)
    if timestamps.min() < start or timestamps.max() > end + pd.Timedelta(days=1):
        raise ContractError('Requested timestamps are outside the inventory temporal extent')


def _verify_fixture(cache_root: Path) -> dict[str, Any] | None:
    manifest_path = cache_root / 'source_fixture_manifest.json'
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('schema_version') != 'source_fixture_manifest.v1':
        raise ContractError('source_fixture_manifest.json has an unsupported schema_version')
    entries = manifest.get('entries') or []
    if not entries:
        raise ContractError('source_fixture_manifest.json must contain at least one entry')
    verified: list[dict[str, Any]] = []
    warnings: list[str] = []
    for i, entry in enumerate(entries):
        rel = entry.get('relative_path')
        if not isinstance(rel, str) or rel.startswith('/'):
            raise ContractError(f'Fixture entry {i} has an invalid relative_path')
        path = (cache_root / rel).resolve()
        if cache_root.resolve() not in (path, *path.parents):
            raise ContractError(f'Fixture entry {i} escapes cache_dir')
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f'Fixture file not found: {rel}')
        expected_size = int(entry.get('size_bytes'))
        actual_size = path.stat().st_size
        if expected_size <= 0 or actual_size != expected_size:
            raise ContractError(f'Fixture size mismatch for {rel}')
        expected_sha = str(entry.get('sha256', '')).lower()
        actual_sha = _sha256(path)
        if not re.fullmatch(r'[0-9a-f]{64}', expected_sha) or actual_sha != expected_sha:
            raise ContractError(f'Fixture SHA-256 mismatch for {rel}')
        suffix = ''.join(path.suffixes).lower()
        if not any(suffix.endswith(ext) for ext in ('.grib', '.grb', '.grib2', '.nc', '.nc4', '.cdf', '.zarr')):
            warnings.append(f'Verified fixture entry {entry.get("entry_id", rel)} has an uncommon extension: {path.name}')
        verified.append({'entry_id': str(entry.get('entry_id') or rel), 'path': path, 'sha256': actual_sha})
    return {'entries': verified, 'warnings': warnings}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _open_fixture_dataset(paths: list[Path]) -> xr.Dataset:
    datasets: list[xr.Dataset] = []
    for path in paths:
        suffixes = ''.join(path.suffixes).lower()
        if suffixes.endswith('.zarr'):
            ds = xr.open_zarr(path, consolidated=False).load()
        elif suffixes.endswith(('.grib', '.grb', '.grib2')):
            ds = xr.open_dataset(path, engine='cfgrib', backend_kwargs={'indexpath': ''}, decode_cf=True, mask_and_scale=True).load()
        else:
            ds = xr.open_dataset(path, decode_cf=True, mask_and_scale=True).load()
        datasets.append(ds)
    if len(datasets) == 1:
        return datasets[0]
    return xr.combine_by_coords(datasets, combine_attrs='drop_conflicts').load()


def _standardize_grid_names(ds: xr.Dataset) -> xr.Dataset:
    renames: dict[str, str] = {}
    for wanted, candidates in [('time', TIME_NAMES), ('latitude', LAT_NAMES), ('longitude', LON_NAMES), ('pressure_level', PLEV_NAMES)]:
        if wanted in ds.dims or wanted in ds.coords:
            continue
        for name in candidates:
            if name in ds.dims or name in ds.coords:
                renames[name] = wanted
                break
    if renames:
        ds = ds.rename(renames)
    missing = [name for name in ('time', 'latitude', 'longitude') if name not in ds.coords and name not in ds.dims]
    if missing:
        raise ContractError(f'Source fixture is missing required grid coordinates: {missing!r}')
    if 'time' in ds.coords:
        times = pd.to_datetime(ds['time'].values, utc=True).tz_convert(None).to_numpy(dtype='datetime64[ns]')
        ds = ds.assign_coords(time=times)
    return ds


def _filter_dataset(ds: xr.Dataset, request: dict[str, Any]) -> xr.Dataset:
    wanted_times = request['timestamps']
    if not wanted_times.size:
        raise ContractError('Runtime lock selected no timestamps')
    source_times = pd.DatetimeIndex(pd.to_datetime(ds['time'].values).astype('datetime64[ns]'))
    missing_times = wanted_times.difference(source_times)
    if len(missing_times):
        raise ContractError(f'Source fixture does not contain all requested timestamps; first missing timestamp is {missing_times[0]}')
    ds = ds.sel(time=wanted_times.to_numpy(dtype='datetime64[ns]'))

    north, west, south, east = request['area']
    lat_values = np.asarray(ds['latitude'].values, dtype=float)
    lat_idx = np.where((lat_values >= south) & (lat_values <= north))[0]
    if lat_idx.size == 0:
        raise ContractError('Requested latitude area selects no source grid cells')
    lon_values = np.asarray(ds['longitude'].values, dtype=float)
    norm_lon = ((lon_values + 180.0) % 360.0) - 180.0
    norm_west = ((west + 180.0) % 360.0) - 180.0
    norm_east = ((east + 180.0) % 360.0) - 180.0
    if abs(east - west) >= 360 or (west == -180 and east == 180):
        lon_mask = np.ones_like(norm_lon, dtype=bool)
    elif norm_west <= norm_east:
        lon_mask = (norm_lon >= norm_west) & (norm_lon <= norm_east)
    else:
        lon_mask = (norm_lon >= norm_west) | (norm_lon <= norm_east)
    lon_idx = np.where(lon_mask)[0]
    if lon_idx.size == 0:
        raise ContractError('Requested longitude area selects no source grid cells')
    return ds.isel(latitude=lat_idx, longitude=lon_idx)


def _build_public_dataset(ds: xr.Dataset, contract: dict[str, Any], inventory: dict[str, Any], request: dict[str, Any]) -> tuple[xr.Dataset, list[dict[str, Any]]]:
    data_vars: dict[str, xr.DataArray] = {}
    channels: list[dict[str, Any]] = []
    for spec in request['field_specs']:
        source_name = _find_source_variable(ds, spec['name'])
        da = ds[source_name]
        plev_dim = _find_pressure_level_name(da, ds)
        unique_plev_dim = _safe_name(f'pressure_level__{spec["field_id"]}')
        if plev_dim:
            idx = _pressure_level_index(ds[plev_dim].values if plev_dim in ds.coords else da[plev_dim].values, spec['pressure_level'])
            da = da.isel({plev_dim: [idx]})
            da = da.rename({plev_dim: unique_plev_dim})
            da = da.assign_coords({unique_plev_dim: np.asarray([_native_level_value(da[unique_plev_dim].values[0], spec['pressure_level'])])})
            da[unique_plev_dim].attrs.update({'long_name': 'pressure level', 'units': 'hPa', 'selector_dimension': 'pressure_level'})
        else:
            da = da.expand_dims({unique_plev_dim: np.asarray([int(spec['pressure_level'])])})
            da[unique_plev_dim].attrs.update({'long_name': 'pressure level', 'units': 'hPa', 'selector_dimension': 'pressure_level'})
        order = [dim for dim in ('time', unique_plev_dim, 'latitude', 'longitude') if dim in da.dims]
        da = da.transpose(*order, ...)
        out_name = _safe_name(spec['field_id'])
        da = da.copy(deep=False)
        da.attrs = dict(da.attrs)
        da.attrs['field_id'] = spec['field_id']
        da.attrs['source_variable_name'] = source_name
        units = (((inventory.get('option_metadata') or {}).get('variable') or {}).get(spec['name']) or {}).get('units')
        if units and not da.attrs.get('units'):
            da.attrs['units'] = units
        da.encoding = {}
        data_vars[out_name] = da
        channels.append({
            'field_id': spec['field_id'],
            'array_path': out_name,
            'selectors': {'pressure_level': spec['pressure_level']},
            'selector_coordinate_paths': {'pressure_level': unique_plev_dim},
        })
    public = xr.Dataset(data_vars=data_vars, attrs={
        'dataset_slug': contract.get('dataset_slug'),
        'dataset_id': DATASET_ID,
        'pipeline_id': PIPELINE_ID,
        'publication_format': 'zarr',
        'zarr_format_version': 3,
        'source_provider': 'ECMWF',
    })
    for coord in public.coords:
        public[coord].encoding = {}
    return public, channels


def _find_source_variable(ds: xr.Dataset, requested_name: str) -> str:
    candidates = FIELD_ALIASES.get(requested_name, (requested_name,))
    for name in candidates:
        if name in ds.data_vars:
            return name
    lower_map = {name.lower(): name for name in ds.data_vars}
    for name in candidates:
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    raise ContractError(f'Source fixture does not contain requested variable {requested_name!r}; available variables are {list(ds.data_vars)!r}')


def _find_pressure_level_name(da: xr.DataArray, ds: xr.Dataset) -> str | None:
    for name in PLEV_NAMES:
        if name in da.dims or name in da.coords:
            return name
    if 'pressure_level' in ds.coords and 'pressure_level' in da.dims:
        return 'pressure_level'
    return None


def _pressure_level_index(values: Any, requested: str) -> int:
    arr = np.asarray(values)
    as_text = [str(v.item() if hasattr(v, 'item') else v) for v in arr]
    if requested in as_text:
        return as_text.index(requested)
    try:
        req_float = float(requested)
        diffs = np.abs(arr.astype(float) - req_float)
        idx = int(np.argmin(diffs))
        if float(diffs[idx]) < 1e-9:
            return idx
    except Exception:
        pass
    raise ContractError(f'Source fixture does not contain pressure_level={requested!r}')


def _native_level_value(value: Any, fallback: str) -> Any:
    try:
        return value.item()
    except Exception:
        try:
            return int(fallback)
        except Exception:
            return fallback


def _publish_zarr_atomically(ds: xr.Dataset, final_store: Path, request: dict[str, Any], channels: list[dict[str, Any]]) -> None:
    tmp_parent = final_store.parent
    tmp_path = Path(tempfile.mkdtemp(prefix=f'.{final_store.name}.tmp-', dir=tmp_parent))
    backup_path = final_store.with_name(f'.{final_store.name}.bak-{os.getpid()}')
    try:
        encoding: dict[str, dict[str, Any]] = {}
        data_compressors = ()
        for name, da in ds.data_vars.items():
            encoding[name] = {
                'chunks': tuple(1 if dim == 'time' or dim.startswith('pressure_level__') else int(da.sizes[dim]) for dim in da.dims),
                'compressors': data_compressors,
            }
        for name in ds.coords:
            encoding[name] = {}
        ds.to_zarr(tmp_path, mode='w', consolidated=True, zarr_format=3, encoding=encoding)
        _validate_published_zarr(tmp_path, request, channels)
        if backup_path.exists():
            shutil.rmtree(backup_path)
        if final_store.exists():
            final_store.rename(backup_path)
        tmp_path.rename(final_store)
        if backup_path.exists():
            shutil.rmtree(backup_path)
    except Exception:
        if tmp_path.exists():
            shutil.rmtree(tmp_path, ignore_errors=True)
        if backup_path.exists() and not final_store.exists():
            backup_path.rename(final_store)
        raise


def _validate_published_zarr(store: Path, request: dict[str, Any], channels: list[dict[str, Any]]) -> None:
    reopened = xr.open_zarr(store, consolidated=True, zarr_format=3)
    try:
        for coord in ('time', 'latitude', 'longitude'):
            if coord not in reopened.coords:
                raise RuntimeError(f'Published Zarr is missing coordinate {coord}')
        published_times = pd.DatetimeIndex(pd.to_datetime(reopened['time'].values).astype('datetime64[ns]'))
        if not published_times.equals(request['timestamps']):
            raise RuntimeError('Published Zarr time coordinate does not match requested timestamps exactly')
        for channel in channels:
            array_path = channel['array_path']
            if array_path not in reopened.data_vars:
                raise RuntimeError(f'Published Zarr is missing channel array {array_path}')
            selector_coord = channel['selector_coordinate_paths']['pressure_level']
            if selector_coord not in reopened[array_path].dims:
                raise RuntimeError(f'Channel {array_path} does not preserve pressure_level as an output dimension')
            if reopened.sizes[selector_coord] != 1:
                raise RuntimeError(f'Channel {array_path} pressure_level selector dimension must have cardinality one')
    finally:
        reopened.close()


def _canonical_field_id(field: dict[str, Any]) -> str:
    selectors = field.get('selectors') or []
    if not selectors:
        return str(field.get('name'))
    parts = [f'{sel.get("dimension")}={json.dumps(str(sel.get("value")), ensure_ascii=False)}' for sel in selectors]
    return f'{field.get("name")}[{",".join(parts)}]'


def _safe_name(value: str) -> str:
    out = re.sub(r'[^0-9A-Za-z_]+', '_', value).strip('_')
    if not out or out[0].isdigit():
        out = f'v_{out}'
    return out


def _unique_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out

```

FILE: pyproject.toml
```
[project]
name = "pipeline-era5-pareto-baseline-r1-20260907"
version = "1.0.0"
description = "Standalone ERA5 pressure-level dataset-family adapter"
requires-python = ">=3.13,<3.14"
dependencies = [
  "numpy>=2.1,<3",
  "pandas>=2.2,<3",
  "xarray>=2025.1,<2027",
  "zarr>=3.0,<4",
  "h5netcdf>=1.4,<2",
  "netCDF4>=1.7,<2",
  "cfgrib>=0.9.14,<0.10"
]

[project.optional-dependencies]
test = [
  "pytest>=8.3,<9"
]

```
