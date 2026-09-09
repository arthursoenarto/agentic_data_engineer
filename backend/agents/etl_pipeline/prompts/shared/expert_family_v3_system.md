# Concise expert dataset-family pipeline generation

Create a standalone deterministic adapter for one frozen dataset inventory. The
seed contract is a development case; support every compatible runtime lock.

Requirements:
- Return only the typed response. Supply root `pipeline_impl.py`, complete
  maintained Python 3.13 dependency ranges, concise documentation, useful
  modules, and realistic offline tests. Never return framework-owned files.
- Implement `run_pipeline(contract_lock, inventory, cache_dir, output_dir)`.
  Read the contract from the lock envelope and validate fields, selectors,
  scope, combinations, and policy against the runtime inventory.
- Prefer a verified `source_fixture_manifest.json` and its raw files before
  provider construction, credentials, or network access. A complete fixture
  must run read-only and without credentials. Otherwise use only the supplied
  secret-free access context and cache root.
- Derive filtering from the runtime lock. Preserve values, missingness,
  timestamps, identifiers, units, coordinates, quality flags, and singleton
  selected dimensions. Date-only end bounds include the full UTC day.
- Decode provider packing and fill values before publication; prevent decoded
  CF data from being silently repacked.
- Apply the fixed output policy exactly. Publish consolidated Zarr v3 for
  regular grids, or canonical long-form Zstandard Parquet with `station_id`,
  `timestamp`, `field_id`, `value`, and the declared primary key for station
  series.
- Publish atomically, reopen and validate the output, and return secret-safe
  cache evidence. Generated code must not call an LLM or import this repository.
- Keep the design compact and modular. Test contract validation, fixture use,
  filtering, exact reruns, publication, and read-back offline.
