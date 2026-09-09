Fill the implementation slots for this pipeline scaffold.

Dataset directory:
{dataset_dir}

Pipeline output directory:
{output_dir}

Network probing during generation is allowed:
{allow_network_probe}

DatasetContract:
{contract_json}

AccessContext:
{access_context_json}

The framework creates the stable runner, requirements file, gitignore, README path, and manifest. Supply `pipeline_impl.py`, only genuinely useful support modules, offline tests, dependencies, and concise README content.

The complete workflow must:
1. resolve `.env` and shell credentials with explicit layer precedence;
2. build provider requests for only the exact field-selector pairs;
3. download to partial paths and write secret-free request-identity sidecars;
4. validate request identity plus scientific structure before raw reuse;
5. transform lazily with explicit source lifetime and no conflicting scalar selector coordinates;
6. stage the derived artifact, validate exact scope and bounded multi-point content, and publish with rollback protection;
7. reopen the published artifact and return nonzero on failure.

Required offline tests:
- shell credentials override dotenv files;
- a nearer dotenv alias wins over a farther canonical value and maps to the canonical process variable;
- requests contain exactly the selected pairs;
- sidecar-free, mismatched-sidecar, and corrupted raw files are not reusable;
- separate pressure-level sources merge without scalar coordinate conflicts;
- synthetic transform/write/reopen validation covers times, global coordinates, selectors, and finite content;
- a failed candidate or post-publication check preserves the previous valid final.

Paths in `files` and `tests` are relative to the pipeline output directory. Do not return any protected scaffold path.
