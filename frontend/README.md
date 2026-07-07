# Frontend

This folder contains the product-facing workbench for the agentic data engineering system.

Even while the interface is small, treat it as production-grade software:

- keep backend and frontend boundaries explicit
- keep API responses typed in TypeScript
- show clear loading, success, empty, and error states
- keep layout responsive and accessible
- never expose API keys or secret values in frontend code
- read and write project artifacts only through the backend API

## Stack

- Next.js App Router
- TypeScript
- React
- FastAPI backend at `http://127.0.0.1:8000`

The frontend intentionally does not call the LLM directly. It submits flexible candidate input to the backend API. A URL is enough for the current first pass; name, slug, description, and user goal are optional. The backend normalizes that input and runs the reusable Python contract drafting agent.

The main page is a dataset workspace with a persistent collapsible dataset rail. Each dataset has tabs for Overview, Contract, and Pipeline Plan. Candidate input lives on a separate route so contract review can focus on the human-facing choices: selected fields, scope, access, defaults, and credential setup.

Credential setup points the backend to environment variable names or provider config files such as `.env` or `~/.cdsapirc`. The UI must never display, store, or submit secret values.

The Overview tab includes editable credential-reference fields when a contract indicates API-key or token access. `Save` writes non-secret reference metadata to `project/datasets/{slug}/credentials.json`; `Verify` asks the backend to resolve those references locally, generate `project/datasets/{slug}/access_probe.py` if needed, run it, and store `access_probe.json` under the dataset folder. The frontend never reads secret values back from the API.

## Local Development

From the repo root, run the backend API:

```bash
uvicorn backend.api.main:app --reload --port 8000
```

From this folder, run the frontend:

```bash
npm install
npm run dev
```

Open the workbench:

```text
http://127.0.0.1:3000
```

Create a new dataset candidate:

```text
http://127.0.0.1:3000/datasets/new
```

If the backend URL changes, set:

```bash
NEXT_PUBLIC_BACKEND_URL=http://127.0.0.1:8000
```

## Checks

```bash
npm run typecheck
npm run lint
npm run build
```
