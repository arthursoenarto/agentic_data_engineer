"use client";

import {
  AlertTriangle,
  ArrowLeft,
  ChevronLeft,
  ChevronRight,
  Database,
  ExternalLink,
  FileJson,
  FileText,
  Info,
  KeyRound,
  LoaderCircle,
  Plus,
  RefreshCw,
  Route,
  ShieldCheck,
  Sparkles,
} from "lucide-react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { FormEvent, ReactNode, useEffect, useMemo, useState } from "react";

import {
  createDatasetContract,
  fetchCredentialStatus,
  fetchDatasetContracts,
  saveDatasetCredentials,
  verifyDatasetAccess,
} from "@/lib/api";
import { ContractEditor } from "@/components/ContractEditor";
import { compactText } from "@/lib/format";
import type {
  DatasetCandidatePayload,
  CredentialReference,
  CredentialRequirement,
  CredentialStatus,
  DatasetContract,
  DatasetContractRun,
} from "@/lib/types";

type RunState = "idle" | "loading" | "success" | "error";
type DatasetTab = "overview" | "contract" | "pipeline_plan";

type CandidateFormState = {
  name: string;
  url: string;
  slug: string;
  description: string;
  userGoal: string;
};

type CredentialHint = {
  title: string;
  envVars: string[];
  filePath?: string;
  fileKey?: string;
  note: string;
};

type ArtifactStatus = {
  label: string;
  state: "available" | "missing" | "needed";
  value: string;
};

const initialFormState: CandidateFormState = {
  name: "",
  url: "",
  slug: "",
  description: "",
  userGoal: "",
};

function toPayload(form: CandidateFormState): DatasetCandidatePayload {
  const payload: DatasetCandidatePayload = {
    url: form.url.trim(),
  };

  const name = form.name.trim();
  const slug = form.slug.trim();
  const description = form.description.trim();
  const userGoal = form.userGoal.trim();
  if (name) {
    payload.name = name;
  }
  if (slug) {
    payload.slug = slug;
  }
  if (description) {
    payload.description = description;
  }
  if (userGoal) {
    payload.user_goal = userGoal;
  }

  return payload;
}

function validatePayload(payload: DatasetCandidatePayload): string | null {
  if (!payload.url) {
    return "Source URL is required.";
  }

  try {
    new URL(payload.url);
  } catch {
    return "Source URL must be a valid URL.";
  }

  if (payload.slug && !/^[A-Za-z0-9_-]+$/.test(payload.slug)) {
    return "Slug may contain only letters, numbers, hyphens, and underscores.";
  }

  return null;
}

function contractText(contract: DatasetContract): string {
  return [
    contract.title,
    contract.provider,
    contract.dataset_family,
    contract.source_url,
    contract.intent,
    contract.summary,
    ...contract.access_methods,
    ...contract.credential_requirements,
    ...contract.assumptions,
    ...contract.risks_or_unknowns,
    ...contract.fields.map(
      (item) =>
        `${item.name} ${item.display_name ?? ""} ${item.selectors.map((selector) => `${selector.dimension} ${selector.value}`).join(" ")} ${item.description ?? ""}`,
    ),
    ...contract.evidence.map((item) => `${item.title ?? ""} ${item.url} ${item.note ?? ""}`),
  ]
    .filter(Boolean)
    .join(" ")
    .toLowerCase();
}

function inferCredentialHints(contract: DatasetContract): CredentialHint[] {
  const text = contractText(contract);
  const needsCredential = /api key|token|credential|auth|x-api-key|personal-access-token|cdsapirc/.test(text);
  if (!needsCredential) {
    return [];
  }

  if (/era5|copernicus|cds\.climate|cdsapirc/.test(text)) {
    return [
      {
        title: "Copernicus CDS / ERA5 access",
        envVars: ["CDSAPI_URL", "CDSAPI_KEY"],
        filePath: "~/.cdsapirc",
        note: "Use the provider token in your shell or local CDS config file. Keep real values out of project artifacts.",
      },
    ];
  }

  if (/openaq|x-api-key/.test(text)) {
    return [
      {
        title: "OpenAQ API access",
        envVars: ["OPENAQ_API_KEY"],
        note: "Use this value as the provider X-API-Key header when the access probe or pipeline is built.",
      },
    ];
  }

  return [
    {
      title: "Dataset API access",
      envVars: ["DATASET_API_KEY"],
      note: "Rename this to a provider-specific environment variable before building the access probe.",
    },
  ];
}

function datasetSlug(run: DatasetContractRun): string {
  if (run.candidate?.slug) {
    return run.candidate.slug;
  }

  const parts = run.contract_file.split("/");
  const datasetIndex = parts.indexOf("datasets");
  if (datasetIndex >= 0 && parts[datasetIndex + 1]) {
    return parts[datasetIndex + 1];
  }

  return run.contract.title.toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_+|_+$/g, "") || "dataset";
}

function artifactStatuses(
  run: DatasetContractRun,
  credentialStatus: CredentialStatus | null,
  credentialHints: CredentialHint[],
): ArtifactStatus[] {
  const requirementsCount = credentialStatus?.requirements.length ?? credentialHints.length;
  const missingCount = credentialStatus?.missing_env_vars.length ?? requirementsCount;
  const probe = credentialStatus?.access_probe;

  return [
    { label: "Candidate", state: run.candidate_file ? "available" : "missing", value: run.candidate_file ? "Available" : "Missing" },
    { label: "Contract", state: run.contract_file ? "available" : "missing", value: run.contract_file ? "Available" : "Missing" },
    {
      label: "Credentials",
      state: requirementsCount === 0 || missingCount === 0 ? "available" : "needed",
      value: requirementsCount === 0 ? "Not required" : missingCount === 0 ? "Saved" : "Missing",
    },
    {
      label: "Access probe",
      state: probe?.ok ? "available" : requirementsCount > 0 ? "needed" : "available",
      value: probe?.ok ? "Verified" : requirementsCount > 0 ? "Not verified" : "Not required",
    },
    { label: "Pipeline plan", state: "missing", value: "Not created" },
  ];
}

export function DatasetContractsPage() {
  const [contracts, setContracts] = useState<DatasetContractRun[]>([]);
  const [activeRun, setActiveRun] = useState<DatasetContractRun | null>(null);
  const [activeTab, setActiveTab] = useState<DatasetTab>("overview");
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [loadState, setLoadState] = useState<RunState>("loading");
  const [error, setError] = useState<string | null>(null);

  async function loadContracts() {
    setLoadState("loading");
    try {
      const payload = await fetchDatasetContracts();
      setContracts(payload.contracts);
      setActiveRun((current) => current ?? payload.contracts[0] ?? null);
      setLoadState("success");
      setError(null);
    } catch (caught) {
      setLoadState("error");
      setError(caught instanceof Error ? caught.message : "Could not load existing contracts.");
    }
  }

  function updateContractRun(updatedRun: DatasetContractRun) {
    setContracts((current) =>
      current.map((run) => (run.contract_file === updatedRun.contract_file ? updatedRun : run)),
    );
    setActiveRun(updatedRun);
  }

  useEffect(() => {
    let isCurrent = true;

    async function loadInitialContracts() {
      try {
        const payload = await fetchDatasetContracts();
        if (!isCurrent) {
          return;
        }
        setContracts(payload.contracts);
        setActiveRun(payload.contracts[0] ?? null);
        setLoadState("success");
      } catch (caught) {
        if (!isCurrent) {
          return;
        }
        setLoadState("error");
        setError(caught instanceof Error ? caught.message : "Could not load existing contracts.");
      }
    }

    void loadInitialContracts();

    return () => {
      isCurrent = false;
    };
  }, []);

  const statusText = useMemo(() => {
    if (loadState === "loading") {
      return "Loading contracts";
    }
    if (loadState === "error") {
      return "Needs attention";
    }
    return "Ready";
  }, [loadState]);

  return (
    <main className={sidebarCollapsed ? "dataset-workspace dataset-workspace-collapsed" : "dataset-workspace"}>
      <aside className={sidebarCollapsed ? "dataset-rail dataset-rail-collapsed" : "dataset-rail"} aria-label="Datasets">
        <div className="rail-header">
          {!sidebarCollapsed ? (
            <div>
              <span className="section-kicker">Datasets</span>
              <h2>Project</h2>
            </div>
          ) : null}
          <button
            aria-label={sidebarCollapsed ? "Expand dataset sidebar" : "Collapse dataset sidebar"}
            className="icon-button"
            type="button"
            onClick={() => setSidebarCollapsed((current) => !current)}
          >
            {sidebarCollapsed ? <ChevronRight size={18} /> : <ChevronLeft size={18} />}
          </button>
        </div>

        <Link className={sidebarCollapsed ? "rail-new rail-new-collapsed" : "rail-new"} href="/datasets/new" title="New dataset">
          <Plus size={18} />
          {!sidebarCollapsed ? <span>New Dataset</span> : null}
        </Link>

        <div className="dataset-list">
          {contracts.length === 0 ? (
            sidebarCollapsed ? null : <p className="empty-copy">No datasets yet.</p>
          ) : (
            contracts.map((run) => (
              <button
                className={sidebarCollapsed ? "dataset-list-item dataset-list-item-collapsed" : "dataset-list-item"}
                data-active={activeRun?.contract_file === run.contract_file}
                key={run.contract_file}
                title={run.contract.title}
                type="button"
                onClick={() => {
                  setActiveRun(run);
                  setActiveTab("overview");
                }}
              >
                <Database size={17} />
                {!sidebarCollapsed ? (
                  <span>
                    <strong>{run.contract.title}</strong>
                    <small>{datasetSlug(run)}</small>
                  </span>
                ) : null}
              </button>
            ))
          )}
        </div>
      </aside>

      <section className="dataset-main">
        <header className="topbar workspace-topbar">
          <div>
            <div className="eyebrow">Agentic Data Engineer</div>
            <h1>{activeRun ? activeRun.contract.title : "Dataset Workspace"}</h1>
          </div>
          <div className="topbar-actions">
            <span className={`status status-${loadState}`}>{statusText}</span>
            <button className="icon-button" type="button" onClick={() => void loadContracts()} title="Refresh contracts">
              {loadState === "loading" ? <LoaderCircle className="spin" size={18} /> : <RefreshCw size={18} />}
            </button>
          </div>
        </header>

        {error ? (
          <div className="alert page-alert" role="alert">
            <AlertTriangle size={18} />
            <span>{error}</span>
          </div>
        ) : null}

        {activeRun ? (
          <DatasetWorkspace run={activeRun} activeTab={activeTab} onRunChange={updateContractRun} onTabChange={setActiveTab} />
        ) : (
          <section className="workspace-panel workspace-panel-wide">
            <EmptyContractState />
          </section>
        )}
      </section>
    </main>
  );
}

export function DatasetInputPage() {
  const router = useRouter();
  const [form, setForm] = useState<CandidateFormState>(initialFormState);
  const [runState, setRunState] = useState<RunState>("idle");
  const [error, setError] = useState<string | null>(null);

  async function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const payload = toPayload(form);
    const validationError = validatePayload(payload);
    if (validationError) {
      setError(validationError);
      setRunState("error");
      return;
    }

    setError(null);
    setRunState("loading");

    try {
      await createDatasetContract(payload);
      setRunState("success");
      router.push("/");
    } catch (caught) {
      setRunState("error");
      setError(caught instanceof Error ? caught.message : "Contract drafting failed.");
    }
  }

  return (
    <main className="workbench input-workbench">
      <header className="topbar">
        <div>
          <div className="eyebrow">Dataset candidate</div>
          <h1>New Dataset</h1>
        </div>
        <div className="topbar-actions">
          <Link className="secondary-link" href="/">
            <ArrowLeft size={18} />
            Contracts
          </Link>
        </div>
      </header>

      <section className="input-page">
        <section className="intake-panel intake-panel-large" aria-labelledby="candidate-heading">
          <div className="panel-heading">
            <span className="section-kicker">Input</span>
            <h2 id="candidate-heading">Dataset candidate</h2>
          </div>

          <form className="candidate-form" onSubmit={(event) => void onSubmit(event)}>
            <label>
              Source URL
              <input
                autoComplete="url"
                name="url"
                onChange={(event) => setForm((current) => ({ ...current, url: event.target.value }))}
                type="url"
                value={form.url}
              />
            </label>

            <label>
              Dataset name <span>Optional</span>
              <input
                autoComplete="off"
                name="name"
                onChange={(event) => setForm((current) => ({ ...current, name: event.target.value }))}
                value={form.name}
              />
            </label>

            <label>
              Dataset slug <span>Optional</span>
              <input
                autoComplete="off"
                name="slug"
                onChange={(event) => setForm((current) => ({ ...current, slug: event.target.value }))}
                value={form.slug}
              />
            </label>

            <label>
              Description <span>Optional</span>
              <textarea
                name="description"
                onChange={(event) => setForm((current) => ({ ...current, description: event.target.value }))}
                value={form.description}
              />
            </label>

            <label>
              User goal <span>Optional</span>
              <textarea
                name="userGoal"
                onChange={(event) => setForm((current) => ({ ...current, userGoal: event.target.value }))}
                value={form.userGoal}
              />
            </label>

            <button className="primary-button" disabled={runState === "loading"} type="submit">
              {runState === "loading" ? <LoaderCircle className="spin" size={18} /> : <Sparkles size={18} />}
              Draft contract
            </button>
          </form>

          {error ? (
            <div className="alert" role="alert">
              <AlertTriangle size={18} />
              <span>{error}</span>
            </div>
          ) : null}
        </section>
      </section>
    </main>
  );
}

function DatasetWorkspace({
  activeTab,
  onRunChange,
  onTabChange,
  run,
}: {
  activeTab: DatasetTab;
  onRunChange: (run: DatasetContractRun) => void;
  onTabChange: (tab: DatasetTab) => void;
  run: DatasetContractRun;
}) {
  return (
    <section className="workspace-panel workspace-panel-wide" aria-live="polite">
      <div className="dataset-page-header">
        <div>
          <span className="section-kicker">Dataset workspace</span>
          <h2>{run.contract.title}</h2>
        </div>
        <a className="source-link" href={run.contract.source_url} rel="noreferrer" target="_blank">
          <ExternalLink size={16} />
          Source
        </a>
      </div>

      <div className="dataset-tabs" role="tablist" aria-label="Dataset workspace sections">
        <TabButton active={activeTab === "overview"} icon={<Info size={16} />} label="Overview" onClick={() => onTabChange("overview")} />
        <TabButton active={activeTab === "contract"} icon={<FileText size={16} />} label="Contract" onClick={() => onTabChange("contract")} />
        <TabButton
          active={activeTab === "pipeline_plan"}
          icon={<Route size={16} />}
          label="Pipeline Plan"
          onClick={() => onTabChange("pipeline_plan")}
        />
      </div>

      {activeTab === "overview" ? <OverviewPanel run={run} /> : null}
      {activeTab === "contract" ? <ContractDetail run={run} onRunChange={onRunChange} /> : null}
      {activeTab === "pipeline_plan" ? <PipelinePlanPanel run={run} /> : null}
    </section>
  );
}

function TabButton({
  active,
  icon,
  label,
  onClick,
}: {
  active: boolean;
  icon: ReactNode;
  label: string;
  onClick: () => void;
}) {
  return (
    <button aria-selected={active} className="dataset-tab" role="tab" type="button" onClick={onClick}>
      {icon}
      {label}
    </button>
  );
}

function EmptyContractState() {
  return (
    <div className="empty-state">
      <FileJson size={34} />
      <h2>No active contract</h2>
      <p>Create a dataset contract or select a generated contract.</p>
    </div>
  );
}

function OverviewPanel({ run }: { run: DatasetContractRun }) {
  const contract = run.contract;
  const credentialHints = useMemo(() => inferCredentialHints(contract), [contract]);
  const slug = datasetSlug(run);
  const [credentialStatus, setCredentialStatus] = useState<CredentialStatus | null>(null);
  const [credentialError, setCredentialError] = useState<string | null>(null);
  const statuses = artifactStatuses(run, credentialStatus, credentialHints);
  const candidate = run.candidate;

  useEffect(() => {
    let isCurrent = true;

    async function loadCredentialStatus() {
      try {
        const status = await fetchCredentialStatus(slug);
        if (!isCurrent) {
          return;
        }
        setCredentialStatus(status);
        setCredentialError(null);
      } catch (caught) {
        if (!isCurrent) {
          return;
        }
        setCredentialError(caught instanceof Error ? caught.message : "Could not load credential status.");
      }
    }

    void loadCredentialStatus();

    return () => {
      isCurrent = false;
    };
  }, [slug]);

  return (
    <div className="tab-panel">
      <section className="overview-hero">
        <div>
          <span className="section-kicker">Overview</span>
          <p>{contract.summary}</p>
        </div>
        <div className="overview-actions">
          <a className="secondary-link" href={contract.source_url} rel="noreferrer" target="_blank">
            <ExternalLink size={16} />
            Dataset source
          </a>
        </div>
      </section>

      <div className="overview-grid">
        <WorkspaceSection title="Status">
          <div className="status-grid">
            {statuses.map((status) => (
              <div className="status-card" data-state={status.state} key={status.label}>
                <strong>{status.label}</strong>
                <span>{status.value}</span>
              </div>
            ))}
          </div>
        </WorkspaceSection>

        <WorkspaceSection title="Important Information">
          <dl className="detail-list">
            <div>
              <dt>Dataset slug</dt>
              <dd>{datasetSlug(run)}</dd>
            </div>
            <div>
              <dt>Candidate file</dt>
              <dd>{run.candidate_file}</dd>
            </div>
            <div>
              <dt>Contract file</dt>
              <dd>{run.contract_file}</dd>
            </div>
            <div>
              <dt>Intent</dt>
              <dd>{compactText(contract.intent)}</dd>
            </div>
            <div>
              <dt>Recommended next step</dt>
              <dd>{compactText(contract.recommended_next_step, "No recommendation returned.")}</dd>
            </div>
          </dl>
        </WorkspaceSection>

        {credentialHints.length ? (
          <WorkspaceSection title="Credential Setup" wide>
            <CredentialManager
              datasetSlug={slug}
              fallbackHints={credentialHints}
              onStatusChange={setCredentialStatus}
              status={credentialStatus}
            />
            {credentialError ? (
              <div className="alert" role="alert">
                <AlertTriangle size={18} />
                <span>{credentialError}</span>
              </div>
            ) : null}
          </WorkspaceSection>
        ) : null}

        <WorkspaceSection title="Dataset Candidate JSON" wide>
          {candidate ? <pre>{JSON.stringify(candidate, null, 2)}</pre> : <p className="empty-copy">Candidate JSON is not available.</p>}
        </WorkspaceSection>
      </div>
    </div>
  );
}

function ContractDetail({ onRunChange, run }: { onRunChange: (run: DatasetContractRun) => void; run: DatasetContractRun }) {
  return <ContractEditor run={run} onRunChange={onRunChange} />;
}

function PipelinePlanPanel({ run }: { run: DatasetContractRun }) {
  return (
    <div className="tab-panel">
      <section className="pipeline-empty">
        <Route size={38} />
        <h2>Pipeline plan not created yet</h2>
        <p>
          This dataset has a contract, but the pipeline planning step has not been implemented yet. When it exists, this tab
          should show `pipeline_plan.json` and the human-readable planning notes for {run.contract.title}.
        </p>
      </section>
    </div>
  );
}

function CredentialManager({
  datasetSlug,
  fallbackHints,
  onStatusChange,
  status,
}: {
  datasetSlug: string;
  fallbackHints: CredentialHint[];
  onStatusChange: (status: CredentialStatus) => void;
  status: CredentialStatus | null;
}) {
  const requirements = useMemo(
    () =>
      status?.requirements ??
      fallbackHints.flatMap((hint) =>
        hint.envVars.map(
          (envVar): CredentialRequirement => ({
            env_var: envVar,
            label: envVar,
            aliases: [],
            note: hint.note,
            file_path_hint: hint.filePath,
            file_key_hint: hint.fileKey,
          }),
        ),
      ),
    [fallbackHints, status?.requirements],
  );
  const [referenceEdits, setReferenceEdits] = useState<Record<string, CredentialReference>>({});
  const [saveState, setSaveState] = useState<RunState>("idle");
  const [verifyState, setVerifyState] = useState<RunState>("idle");
  const [message, setMessage] = useState<string | null>(null);

  function referenceFor(requirement: CredentialRequirement): CredentialReference {
    const saved = status?.references.find((reference) => reference.requirement === requirement.env_var);
    return referenceEdits[requirement.env_var] ?? saved ?? {
      requirement: requirement.env_var,
      source: "env",
      env_var: requirement.env_var,
    };
  }

  function statusFor(requirement: CredentialRequirement) {
    return status?.references.find((reference) => reference.requirement === requirement.env_var);
  }

  function updateReference(requirement: CredentialRequirement, patch: Partial<CredentialReference>) {
    setReferenceEdits((current) => {
      const existing = current[requirement.env_var] ?? referenceFor(requirement);
      const next: CredentialReference = {
        requirement: requirement.env_var,
        source: existing?.source ?? "env",
        env_var: existing?.env_var ?? requirement.env_var,
        file_path: existing?.file_path ?? requirement.file_path_hint ?? "",
        file_key: existing?.file_key ?? requirement.file_key_hint ?? "",
        ...patch,
      };
      return { ...current, [requirement.env_var]: next };
    });
  }

  async function onSave() {
    const payload = requirements.map((requirement) => referenceFor(requirement));
    const invalid = payload.find((reference) =>
      reference.source === "env" ? !reference.env_var?.trim() : !reference.file_path?.trim(),
    );
    if (invalid) {
      setMessage("Each credential reference needs an environment variable name or file path.");
      setSaveState("error");
      return;
    }

    setSaveState("loading");
    setMessage(null);
    try {
      const nextStatus = await saveDatasetCredentials(datasetSlug, payload);
      onStatusChange(nextStatus);
      setReferenceEdits({});
      setSaveState("success");
      setMessage("Credential references saved. Keep actual secret values in local environment files.");
    } catch (caught) {
      setSaveState("error");
      setMessage(caught instanceof Error ? caught.message : "Could not save credentials.");
    }
  }

  async function onVerify() {
    setVerifyState("loading");
    setMessage(null);
    try {
      await verifyDatasetAccess(datasetSlug);
      const nextStatus = await fetchCredentialStatus(datasetSlug);
      onStatusChange(nextStatus);
      setVerifyState(nextStatus.access_probe?.ok ? "success" : "error");
      setMessage(nextStatus.access_probe?.summary ?? "Access probe completed.");
    } catch (caught) {
      setVerifyState("error");
      setMessage(caught instanceof Error ? caught.message : "Access probe failed.");
    }
  }

  return (
    <div className="credential-list">
      <div className="credential-card">
        <div className="credential-title">
          <KeyRound size={17} />
          <strong>Credential setup</strong>
        </div>
        <p>Point each requirement to a local environment variable or provider config file. Do not paste secret values here.</p>

        <div className="credential-fields">
          {requirements.map((requirement) => {
            const reference = referenceFor(requirement);
            const referenceStatus = statusFor(requirement);
            const isSet = referenceStatus?.is_set ?? false;
            return (
              <label className="credential-field" key={requirement.env_var}>
                <span className="credential-field-label">
                  {requirement.label}
                  <em>{isSet ? "Found" : "Missing"}</em>
                </span>
                <div className="credential-reference-row">
                  <select
                    aria-label={`${requirement.label} source`}
                    onChange={(event) =>
                      updateReference(requirement, {
                        source: event.target.value as CredentialReference["source"],
                        env_var: event.target.value === "env" ? reference.env_var ?? requirement.env_var : null,
                        file_path: event.target.value === "file" ? reference.file_path ?? requirement.file_path_hint ?? "" : null,
                        file_key: event.target.value === "file" ? reference.file_key ?? requirement.file_key_hint ?? "" : null,
                      })
                    }
                    value={reference.source}
                  >
                    <option value="env">env</option>
                    <option value="file">file</option>
                  </select>
                  {reference.source === "file" ? (
                    <>
                      <input
                        autoComplete="off"
                        aria-label={`${requirement.label} file path`}
                        onChange={(event) => updateReference(requirement, { file_path: event.target.value })}
                        placeholder={requirement.file_path_hint ?? "Credential file path"}
                        type="text"
                        value={reference.file_path ?? ""}
                      />
                      <input
                        autoComplete="off"
                        aria-label={`${requirement.label} file key`}
                        onChange={(event) => updateReference(requirement, { file_key: event.target.value })}
                        placeholder={requirement.file_key_hint ?? "key"}
                        type="text"
                        value={reference.file_key ?? ""}
                      />
                    </>
                  ) : (
                    <input
                      autoComplete="off"
                      aria-label={`${requirement.label} environment variable`}
                      onChange={(event) => updateReference(requirement, { env_var: event.target.value })}
                      type="text"
                      value={reference.env_var ?? ""}
                    />
                  )}
                </div>
                <small>
                  Probe expects <code>{requirement.env_var}</code>
                  {requirement.aliases.length ? `; aliases: ${requirement.aliases.join(", ")}` : ""}
                  {requirement.file_path_hint ? `; file: ${requirement.file_path_hint}` : ""}
                  {requirement.note ? `; ${requirement.note}` : ""}
                </small>
              </label>
            );
          })}
        </div>

        <div className="credential-actions">
          <button className="secondary-button" disabled={saveState === "loading"} type="button" onClick={() => void onSave()}>
            {saveState === "loading" ? <LoaderCircle className="spin" size={16} /> : <KeyRound size={16} />}
            Save
          </button>
          <button className="primary-button credential-verify" disabled={verifyState === "loading"} type="button" onClick={() => void onVerify()}>
            {verifyState === "loading" ? <LoaderCircle className="spin" size={16} /> : <ShieldCheck size={16} />}
            Verify
          </button>
        </div>

        <div className="credential-grid">
          <div>
            <dt>Reference file</dt>
            <dd>{status?.credential_file ? <code>{status.credential_file}</code> : "Not saved"}</dd>
          </div>
          <div>
            <dt>Available env refs</dt>
            <dd>{status?.set_env_vars.length ? status.set_env_vars.join(", ") : "None yet"}</dd>
          </div>
          <div>
            <dt>Missing requirements</dt>
            <dd>{status?.missing_env_vars.length ? status.missing_env_vars.join(", ") : "None"}</dd>
          </div>
          <div>
            <dt>Access probe</dt>
            <dd>{status?.access_probe ? status.access_probe.status : "Not run"}</dd>
          </div>
        </div>

        {status?.access_probe ? (
          <div className={status.access_probe.ok ? "probe-result probe-result-ok" : "probe-result probe-result-failed"}>
            <strong>{status.access_probe.ok ? "Verified" : "Probe failed"}</strong>
            <span>{status.access_probe.summary}</span>
          </div>
        ) : null}

        {message ? (
          <div className={saveState === "error" || verifyState === "error" ? "alert" : "inline-message"} role="status">
            {saveState === "error" || verifyState === "error" ? <AlertTriangle size={18} /> : null}
            <span>{message}</span>
          </div>
        ) : null}
      </div>
    </div>
  );
}

function WorkspaceSection({
  children,
  title,
  wide = false,
}: {
  children: ReactNode;
  title: string;
  wide?: boolean;
}) {
  return (
    <section className={wide ? "workspace-section workspace-section-wide" : "workspace-section"}>
      <h3>{title}</h3>
      {children}
    </section>
  );
}
