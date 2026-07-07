"use client";

import { AlertTriangle, LoaderCircle, Plus, Save, Trash2 } from "lucide-react";
import { useEffect, useMemo, useState } from "react";

import { fetchDatasetInventory, saveDatasetContract } from "@/lib/api";
import { compactText } from "@/lib/format";
import type { ContractFieldSpec, ContractSelectorSpec, DatasetContract, DatasetContractRun, DatasetInventory } from "@/lib/types";

type SaveState = "idle" | "loading" | "success" | "error";
type TimePresetId = "all" | "every_3_hours" | "every_6_hours" | "daily_00" | "daily_12" | "custom";

type EditableField = {
  id: string;
  name: string;
  selectors: Record<string, string>;
};

type EditorState = {
  fields: EditableField[];
  productType: string;
  startDate: string;
  endDate: string;
  timePreset: TimePresetId;
  selectedTimes: string[];
  areaMode: "global" | "custom";
  area: [number, number, number, number];
  dataFormat: string;
  downloadFormat: string;
  humanConfirmed: boolean;
};

type TimePreset = {
  id: TimePresetId;
  label: string;
  times: string[];
};

const defaultArea: [number, number, number, number] = [90, -180, -90, 180];
const fieldSelectorOptionKeys = ["pressure_level", "model_level", "depth", "height", "band", "wavelength", "station_id", "lead_time", "ensemble_member"];

function record(value: unknown): Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value) ? (value as Record<string, unknown>) : {};
}

function stringArray(value: unknown): string[] {
  return Array.isArray(value) ? value.map((item) => String(item)) : [];
}

function optionValues(inventory: DatasetInventory | null, key: string): string[] {
  return stringArray(inventory?.options[key]);
}

function selectorDimensions(inventory: DatasetInventory | null): string[] {
  if (!inventory) {
    return [];
  }
  return fieldSelectorOptionKeys.filter((key) => optionValues(inventory, key).length > 0);
}

function labelFromName(value: string): string {
  return value.replace(/_/g, " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function variableMetadata(inventory: DatasetInventory | null, name: string): Record<string, unknown> {
  return record(record(inventory?.option_metadata.variable)[name]);
}

function metadataText(metadata: Record<string, unknown>, key: string): string | null {
  const value = metadata[key];
  return typeof value === "string" && value ? value : null;
}

function parseArea(value: unknown): [number, number, number, number] {
  if (!Array.isArray(value) || value.length !== 4) {
    return defaultArea;
  }
  const parsed = value.map((item) => Number(item));
  return parsed.every((item) => Number.isFinite(item)) ? (parsed as [number, number, number, number]) : defaultArea;
}

function parseHour(value: string): number | null {
  const hour = Number(value.slice(0, 2));
  return Number.isInteger(hour) ? hour : null;
}

function stepTimes(availableTimes: string[], stepHours: number): string[] {
  return availableTimes.filter((time) => {
    const hour = parseHour(time);
    return hour !== null && hour % stepHours === 0;
  });
}

function arraysEqual(left: string[], right: string[]): boolean {
  return left.length === right.length && left.every((value, index) => value === right[index]);
}

function presetOptions(availableTimes: string[]): TimePreset[] {
  const every3 = stepTimes(availableTimes, 3);
  const every6 = stepTimes(availableTimes, 6);
  const daily00 = availableTimes.includes("00:00") ? ["00:00"] : [];
  const daily12 = availableTimes.includes("12:00") ? ["12:00"] : [];

  const presets: Array<TimePreset | null> = [
    availableTimes.length ? { id: "all", label: "Every hour", times: availableTimes } : null,
    every3.length ? { id: "every_3_hours", label: "Every 3 hours", times: every3 } : null,
    every6.length ? { id: "every_6_hours", label: "Every 6 hours", times: every6 } : null,
    daily00.length ? { id: "daily_00", label: "Daily 00:00", times: daily00 } : null,
    daily12.length ? { id: "daily_12", label: "Daily 12:00", times: daily12 } : null,
    { id: "custom", label: "Custom", times: [] },
  ];

  return presets.filter((preset): preset is TimePreset => preset !== null);
}

function detectPreset(selectedTimes: string[], availableTimes: string[]): TimePresetId {
  for (const preset of presetOptions(availableTimes)) {
    if (preset.id !== "custom" && arraysEqual(selectedTimes, preset.times)) {
      return preset.id;
    }
  }
  return "custom";
}

function defaultSelectorValues(inventory: DatasetInventory | null, dimensions: string[]): Record<string, string> {
  return Object.fromEntries(dimensions.map((dimension) => [dimension, optionValues(inventory, dimension)[0] ?? ""]));
}

function initialField(variableOptions: string[], inventory: DatasetInventory | null, dimensions: string[]): EditableField {
  const firstOption = variableOptions[0] ?? "";
  return {
    id: `field-${Date.now()}`,
    name: firstOption,
    selectors: defaultSelectorValues(inventory, dimensions),
  };
}

function buildEditorState(contract: DatasetContract, inventory: DatasetInventory | null): EditorState {
  const scope = record(contract.scope);
  const dateRange = record(scope.date_range);
  const timeScope = record(scope.time);
  const geography = record(scope.geography);
  const advanced = record(contract.advanced_options);
  const productDefaults = stringArray(inventory?.defaults.product_type);
  const productOptions = optionValues(inventory, "product_type");
  const availableTimes = optionValues(inventory, "time");
  const selectedTimes = stringArray(timeScope.selected_times);
  const variableOptions = optionValues(inventory, "variable");
  const dimensions = selectorDimensions(inventory);
  const fields = contract.fields.map((field, index) => ({
    id: `${field.name}-${(field.selectors ?? []).map((selector) => `${selector.dimension}:${selector.value}`).join("-") || "base"}-${index}`,
    name: field.name,
    selectors: {
      ...defaultSelectorValues(inventory, dimensions),
      ...Object.fromEntries((field.selectors ?? []).map((selector) => [selector.dimension, String(selector.value)])),
    },
  }));

  return {
    fields: fields.length ? fields : variableOptions.length ? [initialField(variableOptions, inventory, dimensions)] : [],
    productType:
      (typeof scope.product_type === "string" && scope.product_type) ||
      productDefaults[0] ||
      (productOptions.includes("reanalysis") ? "reanalysis" : productOptions[0] ?? ""),
    startDate: typeof dateRange.start_date === "string" ? dateRange.start_date : "",
    endDate: typeof dateRange.end_date === "string" ? dateRange.end_date : "",
    timePreset: detectPreset(selectedTimes, availableTimes),
    selectedTimes,
    areaMode: record(geography).area === "global" ? "global" : "custom",
    area: parseArea(geography.cds_area ?? inventory?.defaults.area),
    dataFormat: (typeof advanced.data_format === "string" && advanced.data_format) || String(inventory?.defaults.data_format ?? ""),
    downloadFormat:
      (typeof advanced.download_format === "string" && advanced.download_format) || String(inventory?.defaults.download_format ?? ""),
    humanConfirmed: contract.human_confirmed,
  };
}

function selectedTimeLabel(state: EditorState): string {
  return state.selectedTimes.length ? state.selectedTimes.join(", ") : "No times selected";
}

function updateFieldMetadata(field: EditableField, inventory: DatasetInventory | null): ContractFieldSpec {
  const metadata = variableMetadata(inventory, field.name);
  const selectors: ContractSelectorSpec[] = Object.entries(field.selectors)
    .filter(([, value]) => value !== "")
    .map(([dimension, value]) => ({
      dimension,
      value,
      unit: inventory?.option_units[dimension] ?? null,
      label: labelFromName(dimension),
    }));

  return {
    name: field.name,
    display_name: metadataText(metadata, "label") ?? labelFromName(field.name),
    selectors,
    units: metadataText(metadata, "units"),
    description: metadataText(metadata, "description"),
  };
}

function buildContract(contract: DatasetContract, state: EditorState, inventory: DatasetInventory | null): DatasetContract {
  const existingScope = { ...contract.scope };
  delete existingScope.pressure_level_units;
  const existingAdvanced = record(contract.advanced_options);
  const timePreset = presetOptions(optionValues(inventory, "time")).find((preset) => preset.id === state.timePreset);
  const timestep = state.timePreset === "custom" ? "custom" : timePreset?.label.toLowerCase() ?? "custom";
  const selectedFields = state.fields.filter((field) => field.name);

  return {
    ...contract,
    fields: selectedFields.map((field) => updateFieldMetadata(field, inventory)),
    scope: {
      ...existingScope,
      product_type: state.productType,
      date_range: {
        start_date: state.startDate,
        end_date: state.endDate,
        inclusive: true,
      },
      time: {
        timezone: "UTC",
        timestep,
        selected_times: state.selectedTimes,
      },
      geography: {
        area: state.areaMode,
        cds_area: state.area,
        cds_area_order: ["north", "west", "south", "east"],
      },
    },
    advanced_options: {
      ...existingAdvanced,
      data_format: state.dataFormat,
      download_format: state.downloadFormat,
    },
    human_editable_fields: [
      "fields",
      "fields.selectors",
      "scope.product_type",
      "scope.date_range",
      "scope.time.selected_times",
      "scope.geography.cds_area",
      "advanced_options.data_format",
      "advanced_options.download_format",
    ],
    human_confirmed: state.humanConfirmed,
  };
}

export function ContractEditor({
  onRunChange,
  run,
}: {
  onRunChange: (run: DatasetContractRun) => void;
  run: DatasetContractRun;
}) {
  const slug = run.contract.dataset_slug;
  const [inventory, setInventory] = useState<DatasetInventory | null>(null);
  const [state, setState] = useState<EditorState>(() => buildEditorState(run.contract, null));
  const [loadState, setLoadState] = useState<SaveState>("loading");
  const [saveState, setSaveState] = useState<SaveState>("idle");
  const [message, setMessage] = useState<string | null>(null);

  useEffect(() => {
    let isCurrent = true;

    async function loadInventory() {
      setLoadState("loading");
      try {
        const payload = await fetchDatasetInventory(slug);
        if (!isCurrent) {
          return;
        }
        setInventory(payload);
        setState(buildEditorState(run.contract, payload));
        setLoadState("success");
        setMessage(null);
      } catch (caught) {
        if (!isCurrent) {
          return;
        }
        setInventory(null);
        setState(buildEditorState(run.contract, null));
        setLoadState("error");
        setMessage(caught instanceof Error ? caught.message : "Could not load dataset inventory.");
      }
    }

    void loadInventory();

    return () => {
      isCurrent = false;
    };
  }, [run.contract, slug]);

  const variableOptions = useMemo(() => optionValues(inventory, "variable"), [inventory]);
  const fieldSelectorDimensions = useMemo(() => selectorDimensions(inventory), [inventory]);
  const productOptions = useMemo(() => optionValues(inventory, "product_type"), [inventory]);
  const dataFormatOptions = useMemo(() => optionValues(inventory, "data_format"), [inventory]);
  const downloadFormatOptions = useMemo(() => optionValues(inventory, "download_format"), [inventory]);
  const availableTimes = useMemo(() => optionValues(inventory, "time"), [inventory]);
  const timePresets = useMemo(() => presetOptions(availableTimes), [availableTimes]);

  function updateState(patch: Partial<EditorState>) {
    setState((current) => ({ ...current, ...patch }));
  }

  function updateField(id: string, patch: Partial<EditableField>) {
    setState((current) => ({
      ...current,
      fields: current.fields.map((field) => (field.id === id ? { ...field, ...patch } : field)),
    }));
  }

  function addField() {
    setState((current) => ({
      ...current,
      fields: [
        ...current.fields,
        {
          id: `field-${Date.now()}`,
          name: variableOptions[0] ?? "",
          selectors: defaultSelectorValues(inventory, fieldSelectorDimensions),
        },
      ],
    }));
  }

  function removeField(id: string) {
    setState((current) => ({
      ...current,
      fields: current.fields.filter((field) => field.id !== id),
    }));
  }

  function onTimePresetChange(value: TimePresetId) {
    const preset = timePresets.find((item) => item.id === value);
    updateState({
      timePreset: value,
      selectedTimes: value === "custom" ? state.selectedTimes : preset?.times ?? state.selectedTimes,
    });
  }

  function toggleTime(time: string) {
    const selected = state.selectedTimes.includes(time)
      ? state.selectedTimes.filter((item) => item !== time)
      : [...state.selectedTimes, time].sort();
    updateState({ selectedTimes: selected, timePreset: "custom" });
  }

  async function onSave() {
    if (!state.fields.some((field) => field.name)) {
      setSaveState("error");
      setMessage("Select at least one field.");
      return;
    }
    const missingSelector = fieldSelectorDimensions.find((dimension) =>
      state.fields.some((field) => field.name && !field.selectors[dimension]),
    );
    if (missingSelector) {
      setSaveState("error");
      setMessage(`Choose ${labelFromName(missingSelector).toLowerCase()} for each selected field.`);
      return;
    }
    if (!state.startDate || !state.endDate) {
      setSaveState("error");
      setMessage("Choose a start and end date.");
      return;
    }
    if (!state.selectedTimes.length) {
      setSaveState("error");
      setMessage("Choose at least one time.");
      return;
    }

    setSaveState("loading");
    setMessage(null);
    try {
      const updatedRun = await saveDatasetContract(slug, buildContract(run.contract, state, inventory));
      onRunChange(updatedRun);
      setSaveState("success");
      setMessage("Contract saved.");
    } catch (caught) {
      setSaveState("error");
      setMessage(caught instanceof Error ? caught.message : "Could not save contract.");
    }
  }

  return (
    <article className="contract-editor tab-panel">
      <div className="contract-editor-header">
        <dl className="metadata-grid contract-metadata-grid">
          <div>
            <dt>Intent</dt>
            <dd>{run.contract.intent}</dd>
          </div>
          <div>
            <dt>Inventory</dt>
            <dd>{inventory ? `${inventory.extractor_name} / ${inventory.extraction_method}` : loadState === "loading" ? "Loading" : "Unavailable"}</dd>
          </div>
          <div>
            <dt>Selected times</dt>
            <dd>{selectedTimeLabel(state)}</dd>
          </div>
        </dl>
        <button className="primary-button contract-save-button" disabled={saveState === "loading"} type="button" onClick={() => void onSave()}>
          {saveState === "loading" ? <LoaderCircle className="spin" size={17} /> : <Save size={17} />}
          Save contract
        </button>
      </div>

      {message ? (
        <div className={saveState === "error" || loadState === "error" ? "alert" : "inline-message"} role="status">
          {saveState === "error" || loadState === "error" ? <AlertTriangle size={18} /> : null}
          <span>{message}</span>
        </div>
      ) : null}

      <div className="contract-editor-grid">
        <section className="workspace-section workspace-section-wide">
          <div className="section-title-row">
            <h3>Selected Fields</h3>
            <button className="secondary-button compact-button" type="button" onClick={addField}>
              <Plus size={15} />
              Add field
            </button>
          </div>

          <div className="editor-variable-list">
            {state.fields.map((field) => {
              const metadata = variableMetadata(inventory, field.name);
              return (
                <div className="editor-variable-row" key={field.id}>
                  <label>
                    Variable
                    <select value={field.name} onChange={(event) => updateField(field.id, { name: event.target.value })}>
                      {variableOptions.length ? (
                        variableOptions.map((option) => (
                          <option key={option} value={option}>
                            {metadataText(variableMetadata(inventory, option), "label") ?? labelFromName(option)}
                          </option>
                        ))
                      ) : (
                        <option value={field.name}>{field.name || "No inventory options"}</option>
                      )}
                    </select>
                  </label>
                  {fieldSelectorDimensions.map((dimension) => {
                    const options = optionValues(inventory, dimension);
                    const unit = inventory?.option_units[dimension];
                    return (
                      <label key={dimension}>
                        {labelFromName(dimension)}
                        <select
                          value={field.selectors[dimension] ?? ""}
                          onChange={(event) =>
                            updateField(field.id, {
                              selectors: { ...field.selectors, [dimension]: event.target.value },
                            })
                          }
                        >
                          <option value="">Choose {labelFromName(dimension).toLowerCase()}</option>
                          {options.map((option) => (
                            <option key={option} value={option}>
                              {option}
                              {unit ? ` ${unit}` : ""}
                            </option>
                          ))}
                        </select>
                      </label>
                    );
                  })}
                  <button
                    aria-label={`Remove ${field.name || "field"}`}
                    className="icon-button remove-variable-button"
                    type="button"
                    onClick={() => removeField(field.id)}
                  >
                    <Trash2 size={16} />
                  </button>

                  {metadataText(metadata, "description") ? <p className="metadata-note">{compactText(metadataText(metadata, "description"), "")}</p> : null}
                </div>
              );
            })}
          </div>
        </section>

        <section className="workspace-section">
          <h3>Time Range</h3>
          <div className="editor-field-grid">
            <label>
              Product type
              <select value={state.productType} onChange={(event) => updateState({ productType: event.target.value })}>
                {productOptions.length ? (
                  productOptions.map((option) => (
                    <option key={option} value={option}>
                      {labelFromName(option)}
                    </option>
                  ))
                ) : (
                  <option value={state.productType}>{state.productType || "Not available"}</option>
                )}
              </select>
            </label>
            <label>
              Start date
              <input type="date" value={state.startDate} onChange={(event) => updateState({ startDate: event.target.value })} />
            </label>
            <label>
              End date
              <input type="date" value={state.endDate} onChange={(event) => updateState({ endDate: event.target.value })} />
            </label>
            <label>
              Time step
              <select value={state.timePreset} onChange={(event) => onTimePresetChange(event.target.value as TimePresetId)}>
                {timePresets.map((preset) => (
                  <option key={preset.id} value={preset.id}>
                    {preset.label}
                  </option>
                ))}
              </select>
            </label>
          </div>
          {state.timePreset === "custom" && availableTimes.length ? (
            <div className="chip-panel">
              <span>Times</span>
              <OptionChipGroup options={availableTimes} selected={state.selectedTimes} onToggle={toggleTime} />
            </div>
          ) : null}
        </section>

        <section className="workspace-section">
          <h3>Area And Format</h3>
          <div className="editor-field-grid">
            <label>
              Area
              <select value={state.areaMode} onChange={(event) => updateState({ areaMode: event.target.value as EditorState["areaMode"] })}>
                <option value="global">Global</option>
                <option value="custom">Custom bounding box</option>
              </select>
            </label>
            <label>
              Data format
              <select value={state.dataFormat} onChange={(event) => updateState({ dataFormat: event.target.value })}>
                {dataFormatOptions.length ? (
                  dataFormatOptions.map((option) => (
                    <option key={option} value={option}>
                      {option.toUpperCase()}
                    </option>
                  ))
                ) : (
                  <option value={state.dataFormat}>{state.dataFormat || "Not available"}</option>
                )}
              </select>
            </label>
            <label>
              Download
              <select value={state.downloadFormat} onChange={(event) => updateState({ downloadFormat: event.target.value })}>
                {downloadFormatOptions.length ? (
                  downloadFormatOptions.map((option) => (
                    <option key={option} value={option}>
                      {labelFromName(option)}
                    </option>
                  ))
                ) : (
                  <option value={state.downloadFormat}>{state.downloadFormat || "Not available"}</option>
                )}
              </select>
            </label>
            <label className="confirm-toggle">
              <input
                checked={state.humanConfirmed}
                type="checkbox"
                onChange={(event) => updateState({ humanConfirmed: event.target.checked })}
              />
              Ready for pipeline planning
            </label>
          </div>

          {state.areaMode === "custom" ? (
            <div className="area-grid">
              {(["North", "West", "South", "East"] as const).map((label, index) => (
                <label key={label}>
                  {label}
                  <input
                    type="number"
                    value={state.area[index]}
                    onChange={(event) => {
                      const next = [...state.area] as [number, number, number, number];
                      next[index] = Number(event.target.value);
                      updateState({ area: next });
                    }}
                  />
                </label>
              ))}
            </div>
          ) : null}
        </section>
      </div>
    </article>
  );
}

function OptionChipGroup({
  onToggle,
  options,
  selected,
  suffix = "",
}: {
  onToggle: (option: string) => void;
  options: string[];
  selected: string[];
  suffix?: string;
}) {
  return (
    <div className="option-chip-grid">
      {options.map((option) => (
        <button
          className="option-chip"
          data-selected={selected.includes(option)}
          key={option}
          type="button"
          onClick={() => onToggle(option)}
        >
          {option}
          {suffix}
        </button>
      ))}
    </div>
  );
}
