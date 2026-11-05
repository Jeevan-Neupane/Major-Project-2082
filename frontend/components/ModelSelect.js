"use client";

import { useEffect, useMemo, useRef, useState } from "react";

/** Turn a registry key like "multiscale_dualviewcnn_verse19" into the
 *  architecture name so options read "Human Name — architecture". */
function modelFamilyLabel(modelKey) {
    if (modelKey.startsWith("multiscale_dualviewcnn")) return "MultiScale DualViewCNN";
    if (modelKey.startsWith("dualviewcnn")) return "DualViewCNN";
    if (modelKey.startsWith("swinx2s_biplanar")) return "SwinX2S (biplanar)";
    if (modelKey.startsWith("swinx2s")) return "SwinX2S";
    if (modelKey.startsWith("transvert")) return "TransVert";
    return modelKey;
}

/** Extract val-Dice score and parameter-count hints from a registry description. */
function parseQuality(description) {
    const text = description || "";
    const diceMatch = text.match(/Dice\s*~\s*([\d.]+)/i);
    const paramMatch = text.match(/~\s*(\d+M)\s*params/i);
    const dice = diceMatch ? parseFloat(diceMatch[1]) : null;
    return {
        dice: dice !== null && !Number.isNaN(dice) ? dice : null,
        params: paramMatch ? paramMatch[1] : null,
    };
}

/* Measurement gauge for the Dice score — reads like a lab readout,
 * not another colored chip in a row of colored chips. */
function DiceGauge({ dice, compact = false }) {
    if (dice === null) {
        return compact ? null : (
            <span className="font-readout rounded border border-bone-800/70 bg-bone-900/30 px-1.5 py-0.5 text-[10px] text-bone-500">
                no val. gauge
            </span>
        );
    }
    // Practical score span: most values live in 0.5–0.95; map with headroom.
    const pct = Math.max(8, Math.min(100, dice * 100));
    const tier = dice >= 0.85 ? "high" : dice >= 0.7 ? "good" : dice >= 0.5 ? "fair" : "low";
    const barTone = { high: "bg-bone-200", good: "bg-bone-300", fair: "bg-bone-500", low: "bg-red-700" }[tier];
    const textTone = { high: "text-bone-100", good: "text-bone-200", fair: "text-bone-400", low: "text-red-400" }[tier];
    return (
        <span className="inline-flex items-center gap-1.5" title={`Validation Dice ≈ ${dice} on the 0–1 scale`}>
            <span className="h-[3px] w-9 overflow-hidden rounded-full bg-bone-900/70">
                <span className={`block h-full ${barTone}`} style={{ width: `${pct}%` }} />
            </span>
            <span className={`font-readout text-[10px] leading-none ${textTone}`}>{dice.toFixed(3)}</span>
        </span>
    );
}

const DICE_LEGEND = [
    { label: "\u2265 0.850", cls: "bg-bone-200" },
    { label: "0.700\u20130.850", cls: "bg-bone-300" },
    { label: "0.500\u20130.700", cls: "bg-bone-500" },
    { label: "< 0.500", cls: "bg-red-700" },
];

export default function ModelSelect({ models = [], groupedModels = {}, selectedInfo, value, onChange, anatomyLabel }) {
    const [open, setOpen] = useState(false);
    const [query, setQuery] = useState("");
    const [highlight, setHighlight] = useState(-1);
    const rootRef = useRef(null);
    const searchRef = useRef(null);
    const listRef = useRef(null);

    useEffect(() => {
        if (!open) return undefined;
        const onDocClick = (e) => {
            if (rootRef.current && !rootRef.current.contains(e.target)) setOpen(false);
        };
        document.addEventListener("mousedown", onDocClick);
        return () => document.removeEventListener("mousedown", onDocClick);
    }, [open]);

    useEffect(() => {
        if (!open) return undefined;
        const onKey = (e) => {
            if (e.key === "Escape") setOpen(false);
        };
        window.addEventListener("keydown", onKey);
        return () => window.removeEventListener("keydown", onKey);
    }, [open]);

    useEffect(() => {
        if (open) {
            setQuery("");
            setHighlight(-1);
            const id = requestAnimationFrame(() => searchRef.current?.focus());
            return () => cancelAnimationFrame(id);
        }
        return undefined;
    }, [open]);

    const filteredGroups = useMemo(() => {
        const q = query.trim().toLowerCase();
        const out = [];
        for (const [group, list] of Object.entries(groupedModels)) {
            const hits = !q
                ? list
                : list.filter(
                      (m) =>
                          modelFamilyLabel(m.model_key).toLowerCase().includes(q) ||
                          m.model_key.toLowerCase().includes(q) ||
                          (m.description || "").toLowerCase().includes(q)
                  );
            if (hits.length) out.push([group, hits]);
        }
        return out;
    }, [groupedModels, query]);

    const groupedList = useMemo(() => {
        const flat = [];
        filteredGroups.forEach(([group, list]) => {
            list.forEach((m) => flat.push({ group, model: m }));
        });
        return flat;
    }, [filteredGroups]);

    const flatOptions = useMemo(() => groupedList.map((entry) => entry.model), [groupedList]);

    const selected = useMemo(
        () => models.find((m) => m.model_key === value) || null,
        [models, value]
    );

    if (!open && !value && models.length === 0) {
        return (
            <div className="h-10 w-full animate-pulse rounded-[3px] border border-bone-900 bg-bone-900/30" />
        );
    }

    function onSearchKeyDown(e) {
        if (e.key === "ArrowDown" || e.key === "ArrowUp") {
            e.preventDefault();
            const count = flatOptions.length;
            if (!count) return;
            setHighlight((prev) => {
                const next =
                    e.key === "ArrowDown"
                        ? (prev + 1) % count
                        : (prev - 1 + count) % count;
                const node = listRef.current?.querySelector(`[data-index="${next}"]`);
                node?.scrollIntoView({ block: "nearest" });
                return next;
            });
        } else if (e.key === "Enter") {
            e.preventDefault();
            if (highlight >= 0 && flatOptions[highlight]) {
                onChange(flatOptions[highlight].model_key);
                setOpen(false);
            }
        }
    }

    const triggerLabel = selected ? humanName(selected) : "Select a model";
    const selectedQuality = selected ? parseQuality(selected.description) : null;

    return (
        <div ref={rootRef} className="relative">
            {/* Trigger — an instrument control, not a generic select. */}
            <button
                type="button"
                disabled={models.length === 0}
                onClick={() => setOpen((v) => !v)}
                aria-haspopup="listbox"
                aria-expanded={open}
                className={`group flex h-10 w-full items-center gap-3 rounded-[3px] border px-3 text-left transition-colors duration-150 focus:outline-none ${
                    open
                        ? "border-bone-400 bg-bone-900/40"
                        : "border-bone-900 bg-gray-900/60 hover:border-bone-700 focus:border-bone-600"
                } ${models.length === 0 ? "cursor-not-allowed opacity-40" : ""}`}
            >
                <span
                    className={`h-4 w-[3px] shrink-0 rounded-full transition-colors duration-150 ${
                        selected ? "bg-bone-300" : open ? "bg-bone-600" : "bg-bone-800"
                    }`}
                    aria-hidden="true"
                />
                <span className="min-w-0 flex-1">
                    <span
                        className={`block truncate text-[13px] leading-tight ${
                            selected ? "font-medium text-bone-100" : "text-gray-400"
                        }`}
                    >
                        {triggerLabel}
                    </span>
                    {selected ? (
                        <span className="font-readout block truncate text-[11px] leading-tight text-gray-500">
                            {selected.model_key}
                        </span>
                    ) : (
                        <span className="font-readout block text-[10px] leading-tight text-gray-600">
                            {anatomyLabel ? `${anatomyLabel} · ${models.length} registered` : `${models.length} registered`}
                        </span>
                    )}
                </span>
                {selected ? (
                    selectedQuality?.dice !== null ? (
                        <DiceGauge dice={selectedQuality.dice} compact />
                    ) : (
                        <span className="font-readout hidden shrink-0 text-[10px] text-bone-500 sm:inline">unverified</span>
                    )
                ) : null}
                <span
                    className={`shrink-0 text-bone-500 transition-transform duration-200 ${
                        open ? "rotate-180" : ""
                    }`}
                    aria-hidden="true"
                >
                    <svg width="10" height="10" viewBox="0 0 10 10" fill="none">
                        <path d="M1.5 3.25 5 6.75l3.5-3.5" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" />
                    </svg>
                </span>
            </button>

            {/* Panel — the reading-room tray. */}
            {open ? (
                <div
                    className="absolute left-0 right-0 z-30 mt-2 origin-top animate-[panelIn_160ms_ease-out] rounded-[3px] border border-bone-800/90 bg-gray-900 shadow-[0_18px_40px_-12px_rgba(0,0,0,0.7)]"
                    role="dialog"
                    aria-label="Model catalogue"
                >
                    <div className="flex items-center gap-2 border-b border-bone-900/80 px-3 py-2">
                        <svg width="11" height="11" viewBox="0 0 12 12" fill="none" className="shrink-0 text-bone-600" aria-hidden="true">
                            <circle cx="5" cy="5" r="3.6" stroke="currentColor" strokeWidth="1.2" />
                            <path d="m8 8 2.4 2.4" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" />
                        </svg>
                        <input
                            ref={searchRef}
                            value={query}
                            onChange={(e) => {
                                setQuery(e.target.value);
                                setHighlight(-1);
                            }}
                            onKeyDown={onSearchKeyDown}
                            placeholder="Filter by name, key, or notes…"
                            className="font-clinical w-full bg-transparent text-[12px] text-bone-100 placeholder:text-gray-600 focus:outline-none"
                            autoComplete="off"
                            aria-label="Filter models"
                        />
                        <span className="font-readout shrink-0 text-[10px] text-gray-600">
                            {flatOptions.length}/{models.length}
                        </span>
                    </div>

                    <ul ref={listRef} role="listbox" aria-label="Available models" className="max-h-72 overflow-y-auto py-1">
                        {flatOptions.length === 0 ? (
                            <li className="font-readout px-3 py-6 text-center text-[11px] text-gray-600">
                                no match for “{query}”
                            </li>
                        ) : null}

                        {filteredGroups.map(([group, list]) => (
                            <li key={group} role="presentation" className="list-none">
                                <div className="sticky top-0 z-10 flex items-baseline gap-2 border-y border-bone-900/70 bg-gray-900/95 px-3 py-1.5 backdrop-blur-sm">
                                    <span className="font-readout text-[12px] uppercase tracking-[0.14em] text-bone-300">
                                        {group}
                                    </span>
                                    <span className="font-readout text-[10px] text-gray-500">
                                        {list.length}
                                    </span>
                                    <span className="ml-auto h-px flex-1 bg-bone-900/60" />
                                </div>
                                {list.map((m) => {
                                    const flatIndex = groupedList.findIndex(
                                        (entry) => entry.model === m
                                    );
                                    const isSelected = value === m.model_key;
                                    const isHighlighted = highlight === flatIndex;
                                    const q = parseQuality(m.description);
                                    return (
                                        <button
                                            key={m.model_key}
                                            type="button"
                                            data-index={flatIndex}
                                            role="option"
                                            aria-selected={isSelected}
                                            onMouseEnter={() => setHighlight(flatIndex)}
                                            onClick={() => {
                                                onChange(m.model_key);
                                                setOpen(false);
                                            }}
                                            className={`flex w-full items-center gap-2.5 px-3 py-1.5 text-left focus:outline-none ${
                                                isHighlighted ? "bg-bone-900/50" : ""
                                            } ${isSelected ? "bg-bone-900/70" : ""}`}
                                        >
                                            <span
                                                className={`w-4 shrink-0 text-center text-[10px] ${
                                                    isSelected ? "text-bone-200" : "text-transparent"
                                                }`}
                                                aria-hidden="true"
                                            >
                                                ✔
                                            </span>
                                            <span className="min-w-0 flex-1">
                                                <span
                                                    className={`block truncate text-[12.5px] leading-tight ${
                                                        isSelected ? "font-medium text-bone-100" : "text-gray-300"
                                                    }`}
                                                >
                                                    {m.description ? humanName(m) : m.model_key}
                                                </span>
                                                <span className="font-readout block truncate text-[9.5px] leading-tight text-gray-600">
                                                    {m.model_key}
                                                </span>
                                            </span>
                                            {q.params ? (
                                                <span className="font-readout hidden shrink-0 text-[9.5px] text-gray-600 sm:inline">
                                                    {q.params}
                                                </span>
                                            ) : null}
                                            <DiceGauge dice={q.dice} compact />
                                        </button>
                                    );
                                })}
                            </li>
                        ))}
                    </ul>

                    {/* Plate footer — measurement legend. */}
                    <div className="flex flex-wrap items-center gap-x-4 gap-y-1 border-t border-bone-900/80 bg-gray-900/95 px-3 py-2">
                        <span className="font-readout text-[11px] uppercase tracking-[0.14em] text-bone-300">
                            val. dice
                        </span>
                        {DICE_LEGEND.map((entry) => (
                            <span key={entry.label} className="inline-flex items-center gap-1">
                                <span className={`h-[3px] w-4 rounded-full ${entry.cls}`} aria-hidden="true" />
                                <span className="font-readout text-[10.5px] text-gray-500">{entry.label}</span>
                            </span>
                        ))}
                    </div>
                </div>
            ) : null}

            {/* Info card — the specimen-plate annotation under the control. */}
            {selectedInfo && value ? (
                <div className="mt-2 rounded-[3px] border border-bone-900/90 bg-gray-900/50 px-3 py-2.5">
                    <div className="font-readout flex flex-wrap items-baseline gap-x-3 gap-y-1 text-[11px] uppercase tracking-[0.14em] text-bone-300">
                        <span>specimen plate</span>
                        <span className="text-gray-600">
                            {modelFamilyLabel(value)}
                        </span>
                        <span className="ml-auto normal-case tracking-normal text-gray-600">
                            {selectedInfo.weights_exist ? "weights loaded" : "weights missing"}
                        </span>
                    </div>
                    <p className="font-clinical mt-1.5 text-[12px] leading-relaxed text-gray-300">
                        {selectedInfo.description}
                    </p>
                </div>
            ) : null}
        </div>
    );
}

/** Registry descriptions such as "SwinX2S-Small for Femur (biplanar; best val
 *  Dice ~0.925)" reduce cleanly to "SwinX2S-Small for Femur". */
function humanName(m) {
    const cleaned = (m.description || "")
        .replace(/\(.*?\)/g, "")
        .replace(/\s+/g, " ")
        .trim();
    if (cleaned && cleaned.length <= 48) return cleaned;
    return modelFamilyLabel(m.model_key);
}
