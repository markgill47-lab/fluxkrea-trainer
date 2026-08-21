/**
 * The gallery inspector: one item, or aggregate stats for a selection.
 *
 * Doc 09: "The inspector shows one item, or aggregate stats plus batch
 * caption tools when multiple are selected." Those are genuinely two
 * different panels, so they are two components rather than one with a
 * pile of conditionals.
 */

import { useEffect, useRef, useState } from "preact/hooks";
import { assets } from "~/api/client";
import type { Item } from "~/api/types";

const QUALITIES = ["good", "ok", "bad"] as const;

interface SingleProps {
  dataset: string;
  item: Item;
  onCaption(stem: string, caption: string): Promise<void>;
  onQuality(stem: string, quality: string | null): void;
  onOpen(stem: string): void;
}

export function ItemInspector({ dataset, item, onCaption, onQuality, onOpen }: SingleProps) {
  const [draft, setDraft] = useState(item.caption ?? "");
  const [saving, setSaving] = useState(false);
  const loaded = useRef(item.stem);

  // Moving to another item replaces the draft, but only when the item
  // actually changed — otherwise a re-render mid-typing would wipe it.
  useEffect(() => {
    if (loaded.current !== item.stem) {
      loaded.current = item.stem;
      setDraft(item.caption ?? "");
    }
  }, [item.stem, item.caption]);

  const dirty = draft !== (item.caption ?? "");

  const save = async () => {
    setSaving(true);
    try {
      await onCaption(item.stem, draft);
    } finally {
      setSaving(false);
    }
  };

  return (
    <aside class="inspector" aria-label="Item">
      <section class="inspector__section">
        <img
          class="inspector__thumb"
          src={assets.thumb(dataset, item.stem, 480, item.token)}
          alt=""
          onClick={() => onOpen(item.stem)}
        />
        <h2 class="inspector__stem mono">{item.stem}</h2>
        <div class="kv">
          <span class="kv__key">dimensions</span>
          <span class="kv__value mono">
            {item.width && item.height ? `${item.width}×${item.height}` : "—"}
          </span>
        </div>
        <div class="kv">
          <span class="kv__key">mask</span>
          <span class="kv__value">{item.has_mask ? "yes" : "—"}</span>
        </div>
        <div class="kv">
          <span class="kv__key">boxes</span>
          <span class="kv__value">{item.boxes || "—"}</span>
        </div>
        <div class="kv">
          <span class="kv__key">reviewed</span>
          <span class="kv__value">{item.reviewed ? "yes" : "—"}</span>
        </div>
      </section>

      <section class="inspector__section">
        <h2 class="inspector__label">Caption</h2>
        <textarea
          class="caption"
          value={draft}
          rows={6}
          placeholder="No caption. The .txt sidecar is what the trainer reads."
          onInput={(event) => setDraft((event.target as HTMLTextAreaElement).value)}
        />
        <div class="caption__foot">
          <span class="caption__count tabular">
            {draft.trim() ? draft.trim().split(/\s+/).length : 0} words · {draft.length} chars
          </span>
          <button class="btn btn--accent" disabled={!dirty || saving} onClick={save}>
            {saving ? "Saving…" : dirty ? "Save" : "Saved"}
          </button>
        </div>
      </section>

      <section class="inspector__section">
        <h2 class="inspector__label">Quality</h2>
        <div class="quality">
          {QUALITIES.map((value) => (
            <button
              key={value}
              class={`chip${item.quality === value ? " chip--on" : ""}`}
              aria-pressed={item.quality === value}
              onClick={() => onQuality(item.stem, item.quality === value ? null : value)}
            >
              {value}
            </button>
          ))}
        </div>
        <p class="hint">
          Derived metadata. Deleting <code class="mono">metadata.json</code> loses nothing the
          trainer reads.
        </p>
      </section>
    </aside>
  );
}

interface MultiProps {
  items: Item[];
  onAppend(text: string): Promise<void>;
  onQualityAll(quality: string | null): void;
  onClear(): void;
}

export function SelectionInspector({ items, onAppend, onQualityAll, onClear }: MultiProps) {
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);

  const withCaption = items.filter((item) => item.has_caption).length;
  const withMask = items.filter((item) => item.has_mask).length;

  const append = async () => {
    if (!text.trim()) return;
    setBusy(true);
    try {
      await onAppend(text.trim());
      setText("");
    } finally {
      setBusy(false);
    }
  };

  return (
    <aside class="inspector" aria-label="Selection">
      <section class="inspector__section">
        <h2 class="inspector__label">Selection</h2>
        <div class="kv">
          <span class="kv__key">items</span>
          <span class="kv__value tabular">{items.length}</span>
        </div>
        <div class="kv">
          <span class="kv__key">with caption</span>
          <span class="kv__value tabular">
            {withCaption}/{items.length}
          </span>
        </div>
        <div class="kv">
          <span class="kv__key">with mask</span>
          <span class="kv__value tabular">
            {withMask}/{items.length}
          </span>
        </div>
        <button class="btn" style={{ marginTop: "8px" }} onClick={onClear}>
          Clear selection
        </button>
      </section>

      <section class="inspector__section">
        <h2 class="inspector__label">Append to captions</h2>
        <textarea
          class="caption"
          rows={3}
          value={text}
          placeholder="e.g. a trigger word"
          onInput={(event) => setText((event.target as HTMLTextAreaElement).value)}
        />
        <div class="caption__foot">
          <span class="hint">Appended to {items.length} captions</span>
          <button class="btn btn--accent" disabled={!text.trim() || busy} onClick={append}>
            {busy ? "Applying…" : "Append"}
          </button>
        </div>
      </section>

      <section class="inspector__section">
        <h2 class="inspector__label">Set quality</h2>
        <div class="quality">
          {QUALITIES.map((value) => (
            <button key={value} class="chip" onClick={() => onQualityAll(value)}>
              {value}
            </button>
          ))}
          <button class="chip" onClick={() => onQualityAll(null)}>
            clear
          </button>
        </div>
      </section>
    </aside>
  );
}
