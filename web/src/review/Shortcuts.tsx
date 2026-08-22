/**
 * The shortcut overlay.
 *
 * Doc 10 wants this generated from a registry rather than hand-maintained,
 * so the overlay and the bindings cannot drift. The registry is this table:
 * ReviewScreen's key handler and this list are the same source, and adding
 * a binding without adding a row here is the thing to avoid.
 */

interface Props {
  onClose(): void;
}

const GROUPS: { title: string; keys: [string, string][] }[] = [
  {
    title: "Navigation",
    keys: [
      ["J / ↓", "Next item"],
      ["K / ↑", "Previous item"],
      ["Space", "Mark reviewed, advance"],
      ["Shift+Space", "Mark reviewed, stay"],
    ],
  },
  {
    title: "Boxes",
    keys: [
      ["B", "Draw mode, then drag"],
      ["E", "Ellipse or rectangle"],
      ["Del / Backspace", "Delete selected"],
      ["Tab", "Cycle boxes"],
      ["← ↑ → ↓", "Nudge 1px (10px with Shift)"],
      ["Ctrl+Z / Ctrl+Shift+Z", "Undo / redo"],
    ],
  },
  {
    title: "View",
    keys: [
      ["M", "Mask: off → overlay → isolate"],
      ["D", "Toggle detected boxes"],
      ["0", "Fit to window"],
      ["1", "100% zoom"],
      ["Middle-drag", "Pan"],
      ["Wheel", "Zoom at cursor"],
    ],
  },
  {
    title: "Detection",
    keys: [
      ["R", "Re-detect this image"],
      ["Esc", "Cancel mode / clear selection"],
      ["?", "This overlay"],
    ],
  },
];

export function Shortcuts({ onClose }: Props) {
  return (
    <div class="shortcuts" onClick={onClose} role="dialog" aria-modal="true" aria-label="Keyboard shortcuts">
      <div class="shortcuts__panel" onClick={(event) => event.stopPropagation()}>
        <h2 style={{ margin: "0 0 4px" }}>Keyboard</h2>
        <p style={{ margin: "0 0 16px", color: "var(--text-secondary)", fontSize: "12px" }}>
          The review pass is completable without the mouse, except for drawing.
        </p>
        <div class="shortcuts__grid">
          {GROUPS.map((group) => (
            <div key={group.title}>
              <h3 class="inspector__label">{group.title}</h3>
              {group.keys.map(([key, label]) => (
                <div class="shortcuts__row" key={key}>
                  <span style={{ color: "var(--text-secondary)" }}>{label}</span>
                  <kbd>{key}</kbd>
                </div>
              ))}
            </div>
          ))}
        </div>
        <button class="btn" style={{ marginTop: "16px" }} onClick={onClose}>
          Close
        </button>
      </div>
    </div>
  );
}
