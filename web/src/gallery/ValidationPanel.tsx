/**
 * The validation report — doc 08.
 *
 * "Grouped, collapsible list of problems. Each row links to the offending
 * item. Counts by severity at the top."
 *
 * Grouped by `kind` rather than by item: 40 images missing a caption is
 * one problem to act on, not 40 problems to read.
 */

import { useMemo, useState } from "preact/hooks";
import type { Problem, ValidationReport } from "~/api/types";

interface Props {
  report: ValidationReport;
  onPick(stem: string): void;
  onClose(): void;
}

const SEVERITY_ORDER = { error: 0, warning: 1, info: 2 } as const;

export function ValidationPanel({ report, onPick, onClose }: Props) {
  const [open, setOpen] = useState<string | null>(null);

  const groups = useMemo(() => {
    const byKind = new Map<string, Problem[]>();
    for (const problem of report.problems) {
      const list = byKind.get(problem.kind) ?? [];
      list.push(problem);
      byKind.set(problem.kind, list);
    }
    return [...byKind.entries()].sort(
      ([, a], [, b]) =>
        SEVERITY_ORDER[a[0]!.severity] - SEVERITY_ORDER[b[0]!.severity] || b.length - a.length,
    );
  }, [report]);

  const errors = report.problems.filter((p) => p.severity === "error").length;
  const warnings = report.problems.filter((p) => p.severity === "warning").length;

  return (
    <section class="problems" aria-label="Validation report">
      <header class="problems__head">
        <strong>Validation</strong>
        <span class="hint tabular">
          {report.items} items · {errors} errors · {warnings} warnings
        </span>
        <span class="topbar__spacer" />
        <button class="btn btn--ghost" onClick={onClose}>
          Close
        </button>
      </header>

      <div class="problems__list">
        {groups.map(([kind, problems]) => (
          <div key={kind}>
            <button
              class={`problems__group problems__group--${problems[0]!.severity}`}
              aria-expanded={open === kind}
              onClick={() => setOpen(open === kind ? null : kind)}
            >
              <span>{open === kind ? "▾" : "▸"}</span>
              <span class="mono">{kind}</span>
              <span class="hint">{problems[0]!.message}</span>
              <span class="topbar__spacer" />
              <span class="tabular">{problems.length}</span>
            </button>

            {open === kind && (
              <ul class="problems__items">
                {problems.slice(0, 200).map((problem, index) => (
                  <li key={index}>
                    <button class="problems__item" onClick={() => problem.stem && onPick(problem.stem)}>
                      <span class="mono">{problem.stem ?? "—"}</span>
                      <span class="hint">{problem.message}</span>
                    </button>
                  </li>
                ))}
                {problems.length > 200 && (
                  <li class="hint" style={{ padding: "4px 10px" }}>
                    … and {problems.length - 200} more
                  </li>
                )}
              </ul>
            )}
          </div>
        ))}
      </div>
    </section>
  );
}
