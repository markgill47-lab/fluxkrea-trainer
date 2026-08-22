/**
 * The first thing a student sees: pick a project, or make one.
 *
 * The gate exists because a lab node is shared. Without it the app opens
 * on whatever dataset happens to be first in the node's registry — which
 * belongs to somebody else, and the first action anybody takes is a batch
 * operation. Making the choice explicit before anything is on screen costs
 * one click and removes that entirely.
 *
 * Which project a browser has open is stored *in that browser*. The daemon
 * deliberately has no notion of a current project: several people share
 * one, and a server-side selection would mean the last person to click
 * changed everybody's screen.
 */

import { useCallback, useEffect, useRef, useState } from "preact/hooks";
import { api, ApiError, isAbort } from "~/api/client";
import type { Project } from "~/api/types";

/** Where this browser remembers its project. Per-origin, so per-node. */
const STORED = "fluxkrea.project";

export function storedProject(): string | null {
  try {
    return localStorage.getItem(STORED);
  } catch {
    // Private browsing, or storage disabled. The gate simply asks again.
    return null;
  }
}

export function rememberProject(id: string | null): void {
  try {
    if (id) localStorage.setItem(STORED, id);
    else localStorage.removeItem(STORED);
  } catch {
    // Not being able to remember is a worse session, not a broken one.
  }
}

interface Props {
  projects: Project[];
  /** Shown when a project was open and has since been deleted elsewhere. */
  notice?: string | null;
  onOpen(id: string): void;
  onCreated(): Promise<unknown> | void;
  onError(message: string | null): void;
}

export function ProjectGate({ projects, notice, onOpen, onCreated, onError }: Props) {
  const [name, setName] = useState("");
  const [busy, setBusy] = useState(false);
  const input = useRef<HTMLInputElement>(null);

  // Focus the field, because on a fresh node this screen has exactly one
  // thing to do and typing should be it.
  useEffect(() => {
    if (!projects.length) input.current?.focus();
  }, [projects.length]);

  const create = useCallback(async () => {
    const trimmed = name.trim();
    if (!trimmed || busy) return;
    setBusy(true);
    try {
      const created = await api.createProject(trimmed);
      setName("");
      await onCreated();
      onOpen(created.id);
      onError(null);
    } catch (error) {
      if (!isAbort(error)) onError(error instanceof ApiError ? error.message : String(error));
    } finally {
      setBusy(false);
    }
  }, [name, busy, onCreated, onOpen, onError]);

  return (
    <div class="gate">
      <div class="gate__panel">
        <h1 class="gate__title">
          {projects.length ? "Open a project" : "Create a project"}
        </h1>
        <p class="gate__lead">
          A project is a set of dataset folders that share one training
          configuration. Everything you do here happens inside one.
        </p>

        {notice && <div class="banner banner--warn">⚠ {notice}</div>}

        {projects.length > 0 && (
          <ul class="gate__list">
            {projects.map((project) => (
              <li key={project.id}>
                <button class="gate__project" onClick={() => onOpen(project.id)}>
                  <span class="gate__project-name">{project.name}</span>
                  <span class="gate__project-meta tabular">
                    {project.datasets.length}{" "}
                    {project.datasets.length === 1 ? "dataset" : "datasets"}
                  </span>
                </button>
              </li>
            ))}
          </ul>
        )}

        <form
          class="gate__new"
          onSubmit={(event) => {
            event.preventDefault();
            void create();
          }}
        >
          <input
            ref={input}
            class="input"
            placeholder={projects.length ? "…or start a new one" : "Project name"}
            value={name}
            maxLength={64}
            onInput={(event) => setName((event.target as HTMLInputElement).value)}
            aria-label="New project name"
          />
          <button class="btn btn--accent" type="submit" disabled={!name.trim() || busy}>
            New project
          </button>
        </form>

        <p class="gate__hint">
          There are no passwords. The project name is what appears beside
          your runs in the shared training queue, so pick one the room can
          tell apart.
        </p>
      </div>
    </div>
  );
}
