/**
 * Settings — the node's configuration, edited where it is used.
 *
 * Three decisions shape this screen.
 *
 * **A field saves when you leave it, not when you press a button.** There
 * is no Save; `PUT /config` takes one dotted key at a time and writes the
 * file, so a form-wide save would be a worse version of what the API
 * already does per field. Each row reports its own state instead.
 *
 * **Locked settings are shown, not hidden.** `daemon.*` and
 * `dataset.roots` decide what this daemon will reach and who can reach it,
 * so the API refuses to write them. Hiding them would make the screen look
 * incomplete; showing them read-only with the reason turns a refusal into
 * an explanation.
 *
 * **The captioner is probed on demand, never on load.** Opening a settings
 * screen should not make a network call to a vision model, and a probe
 * that runs by itself cannot be told apart from one that failed.
 */

import { useCallback, useEffect, useState } from "preact/hooks";
import { api, ApiError, isAbort } from "~/api/client";
import type {
  CaptionerInfo,
  CaptionerProbe,
  ConfigPayload,
  SecretInfo,
} from "~/api/types";
import { Field } from "./Field";

export function SettingsScreen({ onError }: { onError(message: string | null): void }) {
  const [config, setConfig] = useState<ConfigPayload | null>(null);
  const [captioners, setCaptioners] = useState<CaptionerInfo[]>([]);
  const [secrets, setSecrets] = useState<SecretInfo[]>([]);
  const [probe, setProbe] = useState<CaptionerProbe | null>(null);
  const [probing, setProbing] = useState(false);

  const fail = useCallback(
    (error: unknown) => {
      if (!isAbort(error)) onError(error instanceof ApiError ? error.message : String(error));
    },
    [onError],
  );

  useEffect(() => {
    (async () => {
      try {
        const [loaded, backends, keys] = await Promise.all([
          api.config(),
          api.captioners(),
          api.secrets(),
        ]);
        setConfig(loaded);
        setCaptioners(backends.captioners);
        setSecrets(keys.secrets);
      } catch (error) {
        fail(error);
      }
    })();
  }, [fail]);

  /** Write one setting. Returns false so the row can keep its own error. */
  const save = useCallback(
    async (key: string, value: unknown): Promise<string | null> => {
      try {
        const updated = await api.putConfig({ [key]: value });
        setConfig(updated);
        // A saved provider change makes any earlier probe stale.
        if (key.startsWith("captioner.")) setProbe(null);
        return null;
      } catch (error) {
        if (isAbort(error)) return null;
        return error instanceof ApiError ? error.message : String(error);
      }
    },
    [],
  );

  const test = useCallback(async () => {
    setProbing(true);
    setProbe(null);
    try {
      setProbe(await api.testCaptioner());
    } catch (error) {
      fail(error);
    } finally {
      setProbing(false);
    }
  }, [fail]);

  if (!config) {
    return (
      <div class="empty">
        <span class="spinner" aria-label="Loading" />
      </div>
    );
  }

  const captioner = config.captioner;
  const isOllama = captioner.provider === "ollama";
  const claudeKey = secrets.find((entry) => entry.name === "claude");

  return (
    <div class="settings">
      <header class="settings__head">
        <span class="settings__title">Settings</span>
        <span class="topbar__spacer" />
        <span class="settings__source mono" title="the file these are written to">
          {config.source ?? "no config file yet"}
        </span>
      </header>

      <div class="settings__body">
        {/* -- captioning ------------------------------------------------ */}
        <section class="panel">
          <h2 class="panel__title">Captioning</h2>
          <p class="panel__note">
            Ollama runs on this node and sends nothing anywhere. Claude writes better
            captions and sends every image to Anthropic.
          </p>

          <Field
            label="Backend"
            hint="which model writes the captions"
            value={captioner.provider}
            options={captioners.map((entry) => ({
              value: entry.name,
              label: entry.available ? entry.label : `${entry.label} — not installed`,
              disabled: !entry.available,
            }))}
            onSave={(value) => save("captioner.provider", value)}
          />

          {isOllama ? (
            <>
              <Field
                label="Ollama URL"
                hint="where the daemon on this node is listening"
                value={captioner.ollama_url}
                onSave={(value) => save("captioner.ollama_url", value)}
              />
              <Field
                label="Model"
                hint={
                  probe?.models?.length
                    ? `pulled on this node: ${probe.models.join(", ")}`
                    : "run the test to list what is pulled"
                }
                value={captioner.ollama_model}
                onSave={(value) => save("captioner.ollama_model", value)}
              />
            </>
          ) : (
            <>
              <Field
                label="Model"
                hint="an Anthropic model id"
                value={captioner.claude_model}
                onSave={(value) => save("captioner.claude_model", value)}
              />
              <div class="field field--static">
                <label class="field__label">API key</label>
                <div class="field__control">
                  <span class={`chip${claudeKey?.found ? "" : " chip--warn"}`}>
                    {claudeKey?.found ? "found on this node" : "not set"}
                  </span>
                </div>
                <div class="field__hint">
                  {/* The rule that keeps config.toml safe to commit and share. */}
                  Keys never travel over this API or live in the config file. Set{" "}
                  <code class="mono">{claudeKey?.env[0] ?? "FLUXKREA_CLAUDE_API_KEY"}</code> in
                  the environment the daemon starts in.
                </div>
              </div>
            </>
          )}

          <Field
            label="Trigger token"
            hint="prepended to every caption, e.g. a LoRA trigger word"
            value={captioner.prefix}
            placeholder="none"
            onSave={(value) => save("captioner.prefix", value)}
          />
          <Field
            label="Prompt"
            hint="what the model is asked. Empty uses the built-in caption prompt."
            value={captioner.prompt}
            multiline
            placeholder="(the built-in prompt)"
            onSave={(value) => save("captioner.prompt", value)}
          />
          <Field
            label="Max tokens"
            hint="a ceiling on one caption, not a target"
            value={captioner.max_tokens}
            type="number"
            onSave={(value) => save("captioner.max_tokens", Number(value))}
          />
          <Field
            label="Timeout"
            hint="seconds to wait for one image"
            value={captioner.timeout}
            type="number"
            onSave={(value) => save("captioner.timeout", Number(value))}
          />

          <div class="settings__actions">
            <button class="btn" onClick={() => void test()} disabled={probing}>
              {probing ? <span class="spinner" /> : null}
              Test connection
            </button>
            {probe && (
              <span class={`probe ${probe.ok ? "probe--ok" : "probe--bad"}`} role="status">
                {probe.ok ? "✓" : "✕"} {probe.message}
              </span>
            )}
          </div>
        </section>

        {/* -- masking --------------------------------------------------- */}
        <section class="panel">
          <h2 class="panel__title">Masking</h2>
          <p class="panel__note">
            Defaults for new mask exports. White is trained, black is protected — a
            larger expansion protects more of the face.
          </p>

          <Field
            label="Detector"
            value={config.mask.detector}
            hint="face detector used by Detect"
            onSave={(value) => save("mask.detector", value)}
          />
          <Field
            label="Confidence"
            hint="lower finds more faces, and more things that are not faces"
            value={config.mask.confidence}
            type="number"
            step="0.05"
            onSave={(value) => save("mask.confidence", Number(value))}
          />
          <Field
            label="Expand"
            hint="box growth factor; 1.6 covers hair and jaw"
            value={config.mask.expand}
            type="number"
            step="0.1"
            onSave={(value) => save("mask.expand", Number(value))}
          />
          <Field
            label="Expand up"
            hint="extra upward growth, for the hairline"
            value={config.mask.expand_up}
            type="number"
            step="0.1"
            onSave={(value) => save("mask.expand_up", Number(value))}
          />
          <Field
            label="Feather"
            hint="pixels of gradient at the mask boundary"
            value={config.mask.feather}
            type="number"
            onSave={(value) => save("mask.feather", Number(value))}
          />
          <Field
            label="Require review"
            hint="refuse to export masks for images nobody has looked at"
            value={config.mask.require_review}
            onSave={(value) => save("mask.require_review", value === "true")}
            options={[
              { value: "true", label: "yes — export needs a review pass" },
              { value: "false", label: "no" },
            ]}
          />
        </section>

        {/* -- locked ---------------------------------------------------- */}
        <section class="panel">
          <h2 class="panel__title">Not editable here</h2>
          <p class="panel__note">
            These decide what this daemon will reach and who can reach it, so they are
            changed by someone with a shell on the node — edit{" "}
            <code class="mono">{config.source ?? "config.toml"}</code> and restart.
          </p>
          <ul class="locked">
            {config.read_only.map((key) => (
              <li key={key} class="locked__row">
                <code class="mono">{key}</code>
                <span class="locked__value mono">{describeLocked(key, config)}</span>
              </li>
            ))}
          </ul>
        </section>
      </div>
    </div>
  );
}

/** Show what a locked setting currently *is*, since it cannot be changed. */
function describeLocked(key: string, config: ConfigPayload): string {
  if (key === "daemon.*") {
    const daemon = config.daemon as { host?: string; port?: number };
    return `${daemon.host ?? "?"}:${daemon.port ?? "?"}`;
  }
  const [section, name] = key.split(".");
  const values = config[section as keyof ConfigPayload] as Record<string, unknown> | undefined;
  const value = name ? values?.[name] : undefined;
  if (Array.isArray(value)) return value.length ? value.join(", ") : "(none)";
  return value === undefined || value === "" ? "(unset)" : String(value);
}
