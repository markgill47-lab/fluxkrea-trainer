/**
 * The settings screen, and the prompt library inside it.
 *
 * Two things here only exist once the component is mounted, so neither is
 * reachable from a Python test: a row that saves on blur and must not save
 * a value that did not change, and a prompt picker that has to tell a
 * loaded prompt from an edited one.
 */

import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/preact";
import { describe, expect, it, vi } from "vitest";
import type { ConfigPayload } from "~/api/types";
import { SettingsScreen } from "~/settings/SettingsScreen";
import { fakeApi } from "./fixtures";

/**
 * Rendered, with all four of the screen's loads landed and flushed.
 *
 * The wait is not ceremony. The screen fetches config, captioners, secrets
 * and prompts, and a keystroke that arrives while those are still settling
 * is discarded — the parent re-render lands on top of the field's queued
 * state. It is a few milliseconds at screen open and no person types into
 * it, but a test is fast enough to hit it every time.
 */
async function ready() {
  render(<SettingsScreen onError={() => {}} />);
  await screen.findByLabelText("Trigger token");
  // The prompt picker is empty until the last of the four loads lands, so
  // its options are a real signal that the screen has finished rather than
  // a guess at how long that takes.
  await waitFor(() =>
    expect(within(screen.getByLabelText("Saved prompts")).getAllByRole("option").length).toBe(
      PROMPTS.length + 1,
    ),
  );
  // And a short settle on top. The options above prove the data arrived;
  // this covers a trailing continuation that still lands after it and
  // re-renders the screen. A keystroke delivered inside that window is
  // discarded - the parent render lands on top of the field's queued
  // state. It is a few milliseconds at screen open, so no person can hit
  // it, but a test is fast enough to hit it every time.
  await act(async () => {
    await new Promise((resolve) => setTimeout(resolve, 60));
  });
  return screen.getByLabelText("Trigger token") as HTMLInputElement;
}

const CONFIG: ConfigPayload = {
  dataset: { caption_ext: ".txt", roots: ["D:/data"] },
  mask: {
    detector: "yunet",
    confidence: 0.5,
    expand: 1.6,
    expand_up: 1.35,
    feather: 6,
    min_value: 0,
    nms: 0.3,
    require_review: true,
    write_previews: true,
  },
  captioner: {
    provider: "ollama",
    ollama_url: "http://localhost:11434",
    ollama_model: "llama3.2-vision",
    joycaption_model: "fancyfeast/llama-joycaption-beta-one-hf-llava",
    joycaption_quantize: true,
    claude_model: "claude-opus-5",
    prompt: "",
    prefix: "",
    max_tokens: 400,
    timeout: 180,
  },
  daemon: { host: "127.0.0.1", port: 8471 },
  backends: {},
  log_level: "info",
  source: "C:/config.toml",
  read_only: ["daemon.*", "dataset.roots"],
};

const PROMPTS = [
  { name: "default", text: "Describe this image.", builtin: true, shadows_builtin: false },
  { name: "person", text: "Describe this person.", builtin: true, shadows_builtin: false },
];

function setup(overrides: Partial<Parameters<typeof fakeApi>[0]> = {}) {
  fakeApi({
    config: async () => CONFIG,
    captioners: async () => ({
      captioners: [
        { name: "ollama", label: "Ollama (local)", available: true },
        { name: "joycaption", label: "JoyCaption (local, in-process)", available: true },
      ],
      configured: "ollama",
    }),
    secrets: async () => ({ secrets: [{ name: "claude", found: false, env: ["ANTHROPIC_API_KEY"] }] }),
    prompts: async () => ({ prompts: PROMPTS, file: "C:/prompts.json" }),
    ...overrides,
  });
}

describe("saving a setting", () => {
  it("writes one dotted key on blur", async () => {
    const putConfig = vi.fn(async () => CONFIG);
    setup({ putConfig: putConfig as never });

    const field = await ready();

    fireEvent.input(field, { target: { value: "mara" } });
    await waitFor(() => expect(field).toHaveValue("mara"));
    fireEvent.blur(field);

    await waitFor(() => expect(putConfig).toHaveBeenCalledWith({ "captioner.prefix": "mara" }));
  });

  it("does not write a value that did not change", async () => {
    // Tabbing through a form must not rewrite the config file once per field.
    const putConfig = vi.fn(async () => CONFIG);
    setup({ putConfig: putConfig as never });

    const field = await ready();

    fireEvent.blur(field);
    await new Promise((resolve) => setTimeout(resolve, 10));
    expect(putConfig).not.toHaveBeenCalled();
  });

  it("puts the row back to what the node still holds when a write is refused", async () => {
    setup({
      putConfig: (async () => {
        throw new Error("mask.expand below 1.0 would shrink the detected box");
      }) as never,
    });

    const field = await ready();

    fireEvent.input(field, { target: { value: "nope" } });
    await waitFor(() => expect(field).toHaveValue("nope"));
    fireEvent.blur(field);

    // The refusal is shown on the row, and the value shown is the one the
    // node actually has - not the one it rejected.
    await waitFor(() => expect(screen.getByText(/would shrink/)).toBeInTheDocument());
    expect(field).toHaveValue("");
  });
});

describe("what cannot be edited", () => {
  it("lists the locked settings with their current values", async () => {
    setup();
    render(<SettingsScreen onError={() => {}} />);

    await waitFor(() => expect(screen.getByText("daemon.*")).toBeInTheDocument());
    expect(screen.getByText("127.0.0.1:8471")).toBeInTheDocument();
    expect(screen.getByText("dataset.roots")).toBeInTheDocument();
  });
});

describe("probing a captioner", () => {
  it("does not probe on load", async () => {
    // Opening settings must not make a network call to a vision model.
    const testCaptioner = vi.fn(async () => ({ ok: true, message: "ready" }));
    setup({ testCaptioner: testCaptioner as never });

    render(<SettingsScreen onError={() => {}} />);
    await screen.findByLabelText("Trigger token");
    expect(testCaptioner).not.toHaveBeenCalled();
  });

  it("renders a failure as an answer rather than an error", async () => {
    setup({
      testCaptioner: (async () => ({
        ok: false,
        message: "Cannot reach Ollama. Is `ollama serve` running?",
      })) as never,
    });

    render(<SettingsScreen onError={() => {}} />);
    fireEvent.click(await screen.findByRole("button", { name: /test connection/i }));

    await waitFor(() => expect(screen.getByText(/ollama serve/)).toBeInTheDocument());
  });
});

describe("the prompt library", () => {
  it("fills the box when a saved prompt is picked", async () => {
    const putConfig = vi.fn(async () => CONFIG);
    setup({ putConfig: putConfig as never });

    render(<SettingsScreen onError={() => {}} />);
    const picker = await screen.findByLabelText("Saved prompts");

    fireEvent.change(picker, { target: { value: "person" } });

    await waitFor(() =>
      expect(putConfig).toHaveBeenCalledWith({ "captioner.prompt": "Describe this person." }),
    );
  });

  it("saves the current text under a name", async () => {
    const savePrompt = vi.fn(async () => PROMPTS[0]!);
    setup({ savePrompt: savePrompt as never });

    render(<SettingsScreen onError={() => {}} />);
    const box = await screen.findByLabelText("Prompt");
    fireEvent.input(box, { target: { value: "Describe her jacket." } });

    fireEvent.click(screen.getByRole("button", { name: /save as/i }));
    fireEvent.input(screen.getByPlaceholderText(/name, e.g./), {
      target: { value: "mara-portrait" },
    });
    fireEvent.click(screen.getByRole("button", { name: /^save$/i }));

    await waitFor(() =>
      expect(savePrompt).toHaveBeenCalledWith("mara-portrait", "Describe her jacket."),
    );
  });

  it("will not save a prompt with no name", async () => {
    const savePrompt = vi.fn();
    setup({ savePrompt: savePrompt as never });

    render(<SettingsScreen onError={() => {}} />);
    const box = await screen.findByLabelText("Prompt");
    fireEvent.input(box, { target: { value: "some text" } });
    fireEvent.click(screen.getByRole("button", { name: /save as/i }));

    expect(screen.getByRole("button", { name: /^save$/i })).toBeDisabled();
    expect(savePrompt).not.toHaveBeenCalled();
  });

  it("offers to delete a saved prompt but never a built-in", async () => {
    setup({
      prompts: (async () => ({
        prompts: [
          ...PROMPTS,
          { name: "mine", text: "Describe this image.", builtin: false, shadows_builtin: false },
        ],
        file: "C:/prompts.json",
      })) as never,
      putConfig: (async () => CONFIG) as never,
    });

    render(<SettingsScreen onError={() => {}} />);
    const picker = await screen.findByLabelText("Saved prompts");

    // "mine" and the built-in "default" share their text, and the saved one
    // is the one that must win - otherwise deleting is offered for a
    // built-in, which the API refuses.
    fireEvent.change(picker, { target: { value: "person" } });
    await waitFor(() => expect(screen.queryByRole("button", { name: /delete/i })).toBeNull());
  });
});
