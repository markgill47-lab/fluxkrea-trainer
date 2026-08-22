/**
 * The training screen — the bug this suite exists for.
 *
 * A run was configured against a 47-image dataset and trained a 6-image
 * one instead. `TrainScreen` renders the form or the monitor, never both,
 * so toggling the view unmounted the form and destroyed its state; the
 * seeding effect then re-picked the first registered dataset. The run said
 * nothing, and the wrong images trained for real.
 *
 * The first test here is that sequence, exactly.
 */

import { fireEvent, render, screen, waitFor } from "@testing-library/preact";
import { useState } from "preact/hooks";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { api } from "~/api/client";
import { TrainScreen } from "~/train/TrainScreen";
import { dataset, fakeApi, job, jobsResponse, model, plan, project } from "./fixtures";

const DATASETS = [dataset("api", "D:/data/api"), dataset("blizzard-training", "D:/data/blizzard")];

/** Image counts per dataset, so a plan reflects which one was asked about. */
const IMAGES: Record<string, number> = { api: 6, "blizzard-training": 47 };

/**
 * The project's shared config, standing in for the daemon's copy.
 *
 * The form's state lives in the project now rather than in sessionStorage,
 * so "does an edit survive an unmount" is a question about this round trip
 * — and a test that did not model it would be testing nothing.
 */
let stored = project();

function setup(overrides: Partial<typeof api> = {}) {
  stored = project();
  fakeApi({
    jobs: async () => jobsResponse(),
    saveProjectConfig: (async (_id: string, config: Record<string, unknown>) => {
      stored = { ...stored, config };
      return stored;
    }) as never,
    models: async () => ({ models: [model("flux2", "FLUX.2 dev")], backends: {} }),
    planRun: async (id: string, repeats: number, epochs: number) => ({
      ...plan(IMAGES[id] ?? 0, repeats, epochs),
      dataset: id,
    }),
    ...overrides,
  });
}

/** The shell owns the dataset and the project, so the test plays the shell. */
function Shell({ initial = "api" }: { initial?: string }) {
  const [current, setCurrent] = useState(initial);
  // Seeded from the stored copy, so a remount sees what was saved — which
  // is exactly what the shell does after reloading its project list.
  const [open, setOpen] = useState(stored);
  return (
    <TrainScreen
      datasets={DATASETS}
      dataset={current}
      project={open}
      onDataset={setCurrent}
      onProject={setOpen}
      onError={() => {}}
    />
  );
}

const datasetSelect = () => screen.getByLabelText("Dataset") as HTMLSelectElement;
const tab = (name: "Configure" | "Monitor") => screen.getByRole("tab", { name });

describe("the dataset survives everything", () => {
  beforeEach(() => setup());

  it("keeps the chosen dataset across a view toggle", async () => {
    // The reported sequence: pick a dataset, look at the monitor, come back.
    render(<Shell />);
    await waitFor(() => expect(datasetSelect()).toBeInTheDocument());

    fireEvent.change(datasetSelect(), { target: { value: "blizzard-training" } });
    await waitFor(() => expect(datasetSelect().value).toBe("blizzard-training"));

    fireEvent.click(tab("Monitor"));
    fireEvent.click(tab("Configure"));

    await waitFor(() => expect(datasetSelect()).toBeInTheDocument());
    expect(datasetSelect().value).toBe("blizzard-training");
  });

  it("plans against the chosen dataset, not the first registered one", async () => {
    render(<Shell />);
    await waitFor(() => expect(datasetSelect()).toBeInTheDocument());

    fireEvent.change(datasetSelect(), { target: { value: "blizzard-training" } });

    // 47 x 10 x 6, not 6 x 10 x 6. The wrong number here is what the bug
    // looked like from the outside.
    await waitFor(() => expect(screen.getByText("2,820")).toBeInTheDocument());
  });

  it("keeps every other setting across a view toggle too", async () => {
    render(<Shell />);
    await waitFor(() => expect(screen.getByLabelText("Epochs")).toBeInTheDocument());

    fireEvent.input(screen.getByLabelText("Epochs"), { target: { value: "3" } });
    fireEvent.input(screen.getByLabelText("Run name"), { target: { value: "mara-v3" } });

    fireEvent.click(tab("Monitor"));
    fireEvent.click(tab("Configure"));

    await waitFor(() => expect(screen.getByLabelText("Run name")).toHaveValue("mara-v3"));
    expect(screen.getByLabelText("Epochs")).toHaveValue(3);
  });

  it("survives the screen being unmounted entirely", async () => {
    // Switching to another rail tab and back unmounts TrainScreen, not just
    // the form, so in-memory state is not enough on its own. The save is
    // debounced, and this unmounts inside that window on purpose: a
    // cancelled timer here would drop the edit silently.
    const first = render(<Shell />);
    await waitFor(() => expect(datasetSelect()).toBeInTheDocument());
    fireEvent.input(screen.getByLabelText("Epochs"), { target: { value: "9" } });
    await waitFor(() => expect(screen.getByLabelText("Epochs")).toHaveValue(9));
    first.unmount();

    await waitFor(() => expect(stored.config.epochs).toBe(9));

    render(<Shell />);
    await waitFor(() => expect(screen.getByLabelText("Epochs")).toHaveValue(9));
  });

  it("tells the shell when the dataset changes, so the two cannot drift", async () => {
    // Two independent dataset pickers on one screen was the trap under the
    // bug: only one of them was submitted.
    const onDataset = vi.fn();
    render(
      <TrainScreen
        datasets={DATASETS}
        dataset="api"
        project={project()}
        onDataset={onDataset}
        onProject={() => {}}
        onError={() => {}}
      />,
    );
    await waitFor(() => expect(datasetSelect()).toBeInTheDocument());

    fireEvent.change(datasetSelect(), { target: { value: "blizzard-training" } });
    expect(onDataset).toHaveBeenCalledWith("blizzard-training");
  });

  it("follows the shell when the dataset is changed elsewhere", async () => {
    const view = render(
      <TrainScreen
        datasets={DATASETS}
        dataset="api"
        project={project()}
        onDataset={() => {}}
        onProject={() => {}}
        onError={() => {}}
      />,
    );
    await waitFor(() => expect(datasetSelect().value).toBe("api"));

    view.rerender(
      <TrainScreen
        datasets={DATASETS}
        dataset="blizzard-training"
        project={project()}
        onDataset={() => {}}
        onProject={() => {}}
        onError={() => {}}
      />,
    );
    await waitFor(() => expect(datasetSelect().value).toBe("blizzard-training"));
  });
});

describe("what gets submitted", () => {
  it("submits the dataset that is on screen", async () => {
    const submitJob = vi.fn(async (_spec: Record<string, unknown>) => job());
    setup({ submitJob: submitJob as never });

    render(<Shell />);
    await waitFor(() => expect(datasetSelect()).toBeInTheDocument());
    fireEvent.change(datasetSelect(), { target: { value: "blizzard-training" } });
    await waitFor(() => expect(screen.getByText("2,820")).toBeInTheDocument());

    fireEvent.click(screen.getByRole("button", { name: /start training/i }));

    await waitFor(() => expect(submitJob).toHaveBeenCalled());
    const spec = submitJob.mock.calls[0]![0];
    expect(spec.dataset).toBe("blizzard-training");
    expect(spec.steps).toBe(2820);
  });

  it("shows the dataset and its path above the button", async () => {
    // The last line of defence: a typed run name is not evidence of the
    // dataset that was picked, so the commit block restates it.
    setup();
    render(<Shell />);
    await waitFor(() => expect(datasetSelect()).toBeInTheDocument());
    fireEvent.change(datasetSelect(), { target: { value: "blizzard-training" } });

    await waitFor(() => expect(screen.getByText("47 images")).toBeInTheDocument());
    // Twice on purpose: as the dataset row's hint, and again at the point of
    // commitment where it is the thing being confirmed.
    expect(screen.getAllByText("D:/data/blizzard")).toHaveLength(2);
  });
});

describe("the queue", () => {
  /**
   * The form used to lock while any run was going. On a shared node that
   * is the opposite of what a queue is for: it makes the card
   * whoever-got-there-first for the rest of the day.
   */
  it("stays editable while somebody else's run is going", async () => {
    setup({ jobs: async () => jobsResponse([job({ status: "running" })]) });

    render(<Shell />);
    await waitFor(() => expect(tab("Configure")).toBeInTheDocument());
    fireEvent.click(tab("Configure"));

    await waitFor(() => expect(datasetSelect()).toBeEnabled());
    expect(screen.getByLabelText("Epochs")).toBeEnabled();
  });

  it("leaves everything editable when nothing is running", async () => {
    setup();
    render(<Shell />);
    await waitFor(() => expect(datasetSelect()).toBeEnabled());
    expect(screen.getByLabelText("Epochs")).toBeEnabled();
  });

  it("says what pressing the button will actually do", async () => {
    setup({
      jobs: async () =>
        jobsResponse([
          job({ id: "a", status: "queued", project: "alice" }),
          job({ id: "b", status: "queued", project: "tuesday" }),
        ]),
    });

    render(<Shell />);
    await waitFor(() => expect(tab("Configure")).toBeInTheDocument());
    fireEvent.click(tab("Configure"));

    // "Start training" beside a busy node is a promise it cannot keep.
    await waitFor(() =>
      expect(screen.getByRole("button", { name: /add to queue/i })).toBeInTheDocument(),
    );
  });

  it("shows other projects' runs, not only this project's", async () => {
    // The number a student is asking for. A queue filtered to your own
    // runs would say "you are next" while four other people are ahead.
    setup({
      jobs: async () =>
        jobsResponse([
          job({ id: "a", status: "queued", project: "alice" }),
          job({ id: "b", status: "queued", project: "bob" }),
        ]),
    });

    render(<Shell />);
    await waitFor(() => expect(tab("Configure")).toBeInTheDocument());
    fireEvent.click(tab("Configure"));

    await waitFor(() => expect(screen.getByText("alice")).toBeInTheDocument());
    expect(screen.getByText("bob")).toBeInTheDocument();
  });
});

describe("the shared config", () => {
  it("submits the run under the open project", async () => {
    const submitted: Record<string, unknown>[] = [];
    setup({
      submitJob: async (spec: Record<string, unknown>) => {
        submitted.push(spec);
        return { ...job(), id: "new-job" };
      },
      saveProjectConfig: async () => project(),
    });

    render(<Shell />);
    await waitFor(() => expect(datasetSelect()).toBeEnabled());
    await waitFor(() =>
      expect(screen.getByRole("button", { name: /start training/i })).toBeEnabled(),
    );
    fireEvent.click(screen.getByRole("button", { name: /start training/i }));

    await waitFor(() => expect(submitted).toHaveLength(1));
    expect(submitted[0]!.project).toBe("tuesday");
  });

  it("opens on the settings the project already holds", async () => {
    setup();
    const open = project("tuesday", { config: { epochs: 11 } });
    render(
      <TrainScreen
        datasets={DATASETS}
        dataset="api"
        project={open}
        onDataset={() => {}}
        onProject={() => {}}
        onError={() => {}}
      />,
    );

    await waitFor(() =>
      expect(screen.getByLabelText("Epochs")).toHaveValue(11),
    );
  });
});
