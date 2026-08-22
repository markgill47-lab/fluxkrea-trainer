/**
 * The dataset picker and the resize dialog.
 *
 * Both are one click away from something irreversible — forgetting a
 * dataset, rewriting every file in a folder — so what is tested here is
 * what they say before they do it, and that they call what they claim to.
 */

import { fireEvent, render, screen, waitFor } from "@testing-library/preact";
import { describe, expect, it, vi } from "vitest";
import { api } from "~/api/client";
import { DatasetPicker } from "~/datasets/DatasetPicker";
import { ResizeDialog, SIZES } from "~/gallery/ResizeDialog";
import { dataset, fakeApi, item } from "./fixtures";

// --------------------------------------------------------------------------
// the picker
// --------------------------------------------------------------------------

const BROWSE = {
  path: "D:/data",
  parent: null,
  roots: ["D:/data"],
  entries: [
    { path: "D:/data/poses", name: "poses", images: 42, has_masks: true, dataset_id: null },
    { path: "D:/data/api", name: "api", images: 6, has_masks: false, dataset_id: "api" },
    { path: "D:/data/empty", name: "empty", images: 0, has_masks: false, dataset_id: null },
  ],
};

describe("the dataset picker", () => {
  it("registers the folder that was clicked", async () => {
    const registerDataset = vi.fn(async () => dataset("poses"));
    fakeApi({ browse: async () => BROWSE, registerDataset: registerDataset as never });

    render(<DatasetPicker datasets={[]} onClose={() => {}} onChanged={() => {}}
          onRemove={async () => {}} />);
    await waitFor(() => expect(screen.getByText("poses")).toBeInTheDocument());

    fireEvent.click(screen.getByTitle("Register this folder as a dataset"));
    await waitFor(() => expect(registerDataset).toHaveBeenCalledWith("D:/data/poses"));
  });

  it("will not register a folder with no images in it", async () => {
    fakeApi({ browse: async () => BROWSE });
    render(<DatasetPicker datasets={[]} onClose={() => {}} onChanged={() => {}}
          onRemove={async () => {}} />);
    await waitFor(() => expect(screen.getByText("empty")).toBeInTheDocument());

    expect(screen.getByTitle(/No images in this folder/)).toBeDisabled();
  });

  it("shows an already-registered folder as registered rather than offering it again", async () => {
    fakeApi({ browse: async () => BROWSE });
    render(<DatasetPicker datasets={[]} onClose={() => {}} onChanged={() => {}}
          onRemove={async () => {}} />);

    await waitFor(() => expect(screen.getByText("registered as api")).toBeInTheDocument());
  });

  it("removes a dataset from the project without touching the folder", async () => {
    // Out of the project, *not* off the node: on a shared daemon the same
    // folder may be in somebody else's project, and a button in a browser
    // must not be able to take it from them.
    const onRemove = vi.fn(async () => {});
    const forgetDataset = vi.fn(async () => ({ id: "poses", forgotten: true }));
    fakeApi({ browse: async () => BROWSE, forgetDataset: forgetDataset as never });

    render(
      <DatasetPicker
        datasets={[dataset("poses")]}
        onClose={() => {}}
        onChanged={() => {}}
        onRemove={onRemove}
      />,
    );
    await waitFor(() => expect(screen.getByText("Remove")).toBeInTheDocument());

    // The wording is the guarantee, and it is worth pinning: a destructive
    // -sounding button in a browser must not be ambiguous about scope.
    const remove = screen.getByTitle(/folder and its files are left alone/);
    fireEvent.click(remove);
    await waitFor(() => expect(onRemove).toHaveBeenCalledWith("poses"));
    expect(forgetDataset).not.toHaveBeenCalled();
  });

  it("tells the shell to reload after a change", async () => {
    const onChanged = vi.fn();
    fakeApi({
      browse: async () => BROWSE,
      registerDataset: (async () => dataset("poses")) as never,
    });

    render(
      <DatasetPicker
        datasets={[]}
        onClose={() => {}}
        onChanged={onChanged}
        onRemove={async () => {}}
      />,
    );
    await waitFor(() => expect(screen.getByText("poses")).toBeInTheDocument());
    fireEvent.click(screen.getByTitle("Register this folder as a dataset"));

    await waitFor(() => expect(onChanged).toHaveBeenCalled());
  });

  it("closes on Escape", async () => {
    const onClose = vi.fn();
    fakeApi({ browse: async () => BROWSE });
    render(<DatasetPicker datasets={[]} onClose={onClose} onChanged={() => {}}
          onRemove={async () => {}} />);

    fireEvent.keyDown(window, { key: "Escape" });
    expect(onClose).toHaveBeenCalled();
  });

  it("reports a failed browse instead of showing an empty folder list", async () => {
    fakeApi({
      browse: (async () => {
        throw new Error("cannot read D:/data");
      }) as never,
    });

    render(<DatasetPicker datasets={[]} onClose={() => {}} onChanged={() => {}}
          onRemove={async () => {}} />);
    await waitFor(() => expect(screen.getByText(/cannot read/)).toBeInTheDocument());
  });
});

// --------------------------------------------------------------------------
// the resize dialog
// --------------------------------------------------------------------------

const LANDSCAPE = item("a", 2000, 1500); // longest 2000
const SMALL = item("b", 800, 600); //        longest 800
const EXACT = item("c", 1024, 700); //       longest 1024

describe("the resize dialog", () => {
  const show = (items = [LANDSCAPE, SMALL, EXACT], onRun = vi.fn()) => {
    render(<ResizeDialog items={items} running={false} onRun={onRun} onClose={() => {}} />);
    return onRun;
  };

  it("offers only the sizes this lab trains at", () => {
    expect([...SIZES]).toEqual([1024, 2048]);
    show();
    expect(screen.queryByRole("radio", { name: /512/ })).not.toBeInTheDocument();
  });

  it("counts what will shrink, grow and stay put at the chosen size", () => {
    show();
    // Defaults to 1024: one above it, one below, one already there.
    expect(screen.getByText("shrink to 1024px")).toBeInTheDocument();
    expect(screen.getByText("enlarge to 1024px")).toBeInTheDocument();
    expect(screen.getByText("already correct")).toBeInTheDocument();
  });

  it("recounts when the size changes", () => {
    show();
    fireEvent.click(screen.getByRole("radio", { name: /2048/ }));

    // At 2048 nothing shrinks; all three grow.
    expect(screen.queryByText("shrink to 2048px")).not.toBeInTheDocument();
    expect(screen.getByText("enlarge to 2048px")).toBeInTheDocument();
  });

  it("reports images it will not enlarge rather than counting them as work", () => {
    show();
    fireEvent.click(screen.getByRole("checkbox"));

    expect(screen.getByText("left alone (smaller than the target)")).toBeInTheDocument();
    expect(screen.getByText(/1 of 3 files will be rewritten/)).toBeInTheDocument();
  });

  it("refuses to run when nothing would change", () => {
    show([EXACT]);
    expect(screen.getByText("nothing to do at this size")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /resize to 1024px/i })).toBeDisabled();
  });

  it("names the masks that will move with the images", () => {
    show([item("a", 2000, 1500, { has_mask: true })]);
    expect(screen.getByText("masks resized with them, NEAREST")).toBeInTheDocument();
  });

  it("passes the size and the upscale choice through", () => {
    const onRun = show();
    fireEvent.click(screen.getByRole("radio", { name: /2048/ }));
    fireEvent.click(screen.getByRole("button", { name: /resize to 2048px/i }));

    expect(onRun).toHaveBeenCalledWith(2048, true);
  });

  it("counts an item whose size is not cached as work rather than skipping it", async () => {
    // The operation reads the file, so an unknown size is not "no change".
    show([item("d", null as unknown as number, null as unknown as number)]);
    expect(screen.getByText(/size not cached yet/)).toBeInTheDocument();
    expect(screen.getByText(/1 of 1 files will be rewritten/)).toBeInTheDocument();
  });
});
