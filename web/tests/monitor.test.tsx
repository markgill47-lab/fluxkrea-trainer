/**
 * The stat tiles — what they claim, and what they must not.
 *
 * A tile is read at a glance and believed. "0 outliers" is a statement
 * about the images in a dataset, and it needs the backend to have said
 * which image each step used. ai-toolkit does not, so that tile spent a
 * 4,440-step run confidently reporting a number it had no data for.
 */

import { render, screen } from "@testing-library/preact";
import { describe, expect, it } from "vitest";
import type { Job, LossPayload } from "~/api/types";
import { StatTiles } from "~/monitor/StatTiles";
import { job } from "./fixtures";

function loss(overrides: Partial<LossPayload> = {}): LossPayload {
  return {
    id: "j",
    points: [
      { step: 1, value: 0.5 },
      { step: 2, value: 0.4 },
    ],
    ema: [{ step: 2, value: 0.45 }],
    ema_window: 50,
    count: 2,
    decimated: false,
    latest: 0.4,
    latest_ema: 0.45,
    trend: { status: "improving", slope: -0.01, window: 200 },
    attributed: false,
    outliers: [],
    ...overrides,
  };
}

const running: Job = job({ status: "running", progress: { step: 2, total: 100 } });

describe("the outliers tile", () => {
  it("declines to report a count when the trainer names no images", () => {
    render(<StatTiles job={running} loss={loss()} />);

    expect(screen.getByText("no per-image loss from this trainer")).toBeInTheDocument();
    // Not "0" - that is a claim about the dataset, from data that cannot
    // make it.
    expect(screen.queryByText("0")).toBeNull();
  });

  it("reports a real zero when the trainer does name them", () => {
    render(
      <StatTiles job={running} loss={loss({ attributed: true })} />,
    );

    expect(screen.getByText("0")).toBeInTheDocument();
    expect(screen.queryByText("no per-image loss from this trainer")).toBeNull();
  });

  it("names the worst image when there is one", () => {
    const payload = loss({
      attributed: true,
      outliers: [{ image_id: "pose_007.jpg", mean: 0.9, severity: 2.1, samples: 12 }],
    });
    render(<StatTiles job={running} loss={payload} />);

    expect(screen.getByText("1")).toBeInTheDocument();
    expect(screen.getByText("pose_007.jpg")).toBeInTheDocument();
  });
});

describe("the trend tile", () => {
  it("shows the direction beside the EMA", () => {
    render(<StatTiles job={running} loss={loss()} />);
    expect(screen.getByText(/improving/)).toBeInTheDocument();
  });

  it("warns only when the loss is going the wrong way", () => {
    const payload = loss({ trend: { status: "degrading", slope: 0.01, window: 200 } });
    const { container } = render(
      <StatTiles job={running} loss={payload} />,
    );
    expect(container.querySelector(".tile--warn")).not.toBeNull();
  });
});
