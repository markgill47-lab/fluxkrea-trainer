/**
 * The project gate, and the ellipse geometry the Masks screen rests on.
 *
 * The gate is the first thing a student sees on a shared node, and the
 * thing it exists to prevent is invisible: without it the app opens on
 * whatever dataset is first in the node's registry — somebody else's — and
 * the first action anybody takes is a batch operation over it.
 */

import { fireEvent, render, screen, waitFor } from "@testing-library/preact";
import { describe, expect, it, vi } from "vitest";
import type { Box } from "~/api/types";
import { contains, isEllipse, shapeOf } from "~/lib/boxes";
import { ProjectGate, rememberProject, storedProject } from "~/projects/ProjectGate";
import { fakeApi, project } from "./fixtures";

describe("the project gate", () => {
  it("asks for a name when the node has no projects", () => {
    fakeApi({});
    render(
      <ProjectGate projects={[]} onOpen={() => {}} onCreated={() => {}} onError={() => {}} />,
    );

    expect(screen.getByText("Create a project")).toBeInTheDocument();
    expect(screen.getByLabelText("New project name")).toBeInTheDocument();
  });

  it("lists what is already there, and offers a new one alongside", () => {
    fakeApi({});
    render(
      <ProjectGate
        projects={[project("tuesday", { name: "Tuesday", datasets: ["poses", "punches"] })]}
        onOpen={() => {}}
        onCreated={() => {}}
        onError={() => {}}
      />,
    );

    expect(screen.getByText("Open a project")).toBeInTheDocument();
    expect(screen.getByText("Tuesday")).toBeInTheDocument();
    expect(screen.getByText("2 datasets")).toBeInTheDocument();
  });

  it("opens the project that was clicked", () => {
    fakeApi({});
    const onOpen = vi.fn();
    render(
      <ProjectGate
        projects={[project("tuesday", { name: "Tuesday" })]}
        onOpen={onOpen}
        onCreated={() => {}}
        onError={() => {}}
      />,
    );

    fireEvent.click(screen.getByText("Tuesday"));
    expect(onOpen).toHaveBeenCalledWith("tuesday");
  });

  it("creating a project opens it, so nobody lands back on this screen", async () => {
    const onOpen = vi.fn();
    const onCreated = vi.fn();
    fakeApi({ createProject: (async () => project("fight-choreo")) as never });

    render(
      <ProjectGate
        projects={[]}
        onOpen={onOpen}
        onCreated={onCreated}
        onError={() => {}}
      />,
    );

    fireEvent.input(screen.getByLabelText("New project name"), {
      target: { value: "Fight Choreo" },
    });
    fireEvent.click(screen.getByRole("button", { name: "New project" }));

    await waitFor(() => expect(onOpen).toHaveBeenCalledWith("fight-choreo"));
    // Reloaded before opening, or the shell would not have the new project
    // in its list and would bounce straight back to the gate.
    expect(onCreated).toHaveBeenCalled();
  });

  it("refuses to create a project with no name", () => {
    const createProject = vi.fn();
    fakeApi({ createProject: createProject as never });
    render(
      <ProjectGate projects={[]} onOpen={() => {}} onCreated={() => {}} onError={() => {}} />,
    );

    fireEvent.input(screen.getByLabelText("New project name"), { target: { value: "   " } });
    expect(screen.getByRole("button", { name: "New project" })).toBeDisabled();
    expect(createProject).not.toHaveBeenCalled();
  });

  it("says so when the project this browser had open has gone", () => {
    fakeApi({});
    render(
      <ProjectGate
        projects={[]}
        notice="The project you had open (tuesday) is no longer on this node."
        onOpen={() => {}}
        onCreated={() => {}}
        onError={() => {}}
      />,
    );

    expect(screen.getByText(/no longer on this node/)).toBeInTheDocument();
  });
});

describe("the remembered project", () => {
  it("round-trips through this browser's storage", () => {
    rememberProject("tuesday");
    expect(storedProject()).toBe("tuesday");
    rememberProject(null);
    expect(storedProject()).toBeNull();
  });
});

describe("ellipse regions", () => {
  const box = (extra: Partial<Box> = {}): Box => ({
    x: 100,
    y: 100,
    w: 100,
    h: 100,
    src: "yunet",
    ...extra,
  });

  it("a box with no shape is a rectangle", () => {
    // Every box file written before shapes existed holds rectangles, and
    // the daemon omits the field for them.
    expect(shapeOf(box())).toBe("rect");
    expect(isEllipse(box())).toBe(false);
    expect(isEllipse(box({ shape: "ellipse" }))).toBe(true);
  });

  it("hit-tests the shape, not the bounding box", () => {
    // Two ellipses overlapping at the corners is the ordinary case in a
    // group shot; testing bounding boxes there selects the wrong face.
    const ellipse = box({ shape: "ellipse" });
    expect(contains(ellipse, { x: 150, y: 150 })).toBe(true); // centre
    expect(contains(ellipse, { x: 101, y: 101 })).toBe(false); // corner
    expect(contains(box(), { x: 101, y: 101 })).toBe(true); // …but a rect owns it
  });

  it("a degenerate ellipse contains nothing rather than throwing", () => {
    expect(contains(box({ shape: "ellipse", w: 0, h: 0 }), { x: 100, y: 100 })).toBe(false);
  });
});
