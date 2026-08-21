/**
 * The SSE client.
 *
 * Doc 10: "automatic reconnect, exponential backoff, and Last-Event-ID
 * backfill so a laptop waking from sleep does not lose the middle of a
 * run." That is the whole specification, and every part of it earns its
 * place over an SSH tunnel that will drop.
 *
 * `EventSource` does reconnection and `Last-Event-ID` natively, which is
 * most of why doc 06 chose SSE over WebSockets. What it does *not* do is
 * back off — it retries at a fixed interval forever — so a dropped tunnel
 * becomes a request every three seconds until the laptop is closed. The
 * backoff here is the part worth writing.
 */

import type { StreamEvent } from "./types";

export type ConnectionState = "connecting" | "open" | "retrying" | "closed";

export interface StreamOptions {
  /** Resume from this event index. Events are indexed server-side. */
  since?: number;
  onEvent(event: StreamEvent): void;
  onState?(state: ConnectionState): void;
  /**
   * Stop rather than reconnect once this returns true.
   *
   * The daemon closes the stream when a job or task finishes, and
   * `EventSource` cannot tell a clean close from a dropped tunnel — it
   * retries either way. Without this, a completed run reconnects forever
   * and reports itself as "retrying" for as long as the screen is open.
   */
  isComplete?(): boolean;
}

/**
 * The named events the daemon sends.
 *
 * `sse-starlette` sets the event name from the payload's `kind`, so the
 * default `message` handler above never fires for them. Registering each
 * name explicitly is the price of that, and the alternative — a generic
 * event name — would make the stream unreadable with `curl`, which doc 06
 * specifically wanted to keep.
 */
export const EVENT_KINDS = ["progress", "log", "loss", "finished", "event"] as const;

/** Attach the same handler to every named event on a source. */
export function bindKinds(source: EventSource, handler: (event: StreamEvent) => void): void {
  for (const kind of EVENT_KINDS) {
    source.addEventListener(kind, (message) => {
      try {
        handler(JSON.parse((message as MessageEvent).data) as StreamEvent);
      } catch {
        /* ignore a malformed frame */
      }
    });
  }
}

const INITIAL_DELAY = 500;
const MAX_DELAY = 30_000;

/**
 * Follow an event stream until closed.
 *
 * Returns a stop function. The caller owns the lifetime — a screen that
 * unmounts must stop its stream, or a review pass through five jobs leaves
 * five streams running.
 */
export function subscribe(path: string, options: StreamOptions): () => void {
  let source: EventSource | null = null;
  let timer: number | undefined;
  let delay = INITIAL_DELAY;
  let lastIndex = options.since ?? -1;
  let stopped = false;

  const setState = (state: ConnectionState) => options.onState?.(state);

  const connect = () => {
    if (stopped) return;
    setState(lastIndex >= 0 ? "retrying" : "connecting");

    const url = new URL(path, window.location.href);
    // Resume by index rather than relying on Last-Event-ID alone: a fresh
    // EventSource after a long sleep may not send the header, and the gap
    // is exactly what matters.
    if (lastIndex >= 0) url.searchParams.set("since", String(lastIndex));

    source = new EventSource(url);

    source.onopen = () => {
      delay = INITIAL_DELAY;
      setState("open");
    };

    // sse-starlette names each frame after the event's `kind`, so the
    // default `message` handler never fires. Binding each name is the
    // price of a stream that stays readable with curl, which is one of the
    // reasons doc 06 chose SSE in the first place.
    bindKinds(source, (payload) => {
      if (typeof payload.index === "number") lastIndex = payload.index;
      options.onEvent(payload);
    });

    source.onerror = () => {
      source?.close();
      source = null;
      if (stopped) return;
      if (options.isComplete?.()) {
        stopped = true;
        setState("closed");
        return;
      }
      setState("retrying");
      timer = window.setTimeout(connect, delay);
      // Exponential with a ceiling: a node that is down for an hour should
      // be polled every 30s, not 1,200 times.
      delay = Math.min(delay * 2, MAX_DELAY);
    };
  };

  connect();

  return () => {
    stopped = true;
    if (timer) clearTimeout(timer);
    source?.close();
    source = null;
    setState("closed");
  };
}
