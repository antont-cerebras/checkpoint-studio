// How far the read the server is doing *for us* has got.
//
// `POST /api/open` and `POST /api/compare` are synchronous: the answer means "ready", which is what
// spares this app a state machine over a half-loaded checkpoint. The cost is that the request itself
// carries no news, so a wait showed an elapsed timer and nothing else — while a terminal reading the
// same checkpoint counted `1155/1155 S3 objects · reading S3 storage metadata`. Those numbers existed
// the whole time, one layer down, being animated onto the server's own log.
//
// So the numbers get their own tiny endpoint and this polls it. Polling rather than streaming for the
// same reason the jobs store polls (see `stores/jobs`): one small request, no connection to keep alive,
// and a reload picks the read back up.

import { writable } from 'svelte/store';
import { api } from '../lib/api';

/** One checkpoint's share of a read. Mirrors `web::current::SideProgress`. */
export interface SideProgress {
  /** Which checkpoint this row is about. */
  spec: string;
  done: number;
  /** 0 until the reader knows a denominator: a spinner, not a bar at zero. */
  total: number;
  /** `shards`, `S3 objects`, `tensors` — what the count counts. Empty until the reader says. */
  unit: string;
  /** `reading S3 storage metadata` — which step is running, or null before the reader says. */
  stage: string | null;
  /** This one has landed while the other is still going — which the counters cannot say, since a
   * reader that never learned a total finishes at `0/0` like one that has not started. */
  finished: boolean;
}

/** What `GET /api/reading` reports. Mirrors `web::current::ReadingProgress`. */
export interface ReadingProgress {
  seconds: number;
  /** One entry per checkpoint being read: one for an open, **two** while a comparison sets up, since
   * both of its sides are read at the same time. Baseline first. */
  sides: SideProgress[];
}

/** The read in flight, or null when the server is idle. */
export const reading = writable<ReadingProgress | null>(null);

/** Half the jobs store's interval: this is a smaller response, and it backs a bar people watch. */
const POLL_MS = 400;

let timer: ReturnType<typeof setInterval> | undefined;
/** How many screens are watching. Reference-counted so two waits on screen at once (a compare inside a
 * freshly opened checkpoint) do not each start and stop the same timer. */
let watchers = 0;

function poll(): void {
  void api
    .reading()
    .then((r) => reading.set(r.reading))
    // A failed poll is not worth reporting: the request it accompanies will report for itself, and an
    // error here would replace real progress with a complaint about the progress channel.
    .catch(() => reading.set(null));
}

/**
 * Start watching, and return the function that stops.
 *
 * Shaped for `onMount(() => watchReading())`: Svelte calls the returned function on destroy, so the
 * poll's lifetime is exactly the lifetime of the screen showing it.
 */
export function watchReading(): () => void {
  watchers += 1;
  if (timer === undefined) {
    poll(); // straight away, so the first numbers do not wait out an interval
    timer = setInterval(poll, POLL_MS);
  }
  return () => {
    watchers = Math.max(0, watchers - 1);
    if (watchers === 0 && timer !== undefined) {
      clearInterval(timer);
      timer = undefined;
      reading.set(null);
    }
  };
}
