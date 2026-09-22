// web/src/emulator/EmulatorProvider.tsx
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { createBridge, type Debugger, type WasmModule } from "../wasm/bridge";
import { loadEmulatorModule } from "../wasm/loader";
import { parseOpcodes } from "../wasm/opcodes";
import type { EmulatorSnapshot, EmulatorStatus } from "../wasm/types";
import { useFrameLoop } from "./useFrameLoop";
import { NesAudio } from "./audio";
import { parseMovie } from "./movie";
import { DEFAULT_GAME, romUrl } from "../games/catalog";
import { useToast } from "../components/toast/ToastProvider";

// The live-agent server (python -m nesenv.live) streams Server-Sent Events to here.
// This is a local dev / research feature: in production the default points at the
// visitor's own machine (and an HTTPS page blocks plain-http localhost anyway), so
// the Spawn Agent button is hidden unless VITE_LIVE_AGENT_URL is set (see Toolbar).
const LIVE_AGENT_URL =
  (import.meta.env.VITE_LIVE_AGENT_URL as string | undefined) ??
  "http://localhost:8000/stream";

function base64ToBytes(b64: string): Uint8Array {
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i += 1) out[i] = bin.charCodeAt(i);
  return out;
}

export interface EmulatorActions {
  step(): void;
  run(): void;
  stop(): void;
  reset(): void;
  addBreakpoint(addr: number): void;
  removeBreakpoint(addr: number): void;
  toggleBreakpoint(addr: number): void;
  writeMemory(addr: number, value: number): void;
  loadROM(data: Uint8Array): void;
  loadRom(data: Uint8Array): number;
  loadOpcodes(text: string): void;
  setController(state: number, port?: number): void;
  playMovie(data: Uint8Array): void;
  stopMovie(): void;
  connectLiveAgent(url?: string): void;
  disconnectLiveAgent(): void;
}

/** One decision streamed by a narrating live agent (see python/nesenv/live.py). */
export interface AgentDecision {
  n: number;
  frame: number;
  x: number;
  summary: string;
  move: string;
  confidence: number;
  source: string; // "jev" | "jev-fallback" | "jev-cached" | "code" | "heuristic"
  latency_ms: number;
  /** The part of latency_ms the model itself took; the rest is network. */
  server_ms?: number;
  input_tokens: number;
  probabilities?: Record<string, number>;
  obstacle_needs_jump?: number;
  enemy_needs_jump?: number;
  /** What the harness actually did: the move, or "collect <item>" for a chosen target. */
  action?: string;
  target?: string;
  target_probabilities?: Record<string, number>;
  target_confidence?: number;
  hazard_first?: number;
  world?: number;
  stage?: number;
  coins?: number;
  lives?: number;
}

export interface LifeOutcome {
  outcome: string;
  distance: number;
  world?: number;
  stage?: number;
  coins?: number;
}

export interface AgentRun {
  policy: string | null;
  model: string | null;
  moves: string[];
  pricePerMtok: number;
  /** Most recent decisions, oldest first (capped at AGENT_HISTORY). */
  decisions: AgentDecision[];
  /** Decisions that cost an API round trip (the rest were cached or decided in code). */
  calls: number;
  /** Every decision, including the free ones. */
  decided: number;
  fallbacks: number;
  inputTokens: number;
  latencySumMs: number;
  lives: number;
  lastOutcome: LifeOutcome | null;
  /** Every life / level outcome so far, oldest first. */
  outcomes: LifeOutcome[];
}

export const AGENT_HISTORY = 120;
/** Frames held back before live playback starts, and before it resumes after running dry. */
export const LIVE_BUFFER_START = 180;
export const LIVE_BUFFER_RESUME = 60;
export const LIVE_BUFFER_CATCH_UP = 300;
export const LIVE_MIN_FPS = 12;
export const LIVE_RATE_WINDOW_MS = 4000;

/** True when this decision was a real policy call, not a cache hit or a code shortcut. */
export function isModelCall(d: AgentDecision): boolean {
  return d.source !== "jev-cached" && d.source !== "code";
}

export const EMPTY_AGENT_RUN: AgentRun = {
  policy: null,
  model: null,
  moves: [],
  pricePerMtok: 0,
  decisions: [],
  calls: 0,
  decided: 0,
  fallbacks: 0,
  inputTokens: 0,
  latencySumMs: 0,
  lives: 0,
  lastOutcome: null,
  outcomes: [],
};

export interface EmulatorContextValue {
  status: EmulatorStatus;
  snapshot: EmulatorSnapshot | null;
  breakpoints: number[];
  running: boolean;
  framebuffer: Uint8ClampedArray | null;
  movie: { playing: boolean; frame: number; total: number };
  /** `buffered` is how many streamed frames wait in the jitter buffer (about 60 per second of play). */
  liveAgent: { connected: boolean; frame: number; buffered: number; fps: number };
  agentRun: AgentRun;
  dbg: Debugger | null;
  actions: EmulatorActions;
}

const EmulatorContext = createContext<EmulatorContextValue | null>(null);

export function EmulatorProvider(props: {
  children: ReactNode;
  loadModule?: () => Promise<WasmModule>;
}): JSX.Element {
  const { loadModule = loadEmulatorModule } = props;
  const { addToast } = useToast();
  const dbgRef = useRef<Debugger | null>(null);
  const [dbg, setDbg] = useState<Debugger | null>(null);
  const [status, setStatus] = useState<EmulatorStatus>("loading");
  const [snapshot, setSnapshot] = useState<EmulatorSnapshot | null>(null);
  const [breakpoints, setBreakpoints] = useState<number[]>([]);
  const [framebuffer, setFramebuffer] = useState<Uint8ClampedArray | null>(
    null,
  );
  const [movie, setMovie] = useState({ playing: false, frame: 0, total: 0 });
  const movieRafRef = useRef<number | null>(null);
  const [liveAgent, setLiveAgent] = useState({ connected: false, frame: 0, buffered: 0, fps: 0 });
  const liveRafRef = useRef<number | null>(null);
  const [agentRun, setAgentRun] = useState<AgentRun>(EMPTY_AGENT_RUN);
  const liveSourceRef = useRef<EventSource | null>(null);
  const liveRomRef = useRef<Uint8Array | null>(null);
  // Mirror the breakpoints state in a ref so toggleBreakpoint can read the
  // current set synchronously without doing side-effects inside a setState
  // updater (updaters must stay pure).
  const breakpointsRef = useRef<number[]>([]);
  useEffect(() => {
    breakpointsRef.current = breakpoints;
  }, [breakpoints]);

  const publishSnapshot = useCallback((s: EmulatorSnapshot) => {
    setSnapshot(s);
  }, []);

  const handleBreak = useCallback(() => {
    addToast("Breakpoint hit", "warning");
  }, [addToast]);

  const handleBrk = useCallback(() => {
    addToast("Program terminated with BRK", "info");
  }, [addToast]);

  const audioRef = useRef(new NesAudio());

  const handleFrame = useCallback((fb: Uint8ClampedArray) => {
    // Copy out of the WASM heap view so React state holds a stable buffer
    // (the heap view is reused/invalidated by the next frame).
    setFramebuffer(new Uint8ClampedArray(fb));
    // Drain this frame's audio and queue it for playback.
    const bridge = dbgRef.current;
    if (bridge) {
      const samples = bridge.audioDrain(4096);
      if (samples.length) audioRef.current.pump(samples);
    }
  }, []);

  // Release the audio context when the provider unmounts.
  useEffect(() => {
    const audio = audioRef.current;
    return () => audio.close();
  }, []);

  const {
    start: startLoop,
    stop: stopLoop,
    running,
  } = useFrameLoop({
    dbg,
    onFrame: handleFrame,
    onSnapshot: publishSnapshot,
    onBreak: handleBreak,
    onBrk: handleBrk,
  });

  useEffect(() => {
    let cancelled = false;
    loadModule()
      .then((module) => {
        if (cancelled) return;
        const bridge = createBridge(module);
        dbgRef.current = bridge;
        setDbg(bridge);
        setSnapshot(bridge.getSnapshot());
        // Publish the framebuffer once at load so the Screen shows the PPU's
        // current output (or the DEV demo image) at idle — the run loop only
        // calls onFrame while running, so without this the canvas stays black
        // until the user hits Run. Copy out of the heap view (it's reused).
        try {
          const fb = bridge.getFramebuffer();
          setFramebuffer(new Uint8ClampedArray(fb));
        } catch {
          /* no framebuffer export (stale build) — leave canvas blank */
        }
        setStatus("ready");
        // Boot into a bundled, freely-licensed game so a fresh visitor lands on
        // something playable instead of a blank debugger. Best-effort: a failed
        // fetch (offline, missing asset, no fetch in the env) leaves it idle.
        if (typeof fetch !== "function") return;
        fetch(romUrl(DEFAULT_GAME))
          .then((res) => (res.ok ? res.arrayBuffer() : null))
          .then((buf) => {
            if (cancelled || !buf) return;
            if (bridge.loadRom(new Uint8Array(buf)) === 0) {
              setSnapshot(bridge.getSnapshot());
              try {
                setFramebuffer(new Uint8ClampedArray(bridge.getFramebuffer()));
              } catch {
                /* stale build without framebuffer export */
              }
              addToast(
                `${DEFAULT_GAME.title} loaded — press ▶ Run to play`,
                "success",
              );
            }
          })
          .catch(() => {
            /* offline or asset missing — stay idle */
          });
      })
      .catch(() => {
        if (cancelled) return;
        dbgRef.current = null;
        setDbg(null);
        setStatus("error");
      });
    return () => {
      cancelled = true;
    };
  }, [loadModule]);

  const refresh = useCallback(() => {
    const bridge = dbgRef.current;
    if (!bridge) return;
    setSnapshot(bridge.getSnapshot());
  }, []);

  const step = useCallback(() => {
    const bridge = dbgRef.current;
    if (!bridge) return;
    bridge.step();
    refresh();
  }, [refresh]);

  const run = useCallback(() => {
    const bridge = dbgRef.current;
    if (!bridge) return;
    bridge.run();
    audioRef.current.resume();  // user-gesture-initiated; unlocks WebAudio
    addToast("Execution started", "info");
    startLoop();
  }, [addToast, startLoop]);

  const stopMovie = useCallback(() => {
    if (movieRafRef.current !== null) {
      cancelAnimationFrame(movieRafRef.current);
      movieRafRef.current = null;
    }
    setMovie((m) => (m.playing ? { ...m, playing: false } : m));
  }, []);

  const disconnectLiveAgent = useCallback(() => {
    if (liveSourceRef.current) {
      liveSourceRef.current.close();
      liveSourceRef.current = null;
    }
    if (liveRafRef.current !== null) {
      cancelAnimationFrame(liveRafRef.current);
      liveRafRef.current = null;
    }
    setLiveAgent((s) => (s.connected ? { connected: false, frame: 0, buffered: 0, fps: 0 } : s));
  }, []);

  const stop = useCallback(() => {
    const bridge = dbgRef.current;
    if (!bridge) return;
    stopMovie();
    disconnectLiveAgent();
    stopLoop();
    bridge.stop();
    audioRef.current.suspend();
    refresh();
  }, [refresh, stopLoop, stopMovie, disconnectLiveAgent]);

  const reset = useCallback(() => {
    const bridge = dbgRef.current;
    if (!bridge) return;
    stopMovie();
    stopLoop();
    bridge.reset();
    refresh();
  }, [refresh, stopLoop, stopMovie]);

  // Replay a self-contained .nesmovie (ROM + one controller byte per frame).
  // Because the core is deterministic, this reproduces exactly what the agent
  // (or any recorder) did when the movie was captured.
  const playMovie = useCallback(
    (bytes: Uint8Array) => {
      const bridge = dbgRef.current;
      if (!bridge) return;
      stopLoop();
      if (movieRafRef.current !== null) {
        cancelAnimationFrame(movieRafRef.current);
        movieRafRef.current = null;
      }
      let parsed;
      try {
        parsed = parseMovie(bytes);
      } catch (e) {
        addToast(`Invalid movie: ${(e as Error).message}`, "danger");
        return;
      }
      if (bridge.loadRom(parsed.rom) !== 0) {
        addToast("Movie ROM was rejected", "danger");
        return;
      }
      audioRef.current.resume(); // user-gesture-initiated; unlocks WebAudio
      const inputs = parsed.inputs;
      setMovie({ playing: true, frame: 0, total: inputs.length });
      addToast(`Playing movie — ${inputs.length} frames`, "info");

      let i = 0;
      const tick = () => {
        const b = dbgRef.current;
        if (!b) {
          movieRafRef.current = null;
          setMovie((m) => ({ ...m, playing: false }));
          return;
        }
        if (i >= inputs.length) {
          movieRafRef.current = null;
          setMovie((m) => ({ ...m, playing: false }));
          setSnapshot(b.getSnapshot());
          addToast("Movie finished", "info");
          return;
        }
        b.setController(inputs[i]);
        b.runFrame();
        setFramebuffer(new Uint8ClampedArray(b.getFramebuffer()));
        const samples = b.audioDrain(4096);
        if (samples.length) audioRef.current.pump(samples);
        i += 1;
        if (i % 6 === 0) setMovie((m) => ({ ...m, frame: i }));
        movieRafRef.current = requestAnimationFrame(tick);
      };
      movieRafRef.current = requestAnimationFrame(tick);
    },
    [addToast, stopLoop],
  );

  // Watch a live agent: connect to the SSE stream and re-simulate the streamed
  // actions on the local core. Because the core is deterministic, the browser
  // reproduces exactly what the agent is doing on the server, frame for frame.
  //
  // The server pauses for every model call, so its frames arrive in bursts. They
  // are queued here and played back at a steady 60 fps from behind a small buffer
  // (a jitter buffer): a few seconds of delay, no stutter. Agent events ride the
  // same queue so a decision shows up exactly when its frame plays.
  const connectLiveAgent = useCallback(
    (url: string = LIVE_AGENT_URL) => {
      const bridge = dbgRef.current;
      if (!bridge) return;
      stopLoop();
      stopMovie();
      disconnectLiveAgent();
      audioRef.current.resume();

      type Item =
        | { t: "step"; mask: number }
        | { t: "reset" }
        | { t: "agent"; info: Record<string, unknown> }
        | { t: "decision"; d: AgentDecision }
        | { t: "outcome"; o: LifeOutcome };
      const queue: Item[] = [];
      let queuedSteps = 0;
      let played = 0;
      let buffering = true;
      let last = performance.now();
      let carry = 0;

      const es = new EventSource(url);
      liveSourceRef.current = es;
      setAgentRun(EMPTY_AGENT_RUN);

      // Arrival times of recent frames: the server produces fewer than 60 a second
      // while it waits on the model, and playback follows that rate so the game runs
      // steadily (in slow motion when it must) instead of stalling at every call.
      const arrivals: number[] = [];
      const push = (item: Item) => {
        queue.push(item);
        if (item.t === "step") {
          queuedSteps += 1;
          arrivals.push(performance.now());
        }
      };
      const arrivalRate = (now: number) => {
        while (arrivals.length && arrivals[0] < now - LIVE_RATE_WINDOW_MS) arrivals.shift();
        return (arrivals.length * 1000) / LIVE_RATE_WINDOW_MS;
      };
      es.addEventListener("rom", (ev) => {
        const rom = base64ToBytes((ev as MessageEvent).data);
        liveRomRef.current = rom;
        bridge.loadRom(rom);
      });
      es.addEventListener("reset", () => push({ t: "reset" }));
      es.addEventListener("step", (ev) => {
        push({ t: "step", mask: parseInt((ev as MessageEvent).data, 10) || 0 });
      });
      es.addEventListener("agent", (ev) => {
        push({ t: "agent", info: JSON.parse((ev as MessageEvent).data) });
      });
      es.addEventListener("decision", (ev) => {
        push({ t: "decision", d: JSON.parse((ev as MessageEvent).data) as AgentDecision });
      });
      es.addEventListener("outcome", (ev) => {
        push({ t: "outcome", o: JSON.parse((ev as MessageEvent).data) });
      });

      const apply = (item: Item) => {
        if (item.t === "step") {
          bridge.setController(item.mask);
          bridge.runFrame();
          setFramebuffer(new Uint8ClampedArray(bridge.getFramebuffer()));
          const samples = bridge.audioDrain(4096);
          if (samples.length) audioRef.current.pump(samples);
          played += 1;
          queuedSteps -= 1;
        } else if (item.t === "reset") {
          if (liveRomRef.current) bridge.loadRom(liveRomRef.current); // re-boot
        } else if (item.t === "agent") {
          const info = item.info as { policy: string; model: string; moves?: string[]; price_per_mtok?: number };
          setAgentRun((run) => ({
            ...run,
            policy: info.policy,
            model: info.model,
            moves: info.moves ?? [],
            pricePerMtok: info.price_per_mtok ?? 0,
            lives: run.lives + 1,
          }));
        } else if (item.t === "decision") {
          const d = item.d;
          setAgentRun((run) => ({
            ...run,
            decisions: [...run.decisions, d].slice(-AGENT_HISTORY),
            calls: run.calls + (isModelCall(d) ? 1 : 0),
            decided: run.decided + 1,
            fallbacks: run.fallbacks + (d.source === "jev-fallback" ? 1 : 0),
            inputTokens: run.inputTokens + d.input_tokens,
            latencySumMs: run.latencySumMs + (isModelCall(d) ? d.latency_ms : 0),
          }));
        } else {
          setAgentRun((run) => ({
            ...run,
            lastOutcome: item.o,
            outcomes: [...run.outcomes, item.o].slice(-50),
          }));
        }
      };

      const tick = (now: number) => {
        if (liveSourceRef.current !== es) return; // disconnected
        // Start (or resume) only with a cushion of frames, then hold 60 fps from it.
        if (buffering && queuedSteps >= (played === 0 ? LIVE_BUFFER_START : LIVE_BUFFER_RESUME)) {
          buffering = false;
          last = now;
          carry = 0;
        }
        if (!buffering) {
          // Play at the pace frames arrive (never above 60 fps, never below a crawl);
          // with a deep buffer, run at full speed to catch up.
          const fps = queuedSteps > LIVE_BUFFER_CATCH_UP
            ? 60
            : Math.max(LIVE_MIN_FPS, Math.min(60, arrivalRate(now) * 0.95));
          carry += ((now - last) * fps) / 1000;
          last = now;
          let steps = Math.min(Math.floor(carry), 4); // never rush more than 4 frames per tick
          carry -= steps;
          while (steps > 0) {
            if (queuedSteps === 0) {
              buffering = true; // ran dry: refill before playing on
              break;
            }
            const item = queue.shift() as Item;
            apply(item);
            if (item.t === "step") steps -= 1;
          }
          if (played % 6 === 0) {
            setLiveAgent({ connected: true, frame: played, buffered: queuedSteps, fps: Math.round(fps) });
          }
        } else if (queuedSteps % 30 === 0) {
          setLiveAgent({ connected: true, frame: played, buffered: queuedSteps, fps: 0 });
        }
        liveRafRef.current = requestAnimationFrame(tick);
      };
      liveRafRef.current = requestAnimationFrame(tick);

      es.onopen = () => {
        setLiveAgent({ connected: true, frame: 0, buffered: 0, fps: 0 });
        addToast("Live agent connected", "info");
      };
      es.onerror = () => {
        addToast("Live agent not reachable (is `python -m nesenv.live` running?)", "danger");
        disconnectLiveAgent();
      };
    },
    [addToast, stopLoop, stopMovie, disconnectLiveAgent],
  );

  // BRK (opcode 0x00) handling: the C++ core dispatches `nes-brk-encountered`
  // when a BRK is stepped. Auto-stop the loop and surface a toast. (§6, preserve.)
  useEffect(() => {
    const onBrk = () => {
      stopLoop();
      const bridge = dbgRef.current;
      if (bridge) {
        bridge.stop();
        setSnapshot(bridge.getSnapshot());
      }
      addToast("Program terminated with BRK", "info");
    };
    window.addEventListener("nes-brk-encountered", onBrk);
    return () => window.removeEventListener("nes-brk-encountered", onBrk);
  }, [addToast, stopLoop]);

  const addBreakpoint = useCallback((addr: number) => {
    const bridge = dbgRef.current;
    if (!bridge) return;
    bridge.addBreakpoint(addr);
    setBreakpoints((prev) =>
      prev.includes(addr) ? prev : [...prev, addr].sort((a, b) => a - b),
    );
  }, []);

  const removeBreakpoint = useCallback((addr: number) => {
    const bridge = dbgRef.current;
    if (!bridge) return;
    bridge.removeBreakpoint(addr);
    setBreakpoints((prev) => prev.filter((b) => b !== addr));
  }, []);

  const toggleBreakpoint = useCallback(
    (addr: number) => {
      const bridge = dbgRef.current;
      if (!bridge) return;
      // Read the current set synchronously from the ref, perform bridge
      // side-effects + the toast here, and hand a pure value to setState.
      const current = breakpointsRef.current;
      if (current.includes(addr)) {
        bridge.removeBreakpoint(addr);
        setBreakpoints((prev) => prev.filter((b) => b !== addr));
      } else {
        bridge.addBreakpoint(addr);
        setBreakpoints((prev) =>
          prev.includes(addr) ? prev : [...prev, addr].sort((a, b) => a - b),
        );
      }
      addToast(
        `Breakpoint toggled at $${addr.toString(16).toUpperCase().padStart(4, "0")}`,
        "info",
      );
    },
    [addToast],
  );

  const writeMemory = useCallback(
    (addr: number, value: number) => {
      const bridge = dbgRef.current;
      if (!bridge) return;
      bridge.writeMemory(addr, value);
      refresh();
    },
    [refresh],
  );

  const loadROM = useCallback(
    (data: Uint8Array) => {
      const bridge = dbgRef.current;
      if (!bridge) return;
      bridge.loadROM(data);
      refresh();
    },
    [refresh],
  );

  const loadRom = useCallback(
    (data: Uint8Array): number => {
      const bridge = dbgRef.current;
      if (!bridge) return -1;
      stopLoop();
      const status = bridge.loadRom(data);
      if (status === 0) {
        // Do NOT reset() here: load_rom already boots the cartridge (PC <- reset
        // vector). A second reset() would send PC back to $FFFC and the ROM would
        // never run. Just publish the freshly-rendered framebuffer.
        setFramebuffer(new Uint8ClampedArray(bridge.getFramebuffer()));
      }
      refresh();
      return status;
    },
    [refresh, stopLoop],
  );

  const loadOpcodes = useCallback(
    (text: string) => {
      const bytes = parseOpcodes(text);
      loadROM(Uint8Array.from(bytes));
    },
    [loadROM],
  );

  const setController = useCallback((state: number, port = 0) => {
    dbgRef.current?.setController(state, port);
  }, []);

  const actions = useMemo<EmulatorActions>(
    () => ({
      step,
      run,
      stop,
      reset,
      addBreakpoint,
      removeBreakpoint,
      toggleBreakpoint,
      writeMemory,
      loadROM,
      loadRom,
      loadOpcodes,
      setController,
      playMovie,
      stopMovie,
      connectLiveAgent,
      disconnectLiveAgent,
    }),
    [
      step,
      run,
      stop,
      reset,
      addBreakpoint,
      removeBreakpoint,
      toggleBreakpoint,
      writeMemory,
      loadROM,
      loadRom,
      loadOpcodes,
      setController,
      playMovie,
      stopMovie,
      connectLiveAgent,
      disconnectLiveAgent,
    ],
  );

  const value = useMemo<EmulatorContextValue>(
    () => ({
      status,
      snapshot,
      breakpoints,
      running,
      framebuffer,
      movie,
      liveAgent,
      agentRun,
      dbg,
      actions,
    }),
    [status, snapshot, breakpoints, running, framebuffer, movie, liveAgent, agentRun, dbg, actions],
  );

  return (
    <EmulatorContext.Provider value={value}>
      {props.children}
    </EmulatorContext.Provider>
  );
}

export function useEmulator(): EmulatorContextValue {
  const ctx = useContext(EmulatorContext);
  if (!ctx) {
    throw new Error("useEmulator must be used within an EmulatorProvider");
  }
  return ctx;
}
