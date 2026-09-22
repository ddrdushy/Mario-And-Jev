// web/src/components/AgentPanels.test.tsx
import { render, screen } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import {
  AgentDecisionPanel,
  AgentLatencyPanel,
  formatLatency,
  niceCeil,
  percentile,
} from "./AgentPanels";
import {
  EMPTY_AGENT_RUN,
  type AgentDecision,
  type EmulatorContextValue,
} from "../emulator/EmulatorProvider";

vi.mock("../emulator/EmulatorProvider", async (orig) => {
  const actual = await orig<typeof import("../emulator/EmulatorProvider")>();
  return { ...actual, useEmulator: () => mockCtx };
});

let mockCtx: EmulatorContextValue;

function decision(n: number, latency: number, extra: Partial<AgentDecision> = {}): AgentDecision {
  return {
    n,
    frame: 600 + n * 6,
    x: 40 + n * 10,
    summary: "goomba very close (1 tile) ahead, same height as Mario",
    move: "hop",
    confidence: 0.81,
    source: "jev",
    latency_ms: latency,
    input_tokens: 500,
    probabilities: { run_right: 0.1, hop: 0.85, full_jump: 0.05, wait: 0, back_up: 0 },
    obstacle_needs_jump: 0.04,
    enemy_needs_jump: 0.97,
    ...extra,
  };
}

function ctx(decisions: AgentDecision[]): EmulatorContextValue {
  return {
    status: "ready",
    snapshot: null,
    breakpoints: [],
    running: false,
    framebuffer: null,
    movie: { playing: false, frame: 0, total: 0 },
    liveAgent: { connected: true, frame: 0 },
    agentRun: {
      ...EMPTY_AGENT_RUN,
      policy: "jev",
      model: "jev-latest",
      moves: ["run_right", "hop", "full_jump", "wait", "back_up"],
      pricePerMtok: 0.042,
      decisions,
      calls: decisions.length,
      decided: decisions.length,
      inputTokens: decisions.length * 500,
      latencySumMs: decisions.reduce((sum, d) => sum + d.latency_ms, 0),
      lives: 1,
    },
    dbg: null,
    actions: {} as EmulatorContextValue["actions"],
  };
}

describe("latency helpers", () => {
  it("formats across the ms / s boundary", () => {
    expect(formatLatency(0.034)).toBe("0.03 ms");
    expect(formatLatency(4.26)).toBe("4.3 ms");
    expect(formatLatency(142.4)).toBe("142 ms");
    expect(formatLatency(1250)).toBe("1.25 s");
  });

  it("takes nearest-rank percentiles", () => {
    expect(percentile([], 50)).toBe(0);
    expect(percentile([100, 300, 200, 400], 50)).toBe(200);
    expect(percentile([100, 300, 200, 400], 95)).toBe(400);
  });

  it("rounds the axis up to a clean number", () => {
    expect(niceCeil(142)).toBe(200);
    expect(niceCeil(480)).toBe(500);
    expect(niceCeil(0)).toBe(1);
  });
});

describe("AgentDecisionPanel", () => {
  it("waits for the first decision", () => {
    mockCtx = ctx([]);
    render(<AgentDecisionPanel />);
    expect(screen.getByText(/first decision/)).toBeInTheDocument();
  });

  it("shows progress, the collect action and the target choice", () => {
    mockCtx = ctx([
      decision(3, 300, {
        action: "collect question block (close (2 tiles), 4 up)",
        target: "item_1",
        target_probabilities: { none: 0.02, item_1: 0.9, item_2: 0.08 },
        world: 1, stage: 2, coins: 12, lives: 2,
      }),
    ]);
    render(<AgentDecisionPanel />);
    expect(screen.getByTestId("agent-progress")).toHaveTextContent("World 1-2");
    expect(screen.getByTestId("agent-progress")).toHaveTextContent("coins 12");
    expect(screen.getByTestId("agent-move")).toHaveTextContent("collect question block");
    expect(screen.getByTestId("agent-target")).toHaveTextContent("item 1");
  });

  it("shows the latest move, the scene it was given and the policy", () => {
    mockCtx = ctx([decision(1, 120, { move: "run_right" }), decision(2, 150)]);
    render(<AgentDecisionPanel />);
    expect(screen.getByTestId("agent-move")).toHaveTextContent("Hop");
    expect(screen.getByTestId("agent-summary")).toHaveTextContent("goomba very close");
    expect(screen.getByTestId("agent-policy")).toHaveTextContent("jev · jev-latest");
  });
});

describe("AgentLatencyPanel", () => {
  it("ignores cached and code decisions and splits model time from network", () => {
    mockCtx = ctx([
      decision(1, 400, { server_ms: 100 }),
      decision(2, 0, { source: "jev-cached", input_tokens: 0 }),
      decision(3, 0, { source: "code", input_tokens: 0 }),
      decision(4, 600, { server_ms: 150 }),
    ]);
    render(<AgentLatencyPanel />);
    expect(screen.getAllByTestId("agent-latency-column")).toHaveLength(2);
    expect(screen.getByTestId("agent-p50")).toHaveTextContent("400 ms");
    expect(screen.getByText(/Jev 100 ms \+ network 300 ms/)).toBeInTheDocument();
  });

  it("reports the median and one column per decision", () => {
    mockCtx = ctx([decision(1, 100), decision(2, 200), decision(3, 300)]);
    render(<AgentLatencyPanel />);
    expect(screen.getByTestId("agent-p50")).toHaveTextContent("200 ms");
    expect(screen.getAllByTestId("agent-latency-column")).toHaveLength(3);
  });
});
