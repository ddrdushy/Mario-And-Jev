// web/src/components/AgentPanels.tsx
//
// Two cockpit panels that appear while a narrating live agent (python -m nesenv.live
// --agent jev|heuristic) is connected: what it just decided and why, and how long each
// decision took. They read the `decision` events the provider collects in `agentRun`.
import { useState } from "react";
import {
  useEmulator,
  isModelCall,
  type AgentDecision,
} from "../emulator/EmulatorProvider";
import { Module } from "./ui/Module";

const FRAME_MS = 1000 / 60;
const CHART_CALLS = 60; // columns drawn; percentiles still use the whole history

const MOVE_LABELS: Record<string, string> = {
  run_right: "Run right",
  hop: "Hop",
  full_jump: "Full jump",
  wait: "Wait",
  back_up: "Back up",
};

function moveLabel(move: string): string {
  return MOVE_LABELS[move] ?? move;
}

export function formatLatency(ms: number): string {
  if (ms >= 1000) return `${(ms / 1000).toFixed(2)} s`;
  if (ms >= 10) return `${Math.round(ms)} ms`;
  if (ms >= 1) return `${ms.toFixed(1)} ms`;
  return `${ms.toFixed(2)} ms`;
}

export function percentile(values: number[], p: number): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const rank = Math.min(sorted.length - 1, Math.ceil((p / 100) * sorted.length) - 1);
  return sorted[Math.max(0, rank)];
}

/** Round an axis maximum up to 1 / 2 / 5 x 10^n so the ticks are clean numbers. */
export function niceCeil(value: number): number {
  if (value <= 0) return 1;
  const pow = Math.pow(10, Math.floor(Math.log10(value)));
  const unit = value / pow;
  return (unit <= 1 ? 1 : unit <= 2 ? 2 : unit <= 5 ? 5 : 10) * pow;
}

function sourceLabel(source: string): string {
  if (source === "jev") return "Jev";
  if (source === "laya") return "Laya (local)";
  if (source === "jev-fallback") return "Jev, low confidence";
  if (source === "jev-cached") return "Jev, cached answer";
  if (source === "code") return "code, no call needed";
  return "offline rules";
}

interface PanelProps {
  revealDelay?: number;
  className?: string;
}

/** A 0..1 value as a thin horizontal bar with the number at its tip. */
function Meter({
  label,
  value,
  emphasis,
}: {
  label: string;
  value: number;
  emphasis: boolean;
}): JSX.Element {
  const pct = Math.max(0, Math.min(1, value)) * 100;
  return (
    <div className="flex items-center gap-[8px]">
      <span className="w-[74px] flex-none truncate text-[10px] text-[var(--mut)]">
        {label}
      </span>
      <div className="relative h-[8px] min-w-0 flex-1 rounded-[2px] bg-[var(--b2)]">
        <div
          className="h-full rounded-r-[4px]"
          style={{
            width: `${pct}%`,
            minWidth: pct > 0 ? 2 : 0,
            background: emphasis ? "var(--acc)" : "var(--dim)",
          }}
        />
      </div>
      <span className="w-[30px] flex-none text-right font-mono text-[10px] text-[var(--tx)]">
        {value.toFixed(2)}
      </span>
    </div>
  );
}

export function AgentDecisionPanel({ revealDelay, className }: PanelProps): JSX.Element {
  const { agentRun, liveAgent } = useEmulator();
  const { decisions, moves } = agentRun;
  const current: AgentDecision | undefined = decisions[decisions.length - 1];
  const recent = [...decisions].reverse().slice(0, 40);

  return (
    <Module
      title="Agent decisions"
      status={
        agentRun.policy ? (
          <span data-testid="agent-policy">
            {agentRun.policy} · {agentRun.model}
            {liveAgent.connected
              ? ` · ${liveAgent.fps ? `${liveAgent.fps} fps` : "buffering"} · ${(liveAgent.buffered / 60).toFixed(1)}s ahead`
              : ""}
          </span>
        ) : undefined
      }
      revealDelay={revealDelay}
      className={className}
      bodyClassName="nes-scroll flex min-h-0 flex-col gap-[10px] overflow-auto"
    >
      {!current ? (
        <p className="font-mono text-[10px] text-[var(--tx-dim)]">
          Waiting for the agent&apos;s first decision…
        </p>
      ) : (
        <>
          {current.world !== undefined ? (
            <div
              data-testid="agent-progress"
              className="flex items-center justify-between rounded-[var(--radius-sm)] bg-[var(--b2)] px-[8px] py-[5px] font-mono text-[10px] text-[var(--tx)]"
            >
              <span>
                World <b>{current.world}-{current.stage}</b>
              </span>
              <span>
                coins <b>{current.coins}</b>
              </span>
              <span>
                lives <b>{current.lives}</b>
              </span>
              <span className="text-[var(--mut)]">
                {agentRun.outcomes.filter((o) => o.outcome.startsWith("cleared")).length} levels cleared
              </span>
            </div>
          ) : null}
          <div className="flex items-end justify-between gap-[8px]">
            <div className="min-w-0">
              <div className="text-[9px] uppercase tracking-[0.14em] text-[var(--dim)]">
                Decision {current.n} · frame {current.frame} · x {current.x}
              </div>
              <div
                data-testid="agent-move"
                className="truncate font-sans text-[22px] font-semibold leading-tight text-[var(--tx)]"
              >
                {current.action && current.action !== current.move
                  ? current.action
                  : moveLabel(current.move)}
              </div>
              {current.action && current.action !== current.move ? (
                <div className="text-[10px] text-[var(--mut)]">
                  move if no target: {moveLabel(current.move)}
                </div>
              ) : null}
            </div>
            <div className="flex-none text-right">
              <div className="font-mono text-[13px] text-[var(--tx)]">
                {isModelCall(current) ? formatLatency(current.latency_ms) : "no call"}
              </div>
              <div className="flex items-center justify-end gap-[5px] text-[9px] text-[var(--mut)]">
                <span
                  aria-hidden
                  className="h-[6px] w-[6px] rounded-full"
                  style={{
                    background:
                      current.source === "jev-fallback" ? "var(--amb)" : "var(--acc)",
                  }}
                />
                {sourceLabel(current.source)}
              </div>
            </div>
          </div>

          <p
            data-testid="agent-summary"
            className="rounded-[var(--radius-sm)] bg-[var(--b2)] px-[8px] py-[6px] text-[10px] leading-snug text-[var(--mut)]"
          >
            <span className="text-[var(--dim)]">Scene sent: </span>
            {current.summary}
          </p>

          <div className="flex flex-col gap-[5px]">
            <div className="text-[9px] uppercase tracking-[0.14em] text-[var(--dim)]">
              {current.probabilities ? "Move probabilities" : "Confidence"}
            </div>
            {current.probabilities ? (
              (moves.length ? moves : Object.keys(current.probabilities)).map((m) => (
                <Meter
                  key={m}
                  label={moveLabel(m)}
                  value={current.probabilities?.[m] ?? 0}
                  emphasis={m === current.move}
                />
              ))
            ) : null}
            <Meter label="Confidence" value={current.confidence} emphasis />
            {current.obstacle_needs_jump !== undefined ? (
              <Meter
                label="Obstacle?"
                value={current.obstacle_needs_jump}
                emphasis={false}
              />
            ) : null}
            {current.enemy_needs_jump !== undefined ? (
              <Meter label="Enemy?" value={current.enemy_needs_jump} emphasis={false} />
            ) : null}
            {current.hazard_first !== undefined ? (
              <Meter label="Danger first?" value={current.hazard_first} emphasis={false} />
            ) : null}
          </div>
          {current.target_probabilities ? (
            <div className="flex flex-col gap-[5px]" data-testid="agent-target">
              <div className="text-[9px] uppercase tracking-[0.14em] text-[var(--dim)]">
                Which collectable to go for
              </div>
              {Object.entries(current.target_probabilities)
                .sort(([a], [b]) => (a === "none" ? -1 : b === "none" ? 1 : a.localeCompare(b)))
                .map(([id, pr]) => (
                  <Meter
                    key={id}
                    label={id.replace("_", " ")}
                    value={pr}
                    emphasis={id === current.target}
                  />
                ))}
            </div>
          ) : null}

          <div className="min-h-[120px] flex-none">
            <table className="w-full border-collapse font-mono text-[10px]">
              <thead className="sticky top-0 bg-[var(--b1)] text-left text-[var(--dim)]">
                <tr>
                  <th className="py-[2px] pr-[6px] font-normal">#</th>
                  <th className="py-[2px] pr-[6px] font-normal">frame</th>
                  <th className="py-[2px] pr-[6px] font-normal">move</th>
                  <th className="py-[2px] pr-[6px] text-right font-normal">conf</th>
                  <th className="py-[2px] text-right font-normal">latency</th>
                </tr>
              </thead>
              <tbody>
                {recent.map((d) => (
                  <tr key={`${d.frame}-${d.n}`} title={d.summary} className="text-[var(--tx)]">
                    <td className="py-[1px] pr-[6px] text-[var(--dim)]">{d.n}</td>
                    <td className="py-[1px] pr-[6px] text-[var(--mut)]">{d.frame}</td>
                    <td className="py-[1px] pr-[6px]">
                      {d.action && d.action !== d.move ? d.action.replace(/\s*\(.*\)$/, "") : moveLabel(d.move)}
                      {d.source === "jev-fallback" ? (
                        <span className="text-[var(--mut)]"> (fallback)</span>
                      ) : null}
                    </td>
                    <td className="py-[1px] pr-[6px] text-right">{d.confidence.toFixed(2)}</td>
                    <td className="py-[1px] text-right">
                      {isModelCall(d) ? formatLatency(d.latency_ms) : (
                        <span className="text-[var(--dim)]">{d.source === "code" ? "code" : "cached"}</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </Module>
  );
}

function Stat({ label, value, hint }: { label: string; value: string; hint?: string }): JSX.Element {
  return (
    <div className="min-w-0 rounded-[var(--radius-sm)] bg-[var(--b2)] px-[8px] py-[6px]" title={hint}>
      <div className="truncate text-[9px] text-[var(--mut)]">{label}</div>
      <div className="truncate font-sans text-[12px] font-semibold text-[var(--tx)]">{value}</div>
    </div>
  );
}

export function AgentLatencyPanel({ revealDelay, className }: PanelProps): JSX.Element {
  const { agentRun } = useEmulator();
  const [hover, setHover] = useState<number | null>(null);

  // Only real round trips count as latency; cached and code-decided moves were free.
  const calls = agentRun.decisions.filter(isModelCall);
  const totals = calls.map((d) => d.latency_ms);
  const split = calls.filter((d) => (d.server_ms ?? 0) > 0);
  const p50 = percentile(totals, 50);
  const p95 = percentile(totals, 95);
  const worst = totals.length ? Math.max(...totals) : 0;
  const modelP50 = percentile(split.map((d) => d.server_ms ?? 0), 50);
  const networkP50 = percentile(split.map((d) => d.latency_ms - (d.server_ms ?? 0)), 50);
  const cost = (agentRun.inputTokens / 1e6) * agentRun.pricePerMtok;
  const free = agentRun.decided - agentRun.calls;

  const charted = calls.slice(-CHART_CALLS);
  // One slow request would flatten every other column, so the axis stops at p95 and
  // anything taller is drawn full height with a break mark (the tooltip has the value).
  const axisMax = niceCeil(Math.max(p95, 1e-3));
  const ticks = [axisMax, axisMax / 2, 0];
  const hovered = hover !== null ? charted[hover] : undefined;
  const hasSplit = split.length > 0;

  return (
    <Module
      title="Decision latency"
      status={<span>API round trips, last {calls.length}</span>}
      revealDelay={revealDelay}
      className={className}
      bodyClassName="nes-scroll flex min-h-0 flex-col gap-[10px] overflow-auto"
    >
      <div className="flex items-end justify-between gap-[8px]">
        <div>
          <div className="text-[9px] uppercase tracking-[0.14em] text-[var(--dim)]">
            Median round trip
          </div>
          <div
            data-testid="agent-p50"
            className="font-sans text-[30px] font-semibold leading-none text-[var(--tx)]"
          >
            {formatLatency(p50)}
          </div>
        </div>
        <div className="text-right text-[10px] leading-snug text-[var(--mut)]">
          {hasSplit ? (
            <>
              Jev {formatLatency(modelP50)} + network {formatLatency(networkP50)}
              <br />
            </>
          ) : null}
          = {(p50 / FRAME_MS).toFixed(1)} game frames; the emulator waits, none are lost
        </div>
      </div>

      <div className="grid grid-cols-4 gap-[6px]">
        <Stat label="p95" value={formatLatency(p95)} />
        <Stat label="Jev model" value={hasSplit ? formatLatency(modelP50) : "n/a"} hint="Median time the TypeSafe server spent on the request" />
        <Stat label="Network" value={hasSplit ? formatLatency(networkP50) : "n/a"} hint="Median round trip minus the server's own time" />
        <Stat label="Slowest" value={formatLatency(worst)} />
        <Stat label="API calls" value={agentRun.calls.toLocaleString()} hint={`${agentRun.lives} lives`} />
        <Stat
          label="No call"
          value={agentRun.decided ? `${Math.round((100 * free) / agentRun.decided)}%` : "0%"}
          hint="Decisions answered from cache or by code because nothing was ahead"
        />
        <Stat label="Tokens in" value={agentRun.inputTokens.toLocaleString()} />
        <Stat
          label="Est. cost"
          value={`$${cost.toFixed(4)}`}
          hint={`$${agentRun.pricePerMtok} per 1M input tokens`}
        />
      </div>

      {agentRun.outcomes.length ? (
        <ul data-testid="agent-outcome" className="flex flex-col gap-[2px] font-mono text-[10px] text-[var(--mut)]">
          {agentRun.outcomes.slice(-4).map((o, i) => (
            <li key={`${o.outcome}-${o.distance}-${i}`}>
              {o.world !== undefined ? `${o.world}-${o.stage}: ` : ""}
              {o.outcome} at x {o.distance}
              {o.coins !== undefined ? `, ${o.coins} coins` : ""}
            </li>
          ))}
        </ul>
      ) : null}

      <div>
        <div className="mb-[4px] flex items-center justify-between text-[10px] text-[var(--mut)]">
          <span>Latency per call, last {charted.length}</span>
          {hasSplit ? (
            <span className="flex items-center gap-[10px]">
              <span className="flex items-center gap-[4px]">
                <span aria-hidden className="h-[7px] w-[7px] rounded-[1px] bg-[var(--acc)]" />
                Jev model
              </span>
              <span className="flex items-center gap-[4px]">
                <span aria-hidden className="h-[7px] w-[7px] rounded-[1px] bg-[var(--chart-2)]" />
                network
              </span>
            </span>
          ) : null}
        </div>
        <div className="flex gap-[6px] pt-[6px]">
          <div className="relative h-[96px] w-[44px] flex-none font-mono text-[9px] text-[var(--dim)]">
            {ticks.map((t, i) => (
              <span
                key={t}
                className="absolute right-0 -translate-y-1/2"
                style={{ top: `${i * 50}%` }}
              >
                {formatLatency(t)}
              </span>
            ))}
          </div>
          <div
            data-testid="agent-latency-chart"
            className="relative h-[96px] min-w-0 flex-1"
            onMouseLeave={() => setHover(null)}
          >
            {ticks.map((t, i) => (
              <div
                key={t}
                aria-hidden
                className="absolute inset-x-0 h-px bg-[var(--bd)]"
                style={{ top: `${i * 50}%` }}
              />
            ))}
            <div className="absolute inset-0 flex items-end gap-[2px] overflow-hidden">
              {charted.map((d, i) => {
                const clipped = d.latency_ms > axisMax;
                const total = Math.min(d.latency_ms, axisMax);
                const model = Math.min(d.server_ms ?? 0, total);
                const dim = hover !== null && hover !== i;
                return (
                  <div
                    key={`${d.frame}-${d.n}`}
                    data-testid="agent-latency-column"
                    className="flex h-full min-w-0 max-w-[24px] flex-1 flex-col justify-end gap-px"
                    style={{ opacity: dim ? 0.45 : 1 }}
                    onMouseEnter={() => setHover(i)}
                  >
                    {clipped ? (
                      <div aria-hidden className="h-[2px] w-full flex-none bg-[var(--tx)]" />
                    ) : null}
                    <div
                      className="w-full flex-none rounded-t-[2px]"
                      style={{
                        height: `${Math.max(1.5, (100 * (total - model)) / axisMax)}%`,
                        background: hasSplit ? "var(--chart-2)" : "var(--acc)",
                      }}
                    />
                    {model > 0 ? (
                      <div
                        className="w-full flex-none bg-[var(--acc)]"
                        style={{ height: `${Math.max(1.5, (100 * model) / axisMax)}%` }}
                      />
                    ) : null}
                  </div>
                );
              })}
            </div>
            {hovered && hover !== null ? (
              <div
                role="tooltip"
                className="pointer-events-none absolute bottom-full z-10 mb-[4px] w-[200px] rounded-[var(--radius-sm)] border border-[var(--bd2)] bg-[var(--b0)] px-[8px] py-[6px] text-[10px] leading-snug text-[var(--tx)]"
                style={
                  hover < charted.length / 2
                    ? { left: `${(100 * (hover + 1)) / charted.length}%` }
                    : { right: `${100 - (100 * hover) / charted.length}%` }
                }
              >
                <div className="font-mono">
                  {formatLatency(hovered.latency_ms)} · {moveLabel(hovered.move)}
                </div>
                {hovered.server_ms ? (
                  <div className="text-[var(--mut)]">
                    Jev {formatLatency(hovered.server_ms)} + network{" "}
                    {formatLatency(hovered.latency_ms - hovered.server_ms)}
                  </div>
                ) : null}
                <div className="text-[var(--mut)]">
                  decision {hovered.n}, frame {hovered.frame}, confidence{" "}
                  {hovered.confidence.toFixed(2)}
                </div>
                <div className="mt-[3px] text-[var(--mut)]">{hovered.summary}</div>
              </div>
            ) : null}
          </div>
        </div>
        {charted.some((d) => d.latency_ms > axisMax) ? (
          <p className="mt-[4px] pl-[50px] text-[9px] text-[var(--dim)]">
            Columns with a white cap ran past the axis; hover for the real time.
          </p>
        ) : null}
      </div>
    </Module>
  );
}
