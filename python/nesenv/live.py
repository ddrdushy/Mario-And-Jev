"""Stream a live agent to the browser over Server-Sent Events (stdlib only).

The server runs an agent on its own deterministic core and streams one action byte
per frame. NES Studio re-simulates those actions on its WASM core, so the browser
shows exactly what the agent is doing, frame for frame, at tiny bandwidth. There is
no WebSocket dependency: SSE is plain HTTP and works with the built-in http.server.

    python -m nesenv.live "Super Mario Bros.nes" --agent scripted --port 8000
    python -m nesenv.live "Super Mario Bros.nes" --agent jev     # TypeSafe Jev decides
    python -m nesenv.live "Super Mario Bros.nes" --agent duo     # Laya locally, Jev when unsure

Then click "Spawn Agent" in NES Studio. Swap the policy in `agent_action` for a
trained network to watch it learn; the streaming protocol does not change.

Protocol (text/event-stream):
    event: rom    data: <base64 ROM>     (once, on connect)
    event: reset  data:                  (re-boot, at the start of a life cycle)
    event: step   data: <button mask>    (one per frame)

The jev / heuristic agents also narrate themselves, so the UI can show what was decided
and how long it took:
    event: agent     data: {"policy", "model", "moves", "price_per_mtok"}   (once per life)
    event: decision  data: {"frame", "x", "summary", "move", "confidence", "source",
                            "latency_ms", "input_tokens", "probabilities"?, ...}
    event: outcome   data: {"outcome", "distance", "world", "stage", "coins"}  (each life / level)
"""

from __future__ import annotations

import argparse
import base64
import json
import random
import time
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .core import A, B, RIGHT, START, Nes
from .smb import OPER_MODE, PLAYER_PAGE, PLAYER_X

# Module-level config set by main(); each connection reads it.
ROM = b""
AGENT = "scripted"
LOG_DIR = Path(__file__).resolve().parents[2] / "logs"   # <repo>/logs, one .jsonl per game


def gameplay_started(nes: Nes, frame: int) -> bool:
    return nes.peek(OPER_MODE) == 1 and frame > 250


# Reboot a life if the agent makes no forward progress for this many frames
# (covers both getting stuck on an obstacle and dying), so the stream never freezes.
STALL_RESET = 480  # ~8 seconds


def agent_action(nes: Nes, frame: int, state: dict) -> int:
    """Decide a button mask for this frame. This is a fixed heuristic, not a
    learner: replace it with a trained policy to watch an agent actually improve;
    everything downstream (the stream, the browser) stays the same."""
    if AGENT == "random":
        # mostly run right, sometimes jump, so a random run still looks alive
        return random.choice([RIGHT, RIGHT | A, RIGHT | B, RIGHT | A | B, 0, RIGHT])
    # scripted: run right (B), jump on a rhythm and when progress stalls. The jump
    # phase is offset per attempt (state["offset"]) so a reset retries differently.
    progress = nes.peek(PLAYER_PAGE) * 256 + nes.peek(PLAYER_X)
    state["pstall"] = state.get("pstall", 0) + 1 if progress <= state.get("pmax", 0) else 0
    state["pmax"] = max(state.get("pmax", 0), progress)
    mask = RIGHT | B  # hold B to run, so run-jumps clear bigger gaps
    if ((frame + state.get("offset", 0)) % 24) < 13 or state["pstall"] > 4:
        mask |= A
    return mask


def drive(send) -> None:
    """Run the agent forever, calling send(event, data) for each frame."""
    nes = Nes()
    nes.load(ROM)
    send("rom", base64.b64encode(ROM).decode())
    policy = None

    while True:
        send("reset", "")
        nes.reset()
        if AGENT in ("jev", "heuristic", "laya", "duo"):
            policy = drive_player(nes, send, policy)  # reused, so its answer cache survives lives
            continue
        state: dict = {"offset": random.randint(0, 23)}
        max_progress = 0
        stall = 0
        started = False
        for frame in range(100_000):
            if frame < 80:
                mask = 0
            elif frame < 90:
                mask = START
            elif not gameplay_started(nes, frame):
                mask = 0
            else:
                started = True
                mask = agent_action(nes, frame, state)

            send("step", str(mask))
            nes.step(mask)
            time.sleep(1 / 60)  # pace to roughly real time

            if started:
                progress = nes.peek(PLAYER_PAGE) * 256 + nes.peek(PLAYER_X)
                if progress > max_progress:
                    max_progress, stall = progress, 0
                else:
                    stall += 1
                if stall > STALL_RESET:
                    break  # stuck or died: reboot and try again


def drive_player(nes: Nes, send, policy=None):
    """One whole game (all lives, level after level) played by a nesenv.jev Player.
    Returns the policy so the caller can hand it back for the next game."""
    from .jev import MOVE_CRITERIA, PRICE_PER_MTOK, Player, make_policy

    policy = policy or make_policy(AGENT)
    send("agent", json.dumps({
        "policy": policy.name, "model": policy.model,
        "moves": list(MOVE_CRITERIA), "price_per_mtok": PRICE_PER_MTOK,
    }))
    log_path = LOG_DIR / f"live-{AGENT}-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    LOG_DIR.mkdir(exist_ok=True)
    log = open(log_path, "w")
    print(f"logging decisions to {log_path}", flush=True)

    def on_life_end(life: dict) -> None:
        log.write(json.dumps({"life_end": life}) + "\n")
        log.flush()
        print(f"{AGENT}: {life['world']}-{life['stage']} {life['outcome']} at x={life['distance']}, "
              f"{life['coins']} coins", flush=True)

    player = Player(
        nes, policy, log, max_decisions=10**9,
        on_decision=lambda record: send("decision", json.dumps(record)),
        on_life_end=on_life_end,
    )
    tail = 0
    while tail < 120:  # linger two seconds on the game over screen before rebooting
        started = time.perf_counter()
        mask = player.next_mask()
        send("step", str(mask))
        nes.step(mask)
        # a Jev call already took longer than a frame; only sleep off what is left
        time.sleep(max(0.0, 1 / 60 - (time.perf_counter() - started)))
        if player.outcome:
            if tail == 0:
                send("outcome", json.dumps({"outcome": player.outcome, "distance": player.max_x,
                                            "coins": player.coins}))
            tail += 1
    log.close()
    print(f"{AGENT}: {player.outcome}, {player.levels_cleared} levels cleared, {player.coins} coins", flush=True)
    return policy


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # quieter
        pass

    def do_GET(self):
        if self.path.rstrip("/") not in ("", "/stream"):
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        def send(event: str, data: str) -> None:
            self.wfile.write(f"event: {event}\ndata: {data}\n\n".encode())
            self.wfile.flush()

        try:
            drive(send)
        except (BrokenPipeError, ConnectionResetError):
            pass  # viewer closed the tab


def main() -> int:
    global ROM, AGENT
    p = argparse.ArgumentParser(description="Stream a live NES agent over SSE.")
    p.add_argument("rom")
    p.add_argument("--agent", choices=["scripted", "random", "jev", "heuristic", "laya", "duo"], default="scripted")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()

    ROM = open(args.rom, "rb").read()
    AGENT = args.agent
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"live agent ({args.agent}) streaming on http://127.0.0.1:{args.port}/stream")
    print("open NES Studio and click 'Spawn Agent'. Ctrl-C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
