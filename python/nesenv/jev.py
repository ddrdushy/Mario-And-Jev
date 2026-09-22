"""Play Super Mario Bros with TypeSafe's Jev model making the decisions.

Jev is not a text generator: you send it a `state` plus typed questions and it returns
probabilities. So the split here is the one TypeSafe recommends. Code owns everything
deterministic (reading RAM, tile maths, button timing, recording), and Jev is asked one
narrow judgement a few times a second: given this scene, which move?

    scene (from smb_state.read_scene)  ->  Jev: Choice "move" + speculative Nouls
                                       ->  code turns the move into held buttons

The HTTP call uses only the standard library, like the rest of this package. Set
TYPESAFE_API_KEY in the environment or in a .env file at the repo root.
"""

from __future__ import annotations

import http.client
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from .core import A, B, DOWN, LEFT, RIGHT, START, Nes
from .movie import write_movie
from .smb_state import OPER_MODE, REACH_TILES, read_scene

API_HOST = "api.typesafe.ai"
API_PATH = "/v1/systemone"
DEFAULT_MODEL = "jev-latest"
PRICE_PER_MTOK = 0.042  # input tokens; output is free (docs.typesafe.ai/models)

# ---------------------------------------------------------------- the questions
#
# These strings and the thresholds below are the whole "policy". Tune them here.

MOVE_CRITERIA = {
    "run_right": {
        "what": "Keep running right. Nothing needs a jump yet.",
        "when": "No pit, wall, pipe or same-height enemy is within 2 tiles ahead.",
        "also": "A ledge that drops down is safe and is not a pit: run off it, do not jump. "
                "A sideways pipe opening is walked into, never jumped at. "
                "Anything near or far away does not need a move yet.",
    },
    "hop": {
        "what": "A short, low jump to the right.",
        "when": "Exactly one ground enemy at the same height is close or very close ahead, "
                "or a wall only 1 tile high is very close ahead.",
        "not_for": "An enemy that is still near or far away: keep running until it is close. "
                   "Enemies behind Mario or above or below him. Mario standing still (`mario.speed` "
                   "is stopped): a hop from a standstill runs into the enemy, use stomp. "
                   "Koopas are taller than a hop clears: use full_jump for a koopa, and start it "
                   "while the koopa is still near (3-4 tiles).",
    },
    "full_jump": {
        "what": "A full-height running jump to the right.",
        "when": "A pit is close or very close ahead, or a wall or pipe 2 or more tiles "
                "high is close or very close ahead, or two enemies in a row at the same "
                "height are close or very close ahead while Mario is running.",
        "not_for": "A ledge drop, or a pit, wall or pipe that is still near or far away.",
    },
    "stomp": {
        "what": "Jump straight up on the spot, so an enemy walking up passes underneath or gets stomped.",
        "when": "Mario is standing still (`mario.speed` is stopped) and an enemy at the same height "
                "is close or very close ahead.",
    },
    "wait": {
        "what": "Stand still for a moment.",
        "when": "A piranha plant is out of a pipe directly ahead, or an enemy is dropping "
                "onto the spot Mario would land on.",
    },
    "back_up": {
        "what": "Step left to make room.",
        "when": "Mario is pressed against a tall wall or pipe (touching) and needs a run-up.",
    },
    "run_left": {
        "what": "Turn around and run left.",
        "when": "The way right is blocked by a wall or pipe higher than 5 tiles, which no jump "
                "clears, so the route must be behind Mario: back off the ledge he is standing "
                "on, or back to where the floor continues underneath.",
    },
}

BASE_QUESTIONS = {
    "move": {
        "type": "choice",
        "instructions": {
            "question": "Which move should Mario make right now to get further right without dying?",
            "focus": "Read `terrain_ahead` and `enemies`. Only things that are close or "
                     "very close ahead matter for this move.",
        },
        "criteria": MOVE_CRITERIA,
    },
    # Speculative questions: asked every time, read only when `move` is uncertain.
    "obstacle_needs_jump": {
        "type": "noul",
        "instructions": "Is there a pit, wall or pipe in `terrain_ahead` that is touching, "
                        "very close or close?",
    },
    "enemy_needs_jump": {
        "type": "noul",
        "instructions": "Is there an enemy in `enemies` that is ahead, at the same height as "
                        "Mario, and touching, very close or close?",
    },
    "hazard_first": {
        "type": "noul",
        "instructions": {
            "question": "Must Mario deal with a danger before stopping to collect anything?",
            "yes_when": "An enemy at the same height, or a pit, is close or very close ahead.",
            "no_when": "Nothing dangerous is within 2 tiles ahead.",
        },
    },
}

TARGET_QUESTION = {
    "question": "Which collectable should Mario go for next? The goal is to collect every "
                "coin and hit every ? block and coin brick on the way, without dying.",
    "rules": [
        "Prefer the nearest one that is reachable with one jump from the ground.",
        "Pick none only when no listed item is reachable, or every reachable one is already behind Mario.",
    ],
}
TARGET_NONE = "Nothing worth going for right now: no reachable collectable ahead."


def build_questions(scene: dict) -> dict:
    """The fixed questions plus a Choice over this scene's collectables (code lists the
    candidates, Jev picks; the jump geometry is then done in code)."""
    questions = dict(BASE_QUESTIONS)
    items = scene.get("collectables")
    if isinstance(items, list) and items:
        criteria = {"none": TARGET_NONE}
        for item in items:
            criteria[item["id"]] = {
                "what": item["what"],
                "where": f"{item['distance']} ahead, {item['height_tiles']} tiles above the ground",
                "reachable": item["reachable"],
            }
        questions["target"] = {"type": "choice", "instructions": TARGET_QUESTION, "criteria": criteria}
    return questions


MIN_MOVE_CONFIDENCE = 0.35   # below this, fall back to the two Nouls
NOUL_YES = 0.5
HAZARD_MOVES = ("full_jump", "hop", "stomp", "wait", "back_up", "run_left", "up_over")
UNJUMPABLE = 5     # a wall or pipe higher than this cannot be jumped from the ground
RETREAT_FRAMES = 240   # how long run_left keeps going before the policy is asked again
HOP_LANDING = 4.7      # tiles a running hop covers before touching down (measured)
FULL_LANDING = 7.9     # tiles a running full jump covers
TALL_ENEMIES = ("green koopa", "red koopa", "jumping paratroopa", "flying paratroopa", "hammer bro")

# ---------------------------------------------------------------- moves -> buttons

BOOT_FRAMES = 200      # nothing is playable before this, on any SMB cart
DECISION_FRAMES = 6    # how long a ground move is held before asking again
STOMP_FAR = 1.3        # tiles: jump straight up when the enemy is this close; it passes underneath (measured)
STOMP_HOLD = 16        # A frames for that jump: ~3.4 tiles, back down in ~40 frames
HOP_FRAMES = 9         # A held this long gives a low jump
FULL_JUMP_FRAMES = 30  # A held through the whole rise


def move_to_masks(move: str) -> list[int]:
    if move == "hop":
        return [RIGHT | B | A] * HOP_FRAMES
    if move == "full_jump":
        return [RIGHT | B | A] * FULL_JUMP_FRAMES
    if move == "stomp":
        return [A] * 20 + [0] * 26
    if move == "up_over":
        return [A] * 18 + [A | RIGHT] * 14 + [RIGHT] * 22
    if move == "wait":
        return [0] * DECISION_FRAMES
    if move == "back_up":
        return [LEFT] * (DECISION_FRAMES * 2)
    if move == "run_left":
        return [LEFT | B] * (DECISION_FRAMES * 3)
    return [RIGHT | B] * DECISION_FRAMES


# ---------------------------------------------------------------- policies


@dataclass
class Decision:
    move: str
    confidence: float
    source: str                       # "jev", "jev-fallback", "jev-cached", "code" or "heuristic"
    detail: dict = field(default_factory=dict)
    latency_ms: float = 0.0           # wall time spent deciding (the API round trip for Jev)
    input_tokens: int = 0
    server_ms: float = 0.0            # the part of latency_ms Jev itself took; the rest is network
    target: str = "none"              # collectable id to go for, or "none"
    hazard_first: float = 0.0


class HeuristicPolicy:
    """Offline stand-in that reads the same scene. Lets you test the harness with no key."""

    name = "heuristic"
    model = "offline rules"

    def decide(self, scene: dict) -> Decision:
        started = time.perf_counter()
        decision = self._decide(scene)
        decision.latency_ms = (time.perf_counter() - started) * 1000
        return decision

    def _decide(self, scene: dict) -> Decision:
        terrain = scene["terrain_ahead"] if isinstance(scene["terrain_ahead"], list) else []
        enemies = scene["enemies"] if isinstance(scene["enemies"], list) else []
        for f in terrain:
            if f["distance_tiles"] > 2:
                continue
            if f["kind"] == "pit" or (f["kind"] in ("wall", "pipe") and f["height_tiles"] >= 2):
                return Decision("full_jump", 1.0, self.name, hazard_first=1.0)
            if f["kind"] == "wall":
                return Decision("hop", 1.0, self.name, hazard_first=1.0)
        for e in enemies:
            if e["direction"] == "ahead" and e["level"].startswith("same") and e["distance_tiles"] < 2.5:
                if e["type"] == "piranha plant":
                    move = "wait"
                elif scene["mario"]["speed"] == "stopped":
                    move = "stomp"           # a standing hop runs into the enemy; straight up clears it
                else:
                    move = "hop"
                return Decision(move, 1.0, self.name, hazard_first=1.0)
        items = scene["collectables"] if isinstance(scene["collectables"], list) else []
        for item in items:
            if item["reachable"].startswith("yes"):
                return Decision("run_right", 1.0, self.name, target=item["id"])
        return Decision("run_right", 1.0, self.name)


class JevPolicy:
    """Asks Jev. One request per decision, all questions in that one request."""

    name = "jev"

    def __init__(self, api_key: str | None = None, model: str = DEFAULT_MODEL, timeout: float = 10.0) -> None:
        self.api_key = api_key or load_api_key()
        if not self.api_key:
            raise RuntimeError(
                "TYPESAFE_API_KEY is not set. Create a key at https://console.typesafe.ai/keys "
                "and put TYPESAFE_API_KEY=... in the environment or in .env at the repo root."
            )
        self.model = model
        self.timeout = timeout
        self.calls = 0
        self.fallbacks = 0
        self.input_tokens = 0
        self.seconds = 0.0
        self.server_seconds = 0.0
        self.retries = 0
        self.cache_hits = 0
        self.model_seen = ""
        self._conn: http.client.HTTPSConnection | None = None
        # The scene is bucketed text, so consecutive frames often produce the very same
        # state. Same state, same answer: reuse it instead of paying another round trip.
        self._cache: dict[str, Decision] = {}

    def _post(self, body: dict) -> tuple[dict, float]:
        """POST on one kept-alive connection (a fresh TLS handshake per decision costs more
        than the model does). Returns the JSON and the server's own processing time in ms."""
        data = json.dumps(body)
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        delay = 0.5
        for _attempt in range(6):
            try:
                if self._conn is None:
                    self._conn = http.client.HTTPSConnection(API_HOST, timeout=self.timeout)
                self._conn.request("POST", API_PATH, body=data, headers=headers)
                resp = self._conn.getresponse()
                payload = resp.read()
            except (OSError, http.client.HTTPException):
                self._conn = None          # dropped or timed out: reconnect and retry
                self.retries += 1
                time.sleep(delay)
            else:
                if resp.status == 200:
                    return json.loads(payload), float(resp.getheader("x-envoy-upstream-service-time") or 0)
                # 429 rate limit, 529 overloaded, 5xx: back off and retry. Anything else is a bug.
                if resp.status not in (408, 429, 529) and resp.status < 500:
                    raise RuntimeError(f"TypeSafe API {resp.status}: {payload.decode(errors='replace')}")
                self.retries += 1
                retry_after = resp.getheader("retry-after")
                time.sleep(float(retry_after) if retry_after else delay)
            delay = min(delay * 2, 5.0)
        raise RuntimeError("TypeSafe API kept failing after 6 attempts")

    def decide(self, scene: dict) -> Decision:
        state = {k: v for k, v in scene.items() if not k.startswith("_")}
        key = json.dumps(state, sort_keys=True)
        hit = self._cache.get(key)
        if hit:
            self.cache_hits += 1
            return Decision(hit.move, hit.confidence, "jev-cached", hit.detail, target=hit.target,
                            hazard_first=hit.hazard_first)
        decision = self._ask(state)
        self._cache[key] = decision
        return decision

    def _ask(self, state: dict) -> Decision:
        started = time.perf_counter()
        resp, server_ms = self._post({"state": state, "model": self.model, "questions": build_questions(state)})
        latency_ms = (time.perf_counter() - started) * 1000
        self.server_seconds += server_ms / 1000
        tokens = resp.get("usage", {}).get("input_tokens", 0)
        self.seconds += latency_ms / 1000
        self.calls += 1
        self.input_tokens += tokens
        self.model_seen = resp.get("model", self.model_seen)

        answers = resp["answers"]
        move = answers["move"]
        detail = {
            "probabilities": move["probabilities"],
            "obstacle_needs_jump": answers["obstacle_needs_jump"]["noul"],
            "enemy_needs_jump": answers["enemy_needs_jump"]["noul"],
        }
        hazard = answers["hazard_first"]["noul"]
        target = "none"
        if "target" in answers:
            picked = answers["target"]
            target = picked["choice"]
            detail["target_probabilities"] = picked["probabilities"]
            detail["target_confidence"] = picked["confidence"]
        if move["confidence"] >= MIN_MOVE_CONFIDENCE:
            return Decision(move["choice"], move["confidence"], "jev", detail, latency_ms, tokens, server_ms,
                            target, hazard)

        # Uncertain pick: let the two narrow yes/no answers decide instead.
        self.fallbacks += 1
        if detail["obstacle_needs_jump"] > NOUL_YES:
            fallback = "full_jump"
        elif detail["enemy_needs_jump"] > NOUL_YES:
            fallback = "hop"
        else:
            fallback = "run_right"
        return Decision(fallback, move["confidence"], "jev-fallback", detail, latency_ms, tokens, server_ms,
                        target, hazard)

    def usage_line(self) -> str:
        if not self.calls:
            return "no Jev calls made"
        cost = self.input_tokens / 1e6 * PRICE_PER_MTOK
        return (f"{self.calls} Jev calls ({self.model_seen}), {self.cache_hits} cache hits, "
                f"{self.fallbacks} low-confidence fallbacks, {self.retries} retries, "
                f"{self.input_tokens:,} input tokens ~ ${cost:.4f}, "
                f"{self.seconds / self.calls * 1000:.0f} ms/call "
                f"(Jev itself {self.server_seconds / self.calls * 1000:.0f} ms, the rest is network)")


class LayaPolicy:
    """Same questions, answered locally by Laya (github.com/NandhaKishorM/laya), an
    open-weights typed-decision model with the same choice / score / noul contract.
    Needs the `laya` package (see python/README.md); the model loads on first use."""

    name = "laya"

    def __init__(self, model_id: str = "convaiinnovations/laya", device: str | None = None) -> None:
        import laya                                   # heavy: torch + transformers
        started = time.perf_counter()
        self.agent = laya.load(model_id, device=device or "cpu")
        self.model = f"{model_id} ({self.agent.device.type})"
        self.load_seconds = time.perf_counter() - started
        self.calls = 0
        self.seconds = 0.0

    def decide(self, scene: dict) -> Decision:
        state = {k: v for k, v in scene.items() if not k.startswith("_")}
        started = time.perf_counter()
        resp = self.agent.predict(state, build_questions(state))
        latency_ms = (time.perf_counter() - started) * 1000
        self.calls += 1
        self.seconds += latency_ms / 1000
        answers = resp["answers"]
        move = answers["move"]
        detail = {
            "probabilities": move["probabilities"],
            "obstacle_needs_jump": answers["obstacle_needs_jump"]["noul"],
            "enemy_needs_jump": answers["enemy_needs_jump"]["noul"],
        }
        target = "none"
        if "target" in answers:
            target = answers["target"]["choice"]
            detail["target_probabilities"] = answers["target"]["probabilities"]
            detail["target_confidence"] = answers["target"]["confidence"]
        return Decision(move["choice"], move["confidence"], "laya", detail, latency_ms, 0, latency_ms,
                        target, answers["hazard_first"]["noul"])

    def usage_line(self) -> str:
        if not self.calls:
            return "no Laya calls made"
        return f"{self.calls} Laya calls, {self.seconds / self.calls * 1000:.0f} ms/call (local, {self.model})"


class DuoPolicy:
    """Laya first, Jev when Laya is not sure: confidence-gated routing between the two.
    The local model answers the easy scenes for free; the API answers the rest."""

    name = "duo"

    def __init__(self, laya: LayaPolicy, jev: JevPolicy, min_confidence: float = 0.5) -> None:
        self.laya, self.jev, self.min_confidence = laya, jev, min_confidence
        self.model = f"laya -> jev at confidence < {min_confidence}"
        self.laya_kept = 0

    def decide(self, scene: dict) -> Decision:
        first = self.laya.decide(scene)
        if first.confidence >= self.min_confidence:
            self.laya_kept += 1
            return first
        second = self.jev.decide(scene)
        # Both answers are kept in the record: the log shows what Laya would have done.
        second.detail = {**second.detail, "laya_move": first.move, "laya_confidence": round(first.confidence, 3),
                         "laya_ms": round(first.latency_ms, 1)}
        second.latency_ms += first.latency_ms
        return second

    def usage_line(self) -> str:
        return (f"duo: Laya answered {self.laya_kept}, Jev the rest. {self.laya.usage_line()}; "
                f"{self.jev.usage_line()}")


def make_policy(name: str, model: str = DEFAULT_MODEL, laya_device: str | None = None):
    """The policy behind an --agent / --policy name."""
    if name == "jev":
        return JevPolicy(model=model)
    if name == "heuristic":
        return HeuristicPolicy()
    if name == "laya":
        return LayaPolicy(device=laya_device)
    if name == "duo":
        return DuoPolicy(LayaPolicy(device=laya_device), JevPolicy(model=model))
    raise ValueError(f"unknown policy {name!r}")


def load_api_key() -> str | None:
    key = os.environ.get("TYPESAFE_API_KEY")
    if key:
        return key
    here = Path(__file__).resolve()
    for folder in (Path.cwd(), *here.parents[:3]):
        env = folder / ".env"
        if env.is_file():
            for line in env.read_text().splitlines():
                name, _, value = line.strip().partition("=")
                if name.strip() == "TYPESAFE_API_KEY" and value.strip():
                    return value.strip().strip("\"'")
    return None


# ---------------------------------------------------------------- collecting
#
# Jev says *which* item to go for; the jump itself is geometry, so code does it. The
# numbers come from measuring the emulator: peak height in tiles for A held N frames,
# from a standing start (RIGHT released) and at walking speed.

STAND_JUMP = [(2, 1.6), (6, 2.2), (10, 2.8), (16, 3.4), (24, 4.0), (32, 4.1)]   # (hold, height)
WALK_JUMP = [(2, 1.8), (6, 2.4), (10, 2.9), (16, 3.6), (24, 4.2), (32, 4.4)]
WALK_PEAK_DX = {2: 1.0, 6: 1.25, 10: 1.4, 16: 2.0, 24: 2.5, 32: 3.0}            # tiles to the peak
UNDER_PX = 5           # |dx| this small puts Mario's centre inside the block
STOPPED = 4            # |vx| (1/16 px per frame) at or below this counts as standing still
WALK_SPEED = 26        # walking tops out at 24; above this Mario is running


def _hold_for(height: float, table: list[tuple[int, float]]) -> int | None:
    for hold, reach in table:
        if reach >= height + 0.25:
            return hold
    return None


def plan_climb(platform: int, dx: int, vx: int) -> list[int] | None:
    """Get on top of a ledge `platform` tiles up whose edge is `dx` px ahead, with a
    walking jump launched about three tiles before it (hold 32 peaks at 4.4 tiles)."""
    if platform > REACH_TILES:
        return None
    if abs(vx) > WALK_SPEED:
        return [0] * 4
    launch = 44
    if dx > launch + 16:
        return [RIGHT] * 4
    if dx < launch - 24:
        return [LEFT] * 3 if dx < 0 else [A] * 32 + [RIGHT] * 8   # too close: straight up, then step on
    return [RIGHT | A] * 32 + [RIGHT] * 6


SEEK_FRAMES = 120      # give up positioning under a block after this long


def seek_mask(dx: int, vx: int) -> int | None:
    """One frame of walking Mario to dx == 0 and stopping there. None when he is there."""
    if abs(dx) <= UNDER_PX and abs(vx) <= STOPPED:
        return None
    if abs(vx) > STOPPED and (abs(dx) <= UNDER_PX or (dx > 0) != (vx > 0) or abs(dx) < abs(vx)):
        return LEFT if vx > 0 else RIGHT     # brake: a tap the other way skids to a stop
    return RIGHT if dx > 0 else LEFT


# Measured jump arcs as (dx at peak, peak height, landing dx) in tiles; a parabola through
# those three points is close enough to say whether an arc clears a ledge and lands on it.
JUMP_ARCS = {
    "stand_full": (1.0, 4.1, 2.75),
    "walk_full": (3.06, 4.44, 5.19),
    "run_full": (4.06, 4.94, 7.94),
    "walk_hop": (1.25, 2.44, 2.88),
    "run_hop": (2.19, 2.81, 4.69),
}


def _arc_height(arc: tuple[float, float, float], dx: float) -> float:
    peak_dx, peak, land = arc
    if dx <= peak_dx:
        return peak * (1 - ((peak_dx - dx) / peak_dx) ** 2)
    if dx >= land:
        return -1.0
    return peak * (1 - ((dx - peak_dx) / (land - peak_dx)) ** 2)


def lands_on(arc: tuple[float, float, float], start: float, end: float, height: float) -> bool:
    """Does this arc clear the ledge's near edge and come down on top of it?"""
    edge = start - 0.5
    if _arc_height(arc, edge) < height + 0.15:
        return False
    # the arc must still be above the ledge somewhere on it and come down before its end
    for dx in (edge + 0.25 * k for k in range(1, 40)):
        if dx > end + 0.5:
            return False
        if _arc_height(arc, dx) < height:
            return dx >= edge
    return False


def plan_ledge_jump(ledge: dict, vx: int) -> str | None:
    """Which move lands on `ledge` from here at the current speed, or None."""
    start, height = ledge["distance_tiles"], ledge["height_tiles"]
    end = start + ledge["width_tiles"] - 1
    speed = "run" if abs(vx) > WALK_SPEED else "walk" if abs(vx) > STOPPED else "stand"
    order = {"run": ["run_full", "run_hop"], "walk": ["walk_full", "walk_hop"], "stand": ["stand_full"]}[speed]
    if height < 4:                        # a 4-high ledge leaves no margin for an arc
        for name in order:
            if lands_on(JUMP_ARCS[name], start, end, height):
                return "full_jump" if name.endswith("full") else "hop"
    if speed == "stand" and start <= 2 and height <= 4:
        # From the edge: straight up to full height, then drift over onto the ledge.
        return "up_over"
    return None


def plan_collect(item: dict, meta: dict) -> list[int] | None:
    """Masks that move toward `item` and jump for it when in position, or None if this
    item cannot be reached from the ground here. Called again after the masks run out, so
    each plan only needs to be the next short step."""
    dx = item["_dx_px"]
    height = item["height_tiles"]
    vx = meta["vx"]
    if not item["_reachable"]:
        return None
    if item.get("_platform"):
        return plan_climb(item["_platform"], dx, vx)
    if item["kind"] == "coin" and height <= 1:
        return [RIGHT] * 4                       # ground-level coin: just walk through it
    if item["kind"] != "coin":
        # Blocks are hit from directly underneath, from a standstill: a jump that
        # carries momentum drifts past the block before the head gets there.
        if dx > 56:
            return [RIGHT | B] * 4
        hold = _hold_for(height - 1, STAND_JUMP)
        if hold is None:
            return None
        # Positioning is frame-by-frame work, so hand it to Player.seek (see below).
        return [("seek", item["_col"], hold)]
    # A floating coin: walk to the launch point, then jump through it.
    if abs(vx) > WALK_SPEED:
        return [0] * 4                           # running: coast down to a walk first
    hold = _hold_for(height - 1, WALK_JUMP)   # the body only has to reach the coin's tile
    if hold is None:
        return None
    launch = WALK_PEAK_DX[hold] * 16
    if dx > launch + 24:
        return [RIGHT] * 4
    if dx > launch + 8:
        return [RIGHT] * 2
    if dx < launch - 12:
        return [LEFT] * 2 if dx < 0 else [A] * hold  # already close: jump straight up
    return [RIGHT | A] * hold + [RIGHT] * 4


# ---------------------------------------------------------------- the episode


class Player:
    """Drives one machine frame by frame, for as long as the game lasts: through the
    flagpole into the next level, through deaths into the next life. `next_mask()` is all
    a caller needs, so the same object serves the offline recorder and the live stream."""

    def __init__(self, nes: Nes, policy, log=None, max_decisions: int = 3000,
                 on_decision=None, on_life_end=None) -> None:
        self.nes = nes
        self.policy = policy
        self.log = log
        self.max_decisions = max_decisions
        self.on_decision = on_decision    # called with each decision record (a dict)
        self.on_life_end = on_life_end    # called with {"outcome", "distance", "world", "stage", "coins"}
        self.frame = 0
        self.queue: list[int] = []
        self.last_mask = 0
        self.ground_row: int | None = None
        self.lives: int | None = None
        self.level: tuple[int, int] | None = None
        self.decisions = 0
        self.max_x = 0
        self.stall = 0
        self.flagged = False
        self.seek: tuple[int, int, int] | None = None   # (block column, A frames, deadline)
        self.no_collect_until = 0     # after a seek is abandoned for an enemy, run the move instead
        self.retreat: tuple[int, int] | None = None     # (feet row when run_left began, deadline)
        self.pushed_at = -1           # x // 32 where Mario last tried walking into a tall wall
        self.enter_pipe: tuple[dict, int] | None = None   # (pipe top to go down, deadline)
        self.ledge_x, self.ledge_since, self.ledge_until = -1, 0, 0   # lining-up stall detection
        self.coins = 0
        self.outcome: str | None = None   # set only when the whole game is over
        self.lives_played = 0
        self.levels_cleared = 0

    def _boot_mask(self) -> int:
        # Tap Start until a level is running. This gets through a plain SMB title and
        # through the menu of the SMB / Duck Hunt / Track Meet multicart alike.
        return START if self.frame >= 60 and self.frame % 60 < 10 else 0

    def next_mask(self) -> int:
        """The buttons to hold for the coming frame. Call once per nes.step()."""
        mask = self._next_mask()
        self.frame += 1
        self.last_mask = mask
        return mask

    def _life_ended(self, outcome: str, meta: dict) -> None:
        self.lives_played += 1
        if self.on_life_end:
            self.on_life_end({
                "outcome": outcome, "distance": self.max_x,
                "world": meta["world"], "stage": meta["stage"], "coins": meta["coins"],
            })
        self.max_x, self.stall, self.queue, self.seek = 0, 0, [], None

    def _pipe_mask(self, scene: dict) -> int | None:
        """Frame-by-frame: get on top of the remembered pipe, then hold DOWN. None when done."""
        if not self.enter_pipe:
            return None
        target, deadline = self.enter_pipe
        meta = scene["_meta"]
        pipe = meta["pipe_top"]
        if self.frame > deadline or pipe is None or not meta["on_ground"]:
            if self.frame > deadline or pipe is None:
                self.enter_pipe = None
            return None if self.frame > deadline or pipe is None else 0
        dx, height = pipe["dx_px"], pipe["height"]
        if height == 0:                                   # standing on it
            # The game only takes DOWN with Mario's centre within a few px of the pipe's.
            if abs(meta["vx"]) > STOPPED:
                return LEFT if meta["vx"] > 0 else RIGHT
            if abs(dx) <= 3:
                self.queue = [DOWN] * 40
                self.enter_pipe = None
                return self.queue.pop(0)
            self.queue = [RIGHT if dx > 0 else LEFT] * (5 if abs(dx) > 8 else 3) + [0] * 12
            return self.queue.pop(0)
        if dx < 0:
            # The pipe is behind Mario: hop up onto it from this side, drifting left.
            if abs(meta["vx"]) > STOPPED:
                plan = [LEFT if meta["vx"] > 0 else RIGHT, 0]
            elif dx < -48:
                plan = [LEFT] * 4
            else:
                plan = [A | LEFT] * 30 + [LEFT] * 10 + [0] * 6
        else:
            plan = plan_climb(height, dx, meta["vx"])
        if plan is None:
            self.enter_pipe = None
            return None
        self.queue = plan
        return self.queue.pop(0)

    def _seek_mask(self, scene: dict) -> int | None:
        """Frame-by-frame positioning under a block, then the jump. None when not seeking."""
        if not self.seek:
            return None
        col, hold, deadline = self.seek
        meta = scene["_meta"]
        enemies = scene["enemies"] if isinstance(scene["enemies"], list) else []
        danger = any(e["direction"] == "ahead" and e["level"].startswith("same") and e["distance_tiles"] <= 3.5
                     for e in enemies)
        if self.frame > deadline or danger or not meta["on_ground"]:
            self.seek = None
            if danger:
                self.no_collect_until = self.frame + 90   # deal with the enemy first
            return None
        mask = seek_mask(col * 16 + 8 - (meta["x"] + 8), meta["vx"])
        if mask is not None:
            return mask
        self.seek = None
        # Hold still for the whole jump so Mario comes straight back down under the block
        # instead of drifting into whatever is walking up.
        self.queue = [A] * hold + [0] * (46 - hold)
        return self.queue.pop(0)

    def _next_mask(self) -> int:
        # A reset keeps work RAM, so OPER_MODE can still read 1 from the previous life
        # for the first frames; no level starts this early, so treat that as booting.
        if self.nes.peek(OPER_MODE) != 1 or self.frame < BOOT_FRAMES:
            if self.lives is not None and self.nes.peek(OPER_MODE) == 3:
                self.outcome = self.outcome or "game over"
            return self._boot_mask() if self.lives is None else 0

        scene = read_scene(self.nes, self.ground_row)
        meta = scene["_meta"]
        self.coins = meta["coins"]
        if self.lives is None:
            self.lives = meta["lives"]
        if meta["lives"] < self.lives:
            self.lives = meta["lives"]
            self._life_ended("died", meta)
            return 0
        level = (meta["world"], meta["stage"])
        if self.level is not None and level != self.level:
            if not self.flagged:
                # No flagpole on the way here: a pipe or a warp zone took Mario onward.
                self.levels_cleared += 1
                self._life_ended(f"warped to {level[0]}-{level[1]}", meta)
            self.max_x, self.stall, self.queue, self.flagged, self.seek, self.retreat = 0, 0, [], False, None, None
            self.enter_pipe = None
        self.level = level
        if meta["flagpole"] and not self.flagged and self.max_x > 0:
            self.flagged = True
            self.levels_cleared += 1
            self._life_ended(f"cleared {level[0]}-{level[1]}", meta)
            return 0
        if not meta["controllable"] or meta["dying"]:
            return 0

        if meta["x"] > self.max_x:
            self.max_x, self.stall = meta["x"], 0
        else:
            self.stall += 1
            if self.stall > 900:
                self.stall = 0
                self.queue = [RIGHT | B | A] * 24 + [RIGHT | B] * 30   # stuck: brute-force a run-jump

        if self.queue:
            return self.queue.pop(0)
        if self.retreat:
            start_row, deadline = self.retreat
            walls = [f for f in (scene["terrain_ahead"] if isinstance(scene["terrain_ahead"], list) else [])
                     if f["kind"] in ("wall", "pipe") and f.get("first_step_tiles", f["height_tiles"]) > UNJUMPABLE]
            if meta["on_ground"] and (meta["feet_row"] != start_row or not walls or self.frame > deadline):
                self.retreat = None
            else:
                return LEFT | B
        piping = self._pipe_mask(scene)
        if piping is not None:
            return piping
        seeking = self._seek_mask(scene)
        if seeking is not None:
            return seeking
        if not meta["on_ground"]:
            # Airborne with no plan: keep drifting right, unless that drifts onto an enemy.
            enemies_air = scene["enemies"] if isinstance(scene["enemies"], list) else []
            if any(e["direction"] == "ahead" and e["distance_tiles"] <= 3 and not e["level"].startswith("above")
                   for e in enemies_air):
                return LEFT               # air control: kill the forward momentum before landing
            return RIGHT | B

        self.ground_row = meta["feet_row"]
        if self.decisions >= self.max_decisions:
            self.outcome = "decision budget spent"
            return 0
        maneuver = self._enemy_maneuver(scene, meta)
        if maneuver is None and meta["piranha_out_ahead"] and (
                any(f["kind"] in ("pipe", "wall") and f["distance_tiles"] <= 3 for f in
                    (scene["terrain_ahead"] if isinstance(scene["terrain_ahead"], list) else []))):
            # Never jump onto a pipe while its plant is out: it dips back in every few seconds.
            maneuver = ("wait", "wait for the piranha plant to hide", [LEFT if meta["vx"] > STOPPED else 0] * 3)
        if maneuver is not None:
            # Enemies on flat ground are handled by code, no API call: stop, let them walk
            # up, and jump straight up so they pass underneath. A jump over an enemy lands
            # on whatever walks behind it; this does not.
            move, action, self.queue = maneuver
            decision = Decision(move, 1.0, "code", {"enemy_maneuver": True})
        elif scene["summary"] == "nothing ahead":
            # Use code when you can: an empty scene needs no judgement, so no API call.
            decision = Decision("run_right", 1.0, "code")
        else:
            decision = self.policy.decide(scene)
        self.decisions += 1
        if maneuver is not None:
            self._record(decision, scene, meta, action)
            if self.queue[0] & A and self.last_mask & A:
                self.queue.insert(0, 0)
            return self.queue.pop(0)

        # A jump launched three or more tiles early is wasted (and lands who knows where),
        # so a jump move only fires once its reason is within reach. Use code when you can.
        terrain_now = scene["terrain_ahead"] if isinstance(scene["terrain_ahead"], list) else []
        enemies_now = scene["enemies"] if isinstance(scene["enemies"], list) else []
        # The open end of a sideways pipe is a doorway: walk in, and never jump at it
        # (a jump lands on top of it, where there is nowhere to go).
        if any(f["kind"] == "side_pipe" and f["distance_tiles"] <= 4 for f in terrain_now):
            self.queue = [RIGHT | B] * 12
            decision.move = "run_right"
            decision.detail = {**decision.detail, "side_pipe": True}
            self._record(decision, scene, meta, "walk into the sideways pipe")
            return self.queue.pop(0)

        # A wall no jump clears: the route is behind Mario (off the ledge, or under the
        # platform). Repeating full_jump into it is how he hangs; go left instead.
        blocked = [f for f in terrain_now if f["kind"] in ("wall", "pipe")
                   and f.get("first_step_tiles", f["height_tiles"]) > UNJUMPABLE and f["distance_tiles"] <= 2]
        if blocked:
            # A sideways pipe reads as a tall wall too, and the way through it is to walk
            # in. Push into it once; a real wall does not give, and then it's left.
            if self.pushed_at != meta["x"] // 32:
                self.pushed_at = meta["x"] // 32
                self.queue = [RIGHT] * 40
                decision.move = "run_right"
                decision.detail = {**decision.detail, "push_into_wall": True}
                self._record(decision, scene, meta, "walk into it (a side pipe?)")
                return self.queue.pop(0)
            if meta["pipe_top"] and 0 < meta["pipe_top"]["height"] <= REACH_TILES:
                # A dead end with a pipe beside it: the way on is down the pipe.
                self.enter_pipe = (meta["pipe_top"], self.frame + 600)
                decision.move = "enter_pipe"
                decision.detail = {**decision.detail, "dead_end": True}
                self._record(decision, scene, meta, "dead end: get onto the pipe and go down it")
                return self._pipe_mask(scene) or 0
            decision.move = "run_left"
            decision.detail = {**decision.detail, "unjumpable_wall": True}
        if decision.move == "run_left":
            # Keep going left until Mario is on a different level (dropped off the ledge)
            # or the wall is well out of view; a single burst just turns him back into it.
            self.retreat = (meta["feet_row"], self.frame + RETREAT_FRAMES)

        # Obstacles stand still, so a jump 3+ tiles early lands short; enemies walk toward
        # Mario, so a running jump from up to 5 tiles away clears them (and a koopa, being
        # tall, needs that head start).
        obstacles = [f["distance_tiles"] for f in terrain_now if f["kind"] in ("pit", "wall", "pipe")
                     and not (f["kind"] != "pit" and f.get("first_step_tiles", f["height_tiles"]) > UNJUMPABLE)]
        foes = [e["distance_tiles"] for e in enemies_now
                if e["direction"] == "ahead" and e["level"].startswith("same")]
        if decision.move == "stomp":
            too_early = not foes or min(foes) > 1.9
        else:
            too_early = not ((obstacles and min(obstacles) < 3.0) or (foes and min(foes) < 5.0))
        # Where does the jump land? A running hop comes down ~4.7 tiles on, a full jump
        # ~7.9. If that spot is inside a pit, take the other jump when it will do.
        pit_spans = [(f["distance_tiles"], f["distance_tiles"] + f["width_tiles"]) for f in terrain_now
                     if f["kind"] == "pit"]
        def lands_in_pit(move: str) -> bool:
            land = HOP_LANDING if move == "hop" else FULL_LANDING
            return any(start - 0.5 <= land <= end + 0.5 for start, end in pit_spans)
        low_obstacle = all(f.get("first_step_tiles", 9) <= 2 for f in terrain_now
                           if f["kind"] in ("wall", "pipe") and f["distance_tiles"] <= 3)
        if decision.move == "full_jump" and lands_in_pit("full_jump") and low_obstacle and not lands_in_pit("hop"):
            decision.move = "hop"
            decision.detail = {**decision.detail, "landing_adjusted": "hop"}
        elif decision.move == "hop" and lands_in_pit("hop") and not lands_in_pit("full_jump"):
            decision.move = "full_jump"
            decision.detail = {**decision.detail, "landing_adjusted": "full_jump"}

        # A pit with a ledge above beyond it (tree tops): pick the jump whose arc clears
        # the ledge's edge and comes down on it, from the measured jump profiles. If none
        # works from here at this speed, slow down or step closer first.
        ledges = [f for f in terrain_now if f["kind"] == "ledge above"]
        first_pit = min((start for start, _ in pit_spans), default=99)
        step_before_pit = any(f["kind"] in ("wall", "pipe") and f["distance_tiles"] < first_pit for f in terrain_now)
        if (decision.move in ("full_jump", "hop", "run_right") and pit_spans and ledges
                and not step_before_pit and self.frame >= self.ledge_until):
            target = min((l for l in ledges if l["distance_tiles"] >= first_pit),
                         key=lambda l: l["distance_tiles"], default=None)
            if target is not None:
                choice = plan_ledge_jump(target, meta["vx"])
                if choice is None:
                    # Nothing lands from here: walk up to the edge (or brake) and look again,
                    # but not forever: if lining up stalls, hand the pit back to the policy.
                    if meta["x"] != self.ledge_x:
                        self.ledge_x, self.ledge_since = meta["x"], self.frame
                    elif self.frame - self.ledge_since > 60:
                        self.ledge_until = self.frame + 300
                    if abs(meta["vx"]) > STOPPED:
                        self.queue = [LEFT if meta["vx"] > 0 else RIGHT, 0]
                    else:
                        self.queue = [RIGHT] * (2 if first_pit <= 1 else 4) + [0] * 4
                    decision.move = "wait"
                    decision.detail = {**decision.detail, "ledge_approach": target["distance_tiles"]}
                    self._record(decision, scene, meta, f"line up for the ledge {target['height_tiles']} up")
                    return self.queue.pop(0)
                decision.move = choice
                decision.detail = {**decision.detail, "ledge_jump": choice}
                too_early = False

        # Under a low ceiling a jump bonks and drops short: hold the pit jump until the
        # overhang ends or the edge is right here, and make it a hop (A cut short bonks less).
        pits = [f["distance_tiles"] for f in terrain_now if f["kind"] == "pit"]
        if decision.move in ("full_jump", "hop") and pits and meta["headroom"] < 4:
            if min(pits) > 1:
                too_early = True
            else:
                decision.move = "hop"
                decision.detail = {**decision.detail, "low_ceiling_hop": True}
        if decision.move in ("full_jump", "hop", "stomp") and (obstacles or foes) and too_early:
            decision = Decision("run_right", decision.confidence, decision.source, decision.detail,
                                decision.latency_ms, decision.input_tokens, decision.server_ms,
                                decision.target, decision.hazard_first)
            decision.detail = {**decision.detail, "early_jump_held": True}

        # A hop clears one goomba and lands on the next, and never clears a koopa (a tile
        # and a half tall): both take a full jump.
        in_row = [e for e in enemies_now if e["direction"] == "ahead" and e["level"].startswith("same")
                  and e["distance_tiles"] <= 7]
        tall = any(e["type"] in TALL_ENEMIES for e in in_row)
        if decision.move == "hop" and (len(in_row) >= 2 or tall):
            decision.move = "full_jump"
            decision.detail = {**decision.detail, "hop_upgraded": True}

        # Hazards first; otherwise go for the item Jev picked; otherwise the move.
        plan = None
        item = next((i for i in meta["items"] if i["id"] == decision.target), None)
        terrain = scene["terrain_ahead"] if isinstance(scene["terrain_ahead"], list) else []
        enemies = scene["enemies"] if isinstance(scene["enemies"], list) else []
        pit_ahead = any(f["kind"] == "pit" and f["distance_tiles"] <= 2 for f in terrain)
        # Stopping under a block with an enemy walking up is how Mario dies, so let the
        # move deal with anything at his height first; the block is still there afterwards.
        enemy_near = any(e["level"].startswith("same") and (e["direction"] == "ahead" or e["distance_tiles"] <= 4)
                         for e in enemies)
        hazard = decision.move in HAZARD_MOVES and decision.hazard_first > NOUL_YES
        if item and not hazard and not pit_ahead and not enemy_near and self.frame >= self.no_collect_until:
            plan = plan_collect(item, meta)
        action = f"collect {item['kind']} ({item['distance']}, {item['height_tiles']} up)" \
            if plan else decision.move
        self._record(decision, scene, meta, action)
        if plan and isinstance(plan[0], tuple):
            _, col, hold = plan[0]
            self.seek = (col, hold, self.frame + SEEK_FRAMES)
            return self._seek_mask(scene) or 0
        self.queue = plan or move_to_masks(decision.move)
        # A jump only registers on a fresh press, so let go of A for a frame first.
        if self.queue[0] & A and self.last_mask & A:
            self.queue.insert(0, self.queue[0] & ~A)
        return self.queue.pop(0)

    def _enemy_maneuver(self, scene: dict, meta: dict) -> tuple[str, str, list[int]] | None:
        """(move, action, masks) for an enemy walking up at Mario's height, else None."""
        enemies = scene["enemies"] if isinstance(scene["enemies"], list) else []
        terrain = scene["terrain_ahead"] if isinstance(scene["terrain_ahead"], list) else []
        foes = [e["distance_tiles"] for e in enemies
                if e["direction"] == "ahead" and e["level"].startswith("same")
                and e["type"] != "piranha plant"]          # plants sit in pipes: the policy's `wait`
        behind = [e["distance_tiles"] for e in enemies
                  if e["direction"] == "behind" and e["level"].startswith("same")
                  and e["type"] != "piranha plant"]
        vx = meta["vx"]
        if behind and min(behind) <= 2.0 and not (foes and min(foes) <= 5.0):
            # Something walking up from behind, nothing ahead: stop and let it pass under.
            if abs(vx) > STOPPED:
                return "wait", "brake, enemy behind", [LEFT if vx > 0 else RIGHT] * 2
            if min(behind) <= STOMP_FAR:
                return "stomp", "jump straight up, it passes underneath", [A] * STOMP_HOLD + [0] * 44
            return "wait", "wait for the enemy behind", [0] * 3
        if not foes or min(foes) > 5.0:
            return None
        if any(f["kind"] in ("wall", "pipe") and f["distance_tiles"] <= 1 for f in terrain):
            return None                                   # against a wall: the policy's jump first
        if any(f["kind"] == "pit" and f["distance_tiles"] <= 2 for f in terrain):
            # An enemy on the far side of a pit: the jump would land on it. It cannot
            # cross the pit either, so hold here until it has wandered off.
            if min(foes) <= STOMP_FAR and abs(meta["vx"]) <= STOPPED:
                return "stomp", "jump straight up, it passes underneath", [A] * STOMP_HOLD + [0] * 44
            if min(foes) <= 5.0:
                return "wait", "wait for the enemy across the pit to move off", [LEFT if meta["vx"] > STOPPED else 0] * 3
            return None                                   # a pit this close: jumping is the policy's call
        foe = min(foes)
        clear_behind = not (behind and min(behind) < 6)   # room to back away
        running = abs(vx) > WALK_SPEED
        if meta["headroom_ahead"] < 2 and clear_behind:
            # An overhang here or just ahead leaves no room to jump. The enemy is ahead, so
            # the open ground is behind: stop, back out the way Mario came, meet it there.
            if vx > STOPPED:
                return "wait", "low ceiling ahead, brake", [LEFT] * 3
            return "run_left", "low ceiling ahead, back out to open ground", [LEFT | B] * 3
        if meta["headroom"] < 2:
            return "run_right", "low ceiling, run out from under it", [RIGHT | B] * 3
        if meta["headroom"] < 4 and running:
            # Not enough room for a full jump: a hop is all there is, so hop when close.
            if foe <= 2.99:
                return "hop", "low ceiling, running hop over it", move_to_masks("hop")
            return "run_right", "keep running, hop when close", [RIGHT | B] * 3
        if running and foe <= 4.0:
            # No room to stop (a skid from a run slides two tiles): a running hop clears
            # a single enemy reliably, a running full jump a tall one or a pair.
            # A running hop lands ~4.7 tiles on, a full jump ~8: with a second enemy in
            # the line, pick whichever lands in a gap rather than on it.
            second = sorted(foes)[1] if len(foes) >= 2 else None
            tall = any(e["type"] in TALL_ENEMIES for e in enemies
                       if e["direction"] == "ahead" and e["level"].startswith("same"))
            if tall:
                # A koopa is a tile and a half tall: the full jump has to start early.
                return "full_jump", "running full jump over the koopa", move_to_masks("full_jump")
            if foe <= 2.99:
                move = "full_jump" if second is not None and second <= 5.5 else "hop"
                return move, f"running {move.replace('_', ' ')} over it", move_to_masks(move)
            return "run_right", "keep running, hop when close", [RIGHT | B] * 3
        if abs(vx) > STOPPED:
            if not running and foe <= 1.5:
                return "hop", "walking hop over it", move_to_masks("hop")
            return "wait", "brake for the enemy", [LEFT if vx > 0 else RIGHT] * 2
        if meta["ceiling"] and clear_behind:
            # A block overhead bonks the jump short, and a short jump lands on the enemy:
            # get clear of it first, quickly when the enemy is already close.
            if foe > 2.2:
                return "back_up", "back out from under the block", [LEFT] * 4 + [0] * 2
            return "run_left", "run out from under the block", [LEFT | B] * 3
        if foe <= STOMP_FAR or (behind and min(behind) <= STOMP_FAR):
            # Held still for the whole flight so Mario comes straight back down. An enemy
            # coming up from behind gets the same treatment.
            return "stomp", "jump straight up, it passes underneath", [A] * STOMP_HOLD + [0] * 44
        return "wait", "wait for the enemy", [0] * 3

    def _record(self, decision: Decision, scene: dict, meta: dict, action: str) -> None:
        record = {
            "n": self.decisions, "frame": self.frame, "x": meta["x"], "summary": scene["summary"],
            "move": decision.move, "action": action, "target": decision.target,
            "hazard_first": round(decision.hazard_first, 3),
            "confidence": round(decision.confidence, 3),
            "source": decision.source, "latency_ms": round(decision.latency_ms, 2),
            "server_ms": round(decision.server_ms, 2),
            "input_tokens": decision.input_tokens,
            "world": meta["world"], "stage": meta["stage"], "coins": meta["coins"], "lives": meta["lives"],
            **decision.detail,
        }
        if self.log:
            self.log.write(json.dumps(record) + "\n")
        if self.on_decision:
            self.on_decision(record)


def play_episode(rom: bytes, policy, movie_path: str | None = None, log_path: str | None = None,
                 max_frames: int = 60_000, max_decisions: int = 3000) -> dict:
    """Play until game over (or a budget runs out) and optionally save a .nesmovie of it."""
    nes = Nes()
    nes.load(rom)
    history = bytearray()
    log = open(log_path, "w") if log_path else None
    lives: list[dict] = []
    player = Player(nes, policy, log, max_decisions, on_life_end=lives.append)
    tail = 0
    try:
        while player.frame < max_frames and tail < 120:   # 2s of footage after the end
            mask = player.next_mask()
            nes.step(mask)
            history.append(mask)
            if player.outcome:
                tail += 1
    finally:
        if log:
            log.close()
    if movie_path:
        write_movie(movie_path, rom, bytes(history))
    return {
        "outcome": player.outcome or "frame budget spent",
        "lives": lives,
        "levels_cleared": player.levels_cleared,
        "coins": player.coins,
        "frames": player.frame,
        "decisions": player.decisions,
    }
