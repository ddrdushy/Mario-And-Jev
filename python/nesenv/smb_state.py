"""Turn Super Mario Bros RAM into a small, *semantic* description of the scene.

This is the text a language-style decision model reads instead of pixels. Everything
numeric (pixel maths, tile lookups, distance buckets) is done here in code, so the model
is only asked for the judgement: given what is ahead, what should Mario do?

    from nesenv.smb_state import read_scene
    scene = read_scene(nes)     # -> plain dict, JSON-serialisable
"""

from __future__ import annotations

from .core import Nes

# Player.
PLAYER_X = 0x0086
PLAYER_PAGE = 0x006D
PLAYER_Y = 0x00CE         # top of the 32px box; small Mario occupies the lower half
PLAYER_Y_PAGE = 0x00B5    # 1 while on screen, >1 once he has fallen below it
PLAYER_STATE = 0x000E     # 8 = normal control
PLAYER_FLOAT = 0x001D     # 0 ground, 1 jumping, 2 falling, 3 sliding down the flagpole
PLAYER_X_SPEED = 0x0057   # signed
PLAYER_Y_SPEED = 0x009F   # signed, negative = rising
POWERUP = 0x0756          # 0 small, 1 big, 2 fire
OPER_MODE = 0x0770
OPER_TASK = 0x0772        # 3 while a level is actually being played
LIVES = 0x075A
WORLD = 0x075F
STAGE = 0x075C
COINS = 0x075E

# Enemy slots (five enemies plus the power-up slot).
ENEMY_ACTIVE = 0x000F
ENEMY_TYPE = 0x0016
ENEMY_STATE = 0x001E
ENEMY_PAGE = 0x006E
ENEMY_X = 0x0087
ENEMY_Y = 0x00CF
ENEMY_SLOTS = 6

# The level's tile buffer: two 16x13 pages, reused as the screen scrolls.
TILES = 0x0500
TILE_ROWS = 13
PAGE_TILES = 16 * TILE_ROWS

ENEMY_NAMES = {
    0x00: "green koopa", 0x01: "red koopa", 0x02: "buzzy beetle", 0x03: "red koopa",
    0x05: "hammer bro", 0x06: "goomba", 0x07: "blooper", 0x08: "bullet bill",
    0x0A: "cheep cheep", 0x0B: "cheep cheep", 0x0D: "piranha plant",
    0x0E: "jumping paratroopa", 0x0F: "flying paratroopa", 0x10: "flying paratroopa",
    0x11: "lakitu", 0x12: "spiny", 0x14: "flying cheep cheep", 0x15: "bowser fire",
    0x2D: "bowser",
}
POWERUP_TYPE = 0x2E
PLATFORM_TYPES = range(0x24, 0x2D)

EMPTY_TILES = {0x00, 0xC2, 0xC3}      # air and coins
FLAG_TILES = {0x24, 0x25}             # flagpole: touchable, not solid
PIPE_TILES = range(0x10, 0x22)

# Things worth going out of the way for (metatile ids, checked against 1-1's buffer).
COIN_TILES = {0xC2, 0xC3}
ITEM_TILES = {
    0xC0: ("question block", "a ? block with a coin inside"),
    0xC1: ("question block", "a ? block with a power-up inside"),
    0x58: ("coin brick", "a brick that pays out many coins when hit repeatedly"),
    0x55: ("item brick", "a brick with an item inside"),
    0x57: ("item brick", "a brick with an item inside"),
    0x59: ("item brick", "a brick with an item inside"),
    0x5A: ("item brick", "a brick with an item inside"),
    0x5B: ("item brick", "a brick with an item inside"),
}
MAX_ITEMS = 4        # nearest collectables offered to the model per decision
REACH_TILES = 4      # a standing/walking jump tops out just above 4 tiles

LOOKAHEAD = 8  # columns described in front of Mario


def _signed(b: int) -> int:
    return b - 256 if b > 127 else b


def _tile(ram: bytes, level_x: int, row: int) -> int:
    if not 0 <= row < TILE_ROWS:
        return 0
    page = (level_x // 256) % 2
    col = (level_x % 256) // 16
    return ram[TILES + page * PAGE_TILES + row * 16 + col]


def _solid(t: int) -> bool:
    return t not in EMPTY_TILES and t not in FLAG_TILES


def _distance_words(tiles: float) -> str:
    if tiles < 1.0:
        return "touching"
    if tiles < 2.0:
        return "very close (1 tile)"
    if tiles < 3.0:
        return "close (2 tiles)"
    if tiles < 5.0:
        return "near (3-4 tiles)"
    return "far (5+ tiles)"


def _column(ram: bytes, level_x: int, feet_row: int, body_rows: int) -> dict:
    """Describe one 16px column relative to the row Mario stands on."""
    blocked = any(_solid(_tile(ram, level_x, feet_row - 1 - i)) for i in range(body_rows))
    if blocked:
        top = feet_row - 1
        while top - 1 >= 0 and _solid(_tile(ram, level_x, top - 1)):
            top -= 1
        # a low ceiling with a gap under it is still a wall to a running Mario
        while top > 0 and not _solid(_tile(ram, level_x, top)):
            top -= 1
        kind = "pipe" if _tile(ram, level_x, top) in PIPE_TILES else "wall"
        return {"kind": kind, "height": feet_row - top}
    if any(_tile(ram, level_x, r) in FLAG_TILES for r in range(TILE_ROWS)):
        return {"kind": "flagpole", "height": 0}
    for row in range(feet_row, TILE_ROWS):
        if _solid(_tile(ram, level_x, row)):
            drop = row - feet_row
            return {"kind": "ground" if drop == 0 else "drop", "height": -drop}
    return {"kind": "pit", "height": 0}


def _terrain(ram: bytes, mario_x: int, feet_row: int, body_rows: int) -> tuple[list[dict], list[str]]:
    """Run-length the columns ahead into features, plus a words-only summary."""
    cols = [_column(ram, mario_x + 16 * i, feet_row, body_rows) for i in range(1, LOOKAHEAD + 1)]
    features: list[dict] = []
    for i, col in enumerate(cols, start=1):
        if col["kind"] == "ground":
            continue
        last = features[-1] if features else None
        if last and last["kind"] == col["kind"] and last["_end"] == i - 1:
            last["_end"] = i
            last["width_tiles"] += 1
            last["height_tiles"] = max(last["height_tiles"], col["height"])
            continue
        features.append({
            "kind": col["kind"],
            "distance": _distance_words(i),
            "distance_tiles": i,
            "width_tiles": 1,
            "height_tiles": col["height"],
            "_end": i,
        })
    words = []
    for f in features:
        del f["_end"]
        if f["kind"] == "pit":
            f.pop("height_tiles")
            words.append(f"pit {f['width_tiles']} tiles wide, {f['distance']} ahead")
        elif f["kind"] == "drop":
            f["height_tiles"] = -f["height_tiles"]
            words.append(f"ledge drops {f['height_tiles']} tiles, {f['distance']} ahead")
        elif f["kind"] == "flagpole":
            f.pop("height_tiles")
            words.append(f"goal flagpole {f['distance']} ahead")
        else:
            words.append(f"{f['kind']} {f['height_tiles']} tiles high, {f['distance']} ahead")
    return features, words


def _enemies(ram: bytes, mario_x: int, mario_y: int) -> tuple[list[dict], dict | None]:
    hostile: list[dict] = []
    powerup = None
    for i in range(ENEMY_SLOTS):
        if not ram[ENEMY_ACTIVE + i]:
            continue
        kind = ram[ENEMY_TYPE + i]
        ex = ram[ENEMY_PAGE + i] * 256 + ram[ENEMY_X + i]
        ey = ram[ENEMY_Y + i]
        dx, dy = ex - mario_x, ey - mario_y
        if abs(dx) > 16 * LOOKAHEAD:
            continue
        where = "ahead" if dx >= 0 else "behind"
        if dy < -20:
            level = "above Mario"
        elif dy > 20:
            level = "below Mario"
        else:
            level = "same height as Mario"
        entry = {
            "direction": where,
            "distance": _distance_words(abs(dx) / 16),
            "distance_tiles": round(abs(dx) / 16, 1),
            "level": level,
        }
        if kind == POWERUP_TYPE:
            powerup = entry
        elif kind in ENEMY_NAMES:
            if ram[ENEMY_STATE + i] & 0x20:   # already stomped / kicked away
                continue
            hostile.append({"type": ENEMY_NAMES[kind], **entry})
    hostile.sort(key=lambda e: e["distance_tiles"])
    return hostile, powerup


def _items(ram: bytes, mario_x: int, feet_row: int) -> list[dict]:
    """Coins and item blocks from Mario's own column to LOOKAHEAD tiles ahead."""
    found: list[dict] = []
    centre = mario_x + 8
    first_col = max(0, centre // 16 - 6)   # a few columns behind too: worth walking back for
    for col in range(first_col, centre // 16 + LOOKAHEAD + 1):
        for row in range(TILE_ROWS):
            t = _tile(ram, col * 16, row)
            if t in COIN_TILES:
                kind, what = "coin", "a coin"
            elif t in ITEM_TILES:
                kind, what = ITEM_TILES[t]
            else:
                continue
            dx_px = col * 16 + 8 - centre
            height = feet_row - row           # tiles above the ground Mario stands on
            if height < 0:
                continue                       # below Mario: a pit-floor coin, ignore
            # A block is hit from underneath, so the head must reach its bottom edge; a
            # coin only needs Mario's body to pass through it.
            needed = height - 1
            reachable = needed <= REACH_TILES
            # Too high from the ground? Maybe there is something to stand on under it
            # (the classic ? block above a row of bricks). Code will climb onto that first.
            platform = None
            if not reachable:
                for prow in range(row + 1, feet_row):
                    if _solid(_tile(ram, col * 16, prow)):
                        platform = feet_row - prow       # its top, in tiles above the ground
                        break
                if platform is not None and platform <= REACH_TILES and (platform - height) >= -REACH_TILES - 1:
                    reachable = True
                    what += f", above a ledge {platform} tiles up that Mario can stand on"
            found.append({
                "kind": kind,
                "what": what,
                "distance": _distance_words(abs(dx_px) / 16) + (" behind" if dx_px < -8 else ""),
                "distance_tiles": round(dx_px / 16, 1),
                "height_tiles": height,
                "reachable": "yes, with one jump from the ground" if reachable
                             else "no, too high from here; needs something to stand on",
                "_dx_px": dx_px, "_row": row, "_col": col, "_reachable": reachable,
                "_platform": platform,
            })
    found.sort(key=lambda i: (abs(i["_dx_px"]), i["height_tiles"]))
    return found[:MAX_ITEMS]


def read_scene(nes: Nes, ground_row: int | None = None) -> dict:
    """The current scene as a JSON-serialisable dict.

    `ground_row` is the row Mario last stood on; pass it while he is airborne so the
    terrain ahead is still described relative to the ground he jumped from.
    """
    ram = nes.ram()
    x = ram[PLAYER_PAGE] * 256 + ram[PLAYER_X]
    y = ram[PLAYER_Y]
    big = ram[POWERUP] > 0
    float_state = ram[PLAYER_FLOAT]
    on_ground = float_state == 0
    feet_row = y // 16 if on_ground or ground_row is None else ground_row
    vx = _signed(ram[PLAYER_X_SPEED])
    vy = _signed(ram[PLAYER_Y_SPEED])

    if on_ground:
        motion = "standing on the ground"
    elif float_state == 3:
        motion = "sliding down the flagpole"
    elif vy < 0:
        motion = "in the air, rising"
    else:
        motion = "in the air, falling"
    speed = "stopped" if abs(vx) < 4 else "walking" if abs(vx) < 28 else "running"

    features, words = _terrain(ram, x + 8, feet_row, 2 if big else 1)
    hostile, powerup = _enemies(ram, x, y)
    items = _items(ram, x, feet_row)
    for n, item in enumerate(items, start=1):
        item["id"] = f"item_{n}"

    scene = {
        "mario": {
            "size": "big" if big else "small",
            "motion": motion,
            "speed": speed,
            "facing_goal": "the goal is to the right",
        },
        "terrain_ahead": features or "flat open ground for the next 8 tiles",
        "enemies": hostile or "none nearby",
    }
    if powerup:
        scene["power_up_item"] = powerup
    scene["collectables"] = [
        {k: v for k, v in item.items() if not k.startswith("_")} for item in items
    ] or "none in view"
    scene["summary"] = "; ".join(
        words
        + [f"{e['type']} {e['distance']} {e['direction']}, {e['level']}" for e in hostile]
        + [f"{i['kind']} {i['distance']} ahead, {i['height_tiles']} tiles up" for i in items]
    ) or "nothing ahead"
    # Bookkeeping for the runner; not sent to the model.
    # A block directly overhead cuts a vertical jump short (head bonk), which matters
    # when Mario wants to jump an enemy from a standstill.
    ceiling = any(_solid(_tile(ram, x + 8, feet_row - h)) for h in range(2, 6))

    def free_above(px: int) -> int:
        n = 0
        for h in range(2 if not big else 3, feet_row + 1):
            if _solid(_tile(ram, px, feet_row - h)):
                break
            n += 1
        return n
    headroom = free_above(x + 8)   # free tiles above Mario's head in his own column
    scene["_meta"] = {
        "x": x, "y": y, "feet_row": feet_row, "on_ground": on_ground, "ceiling": ceiling,
        "headroom": headroom,
        "controllable": ram[PLAYER_STATE] == 8 and ram[OPER_MODE] == 1 and ram[OPER_TASK] == 3,
        "dying": ram[PLAYER_STATE] in (6, 0x0B) or ram[PLAYER_Y_PAGE] > 1,
        "flagpole": ram[OPER_TASK] == 3 and (float_state == 3 or ram[PLAYER_STATE] in (4, 5)),
        "game_over": ram[OPER_MODE] == 3,
        "lives": ram[LIVES], "world": ram[WORLD] + 1, "stage": ram[STAGE] + 1,
        "coins": ram[COINS], "vx": vx,
        "items": items,
    }
    return scene
