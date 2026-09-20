"""Agentick hard spike (frozen AGENTICK_HARD.md; SokobanPush only).

Fixed adapter (identical rule, all modes): parse obs -> egocentric
(facing, box dir/dist, target dir/dist, wall-ahead) -> frozen push
priority (push-if-aligned / reposition / approach / fixed fallback).
Reads info['valid_actions'] ONLY. Modes ascii/language/state_dict.
SokobanPush-v0 easy/dense. Consumers: adapter (tested), random floor
(descriptive), oracle (normalization only). Seed rule: start 0-4, replace
with next integers until 5 oracle-solved seeds (recorded).
Writes theory/agentick_hard_raw.json (+ verdict appended by runner).
Run under /tmp/agentick-spike (py3.12); env never committed.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

OUT = Path("/Users/sanamimani/paper2_semantic_decision_benchmark/results/v3main/theory")
SEEDS0 = [0, 1, 2, 3, 4]
MODES = ["ascii", "language", "state_dict"]
TASK, DIFF, REWARD = "SokobanPush-v0", "easy", "dense"

import agentick  # noqa: E402

ANAME = ["noop", "move_up", "move_down", "move_left", "move_right", "interact"]
AIDX = {n: i for i, n in enumerate(ANAME)}
EGO = {
    "north": {"ahead": "move_up", "left": "move_left", "right": "move_right",
              "behind": "move_left", "here": "interact"},
    "south": {"ahead": "move_down", "left": "move_right", "right": "move_left",
              "behind": "move_left", "here": "interact"},
    "east": {"ahead": "move_right", "left": "move_up", "right": "move_down",
             "behind": "move_left", "here": "interact"},
    "west": {"ahead": "move_left", "left": "move_down", "right": "move_up",
             "behind": "move_left", "here": "interact"},
}
FALLBACK = ["move_up", "move_right", "move_down", "move_left", "interact", "noop"]
ADIR = {"north": (0, -1), "east": (1, 0), "south": (0, 1), "west": (-1, 0)}
RDIR = {"north": (1, 0), "east": (0, 1), "south": (-1, 0), "west": (0, -1)}
ACHAR = {"^": "north", "v": "south", ">": "east", "<": "west"}


def strip_ansi(s: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", s)


def ego_of(dx: int, dy: int, facing: str):
    if (dx, dy) == (0, 0):
        return "here", 0
    ax, ay = ADIR[facing]
    rx, ry = RDIR[facing]
    fwd, lat = dx * ax + dy * ay, dx * rx + dy * ry
    dist = abs(dx) + abs(dy)
    if fwd > 0:
        side = "ahead" if lat == 0 else ("ahead_left" if lat < 0 else "ahead_right")
    elif fwd < 0:
        side = "behind" if lat == 0 else ("behind_left" if lat < 0 else "behind_right")
    else:
        side = "left" if lat < 0 else "right"
    return side, dist


def parse_ascii(obs: str):
    t = strip_ansi(obs)
    lines = [ln for ln in t.split("\n") if ln and not ln.startswith("---")
             and ":" not in ln]
    apos, facing, boxes, tgts = None, None, [], []
    for y, ln in enumerate(lines):
        for x, ch in enumerate(ln.split(" ")):
            if ch in ACHAR:
                apos, facing = (x, y), ACHAR[ch]
            elif ch == "B":
                boxes.append((x, y))
            elif ch == "T":
                tgts.append((x, y))
    if apos is None or facing is None or not boxes or not tgts:
        return None
    b, g = boxes[0], tgts[0]
    bs, bd = ego_of(b[0] - apos[0], b[1] - apos[1], facing)
    gs, gd = ego_of(g[0] - apos[0], g[1] - apos[1], facing)
    ahead_cell = (apos[0] + ADIR[facing][0], apos[1] + ADIR[facing][1])
    wall = True
    for y, ln in enumerate(lines):
        for x, ch in enumerate(ln.split(" ")):
            if (x, y) == ahead_cell and ch != "#":
                wall = False
    return {"facing": facing, "box": (bs, bd), "target": (gs, gd),
            "wall_ahead": wall}


def parse_state(obs: dict):
    try:
        apos = tuple(obs["agent"]["position"])
        facing = obs["agent"]["orientation"]
        boxes = [(x, y) for y, row in enumerate(obs["grid"]["objects"])
                 for x, v in enumerate(row) if v == 5]
        tgts = [(x, y) for y, row in enumerate(obs["grid"]["objects"])
                for x, v in enumerate(row) if v == 6]
        if not boxes or not tgts:
            return None
        ter = obs["grid"]["terrain"]
        ax, ay = apos[0] + ADIR[facing][0], apos[1] + ADIR[facing][1]
        bs, bd = ego_of(boxes[0][0] - apos[0], boxes[0][1] - apos[1], facing)
        gs, gd = ego_of(tgts[0][0] - apos[0], tgts[0][1] - apos[1], facing)
        return {"facing": facing, "box": (bs, bd), "target": (gs, gd),
                "wall_ahead": bool(ter[ay][ax] == 1)}
    except Exception:
        return None


MENT_RE = re.compile(r"(target|box|wall|tile \d+)\s+([a-z ]+?)\s*\((\d+) steps\)")


def canon(rel: str):
    rel = rel.replace("to your left", "left").replace("to your right", "right")
    toks = re.findall(r"\b(ahead|behind|left|right)\b", rel)
    if not toks:
        return None
    order = {"ahead": 0, "behind": 1, "left": 2, "right": 3}
    return "_".join(sorted(set(toks), key=lambda t: order[t]))


def parse_language(obs: str):
    mfac = re.search(r"facing (\w+)\.", obs)
    if not mfac or mfac.group(1) not in ADIR:
        return None
    facing = mfac.group(1)
    sees = obs.split("You see:")[-1]
    box = tgt = None
    for kind, rel, dist in MENT_RE.findall(sees + " " + obs.split("Goal:")[-1]
                                           if "Goal:" in obs else sees):
        c = canon(rel)
        if c is None:
            continue
        if kind == "box" and box is None:
            box = (c, int(dist))
        elif kind == "target" and tgt is None:
            tgt = (c, int(dist))
    if box is None or tgt is None:
        return None
    return {"facing": facing, "box": box, "target": tgt,
            "wall_ahead": bool(re.search(r"wall ahead \(blocking\)", obs))}


def primary_side(side: str) -> str:
    # diagonals -> forward-first component ("ahead_X"->ahead; "behind_X"->lateral X)
    if side.startswith("ahead"):
        return "ahead"
    if side.startswith("behind_"):
        return side.split("_")[1]
    return side


def decide(p: dict, valid: list[str]) -> int:
    f = p["facing"]
    bs, bd = p["box"]
    gs, gd = p["target"]
    if bd == 1 and bs == gs and gd > bd:
        want = EGO[f][primary_side(bs)]  # PUSH: box between agent and target
    elif bd == 1:
        want = EGO[f][primary_side(gs)]  # REPOSITION toward target side
    else:
        want = EGO[f][primary_side(bs)]  # APPROACH box
    if want in valid:
        return AIDX[want]
    for fb in FALLBACK:
        if fb in valid:
            return AIDX[fb]
    return AIDX["noop"]


PARSERS = {"ascii": parse_ascii, "language": parse_language, "state_dict": parse_state}


def run_episode(mode: str, seed: int, consumer: str, rng=None):
    env = agentick.make(TASK, difficulty=DIFF, render_mode=mode, seed=seed)
    obs, info = env.reset(seed=seed)
    total, parse_ok, steps = 0.0, 0, 0
    done = False
    while not done:
        valid = list(info.get("valid_actions", []))
        if consumer == "adapter":
            pr = PARSERS[mode](obs)
            if pr is None:
                a = AIDX["noop"] if "noop" in valid else AIDX[valid[0]]
            else:
                parse_ok += 1
                a = decide(pr, valid)
            steps += 1
        elif consumer == "random":
            a = int(rng.integers(0, 6))
        else:
            raise ValueError(consumer)
        obs, rew, term, trunc, info = env.step(a)
        total += float(rew)
        done = bool(term or trunc)
    env.close()
    return {"return": total, "parse_rate": (parse_ok / steps) if steps else 0.0}


def oracle_return(seed: int) -> float:
    from agentick.oracles import get_oracle
    env = agentick.make(TASK, difficulty=DIFF, render_mode="state_dict", seed=seed)
    obs, info = env.reset(seed=seed)
    o = get_oracle(TASK, env)
    o.reset(obs, info)
    total, done = 0.0, False
    while not done:
        a = int(o.act(obs, info))
        obs, rew, term, trunc, info = env.step(a)
        total += float(rew)
        done = bool(term or trunc)
    env.close()
    return total


def main():
    import numpy as np
    # seed rule: start 0-4, replace until 5 oracle-solved
    seeds, cand, replaced = [], 0, []
    while len(seeds) < 5:
        r = oracle_return(cand)
        if r > 0:
            seeds.append(cand)
        else:
            replaced.append(cand)
        cand += 1
    print("seeds:", seeds, "replaced (oracle-failed):", replaced, flush=True)
    rec = {"task": TASK, "difficulty": DIFF, "reward": REWARD, "seeds": seeds,
           "replaced_seeds": replaced, "episodes": []}
    for mode in MODES:
        for seed in seeds:
            r = run_episode(mode, seed, "adapter")
            r.update({"mode": mode, "seed": seed, "consumer": "adapter"})
            rec["episodes"].append(r)
        rng = np.random.default_rng(9200)
        for seed in seeds:
            r = run_episode(mode, seed, "random", rng)
            r.update({"mode": mode, "seed": seed, "consumer": "random"})
            rec["episodes"].append(r)
    oret = {s: oracle_return(s) for s in seeds}
    rec["oracle_returns"] = {str(k): v for k, v in oret.items()}
    json.dump(rec, open(OUT / "agentick_hard_raw.json", "w"), indent=1)
    print("raw written; oracle returns:", oret, flush=True)


if __name__ == "__main__":
    main()
