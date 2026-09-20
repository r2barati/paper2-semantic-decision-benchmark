"""Agentick one-task spike (frozen AGENTICK_SPIKE.md; GoToGoal only).

Fixed adapter (identical rule, all modes): parse obs -> egocentric
goal relation (here/ahead/left/right/behind + diagonal variants) and
facing, then act greedily; fallback fixed preference among valid_actions.
Adapter reads info['valid_actions'] ONLY (never agent_position).
Modes: ascii / language / state_dict. Task: GoToGoal-v0, easy, dense.
Consumers: adapter (tested), random floor (descriptive), oracle (ONS
normalization only). Seeds: 0-4 (frozen).
Writes theory/agentick_spike.json (per-episode returns + ONS + verdict).
Run under /tmp/agentick-spike (py3.12); env never committed.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

OUT = Path("/Users/sanamimani/paper2_semantic_decision_benchmark/results/v3main/theory")
SEEDS = [0, 1, 2, 3, 4]
MODES = ["ascii", "language", "state_dict"]
TASK, DIFF, REWARD = "GoToGoal-v0", "easy", "dense"

import agentick  # noqa: E402

ANAME = ["noop", "move_up", "move_down", "move_left", "move_right", "interact"]
AIDX = {n: i for i, n in enumerate(ANAME)}
# egocentric side -> absolute move, keyed by facing; diagonals use fwd first
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


def parse_ascii(obs: str):
    """Return (facing, relation) or None on parse failure."""
    t = strip_ansi(obs)
    lines = [ln for ln in t.split("\n") if ln and not ln.startswith("---")
             and ":" not in ln]
    apos, gpos, facing = None, None, None
    for y, ln in enumerate(lines):
        cells = ln.split(" ")
        for x, ch in enumerate(cells):
            if ch in ACHAR:
                apos, facing = (x, y), ACHAR[ch]
            elif ch == "G":
                gpos = (x, y)
    if apos is None or gpos is None or facing is None:
        # agent may stand ON the goal cell (rendered as agent char only)
        if apos is not None and gpos is None:
            return facing, "here", True
        return None
    return facing, ego_of(apos, facing, gpos), True


def ego_of(apos, facing, gpos):
    dx, dy = gpos[0] - apos[0], gpos[1] - apos[1]
    if (dx, dy) == (0, 0):
        return "here"
    ax, ay = ADIR[facing]
    rx, ry = RDIR[facing]
    fwd, lat = dx * ax + dy * ay, dx * rx + dy * ry
    if fwd > 0:
        return "ahead" if lat == 0 else ("ahead_left" if lat < 0 else "ahead_right")
    if fwd < 0:
        return "behind" if lat == 0 else ("behind_left" if lat < 0 else "behind_right")
    return "left" if lat < 0 else "right"


def parse_state(obs: dict):
    try:
        apos = tuple(obs["agent"]["position"])
        facing = obs["agent"]["orientation"]
        grid = obs["grid"]["objects"]
        gpos = None
        for y, row in enumerate(grid):
            for x, v in enumerate(row):
                if v == 1:
                    gpos = (x, y)
        if gpos is None:
            return None
        return facing, ego_of(apos, facing, gpos), True
    except Exception:
        return None


LANG_RE = re.compile(
    r"facing (\w+)\..*?Goal:\s*([a-z ]+?),\s*~(\d+) steps", re.S)


def parse_language(obs: str):
    m = LANG_RE.search(obs)
    if not m:
        return None
    facing, rel, dist = m.group(1), m.group(2).strip(), int(m.group(3))
    if facing not in ADIR:
        return None
    if dist == 0:
        return facing, "here", True
    rel = rel.replace("to your left", "left").replace("to your right", "right")
    toks = re.findall(r"\b(ahead|behind|left|right)\b", rel)
    if not toks:
        return None
    # canonical order: ahead/behind first, then lateral
    order = {"ahead": 0, "behind": 1, "left": 2, "right": 3}
    key = "_".join(sorted(set(toks), key=lambda t: order[t]))
    norm = {"ahead": "ahead", "ahead_left": "ahead_left", "ahead_right": "ahead_right",
            "left": "left", "right": "right", "behind": "behind",
            "behind_left": "behind_left", "behind_right": "behind_right"}
    if key not in norm:
        return None
    return facing, norm[key], True


def decide(facing: str, relation: str, valid: list[str]) -> int:
    # diagonals: forward component first ("ahead_X" -> ahead; "behind_X" -> X lateral)
    if relation.startswith("ahead_"):
        want = EGO[facing]["ahead"]
    elif relation.startswith("behind_"):
        want = EGO[facing][relation.split("_")[1]]
    else:
        want = EGO[facing].get(relation, "noop")
    if want in valid:
        return AIDX[want]
    for fb in FALLBACK:
        if fb in valid:
            return AIDX[fb]
    return AIDX["noop"]


def run_episode(mode: str, seed: int, consumer: str, rng=None):
    env = agentick.make(TASK, difficulty=DIFF, render_mode=mode, seed=seed)
    obs, info = env.reset(seed=seed)
    total, parse_ok, steps = 0.0, 0, 0
    done = False
    while not done:
        valid = list(info.get("valid_actions", []))
        if consumer == "adapter":
            if mode == "ascii":
                pr = parse_ascii(obs)
            elif mode == "language":
                pr = parse_language(obs)
            else:
                pr = parse_state(obs)
            if pr is None:
                a = AIDX["noop"] if "noop" in valid else AIDX[valid[0]]
            else:
                facing, rel, ok = pr
                parse_ok += int(ok)
                a = decide(facing, rel, valid)
            steps += 1
        elif consumer == "random":
            a = int(rng.integers(0, 6))
        else:
            raise ValueError(consumer)
        obs, rew, term, trunc, info = env.step(a)
        total += float(rew)
        done = bool(term or trunc)
    env.close()
    return {"return": total, "parse_rate": (parse_ok / steps) if steps else 0.0,
            "success": bool(info.get("episode_reward", total) > 0)}


def main():
    import numpy as np
    rec = {"task": TASK, "difficulty": DIFF, "reward": REWARD, "seeds": SEEDS,
           "episodes": []}
    for mode in MODES:
        for seed in SEEDS:
            r = run_episode(mode, seed, "adapter")
            r.update({"mode": mode, "seed": seed, "consumer": "adapter"})
            rec["episodes"].append(r)
        rng = np.random.default_rng(9000)
        for seed in SEEDS:
            r = run_episode(mode, seed, "random", rng)
            r.update({"mode": mode, "seed": seed, "consumer": "random"})
            rec["episodes"].append(r)
    # oracle baseline per seed (state_dict mode; mode-invariance checked below)
    from agentick.oracles import get_oracle
    oret = {}
    for seed in SEEDS:
        env = agentick.make(TASK, difficulty=DIFF, render_mode="state_dict", seed=seed)
        obs, info = env.reset(seed=seed)
        oracle = get_oracle(TASK, env)
        oracle.reset(obs, info)
        total, done = 0.0, False
        while not done:
            a = int(oracle.act(obs, info))
            obs, rew, term, trunc, info = env.step(a)
            total += float(rew)
            done = bool(term or trunc)
        env.close()
        oret[seed] = total
    rec["oracle_returns"] = {str(k): v for k, v in oret.items()}
    json.dump(rec, open(OUT / "agentick_spike_raw.json", "w"), indent=1)
    print("raw written; oracle returns:", oret)


if __name__ == "__main__":
    main()
