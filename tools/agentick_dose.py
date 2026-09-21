"""Agentick dose-response (frozen DESIGN-TRACKA A2; CPU-only).

Authored fixed policies reading ONLY permitted obs + valid_actions:
- SokobanPush: BFS shortest push-plan on parsed absolute state
  (depth<=40, nodes<=20000, corner-deadlock prune), replan each step.
- SequenceMemory: record shown gem positions, revisit in order via BFS
  shortest-path movement (parsed walkability; language assumes open room
  plus mentioned blocking walls).
Dose: radius-k entity masking (boxes/targets/gems) on the PARSED state,
k in {full,3,1,0} (Manhattan from agent; k=0 hides all entities).
Walls (terrain) always visible. Same policy/seeds across modes.
Competence gate (frozen, first): full-parse policy solves >=80% of
oracle-solved instances over 10 seeds per task, else that task STOPS.
Primary: paired J_full - J_k + monotonic-degradation test (no slope).
Cross-format secondary. Oracle = normalization only. Random floor.
OUT overridable via PAPER2_AGENTICK_OUT (default: theory/ desktop path).
"""
from __future__ import annotations

import json
import os
import re
from collections import deque
from pathlib import Path

OUT = Path(os.environ.get(
    "PAPER2_AGENTICK_OUT",
    "/Users/sanamimani/paper2_semantic_decision_benchmark/results/v3main/theory"))
TASKS = ["SokobanPush-v0", "SequenceMemory-v0"]
MODES = ["ascii", "language", "state_dict"]
DOSES = ["full", 3, 1, 0]
DIFF, REWARD = "easy", "dense"
MAXDEPTH, MAXNODES = 40, 20000

import agentick  # noqa: E402

ANAME = ["noop", "move_up", "move_down", "move_left", "move_right", "interact"]
AIDX = {n: i for i, n in enumerate(ANAME)}
FALLBACK = ["move_up", "move_right", "move_down", "move_left", "interact", "noop"]
ADIR = {"north": (0, -1), "east": (1, 0), "south": (0, 1), "west": (-1, 0)}
RDIR = {"north": (1, 0), "east": (0, 1), "south": (-1, 0), "west": (0, -1)}
ACHAR = {"^": "north", "v": "south", ">": "east", "<": "west"}
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


def parse_grid_state(obs, is_ascii: bool):
    """Absolute parsed state. Returns (state|None, ok). Token scan is
    character-wise: co-located glyphs render joined (e.g. '<G' = agent
    facing west sharing a cell with G). G is collected as goals."""
    if is_ascii:
        t = strip_ansi(obs)
        lines = [ln for ln in t.split("\n") if ln and not ln.startswith("---")
                 and ":" not in ln]
        H = len(lines)
        W = 0
        walls, boxes, tgts, goals, gems = set(), [], [], [], []
        apos, facing = None, None
        for y, ln in enumerate(lines):
            cells = ln.split(" ")
            while cells and cells[-1] == "":
                cells.pop()  # trailing padding is not a column
            for x, tok in enumerate(cells):
                W = max(W, x + 1)
                for ch in tok:
                    if ch in ACHAR and apos is None:
                        apos, facing = (x, y), ACHAR[ch]
                    elif ch == "#":
                        walls.add((x, y))
                    elif ch == "B":
                        boxes.append((x, y))
                    elif ch == "T":
                        tgts.append((x, y))
                    elif ch == "G":
                        goals.append((x, y))
                    elif ch.islower():
                        gems.append((x, y))
        if apos is None or facing is None:
            return None, False
        return {"agent": apos, "facing": facing, "walls": walls, "boxes": boxes,
                "targets": tgts, "goals": goals, "gems": gems,
                "dims": (W, H)}, True
    try:
        apos = tuple(obs["agent"]["position"])
        facing = obs["agent"]["orientation"]
        ter, obj = obs["grid"]["terrain"], obs["grid"]["objects"]
        H, W = obs["grid"]["height"], obs["grid"]["width"]
        walls = {(x, y) for y in range(H) for x in range(W) if ter[y][x] == 1}
        boxes = [(x, y) for y in range(H) for x in range(W) if obj[y][x] == 5]
        tgts = [(x, y) for y, row in enumerate(obj) for x, v in enumerate(row)
                if v == 6]
        gems = [(x, y) for y, row in enumerate(obj) for x, v in enumerate(row)
                if v not in (0, 5, 6)]
        goals = [(x, y) for y, row in enumerate(obj) for x, v in enumerate(row)
                 if v == 1]
        return {"agent": apos, "facing": facing, "walls": walls, "boxes": boxes,
                "targets": tgts, "goals": goals, "gems": gems,
                "dims": (W, H)}, True
    except Exception:
        return None, False


MENT_RE = re.compile(r"(target|box|gem|wall|tile \d+)\s+([a-z ]+?)\s*\((\d+) steps\)")


def canon(rel: str):
    rel = rel.replace("to your left", "left").replace("to your right", "right")
    toks = re.findall(r"\b(ahead|behind|left|right)\b", rel)
    if not toks:
        return None
    order = {"ahead": 0, "behind": 1, "left": 2, "right": 3}
    return "_".join(sorted(set(toks), key=lambda t: order[t]))


def primary_side(side: str) -> str:
    if side.startswith("ahead"):
        return "ahead"
    if side.startswith("behind_"):
        return side.split("_")[1]
    return side


class LanguageTracker:
    """Dead-reckoning pose + entity map for language mode (relative frame).

    Same-policy requirement forces a shared consumer: the tracker converts
    relative mentions into (dx,dy)-offset cells from episode start, moves
    update the frame deterministically. Walls: only mentioned blocking walls
    (assumed open otherwise -- documented approximation)."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.pos = (0, 0)
        self.facing = None
        self.boxes = {}
        self.targets = {}
        self.gems = {}
        self.blockahead = False
        self.ok = True

    def observe(self, obs: str):
        mfac = re.search(r"facing (\w+)\.", obs)
        if not mfac or mfac.group(1) not in ADIR:
            self.ok = False
            return
        self.facing = mfac.group(1)
        sees = obs.split("You see:")[-1] if "You see:" in obs else ""
        for kind, rel, dist in MENT_RE.findall(sees):
            c = canon(rel)
            if c is None or kind.startswith("tile"):
                continue
            d = self._offset(c, int(dist))
            if d is None:
                continue
            cell = (self.pos[0] + d[0], self.pos[1] + d[1])
            if kind == "box":
                self.boxes[cell] = True
            elif kind == "target":
                self.targets[cell] = True
            elif kind == "gem":
                self.gems[cell] = True
        self.blockahead = bool(re.search(r"wall ahead \(blocking\)", obs))

    def _offset(self, side: str, dist: int):
        # direction unit vector in absolute (x-right, y-down) frame
        if side == "here":
            return (0, 0)
        s = primary_side(side)
        base = {"ahead": ADIR[self.facing], "behind": (-ADIR[self.facing][0],
                                                       -ADIR[self.facing][1]),
                "left": (-RDIR[self.facing][0], -RDIR[self.facing][1]),
                "right": RDIR[self.facing]}[s]
        if "_" in side:  # diagonal: split dist, forward gets ceil
            lat = side.split("_")[1]
            lv = {"left": (-RDIR[self.facing][0], -RDIR[self.facing][1]),
                  "right": RDIR[self.facing]}[lat]
            fwd = (dist + 1) // 2
            latd = dist - fwd
            return (base[0] * fwd + lv[0] * latd, base[1] * fwd + lv[1] * latd)
        return (base[0] * dist, base[1] * dist)

    def move(self, action_name: str):
        d = {"move_up": (0, -1), "move_down": (0, 1), "move_left": (-1, 0),
             "move_right": (1, 0)}.get(action_name, (0, 0))
        self.pos = (self.pos[0] + d[0], self.pos[1] + d[1])


def bfs_push(agent, boxes, targets, walls, dims):
    """Shortest push plan (first action) or None. Bounded: depth<=40,
    nodes<=20000. Corner-deadlock prune (non-target corners)."""
    from collections import deque
    W, H = dims
    tset, b0 = set(targets), frozenset(boxes)

    def dead(b):
        x, y = b
        if b in tset:
            return False
        return ((x - 1, y) in walls or x - 1 < 0) and \
               ((x, y - 1) in walls or y - 1 < 0) or \
               ((x + 1, y) in walls or x + 1 >= W) and \
               ((x, y - 1) in walls or y - 1 < 0) or \
               ((x - 1, y) in walls or x - 1 < 0) and \
               ((x, y + 1) in walls or y + 1 >= H) or \
               ((x + 1, y) in walls or x + 1 >= W) and \
               ((x, y + 1) in walls or y + 1 >= H)

    if any(dead(b) for b in b0):
        return None
    DIRS = {"move_up": (0, -1), "move_down": (0, 1), "move_left": (-1, 0),
            "move_right": (1, 0)}
    start = (agent, b0)
    q = deque([(start, [])])
    seen = {start}
    while q and len(seen) <= MAXNODES:
        (a, bs), path = q.popleft()
        if len(path) > MAXDEPTH:
            continue
        if set(bs) <= tset:
            return path[0] if path else "noop"
        for name, (dx, dy) in DIRS.items():
            na = (a[0] + dx, a[1] + dy)
            if na in walls or not (0 <= na[0] < W and 0 <= na[1] < H):
                continue
            nbs = set(bs)
            if na in nbs:
                nb = (na[0] + dx, na[1] + dy)
                if nb in walls or nb in nbs or not (0 <= nb[0] < W and
                                                    0 <= nb[1] < H):
                    continue
                if dead(nb):
                    continue
                nbs = (set(nbs) - {na}) | {nb}
            key = (na, frozenset(nbs))
            if key in seen:
                continue
            seen.add(key)
            q.append((key, path + [name]))
    return None


def bfs_move(agent, goal, walls, dims, blocked=frozenset()):
    """Shortest-path first move (agent-only) or None."""
    W, H = dims
    DIRS = [("move_up", (0, -1)), ("move_down", (0, 1)),
            ("move_left", (-1, 0)), ("move_right", (1, 0))]
    q = deque([(agent, [])])
    seen = {agent}
    while q and len(seen) <= MAXNODES:
        a, path = q.popleft()
        if a == goal:
            return path[0] if path else "noop"
        if len(path) >= MAXDEPTH:
            continue
        for name, (dx, dy) in DIRS:
            na = (a[0] + dx, a[1] + dy)
            if na in walls or na in blocked or not (0 <= na[0] < W and
                                                    0 <= na[1] < H):
                continue
            if na in seen:
                continue
            seen.add(na)
            q.append((na, path + [name]))
    return None


def mask_entities(state: dict, dose, agent_pos):
    """Radius-k masking of parsed entities (boxes/targets/gems) by Manhattan
    distance from agent. k='full' -> unchanged. k=0 -> hide all."""
    if dose == "full":
        return state
    out = dict(state)
    for key in ("boxes", "targets", "gems"):
        out[key] = [c for c in state.get(key, [])
                    if abs(c[0] - agent_pos[0]) + abs(c[1] - agent_pos[1]) <= dose]
    return out


class SokobanPolicy:
    """Fixed push policy: BFS plan on parsed state each step; fixed fallback."""

    def __init__(self):
        self.parses = 0
        self.steps = 0

    def act(self, parsed, valid):
        self.steps += 1
        if parsed is None:
            return fallback(valid)
        self.parses += 1
        mv = bfs_push(parsed["agent"], parsed["boxes"], parsed["targets"],
                      parsed["walls"], parsed["dims"])
        if mv is not None and mv in valid:
            return AIDX[mv]
        return fallback(valid)


class SeqMemPolicy:
    """Fixed record-then-revisit policy with recall LATCH. Show phase: wait
    (noop) while recording shown cells in order. Once nothing is shown with
    non-empty memory, latch into recall: visit mem cells in recorded order
    via BFS, advancing the visit index on arrival (never revisiting). The
    latch prevents visibility flicker from thrashing show/recall modes."""

    def __init__(self):
        self.mem = []
        self.seen_mem = set()
        self.recall = False
        self.visit_idx = 0
        self.parses = 0
        self.steps = 0

    def reset(self):
        self.mem, self.seen_mem = [], set()
        self.recall, self.visit_idx = False, 0

    def act(self, parsed, valid):
        self.steps += 1
        if parsed is None:
            return fallback(valid)
        self.parses += 1
        shown = list(parsed["gems"]) + list(parsed.get("goals", []))
        if not self.recall:
            for g in shown:
                if g not in self.seen_mem:
                    self.seen_mem.add(g)
                    self.mem.append(g)
            if not shown and self.mem:
                self.recall = True
            else:
                return AIDX["noop"] if "noop" in valid else fallback(valid)
        while self.visit_idx < len(self.mem) and \
                tuple(self.mem[self.visit_idx]) == tuple(parsed["agent"]):
            self.visit_idx += 1
        if self.visit_idx >= len(self.mem):
            return AIDX["noop"] if "noop" in valid else fallback(valid)
        mv = bfs_move(tuple(parsed["agent"]), tuple(self.mem[self.visit_idx]),
                      parsed["walls"], parsed["dims"])
        if mv is not None and mv in valid:
            return AIDX[mv]
        return fallback(valid)


def fallback(valid):
    for fb in FALLBACK:
        if fb in valid:
            return AIDX[fb]
    return AIDX["noop"]


def parse_obs(task, mode, obs, tracker=None):
    if mode in ("ascii", "state_dict"):
        return parse_grid_state(obs, mode == "ascii")
    # language
    if task == "SokobanPush-v0":
        mfac = re.search(r"facing (\w+)\.", obs)
        if not mfac or mfac.group(1) not in ADIR:
            return None, False
        facing = mfac.group(1)
        sees = obs.split("You see:")[-1] if "You see:" in obs else ""
        boxes, tgts, wall = [], [], False
        for kind, rel, dist in MENT_RE.findall(sees):
            c = canon(rel)
            if c is None or kind.startswith("tile"):
                continue
            n = _lang_cell(facing, c, int(dist))
            if n is None:
                continue
            if kind == "box":
                boxes.append(n)
            elif kind == "target":
                tgts.append(n)
        wall = bool(re.search(r"wall ahead \(blocking\)", obs))
        if not boxes or not tgts:
            return None, False
        # pseudo-absolute frame anchored at episode start (tracker frame)
        ax, ay = tracker.pos
        B = [(ax + dx, ay + dy) for dx, dy in boxes]
        T = [(ax + dx, ay + dy) for dx, dy in tgts]
        walls = set()
        if wall:
            fx, fy = ADIR[facing]
            walls.add((ax + fx, ay + fy))
        return {"agent": (ax, ay), "facing": facing, "walls": walls,
                "boxes": B, "targets": T, "gems": [], "dims": (7, 7)}, True
    # SequenceMemory language: tracker frame directly
    tracker.observe(obs)
    if not tracker.ok or tracker.facing is None:
        return None, False
    ax, ay = tracker.pos
    f = tracker.facing
    gems = list(tracker.gems.keys())
    G = gems  # already absolute-in-frame
    walls = set()
    if tracker.blockahead:
        fx, fy = ADIR[f]
        walls.add((ax + fx, ay + fy))
    return {"agent": (ax, ay), "facing": f, "walls": walls, "boxes": [],
            "targets": [], "gems": G, "dims": (7, 7)}, True


def _lang_cell(facing, side, dist):
    # egocentric offset vector (shared convention with LanguageTracker)
    if side == "here":
        return (0, 0)
    s = primary_side(side)
    base = {"ahead": ADIR[facing],
            "behind": (-ADIR[facing][0], -ADIR[facing][1]),
            "left": (-RDIR[facing][0], -RDIR[facing][1]),
            "right": RDIR[facing]}[s]
    if "_" in side:
        lat = side.split("_")[1]
        lv = {"left": (-RDIR[facing][0], -RDIR[facing][1]),
              "right": RDIR[facing]}[lat]
        fwd = (dist + 1) // 2
        latd = dist - fwd
        return (base[0] * fwd + lv[0] * latd, base[1] * fwd + lv[1] * latd)
    return (base[0] * dist, base[1] * dist)


DOSE_LEVELS = ["full", 3, 1, 0]
N_SEEDS = 10


def run_episode(task, mode, seed, consumer, dose="full", rng=None, policy=None,
                tracker=None):
    env = agentick.make(task, difficulty=DIFF, render_mode=mode, seed=seed)
    obs, info = env.reset(seed=seed)
    if isinstance(policy, SeqMemPolicy):
        policy.reset()
    if tracker is not None:
        tracker.reset()
    total, parse_ok, steps = 0.0, 0, 0
    done = False
    while not done:
        valid = list(info.get("valid_actions", []))
        if consumer == "policy":
            pr = parse_obs(task, mode, obs, tracker)
            if isinstance(pr, tuple):
                pr, ok = pr
            else:
                ok = pr is not None
            if pr is None:
                a = AIDX["noop"] if "noop" in valid else AIDX[valid[0]]
            else:
                parse_ok += int(ok)
                masked = mask_entities(pr, dose, pr["agent"])
                if task == "SokobanPush-v0":
                    if not masked["boxes"] or not masked["targets"]:
                        a = fallback(valid)
                    else:
                        mv = bfs_push(masked["agent"], masked["boxes"],
                                      masked["targets"], masked["walls"],
                                      masked["dims"])
                        a = AIDX[mv] if mv in valid else fallback(valid)
                else:
                    a = policy.act(masked, valid)
            steps += 1
            if tracker is not None and mode == "language":
                aname = ANAME[a]
                tracker.move(aname)
        elif consumer == "random":
            a = int(rng.integers(0, 6))
        else:
            raise ValueError(consumer)
        obs, rew, term, trunc, info = env.step(a)
        total += float(rew)
        done = bool(term or trunc)
    env.close()
    return {"return": total, "parse_rate": (parse_ok / steps) if steps else 0.0}


def oracle_return(task, seed):
    from agentick.oracles import get_oracle
    env = agentick.make(task, difficulty=DIFF, render_mode="state_dict", seed=seed)
    obs, info = env.reset(seed=seed)
    o = get_oracle(task, env)
    o.reset(obs, info)
    total, done = 0.0, False
    while not done:
        a = int(o.act(obs, info))
        obs, rew, term, trunc, info = env.step(a)
        total += float(rew)
        done = bool(term or trunc)
    env.close()
    return total


def solve_seeds(task, n=N_SEEDS):
    seeds, cand, replaced = [], 0, []
    while len(seeds) < n:
        r = oracle_return(task, cand)
        if r > 0:
            seeds.append(cand)
        else:
            replaced.append(cand)
        cand += 1
        assert cand < 60, "too many oracle-failed seeds"
    return seeds, replaced


def main():
    import numpy as np
    rec = {"tasks": TASKS, "modes": MODES, "doses": DOSE_LEVELS, "reward": REWARD,
           "episodes": [], "competence": {}, "oracle_returns": {}}
    for task in TASKS:
        seeds, replaced = solve_seeds(task)
        print(f"{task}: seeds={seeds} replaced={replaced}", flush=True)
        rec["competence"][task] = {"seeds": seeds, "replaced": replaced}
        oret = {s: oracle_return(task, s) for s in seeds}
        rec["oracle_returns"][task] = {str(k): v for k, v in oret.items()}
        # competence gate FIRST on full-parse policy
        policy = SokobanPolicy() if task == "SokobanPush-v0" else SeqMemPolicy()
        from collections import defaultdict
        wins = 0
        for s in seeds:
            tr = LanguageTracker()
            r = run_episode(task, "state_dict", s, "policy",
                            dose="full", policy=policy, tracker=tr)
            wins += int(r["return"] > 0)
        gate = wins / len(seeds)
        rec["competence"][task]["full_parse_solves"] = wins
        rec["competence"][task]["gate_pass"] = bool(gate >= 0.8)
        print(f"{task}: competence {wins}/{len(seeds)} pass={gate >= 0.8}",
              flush=True)
        if gate < 0.8:
            continue  # STOP for this task: no masked episodes counted
        for mode in MODES:
            for dose in DOSE_LEVELS:
                for s in seeds:
                    tr = LanguageTracker()
                    pol = SokobanPolicy() if task == "SokobanPush-v0" \
                        else SeqMemPolicy()
                    r = run_episode(task, mode, s, "policy", dose=dose,
                                    policy=pol, tracker=tr)
                    r.update({"task": task, "mode": mode, "dose": dose,
                              "seed": s, "consumer": "policy"})
                    rec["episodes"].append(r)
        rng = np.random.default_rng(9300)
        for mode in MODES:
            for s in seeds:
                r = run_episode(task, mode, s, "random", rng=rng)
                r.update({"task": task, "mode": mode, "dose": "full",
                          "seed": s, "consumer": "random"})
                rec["episodes"].append(r)
    json.dump(rec, open(OUT / "agentick_dose_raw.json", "w"), indent=1)
    print("raw written", flush=True)


if __name__ == "__main__":
    main()
