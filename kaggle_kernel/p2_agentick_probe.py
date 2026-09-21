"""Agentick Kaggle-compat probe (CPU-only): can the dose-response run here?

Tests, in order: git availability, pip install of the pinned agentick source,
import, env creation (GoToGoal-v0 x ascii/language/state_dict), short random
episodes per mode, oracle import + one oracle episode, gymnasium/numpy
versions. NO adapter code, NO verdict logic -- install/runtime compat only.
Outputs: agentick_probe.json
"""

import json
import os
import subprocess
import sys
import time

AGENTICK_URL = "https://github.com/roger-creus/agentick.git"
AGENTICK_COMMIT = "279fe5f34a35196ba3904550f911d5c8ade3c7c7"
TASK, DIFF = "GoToGoal-v0", "easy"


def main():
    t_all = time.time()
    rep = {"agentick_url": AGENTICK_URL, "commit": AGENTICK_COMMIT}
    r = subprocess.run(["git", "--version"], capture_output=True, text=True)
    rep["git"] = r.stdout.strip() if r.returncode == 0 else "MISSING"
    print("git:", rep["git"], flush=True)
    if r.returncode != 0:
        rep["compatible"] = False
        rep["reason"] = "no git binary"
        json.dump(rep, open("/kaggle/working/agentick_probe.json", "w"), indent=2)
        print("INCOMPATIBLE: no git", flush=True)
        return
    ins = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q",
         f"git+{AGENTICK_URL}@{AGENTICK_COMMIT}"],
        capture_output=True, text=True, timeout=1800)
    rep["pip_rc"] = ins.returncode
    print("pip tail:", ins.stdout[-500:] + ins.stderr[-500:], flush=True)
    if ins.returncode != 0:
        rep["compatible"] = False
        rep["reason"] = "pip install failed"
        json.dump(rep, open("/kaggle/working/agentick_probe.json", "w"), indent=1)
        print("INCOMPATIBLE: pip failed", flush=True)
        return
    import agentick  # noqa: E402
    import numpy  # noqa: E402
    import gymnasium  # noqa: E402
    rep["numpy"] = numpy.__version__
    rep["gymnasium"] = gymnasium.__version__
    assert "GoToGoal-v0" in agentick.list_tasks(), "task missing"
    modes_ok = {}
    for mode in ("ascii", "language", "state_dict"):
        env = agentick.make(TASK, difficulty=DIFF, render_mode=mode, seed=0)
        obs, info = env.reset(seed=0)
        total, done, n = 0.0, False, 0
        rng = __import__("numpy").random.default_rng(7)
        while not done and n < 50:
            a = int(rng.integers(0, env.action_space.n))
            obs, rew, term, trunc, info = env.step(a)
            total += float(rew)
            done = bool(term or trunc)
            n += 1
        env.close()
        modes_ok[mode] = {"episodes": 1, "return": round(total, 4), "steps": n}
        print(f"  {mode}: return={total:.3f} steps={n}", flush=True)
    rep["modes"] = modes_ok
    from agentick.oracles import get_oracle  # noqa: E402
    env = agentick.make(TASK, difficulty=DIFF, render_mode="state_dict", seed=0)
    obs, info = env.reset(seed=0)
    oracle = get_oracle(TASK, env)
    oracle.reset(obs, info)
    total, done = 0.0, False
    while not done:
        a = int(oracle.act(obs, info))
        obs, rew, term, trunc, info = env.step(a)
        total += float(rew)
        done = bool(term or trunc)
    env.close()
    rep["oracle_return"] = round(total, 4)
    rep["compatible"] = True
    rep["runtime_min"] = round((time.time() - t_all) / 60, 2)
    json.dump(rep, open("/kaggle/working/agentick_probe.json", "w"), indent=2)
    print("COMPATIBLE:", json.dumps(
        {k: v for k, v in rep.items() if k != "modes"}), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
