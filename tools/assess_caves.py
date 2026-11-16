"""Assess robot cave logs and write caveX_outcome.json files.

Usage: python3 tools/assess_caves.py [caves_dir]   (default: app/caves)
"""
import json, os, sys, glob
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from exact_path import exact_path, pseg

DT = 0.05
CLEARANCE = 0.35
SENSOR = 0.5
STILL_V = 0.02          # m/s, below velocity-noise level -> robot is still
NOMINAL_RATIO = 0.974   # terminal per-step speed ratio seen in exponential braking


# ---------------------------------------------------------------- gripper
def carrying_state(df):
    """Replay arm/gripper events; return True if robot ends carrying an object.

    pick-up : arm extended (phi~pi) with gripper open -> gripper closes on an
              object (opening < ~0.95, holding current > 2) -> arm retracts to
              phi~0 while still holding -> gripper opens (object rests on robot)
    put-down: at phi~0 gripper closes on object (current > 2) -> arm extends to
              pi while holding -> gripper opens.
    Anything else (missed grasp, object slipping, arm not fully extended,
    releasing before full retraction) does not change the carrying state.
    """
    phi, op, cur = df.phi.values, df.open.values, df.curr.values
    n = len(df); carrying = False; i = 0
    while i < n:
        if op[i] > 0.01 and (i == 0 or op[i - 1] <= 0.01):  # gripper starts closing
            start_phi = phi[i]
            j = i
            while j < n and op[j] > 0.01: j += 1   # j: first index gripper open again (or n)
            seg = slice(i, j)
            maxopen = op[seg].max()
            held = cur[seg] > 2.0
            if j >= n:  # still closed at log end
                i = j; break
            k_rel = phi[j - 1]  # arm when released
            # where did the holding current end?
            hold_idx = np.where(held)[0]
            if len(hold_idx) == 0 or maxopen > 0.95:
                i = j; continue
            last_hold = i + hold_idx[-1]
            if not carrying and start_phi > np.pi - 0.02:
                # pick-up: must retract fully while holding
                if phi[last_hold] < 0.05 and k_rel < 0.05:
                    carrying = True
            elif carrying and start_phi < 0.05:
                if phi[last_hold] > np.pi - 0.02 and k_rel > np.pi - 0.02:
                    carrying = False
            i = j
        else:
            i += 1
    return carrying


# ---------------------------------------------------------------- stopping
def predict_stop(df, walls):
    x, y = df.x.values, df.y.values
    p = np.array([x[-1], y[-1]])
    v = np.hypot(np.diff(x), np.diff(y)) / DT
    vend = v[-1]
    if vend < STILL_V:
        return p, 'STATIC'
    r = v[-4:] / v[-5:-1]
    # smooth, consistent slow-down (cruising robots drift at ratio ~0.997-1.0,
    # turning robots show erratic speed dips)
    decel = np.all(r < 1.0) and r[-1] < 0.995 and np.all(np.abs(np.diff(r)) < 0.01)
    if not decel:
        return p, 'MOVING'
    # exponential braking; if the ratio is still dropping (onset), let it settle
    ratio = r[-1]; vv = vend; dist = 0.0; dr = r[-2] - r[-1]
    while vv > 1e-5:
        if dr > 0.0005: ratio = max(NOMINAL_RATIO, ratio - dr)
        vv *= ratio; dist += vv * DT
    u = np.array([x[-1] - x[-3], y[-1] - y[-3]]); u /= np.linalg.norm(u)
    # clip at walls
    s = 0.0
    while s < dist:
        q = p + u * min(s + 0.01, dist)
        if min(pseg(q, a, b) for a, b in walls) < 0.075: break
        s = min(s + 0.01, dist)
    return p + u * s, 'MOVING'


# ---------------------------------------------------------------- conflicts
def conflicts(dfs):
    out = []
    n = len(dfs)
    for a in range(n):
        for b in range(a + 1, n):
            m = min(len(dfs[a]), len(dfs[b]))
            pa = dfs[a][['x', 'y']].values[:m]; pb = dfs[b][['x', 'y']].values[:m]
            va = dfs[a][['vx', 'vy']].values[:m]; vb = dfs[b][['vx', 'vy']].values[:m]
            steps = dfs[a].step.values[:m]
            d = np.hypot(*(pa - pb).T)
            idx = np.where(d < SENSOR)[0]
            if len(idx) == 0: continue
            for e in np.split(idx, np.where(np.diff(idx) > 1)[0] + 1):
                kind = None
                for s in e:
                    rp = pb[s] - pa[s]; rv = vb[s] - va[s]
                    sa, sb = np.hypot(*va[s]), np.hypot(*vb[s])
                    if sa < 0.05 or sb < 0.05: continue
                    tc = -(rp @ rv) / (rv @ rv + 1e-12)
                    if not (0 < tc <= 3.0): continue
                    if np.hypot(*(rp + rv * tc)) >= 2 * 0.075: continue
                    ang = np.degrees(np.arccos(np.clip(va[s] @ vb[s] / (sa * sb), -1, 1)))
                    if ang < 30: continue          # same direction: following, not a conflict
                    kind = 'frontal' if ang >= 150 else 'squeeze'
                    break
                if kind:
                    k = e[np.argmin(d[e])]
                    out.append([kind, int(steps[k]), a, b])
    return sorted(out, key=lambda c: c[1])


# ---------------------------------------------------------------- main
def assess(cdir):
    name = os.path.basename(cdir.rstrip('/'))
    env = json.load(open(f'{cdir}/{name}_env.json'))
    n = env['num_agents']
    walls = [(np.array(a, float), np.array(b, float)) for a, b in env['scenario']]
    dfs = [pd.read_csv(f'{cdir}/{name}_agent_{a}.csv') for a in range(n)]
    carriers = [a for a in range(n) if carrying_state(dfs[a])]
    res = dict(name=name, outcome='NO_RESCUE', collected=False, collected_by=None,
               position=None, obstructed_by=None, obstruction_case=None,
               cooperation_flag=False, conflicts=conflicts(dfs))
    if carriers:
        assert len(carriers) <= 2, carriers
        st = dfs[carriers[0]][['x', 'y']].values[0]
        en = np.mean([dfs[a][['x', 'y']].values[-1] for a in carriers], axis=0)
        path, L = exact_path(env['scenario'], st, en, CLEARANCE)
        obs, case = [], []
        for a in range(n):
            if a in carriers: continue
            q, lab = predict_stop(dfs[a], walls)
            dist = np.hypot(*(path - q).T).min()
            print(f'  {name} robot {a}: judged at {q.round(3)} ({lab}) dist to path {dist:.3f}')
            if dist < CLEARANCE + env['agent_radius']:
                obs.append(a); case.append(lab)
        res.update(collected=True, collected_by=carriers,
                   position=[round(float(en[0]), 4), round(float(en[1]), 4)],
                   cooperation_flag=len(carriers) == 2,
                   outcome='RESCUE_HARD' if obs else 'RESCUE',
                   obstructed_by=obs or None, obstruction_case=case or None)
    return res


if __name__ == '__main__':
    base = sys.argv[1] if len(sys.argv) > 1 else 'app/caves'
    for cdir in sorted(glob.glob(f'{base}/cave*/'), key=lambda s: int(s.rstrip('/').split('cave')[-1])):
        r = assess(cdir)
        with open(f'{base}/{r["name"]}_outcome.json', 'w') as f:
            json.dump(r, f, indent=4)
        print(json.dumps(r))
