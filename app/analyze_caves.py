#!/usr/bin/env python3
"""Assess robot logs in each cave and write caveX_outcome.json files."""
import glob
import heapq
import json
import math
import os
import sys

import numpy as np
import pandas as pd

CLEARANCE = 0.35
SENSOR_RANGE = 0.5
DT = 0.05
PI = math.pi


# ----------------------------------------------------------------------------
# geometry
# ----------------------------------------------------------------------------
def seg_dist(px, py, segs):
    """Distance from points (arrays) to the closest of the segments."""
    px = np.asarray(px, float)[..., None]
    py = np.asarray(py, float)[..., None]
    ax, ay, bx, by = segs[:, 0], segs[:, 1], segs[:, 2], segs[:, 3]
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    L2 = np.where(L2 == 0, 1e-12, L2)
    t = np.clip(((px - ax) * dx + (py - ay) * dy) / L2, 0, 1)
    cx, cy = ax + t * dx, ay + t * dy
    return np.sqrt((px - cx) ** 2 + (py - cy) ** 2).min(axis=-1)


def point_seg_dist(p, a, b):
    p, a, b = map(np.asarray, (p, a, b))
    d = b - a
    L2 = d @ d
    t = 0.0 if L2 == 0 else float(np.clip((p - a) @ d / L2, 0, 1))
    return float(np.linalg.norm(p - (a + t * d)))


def polyline_dist(p, path):
    return min(point_seg_dist(p, path[i], path[i + 1]) for i in range(len(path) - 1)) \
        if len(path) > 1 else float(np.linalg.norm(np.asarray(p) - path[0]))


# ----------------------------------------------------------------------------
# shortest path with wall clearance
# ----------------------------------------------------------------------------
class Planner:
    def __init__(self, segs, res=0.01):
        self.segs = segs
        pts = segs.reshape(-1, 2)
        self.x0, self.y0 = pts.min(0) - 0.05
        x1, y1 = pts.max(0) + 0.05
        self.res = res
        self.nx = int(math.ceil((x1 - self.x0) / res)) + 1
        self.ny = int(math.ceil((y1 - self.y0) / res)) + 1
        xs = self.x0 + np.arange(self.nx) * res
        ys = self.y0 + np.arange(self.ny) * res
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        self.dist = seg_dist(X.ravel(), Y.ravel(), segs).reshape(X.shape)
        self.free = self.dist >= CLEARANCE

    def clear(self, p):
        return float(seg_dist([p[0]], [p[1]], self.segs)[0])

    def line_free(self, a, b, step=0.005):
        a, b = np.asarray(a), np.asarray(b)
        n = max(2, int(np.linalg.norm(b - a) / step) + 1)
        t = np.linspace(0, 1, n)
        P = a + t[:, None] * (b - a)
        return bool((seg_dist(P[:, 0], P[:, 1], self.segs) >= CLEARANCE - 1e-9).all())

    def cell(self, p):
        return (int(round((p[0] - self.x0) / self.res)), int(round((p[1] - self.y0) / self.res)))

    def pt(self, c):
        return np.array([self.x0 + c[0] * self.res, self.y0 + c[1] * self.res])

    def nearest_free(self, c):
        if self.free[c]:
            return c
        idx = np.argwhere(self.free)
        k = np.argmin(((idx - np.array(c)) ** 2).sum(1))
        return tuple(idx[k])

    def shortest(self, start, goal):
        start, goal = np.asarray(start, float), np.asarray(goal, float)
        s, g = self.nearest_free(self.cell(start)), self.nearest_free(self.cell(goal))
        moves = []
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                if (dx, dy) != (0, 0) and math.gcd(abs(dx), abs(dy)) == 1:
                    moves.append((dx, dy, math.hypot(dx, dy)))
        INF = float("inf")
        cost = np.full(self.free.shape, INF)
        prev = {}
        cost[s] = 0
        gp = self.pt(g)
        h = lambda c: math.hypot(*(self.pt(c) - gp))
        pq = [(h(s), 0.0, s)]
        while pq:
            _, cc, c = heapq.heappop(pq)
            if cc > cost[c]:
                continue
            if c == g:
                break
            for dx, dy, w in moves:
                n = (c[0] + dx, c[1] + dy)
                if not (0 <= n[0] < self.nx and 0 <= n[1] < self.ny) or not self.free[n]:
                    continue
                if abs(dx) == 2 or abs(dy) == 2:  # knight move: check middle cells
                    m1 = (c[0] + int(math.copysign(1, dx)) if dx else c[0],
                          c[1] + int(math.copysign(1, dy)) if dy else c[1])
                    if not self.free[m1]:
                        continue
                nc = cc + w * self.res
                if nc < cost[n]:
                    cost[n] = nc
                    prev[n] = c
                    heapq.heappush(pq, (nc + h(n), nc, n))
        if cost[g] == INF:
            return None
        cells = [g]
        while cells[-1] != s:
            cells.append(prev[cells[-1]])
        cells.reverse()
        pts = [start] + [self.pt(c) for c in cells] + [goal]
        return self.smooth(pts)

    def smooth(self, pts):
        """String pulling, then iterative vertex relaxation."""
        out = [pts[0]]
        i = 0
        while i < len(pts) - 1:
            j = len(pts) - 1
            while j > i + 1 and not self.line_free(pts[i], pts[j]):
                j -= 1
            out.append(pts[j])
            i = j
        out = [np.asarray(p, float) for p in out]
        # relaxation: pull interior vertices toward the chord while staying free
        for _ in range(200):
            moved = False
            for k in range(1, len(out) - 1):
                a, b, v = out[k - 1], out[k + 1], out[k]
                d = b - a
                L2 = d @ d
                t = np.clip((v - a) @ d / L2, 0, 1) if L2 > 0 else 0
                target = a + t * d
                lo, hi = 0.0, 1.0
                for _ in range(20):
                    mid = (lo + hi) / 2
                    cand = v + mid * (target - v)
                    if self.line_free(a, cand) and self.line_free(cand, b):
                        lo = mid
                    else:
                        hi = mid
                if lo > 1e-3 and np.linalg.norm(lo * (target - v)) > 1e-5:
                    out[k] = v + lo * (target - v)
                    moved = True
            if not moved:
                break
        # drop collinear / redundant vertices
        k = 1
        while k < len(out) - 1:
            if self.line_free(out[k - 1], out[k + 1]):
                out.pop(k)
            else:
                k += 1
        return out


def path_length(p):
    return sum(np.linalg.norm(p[i + 1] - p[i]) for i in range(len(p) - 1))


# ----------------------------------------------------------------------------
# gripper / carrying analysis
# ----------------------------------------------------------------------------
HOLD_CURR = 2.0      # current while firmly holding an object
CONTACT_CURR = 0.2   # current above this = gripper squeezing something
EXT, RET = PI - 0.1, 0.1


def grip_episodes(d):
    """Return list of (kind, start, end) grip episodes.

    kind: 'pickup'  - closed with arm extended, firm hold, retracted, released
          'putdown' - closed with arm retracted, firm hold, extended, released
          'failed'  - anything else (slip, drop, never retracted, ...)
    """
    curr, phi, op = d.curr.values, d.phi.values, d.open.values
    n = len(d)
    eps = []
    k = 0
    while k < n:
        if curr[k] > CONTACT_CURR:
            s = k
            while k < n and curr[k] > CONTACT_CURR:
                k += 1
            e = k - 1
            seg = slice(s, e + 1)
            firm = (curr[seg] > HOLD_CURR).sum() >= 5 and op[seg].max() < 0.99
            ended = e < n - 1  # gripper released before end of log
            kind = "failed"
            if firm:
                # was the hold kept through the arm motion?
                held = curr[seg] > HOLD_CURR
                hs = s + int(np.argmax(held))
                he = s + len(held) - 1 - int(np.argmax(held[::-1]))
                if phi[s] >= EXT and phi[he] <= RET:
                    kind = "pickup"
                elif phi[s] <= RET and phi[he] >= EXT:
                    kind = "putdown"
            eps.append((kind, s, e, ended))
        else:
            k += 1
    return eps


def carrying_state(d):
    """True if robot carries an object resting on it at the end of the log."""
    carrying = False
    last = None
    for kind, s, e, ended in grip_episodes(d):
        if kind == "pickup":
            carrying, last = True, (s, e)
        elif kind == "putdown":
            carrying, last = False, (s, e)
    return carrying, last


# ----------------------------------------------------------------------------
# end-of-log motion
# ----------------------------------------------------------------------------
STILL_SPEED = 0.02   # m/s, below this the robot is considered still


def end_motion(d):
    """Characterise the motion at the end of a robot log.

    Speeds are derived from logged positions (the velocity readings are noisy).
    A robot is braking when its speed decreases over the last few samples.
    Two braking laws are recognised: exponential decay (deceleration
    proportional to speed, v' = -k v) and constant deceleration (possibly after
    a deceleration ramp-up).  The expected stopping point is extrapolated along
    the current heading with the matching law.
    """
    x, y = d.x.values, d.y.values
    sp = np.hypot(np.diff(x), np.diff(y)) / DT
    p_end = np.array([x[-1], y[-1]])
    v_end = float(sp[-1])
    res = dict(v=v_end, moving=v_end > STILL_SPEED, braking=False, end=p_end,
               stop=p_end.copy(), model=None, dist=0.0)
    if not res["moving"] or len(sp) < 8:
        return res
    hv = p_end - np.array([x[-2], y[-2]])
    u = hv / np.linalg.norm(hv)
    # trailing run of non-increasing speed
    j = len(sp) - 1
    while j > 0 and sp[j] <= sp[j - 1] + 1e-3 and len(sp) - j < 40:
        j -= 1
    run = sp[j:]
    acc = np.diff(run) / DT
    if len(run) < 4:
        return res
    a3 = -float(acc[-3:].mean())
    if a3 < 0.01 or a3 / v_end < 0.1:
        return res
    res["braking"] = True
    a_end = -float(acc[-3:].mean())
    n = len(acc)
    model = "lin"
    if n >= 8:
        a_mid = -float(acc[n // 2 - 2:n // 2 + 1].mean())
        v_mid = float(run[n // 2])
        k_mid, k_end = a_mid / v_mid, a_end / v_end
        # exponential braking: decel/speed stays constant while decel decreases
        if a_mid > a_end * 1.03 and abs(k_end - k_mid) < 0.15 * k_end:
            model = "exp"
    if model == "exp":
        dist = v_end / (a_end / v_end)
    else:
        dist = v_end * v_end / (2 * a_end)
    res.update(model=model, dist=dist, stop=p_end + u * dist, a=a_end, k=a_end / v_end)
    return res


def load_cave(cdir):
    name = os.path.basename(cdir.rstrip("/"))
    env = json.load(open(os.path.join(cdir, f"{name}_env.json")))
    segs = np.array([[a[0], a[1], b[0], b[1]] for a, b in env["scenario"]], float)
    logs = [pd.read_csv(os.path.join(cdir, f"{name}_agent_{i}.csv")) for i in range(env["num_agents"])]
    return name, env, segs, logs


# ----------------------------------------------------------------------------
# conflicts
# ----------------------------------------------------------------------------
CPA_HORIZON = 3.0      # s, how far ahead a collision counts as imminent
FRONTAL_ANGLE = 135.0  # deg between headings above which a conflict is frontal
MIN_APPROACH_ANGLE = 45.0  # robots heading the same way are not in conflict
MIN_FLAG_STEPS = 1


def find_conflicts(logs, radius):
    out = []
    coll = 2 * radius
    for i in range(len(logs)):
        for j in range(i + 1, len(logs)):
            n = min(len(logs[i]), len(logs[j]))
            A = logs[i][["x", "y"]].values[:n]
            B = logs[j][["x", "y"]].values[:n]
            steps = logs[i].step.values[:n]
            VA = np.gradient(A, DT, axis=0)
            VB = np.gradient(B, DT, axis=0)
            R = B - A
            VR = VB - VA
            D = np.linalg.norm(R, axis=1)
            vr2 = (VR * VR).sum(1)
            tc = np.where(vr2 > 1e-9, -(R * VR).sum(1) / np.maximum(vr2, 1e-12), np.inf)
            tcc = np.where(np.isfinite(tc), np.clip(tc, 0, None), 0.0)
            dc = np.linalg.norm(R + tcc[:, None] * VR, axis=1)
            with np.errstate(invalid="ignore"):
                dc = np.where(np.isfinite(tc), dc, D)
            cosang = (VA * VB).sum(1) / (np.linalg.norm(VA, axis=1) * np.linalg.norm(VB, axis=1) + 1e-12)
            ang_all = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
            flag = ((D < SENSOR_RANGE) & (tc > 0) & (tc < CPA_HORIZON) & (dc < coll)
                    & (ang_all > MIN_APPROACH_ANGLE))
            # require the collision course to persist for a few samples
            fl = flag.copy()
            for k in range(1, MIN_FLAG_STEPS):
                fl[:-k] &= flag[k:]
                fl[-k:] = False
            flag = fl
            # encounters = maximal runs with D < sensor range
            t = 0
            while t < n:
                if D[t] < SENSOR_RANGE:
                    s = t
                    while t < n and D[t] < SENSOR_RANGE:
                        t += 1
                    e = t
                    f = np.flatnonzero(flag[s:e])
                    if len(f):
                        f0 = s + f[0]
                        va, vb = VA[f0], VB[f0]
                        cosang = va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-12)
                        ang = math.degrees(math.acos(max(-1, min(1, cosang))))
                        kind = "frontal" if ang >= FRONTAL_ANGLE else "squeeze"
                        tm = s + int(np.argmin(D[s:e]))
                        out.append(dict(kind=kind, step=int(steps[tm]), i=i, j=j, angle=ang,
                                        dmin=float(D[tm]), first=int(steps[f0]),
                                        dcpa=float(dc[f0]), tcpa=float(tc[f0])))
                else:
                    t += 1
    out.sort(key=lambda c: (c["step"], c["i"], c["j"]))
    return out


# ----------------------------------------------------------------------------
# main assessment
# ----------------------------------------------------------------------------
def assess(cdir, verbose=False):
    name, env, segs, logs = load_cave(cdir)
    radius = float(env.get("agent_radius", 0.075))
    carriers = [i for i, d in enumerate(logs) if carrying_state(d)[0]]
    res = dict(name=name, outcome="NO_RESCUE", collected=False, collected_by=None,
               position=None, obstructed_by=None, obstruction_case=None,
               cooperation_flag=False, conflicts=[])
    if carriers:
        lead = min(carriers)
        start = logs[lead][["x", "y"]].values[0]
        goal = np.mean([logs[i][["x", "y"]].values[-1] for i in carriers], axis=0)
        planner = Planner(segs)
        path = planner.shortest(start, goal)
        obs, cases = [], []
        for i, d in enumerate(logs):
            if i in carriers:
                continue
            m = end_motion(d)
            judge = m["stop"] if m["braking"] else m["end"]
            dist = polyline_dist(judge, path)
            if verbose:
                print(f"  {name} robot {i}: v={m['v']:.3f} braking={m['braking']} "
                      f"model={m['model']} dist_to_path={dist:.3f}")
            if dist < CLEARANCE + radius:
                obs.append(i)
                cases.append("MOVING" if m["moving"] else "STATIC")
        res.update(collected=True, collected_by=sorted(carriers),
                   position=[round(float(goal[0]), 4), round(float(goal[1]), 4)],
                   cooperation_flag=len(carriers) == 2,
                   outcome="RESCUE_HARD" if obs else "RESCUE",
                   obstructed_by=obs or None, obstruction_case=cases or None)
        if verbose:
            print(f"  {name} path len {path_length(path):.3f}: {np.round(path, 3).tolist()}")
    confs = find_conflicts(logs, radius)
    res["conflicts"] = [[c["kind"], c["step"], c["i"], c["j"]] for c in confs]
    return res


def main(root):
    for cdir in sorted(glob.glob(os.path.join(root, "cave*")),
                       key=lambda p: int("".join(ch for ch in os.path.basename(p) if ch.isdigit()) or 0)):
        if not os.path.isdir(cdir):
            continue
        res = assess(cdir, verbose="-v" in sys.argv)
        with open(os.path.join(cdir, f"{res['name']}_outcome.json"), "w") as f:
            json.dump(res, f, indent=4)
        print(json.dumps(res))


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    default = "/app/caves" if os.path.isdir("/app/caves") else os.path.join(os.path.dirname(os.path.abspath(__file__)), "caves")
    main(args[0] if args else default)
