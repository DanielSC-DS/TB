"""Cave rescue assessment.

For each caveX folder: detect which robot(s) carry the object at the end of the
logs, plan the shortest wall-clearance (0.35) path for the rescue team, find
robots obstructing it (using their expected stop position when braking), and
detect frontal / squeeze conflicts between robots. Writes caveX_outcome.json.
"""
import json, os, sys
import numpy as np
from geom import load, seg_dist, pt_poly_dist, Planner, DT, BASE
from carry import gripper_events, carrying_at_end
from conf2 import ang

CLEARANCE = 0.35       # rescue corridor half-width, from the path centre line
STILL_V = 0.01         # m/s, below this a robot is considered still
BRAKE_DECEL = 0.05     # m/s^2, deceleration that counts as braking (vs cruising)
SENSE_R = 0.5          # detection range of the robot sensor
CPA_MAX = 0.15         # closest-approach distance that means collision (2 radii)
TC_MAX = 3.0           # s, horizon for an "imminent" collision
V_MIN = 0.05           # m/s, both robots must be moving
FRONTAL_ANG = 150.0    # deg between headings: opposite directions
SAME_DIR_ANG = 45.0    # deg: below this the robots travel the same way (no conflict)


def terminal_ratio(s):
    """Per-step speed ratio of this robot's own exponential (P-controller) stops."""
    rs = [s[i + 1] / s[i] for i in range(len(s) - 1)
          if 0.03 < s[i] < 0.15 and s[i + 1] < s[i] and s[i + 1] / s[i] > 0.9]
    return float(np.median(rs)) if rs else 0.974


def stop_prediction(d):
    p = d[['x', 'y']].values
    s = np.linalg.norm(np.diff(p, axis=0), axis=1) / DT   # speed from positions (velocity readings are noisy)
    h = p[-1] - p[-2]
    u = h / np.linalg.norm(h) if np.linalg.norm(h) > 1e-12 else np.zeros(2)
    res = dict(last=p[-1], v=s[-1], moving=s[-1] >= STILL_V, dist=0.0, model='still')
    if not res['moving']:
        return res
    dv = np.diff(s[-7:])
    decel = -dv[-3:].mean() / DT
    braking = (decel >= BRAKE_DECEL and np.all(dv[-4:] < 0)           # sustained speed reduction
               and np.abs(np.diff(dv[-4:])).max() < 0.005)            # smooth, not sensor jitter
    if not braking:
        res['model'] = 'cruise'           # not reducing speed: judged where it was last seen
        return res
    mags = -dv[-5:]
    if np.all(np.diff(mags) > 0) and mags[-1] > 0:
        # braking onset (deceleration still building up): the robot's own
        # exponential stopping behaviour gives the expected remaining distance
        r = terminal_ratio(s)
        res.update(model='onset-exp', dist=s[-1] * DT * r / (1 - r))
    else:
        tail = s[-6:]; k = np.arange(6)
        lin = np.polyfit(k, tail, 1); el = np.abs(np.polyval(lin, k) - tail).max()
        ex = np.polyfit(k, np.log(tail), 1); ee = np.abs(np.exp(np.polyval(ex, k)) - tail).max()
        if ee <= el:
            r = np.exp(ex[0]); v = np.exp(np.polyval(ex, 5))
            res.update(model='exp', dist=v * DT * r / (1 - r))
        else:
            a = -lin[0] / DT; v = np.polyval(lin, 5)
            res.update(model='lin', dist=v * v / (2 * a))
    res['stop'] = p[-1] + u * res['dist']
    return res


def find_conflicts(ags):
    out = []
    for i in range(len(ags)):
        for j in range(i + 1, len(ags)):
            a = ags[i].set_index('step'); b = ags[j].set_index('step')
            st = a.index.intersection(b.index)
            pa = a.loc[st, ['x', 'y']].values; pb = b.loc[st, ['x', 'y']].values
            va = np.gradient(pa, axis=0) / DT; vb = np.gradient(pb, axis=0) / DT
            dist = np.linalg.norm(pa - pb, axis=1)
            k = 0
            while k < len(st):
                if dist[k] >= SENSE_R:
                    k += 1; continue
                e = k
                while e + 1 < len(st) and dist[e + 1] < SENSE_R: e += 1
                m = k + int(np.argmin(dist[k:e + 1]))
                for t in range(k, m + 1):
                    if min(np.linalg.norm(va[t]), np.linalg.norm(vb[t])) < V_MIN: continue
                    dp = pb[t] - pa[t]; dvr = vb[t] - va[t]
                    tc = -np.dot(dp, dvr) / max(np.dot(dvr, dvr), 1e-9)
                    if not (0 < tc <= TC_MAX): continue
                    if np.linalg.norm(dp + tc * dvr) >= CPA_MAX: continue
                    an = ang(va[t], vb[t])
                    if an < SAME_DIR_ANG: continue
                    out.append(['frontal' if an >= FRONTAL_ANG else 'squeeze', int(st[m]), i, j])
                    break
                k = e + 1
    return sorted(out, key=lambda x: x[1])


def assess(c, verbose=True):
    env, W, ags = load(c)
    r = env['agent_radius']
    carriers = [a for a, d in enumerate(ags) if carrying_at_end(d)]
    out = dict(name=env['name'], outcome='NO_RESCUE', collected=False, collected_by=None,
               position=None, obstructed_by=None, obstruction_case=None,
               cooperation_flag=False, conflicts=find_conflicts(ags))
    if verbose:
        for a, d in enumerate(ags): print(f'  agent {a}: {gripper_events(d)}')
    if carriers:
        start = ags[min(carriers)][['x', 'y']].values[0]
        goal = np.mean([ags[a][['x', 'y']].values[-1] for a in carriers], axis=0)
        path, info = Planner(W, CLEARANCE).plan(start, goal)
        assert path is not None, 'no clear path'
        obs, case = [], []
        for a, d in enumerate(ags):
            if a in carriers: continue
            sp = stop_prediction(d)
            judged = sp.get('stop', sp['last'])
            dd = pt_poly_dist(judged, path)
            hit = dd < CLEARANCE + r
            if verbose:
                print(f"  agent {a}: v={sp['v']:.3f} {sp['model']} stop_dist={sp['dist']:.2f} "
                      f"d_last={pt_poly_dist(sp['last'], path):.3f} d_judged={dd:.3f} -> {'OBSTRUCTS' if hit else '-'}")
            if hit:
                obs.append(a); case.append('MOVING' if sp['moving'] else 'STATIC')
        out.update(collected=True, collected_by=carriers,
                   position=[round(float(goal[0]), 4), round(float(goal[1]), 4)],
                   cooperation_flag=len(carriers) == 2,
                   outcome='RESCUE_HARD' if obs else 'RESCUE',
                   obstructed_by=obs or None, obstruction_case=case or None)
        if verbose: print(f"  path len {info['len']:.3f}: {np.round(path, 3).tolist()}")
    return out


if __name__ == '__main__':
    caves = sorted(int(n[4:]) for n in os.listdir(BASE) if n.startswith('cave') and n[4:].isdigit())
    for c in caves:
        print(f'== cave{c}')
        o = assess(c)
        print('  ', json.dumps(o))
        with open(os.path.join(BASE, f'cave{c}_outcome.json'), 'w') as f:
            json.dump(o, f, indent=4)
