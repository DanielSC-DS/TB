import json, heapq, numpy as np, pandas as pd, os
BASE = os.path.join(os.path.dirname(__file__), '..', 'caves')
DT = 0.05

def load(c):
    env = json.load(open(f'{BASE}/cave{c}/cave{c}_env.json'))
    ags = [pd.read_csv(f'{BASE}/cave{c}/cave{c}_agent_{a}.csv') for a in range(env['num_agents'])]
    return env, np.array(env['scenario'], float), ags

def seg_dist(P, W):
    """min distance from points P (N,2) to segments W (M,2,2)"""
    P = np.asarray(P, float).reshape(-1, 2)
    A = W[:, 0][None]; B = W[:, 1][None]
    AB = B - A; AP = P[:, None] - A
    t = np.clip((AP * AB).sum(-1) / (AB * AB).sum(-1), 0, 1)
    D = AP - t[..., None] * AB
    return np.sqrt((D ** 2).sum(-1)).min(1)

def pt_poly_dist(p, poly):
    """min distance from point p to polyline poly (K,2)"""
    W = np.stack([poly[:-1], poly[1:]], 1)
    if len(W) == 0: return np.linalg.norm(poly[0] - p)
    return seg_dist(p, W)[0]

class Planner:
    def __init__(self, W, clr, res=0.01, size=5.0):
        self.W, self.clr, self.res = W, clr, res
        n = int(round(size / res)) + 1
        self.n = n
        xs = np.arange(n) * res
        X, Y = np.meshgrid(xs, xs, indexing='ij')
        P = np.c_[X.ravel(), Y.ravel()]
        D = np.concatenate([seg_dist(P[i:i + 20000], W) for i in range(0, len(P), 20000)])
        self.D = D.reshape(n, n)
        self.free = self.D >= clr

    def idx(self, p):
        return tuple(np.clip(np.round(np.asarray(p) / self.res).astype(int), 0, self.n - 1))

    def los(self, a, b):
        L = np.linalg.norm(b - a)
        k = max(2, int(L / (self.res / 2)) + 1)
        t = np.linspace(0, 1, k)[:, None]
        pts = a + t * (b - a)
        return seg_dist(pts, self.W).min() >= self.clr - 1e-9

    def plan(self, s, g):
        s = np.asarray(s, float); g = np.asarray(g, float)
        si, gi = self.idx(s), self.idx(g)
        free = self.free.copy()
        info = dict(start_clear=seg_dist(s, self.W)[0], goal_clear=seg_dist(g, self.W)[0])
        free[si] = True; free[gi] = True
        moves = [(dx, dy) for dx in (-2, -1, 0, 1, 2) for dy in (-2, -1, 0, 1, 2)
                 if (dx, dy) != (0, 0) and np.gcd(abs(dx), abs(dy)) == 1]
        dist = {si: 0.0}; prev = {}
        pq = [(0.0, si)]; n = self.n
        while pq:
            d, u = heapq.heappop(pq)
            if u == gi: break
            if d > dist.get(u, 1e18): continue
            for dx, dy in moves:
                v = (u[0] + dx, u[1] + dy)
                if not (0 <= v[0] < n and 0 <= v[1] < n) or not free[v]: continue
                if abs(dx) == 2 or abs(dy) == 2:
                    m1 = (u[0] + int(np.sign(dx)) * (abs(dx) == 2), u[1] + int(np.sign(dy)) * (abs(dy) == 2))
                    if not free[m1]: continue
                nd = d + np.hypot(dx, dy) * self.res
                if nd < dist.get(v, 1e18):
                    dist[v] = nd; prev[v] = u; heapq.heappush(pq, (nd, v))
        if gi not in dist:
            info['found'] = False
            return None, info
        path = [gi]
        while path[-1] != si: path.append(prev[path[-1]])
        path = np.array(path[::-1], float) * self.res
        path[0] = s; path[-1] = g
        info['found'] = True; info['grid_len'] = dist[gi]
        sm = self.smooth(path)
        info['len'] = np.linalg.norm(np.diff(sm, axis=0), axis=1).sum()
        return sm, info

    def smooth(self, path):
        out = [path[0]]; i = 0; N = len(path)
        while i < N - 1:
            # furthest j visible from i (binary-ish scan from end)
            j = N - 1
            lo = i + 1
            # linear scan backwards with step for speed
            best = lo
            for j in range(N - 1, i, -1):
                if self.los(path[i], path[j]): best = j; break
            out.append(path[best]); i = best
        return np.array(out)
