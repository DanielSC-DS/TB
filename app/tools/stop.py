from geom import *
STILL_V = 0.01   # m/s: below this the robot is considered still

def stop_pred(d, M=6):
    p = d[['x', 'y']].values
    s = np.linalg.norm(np.diff(p, axis=0), axis=1) / DT   # speed from positions (cleaner than noisy v readings)
    tail = s[-M:]; k = np.arange(M)
    heading = p[-1] - p[-2]; nh = np.linalg.norm(heading)
    u = heading / nh if nh > 1e-12 else np.zeros(2)
    out = dict(v_end=s[-1], u=u, last=p[-1])
    dv = np.diff(tail)
    if s[-1] < STILL_V:
        out.update(model='still', dist=0.0, moving=False)
    elif (dv < 0).sum() < len(dv) - 1 or dv.mean() > -1e-4:
        out.update(model='cruise', dist=0.0, moving=True)
    else:
        lin = np.polyfit(k, tail, 1); rl = np.abs(np.polyval(lin, k) - tail).max()
        ex = np.polyfit(k, np.log(tail), 1); re = np.abs(np.exp(np.polyval(ex, k)) - tail).max()
        if re <= rl:
            r = np.exp(ex[0]); v = np.exp(np.polyval(ex, M - 1))
            out.update(model='exp', r=r, dist=v * DT * r / (1 - r))
        else:
            a = -lin[0] / DT; v = np.polyval(lin, M - 1)
            out.update(model='lin', a=a, dist=v * v / (2 * a))
        out['moving'] = True
        out['dist_lin'] = s[-1] ** 2 / (2 * max(1e-6, -dv.mean() / DT))
        r = np.clip((tail[1:] / tail[:-1]).mean(), 0, 0.9999)
        out['dist_exp'] = s[-1] * DT * r / (1 - r)
    out['stop'] = p[-1] + u * out['dist']
    return out
