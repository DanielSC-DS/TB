import numpy as np

def gripper_events(d):
    """Detect complete pick-up and put-down sequences.
    pick: arm fully extended (phi~pi) with gripper open -> gripper closes and
          stalls on an object (high current, opening < 1) -> arm retracts to ~0
          while still holding -> gripper opens (object rests on robot).
    drop: gripper closes at phi~0 on the object (stall) -> arm extends to pi
          while holding -> gripper opens at pi."""
    phi, op, cur = d.phi.values, d.open.values, d.curr.values
    st = d.step.values
    N = len(d); ev = []
    i = 0
    HOLD = 2.0
    while i < N:
        # find closing start
        if op[i] < 0.005 and i + 1 < N and op[i + 1] >= 0.005:
            start_phi = phi[i]
            j = i + 1
            # wait until stall (current high and opening constant) or full close
            while j < N and not (cur[j] > HOLD and abs(op[j] - op[j - 1]) < 1e-6):
                j += 1
            if j >= N: break
            grip = op[j]
            if grip > 0.99:  # closed on nothing
                i = j; continue
            # holding: track until the gripper starts opening
            k = j
            while k < N and op[k] >= grip - 1e-3:
                k += 1
            phi_at_open = phi[k - 1] if k <= N else phi[-1]
            completed = k < N
            if start_phi > 3.1 and completed and phi_at_open < 0.1:
                ev.append(('pick', int(st[j]), int(st[k - 1])))
            elif start_phi < 0.1 and completed and phi_at_open > 3.1:
                ev.append(('drop', int(st[j]), int(st[k - 1])))
            elif not completed:
                ev.append(('holding_at_end', int(st[j]), start_phi))
            else:
                ev.append(('aborted', int(st[j]), round(float(start_phi), 2), round(float(phi_at_open), 2)))
            i = k
        else:
            i += 1
    return ev

def carrying_at_end(d):
    c = False
    for e in gripper_events(d):
        if e[0] == 'pick': c = True
        elif e[0] == 'drop': c = False
    return c
