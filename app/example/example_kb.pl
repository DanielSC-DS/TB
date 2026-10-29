% ---- A small readable knowledge base: "should the robot pick up the sample?" ------------------
% bfact(Id, Atom, From, To): the robot BELIEVES Atom is true for From =< T < To   (To = inf: never ends)
bfact(f1, sees_sample,      0, 20).      % camera 1 sees the sample
bfact(f2, arm_free,         0, 50).
bfact(f3, map_says_barren, 10, 14).      % an old map flags the spot as barren for a while
bfact(f4, sees_sample,     15, 40).      % camera 2 sees the sample (second, independent source)
bfact(f5, low_battery,     25, inf).

% rule(Id, Head, Body): Head holds if every literal in Body holds; not(X) holds if X is NOT derivable
rule(r1, barren,      [map_says_barren]).
rule(r2, can_pick,    [sees_sample, arm_free, not(barren)]).
rule(r3, decide_pick, [can_pick, not(low_battery)]).

% world(Atom, From, To): ground truth. If nothing covers time T, the atom was really false at T.
world(sees_sample,      0, 12).          % the sample was really visible only until t = 12
world(arm_free,         0, 50).
world(map_says_barren, 10, 14).
world(low_battery,     25, inf).
