/*  solution.pl

    Queries over the knowledge bases built by a team of exploring robots.

    The knowledge bases are read (never modified) from /app/KB/:
        agent_memoryX.txt     points, edges, outlines, objects, picked objects
        agent_navMemoryX.txt  experience graph (nodeNav), paths (pathNav),
                              start node, nodes close to objects, ...

    Exported predicates
        current_can_retrieve(+R, +C, +P, ?Cond)
            Cond in {none, explore, not_reachable, not_object}
        current_can_reach(+R, +P, ?Cond)
            Cond in {none, explore, unreachable}
        new_can_retrieve(+C, +P)
        new_can_reach(+Start, +P)
        new_can_reach(+P)

    World model (all distances on the XY plane, Z is ignored)
      * Robots have a diameter of 0.15m (radius 0.075m) and see 0.5m around
        them.  A robot reaches a point (or an object) that is on sight from a
        place it can get to.
      * Obstacles are the known edges.  Edges that belong to an object
        outline, or to an object that has been picked up, are not obstacles.
        Gaps between known edge end points narrower than the robot diameter
        are closed, since a robot cannot pass through them.
      * Places a robot knows how to get to: its experience graph (start node,
        nodeNav nodes linked through their connections), the nodes of the
        paths it computed, the checkpoints it explored and its own
        positions.  Two such places are also linked when they are within the
        line of sight and the straight segment between them leaves enough
        clearance for the robot body.
      * A robot knows how to reach P (Cond none) when P is on sight from one
        of the places above that is connected to its start.
      * A robot knows P cannot be reached (unreachable / not_reachable) when,
        even assuming every unknown area to be free, there is no place it
        could move to (flood fill of the free space from the places it
        visited, using the robot radius as clearance) from which P is on
        sight.
      * Otherwise the robot would need to explore (explore).
      * A robot knows an object of color C is at P when it has an object of
        color C whose outline is centred at P, or a node close to an object
        of color C (nodeObject) that has P on sight.
      * A robot knows there is no object at P when it picked up an object at
        P, or when it knows that a different object is at P.
      * A new robot uses the knowledge of the whole team: the union of all
        maps and graphs.  Objects picked up by any robot are gone.
*/

:- module(solution,
          [ current_can_retrieve/4,
            new_can_retrieve/2,
            new_can_reach/1,
            new_can_reach/2,
            current_can_reach/3
          ]).

:- use_module(library(lists)).
:- use_module(library(apply)).
:- use_module(library(assoc)).
:- use_module(library(pairs)).

:- dynamic kb_fact/2.            % kb_fact(RobotId, Term)
:- dynamic kb_robot/1.           % kb_robot(RobotId)
:- dynamic kb_loaded/0.

% Caches of derived data (the knowledge bases themselves are never touched)
:- dynamic c_obs/2.              % c_obs(Scope, Obstacles)
:- dynamic c_graph/5.            % c_graph(Scope, Vertices, VAssoc, LAssoc, Seeds)
:- dynamic c_reach/2.            % c_reach(Scope, Points)
:- dynamic c_grid/6.             % c_grid(R, X0, Y0, G, NX, NY)
:- dynamic c_cell/3.             % c_cell(R, I, J)  free cell reached by flood
:- dynamic c_border/1.           % c_border(R)      flood reached grid border
:- dynamic c_flooded/1.          % c_flooded(R)

kb_dir('/app/KB/').

sight(0.5).                      % line of sight
radius(0.075).                   % robot radius (diameter 0.15)
diameter(0.15).
clearance(0.07).                 % clearance used to move between places
obj_tol(0.12).                   % tolerance to match an object location
grid_step(0.05).                 % flood fill resolution
grid_margin(0.6).

% ---------------------------------------------------------------------------
% Loading the knowledge bases
% ---------------------------------------------------------------------------

ensure_kb :- kb_loaded, !.
ensure_kb :-
    retractall(kb_fact(_, _)),
    retractall(kb_robot(_)),
    clear_caches,
    kb_dir(Dir),
    (   exists_directory(Dir)
    ->  directory_files(Dir, Files),
        findall(XA, ( member(F, Files), kb_file(F, _, XA) ), XAs0),
        sort(XAs0, XAs),
        forall(member(XA, XAs), load_robot(Dir, XA))
    ;   true
    ),
    assertz(kb_loaded).

clear_caches :-
    retractall(c_obs(_, _)),
    retractall(c_graph(_, _, _, _, _)),
    retractall(c_reach(_, _)),
    retractall(c_grid(_, _, _, _, _, _)),
    retractall(c_cell(_, _, _)),
    retractall(c_border(_)),
    retractall(c_flooded(_)).

kb_file(F, Kind, XA) :-
    atom(F),
    member(Kind-Prefix, [mem-'agent_memory', nav-'agent_navMemory']),
    atom_concat(Prefix, Rest, F),
    atom_concat(XA, '.txt', Rest),
    XA \== ''.

load_robot(Dir, XA) :-
    atomic_list_concat([Dir, 'agent_memory', XA, '.txt'], MemF),
    atomic_list_concat([Dir, 'agent_navMemory', XA, '.txt'], NavF),
    read_kb_file(MemF, MemTerms),
    read_kb_file(NavF, NavTerms),
    (   memberchk(agent_ID(Id0), MemTerms), nonvar(Id0)
    ->  Id = Id0
    ;   atom_number(XA, N)
    ->  Id = N
    ;   Id = XA
    ),
    (   kb_robot(Id) -> true ; assertz(kb_robot(Id)) ),
    forall(member(T, MemTerms), assertz(kb_fact(Id, T))),
    forall(member(T, NavTerms), assertz(kb_fact(Id, T))).

read_kb_file(File, Terms) :-
    exists_file(File),
    !,
    catch(setup_call_cleanup(open(File, read, S),
                             read_all(S, Terms),
                             close(S)),
          _, Terms = []).
read_kb_file(_, []).

read_all(S, Terms) :-
    catch(read_term(S, T, []), _, T = '$kb_skip'),
    (   T == end_of_file
    ->  Terms = []
    ;   T == '$kb_skip'
    ->  (   at_end_of_stream(S) -> Terms = [] ; read_all(S, Terms) )
    ;   compound(T), T = (:- _)
    ->  read_all(S, Terms)
    ;   Terms = [T|Rest],
        read_all(S, Rest)
    ).

kb(R, T) :- kb_fact(R, T).

scope_robot(robot(R), R).
scope_robot(all, R) :- kb_robot(R).

% ---------------------------------------------------------------------------
% Geometry (XY plane)
% ---------------------------------------------------------------------------

pos_xy(L, p(X, Y)) :-
    nonvar(L),
    L = [X0, Y0|_],
    number(X0), number(Y0),
    X is float(X0),
    Y is float(Y0).

key(p(X, Y), k(IX, IY)) :-
    IX is round(X*100),
    IY is round(Y*100).

dist(p(X1, Y1), p(X2, Y2), D) :-
    DX is X1-X2, DY is Y1-Y2,
    D is sqrt(DX*DX+DY*DY).

orient(p(AX, AY), p(BX, BY), p(CX, CY), O) :-
    O is (BX-AX)*(CY-AY) - (BY-AY)*(CX-AX).

% Segments A-B and C-D cross each other at a single interior point
proper_cross(A, B, C, D) :-
    orient(C, D, A, O1), orient(C, D, B, O2),
    O1*O2 < 0,
    orient(A, B, C, O3), orient(A, B, D, O4),
    O3*O4 < 0.

pt_seg_dist(p(PX, PY), p(AX, AY), p(BX, BY), D) :-
    DX is BX-AX, DY is BY-AY,
    L2 is DX*DX+DY*DY,
    (   L2 =< 1.0e-18
    ->  T = 0.0
    ;   T0 is ((PX-AX)*DX+(PY-AY)*DY)/L2,
        T is max(0.0, min(1.0, T0))
    ),
    EX is PX-AX-T*DX, EY is PY-AY-T*DY,
    D is sqrt(EX*EX+EY*EY).

seg_seg_dist(P1, P2, P3, P4, D) :-
    (   proper_cross(P1, P2, P3, P4)
    ->  D = 0.0
    ;   pt_seg_dist(P1, P3, P4, D1),
        pt_seg_dist(P2, P3, P4, D2),
        pt_seg_dist(P3, P1, P2, D3),
        pt_seg_dist(P4, P1, P2, D4),
        D is min(min(D1, D2), min(D3, D4))
    ).

mk_seg(A-B, s(A, B, MinX, MinY, MaxX, MaxY)) :-
    A = p(AX, AY), B = p(BX, BY),
    MinX is min(AX, BX), MaxX is max(AX, BX),
    MinY is min(AY, BY), MaxY is max(AY, BY).

% P is on sight from A: close enough and no obstacle in between.  The end of
% the sight segment is pulled back 1cm so that points lying on a wall face
% can be seen from the free side.
visible(Obs, A, P) :-
    dist(A, P, L),
    sight(S),
    L =< S + 1.0e-9,
    (   L < 0.011
    ->  true
    ;   A = p(AX, AY), P = p(PX, PY),
        K is (L-0.01)/L,
        QX is AX+(PX-AX)*K, QY is AY+(PY-AY)*K,
        Q = p(QX, QY),
        BX0 is min(AX, QX), BX1 is max(AX, QX),
        BY0 is min(AY, QY), BY1 is max(AY, QY),
        \+ ( member(s(C, D, MinX, MinY, MaxX, MaxY), Obs),
             MaxX >= BX0, MinX =< BX1, MaxY >= BY0, MinY =< BY1,
             proper_cross(A, Q, C, D) )
    ).

% The robot body can travel along the straight segment A-B
passable(Obs, A, B) :-
    clearance(Cl),
    A = p(AX, AY), B = p(BX, BY),
    BX0 is min(AX, BX) - Cl, BX1 is max(AX, BX) + Cl,
    BY0 is min(AY, BY) - Cl, BY1 is max(AY, BY) + Cl,
    \+ ( member(s(C, D, MinX, MinY, MaxX, MaxY), Obs),
         MaxX >= BX0, MinX =< BX1, MaxY >= BY0, MinY =< BY1,
         seg_seg_dist(A, B, C, D, Dd),
         Dd < Cl ).

% ---------------------------------------------------------------------------
% Obstacles known in a scope (robot(R) or all)
% ---------------------------------------------------------------------------

rpoint(R, Id, P) :-
    kb(R, point(Id, L)),
    pos_xy(L, P),
    !.

obstacles(Scope, Obs) :-
    c_obs(Scope, Obs0), !,
    Obs = Obs0.
obstacles(Scope, Obs) :-
    findall(Q, ( scope_robot(Scope, R),
                 kb(R, objectPickedUp(_, _, L)),
                 pos_xy(L, Q) ),
            Picked),
    findall(A-B, ( scope_robot(Scope, R),
                   kb(R, edge(_, EP)),
                   nonvar(EP), EP = [I, J|_],
                   rpoint(R, I, A),
                   rpoint(R, J, B),
                   \+ object_edge(R, I, J),
                   \+ picked_edge(Picked, A, B) ),
            Segs0),
    sort(Segs0, Segs),
    findall(E, ( member(A-B, Segs), ( E = A ; E = B ) ), Ends0),
    sort(Ends0, Ends),
    diameter(Dm),
    findall(A-B, ( append(_, [A|Rest], Ends),
                   member(B, Rest),
                   dist(A, B, D),
                   D > 1.0e-6, D < Dm ),
            Gaps),
    append(Segs, Gaps, All),
    maplist(mk_seg, All, Obs),
    assertz(c_obs(Scope, Obs)).

object_edge(R, I, J) :-
    kb(R, object(_, Pts, _)),
    is_list(Pts),
    memberchk(I, Pts),
    memberchk(J, Pts),
    !.

picked_edge(Picked, A, B) :-
    member(Q, Picked),
    dist(A, Q, D1), D1 =< 0.15,
    dist(B, Q, D2), D2 =< 0.15,
    !.

% ---------------------------------------------------------------------------
% Places a robot knows how to get to
% ---------------------------------------------------------------------------

robot_pos(R, P) :- robot_seed(R, P).
robot_pos(R, P) :- kb(R, nodeNav(_, L, _, _, _)), pos_xy(L, P).
robot_pos(R, P) :-
    kb(R, pathNav(_, O, Path, Opt)),
    (   pos_xy(O, P)
    ;   member(Pa, [Path, Opt]),
        is_list(Pa),
        member(E, Pa),
        path_elem_pos(E, P)
    ).
robot_pos(R, P) :- kb(R, agent_lastPos(L, _)), pos_xy(L, P).
robot_pos(R, P) :- kb(R, agent_checkpointCurrent(L)), pos_xy(L, P).
robot_pos(R, P) :- kb(R, agent_checkpointLast(L)), pos_xy(L, P).
robot_pos(R, P) :-
    kb(R, agent_checkpointExplored(Ls)),
    is_list(Ls),
    member(L, Ls),
    pos_xy(L, P).

robot_seed(R, P) :- kb(R, nodeNavStart(L)), pos_xy(L, P).
robot_seed(R, P) :- kb(R, agent_Pos(L, _)), pos_xy(L, P).

path_elem_pos(E, P) :-
    nonvar(E),
    E = [_, _, L|_],
    pos_xy(L, P).

robot_link(R, A, B) :-
    kb(R, nodeNav(_, L, _, Con, _)),
    is_list(Con),
    pos_xy(L, A),
    member(C, Con),
    kb(R, nodeNav(C, L2, _, _, _)),
    pos_xy(L2, B).
robot_link(R, A, B) :-
    kb(R, nodeNav(_, L, Par, _, _)),
    nonvar(Par), Par \== [],
    pos_xy(L, A),
    kb(R, nodeNav(Par, L2, _, _, _)),
    pos_xy(L2, B).
robot_link(R, A, B) :-
    kb(R, pathNav(_, O, Path, Opt)),
    member(Pa, [Path, Opt]),
    is_list(Pa), Pa \== [],
    findall(Q, ( member(E, Pa), path_elem_pos(E, Q) ), Qs),
    (   pos_xy(O, P0) -> Chain = [P0|Qs] ; Chain = Qs ),
    append(_, [A, B|_], Chain).

kp(P, K-P) :- key(P, K).

scope_graph(Scope, Vs, VA, LA, Seeds) :-
    c_graph(Scope, Vs0, VA0, LA0, Seeds0), !,
    Vs = Vs0, VA = VA0, LA = LA0, Seeds = Seeds0.
scope_graph(Scope, Vs, VA, LA, Seeds) :-
    findall(P, ( scope_robot(Scope, R), robot_pos(R, P) ), Ps0),
    findall(A-B, ( scope_robot(Scope, R), robot_link(R, A, B) ), Ls0),
    findall(A, member(A-_, Ls0), LPs1),
    findall(B, member(_-B, Ls0), LPs2),
    append([Ps0, LPs1, LPs2], Ps1),
    maplist(kp, Ps1, KPs0),
    sort(1, @<, KPs0, Vs),
    list_to_assoc(Vs, VA),
    findall(X-Y, ( member(A-B, Ls0),
                   key(A, KA), key(B, KB),
                   KA \== KB,
                   ( X-Y = KA-KB ; X-Y = KB-KA ) ),
            KL0),
    sort(KL0, KL),
    group_pairs_by_key(KL, Groups),
    list_to_assoc(Groups, LA),
    findall(K, ( scope_robot(Scope, R), robot_seed(R, P), key(P, K) ), Ss0),
    sort(Ss0, Ss1),
    (   Ss1 == [] -> pairs_keys(Vs, Seeds) ; Seeds = Ss1 ),
    assertz(c_graph(Scope, Vs, VA, LA, Seeds)).

% Breadth first traversal of the graph of places
bfs([], _, _, _, _, Vis, Vis).
bfs([K|Q], Vs, VA, LA, Obs, Vis0, Vis) :-
    (   get_assoc(K, VA, P)
    ->  (   get_assoc(K, LA, Ls) -> true ; Ls = [] ),
        findall(K2, ( member(K2, Ls),
                      \+ get_assoc(K2, Vis0, _),
                      get_assoc(K2, VA, _) ),
                N1),
        sight(S),
        findall(K2, ( member(K2-P2, Vs),
                      \+ get_assoc(K2, Vis0, _),
                      dist(P, P2, D), D =< S,
                      passable(Obs, P, P2) ),
                N2),
        append(N1, N2, N0),
        sort(N0, Ns),
        foldl(mark, Ns, Vis0, Vis1),
        append(Q, Ns, Q1)
    ;   Vis1 = Vis0,
        Q1 = Q
    ),
    bfs(Q1, Vs, VA, LA, Obs, Vis1, Vis).

mark(K, V0, V) :- put_assoc(K, V0, t, V).

% Places reachable from the seeds (start positions) of the scope
reach_set(Scope, Pts) :-
    c_reach(Scope, Pts0), !,
    Pts = Pts0.
reach_set(Scope, Pts) :-
    scope_graph(Scope, Vs, VA, LA, Seeds),
    obstacles(Scope, Obs),
    empty_assoc(E),
    foldl(mark, Seeds, E, V0),
    bfs(Seeds, Vs, VA, LA, Obs, V0, V),
    assoc_to_keys(V, Ks),
    findall(P, ( member(K, Ks), get_assoc(K, VA, P) ), Pts),
    assertz(c_reach(Scope, Pts)).

% The scope knows how to reach P
known_reach(Scope, P) :-
    reach_set(Scope, Pts),
    obstacles(Scope, Obs),
    member(V, Pts),
    visible(Obs, V, P),
    !.

% ---------------------------------------------------------------------------
% Optimistic reachability of a robot (unknown space assumed free)
% ---------------------------------------------------------------------------

flood(R) :- c_flooded(R), !.
flood(R) :-
    obstacles(robot(R), Obs),
    reach_set(robot(R), Seeds),
    findall(X-Y, ( kb(R, point(_, L)), pos_xy(L, p(X, Y)) ), XY1),
    findall(X-Y, member(p(X, Y), Seeds), XY2),
    append(XY1, XY2, XY),
    (   XY == []
    ->  true
    ;   pairs_keys_values(XY, Xs, Ys),
        min_list(Xs, MinX), max_list(Xs, MaxX),
        min_list(Ys, MinY), max_list(Ys, MaxY),
        grid_step(G), grid_margin(M),
        X0 is MinX - M, Y0 is MinY - M,
        NX is ceiling((MaxX + M - X0)/G),
        NY is ceiling((MaxY + M - Y0)/G),
        assertz(c_grid(R, X0, Y0, G, NX, NY)),
        findall(I-J, ( member(p(X, Y), Seeds),
                       I is round((X-X0)/G), J is round((Y-Y0)/G),
                       I >= 0, I =< NX, J >= 0, J =< NY,
                       cell_free(Obs, X0, Y0, G, I, J) ),
                SC0),
        sort(SC0, SC),
        forall(member(I-J, SC), assertz(c_cell(R, I, J))),
        flood_loop(SC, R, Obs, X0, Y0, G, NX, NY)
    ),
    assertz(c_flooded(R)).

cell_free(Obs, X0, Y0, G, I, J) :-
    radius(Rd),
    CX is X0 + I*G, CY is Y0 + J*G,
    \+ ( member(s(A, B, MinX, MinY, MaxX, MaxY), Obs),
         MinX - Rd =< CX, CX =< MaxX + Rd,
         MinY - Rd =< CY, CY =< MaxY + Rd,
         pt_seg_dist(p(CX, CY), A, B, D),
         D < Rd ).

nb4(I, J, I2, J) :- I2 is I+1.
nb4(I, J, I2, J) :- I2 is I-1.
nb4(I, J, I, J2) :- J2 is J+1.
nb4(I, J, I, J2) :- J2 is J-1.

flood_loop([], _, _, _, _, _, _, _).
flood_loop([I-J|T], R, Obs, X0, Y0, G, NX, NY) :-
    findall(I2-J2,
            ( nb4(I, J, I2, J2),
              (   I2 >= 0, I2 =< NX, J2 >= 0, J2 =< NY
              ->  \+ c_cell(R, I2, J2),
                  cell_free(Obs, X0, Y0, G, I2, J2)
              ;   (   c_border(R) -> true ; assertz(c_border(R)) ),
                  fail
              ) ),
            New),
    forall(member(I2-J2, New), assertz(c_cell(R, I2, J2))),
    append(New, T, T1),
    flood_loop(T1, R, Obs, X0, Y0, G, NX, NY).

% Robot R may be able to reach P (after exploring)
maybe_reach(R, P) :-
    flood(R),
    c_grid(R, X0, Y0, G, NX, NY),
    obstacles(robot(R), Obs),
    P = p(PX, PY),
    I0 is round((PX-X0)/G),
    J0 is round((PY-Y0)/G),
    (   ( I0 < 0 ; I0 > NX ; J0 < 0 ; J0 > NY )
    ->  c_border(R)
    ;   sight(S),
        K is ceiling(S/G) + 1,
        IL is max(0, I0-K), IH is min(NX, I0+K),
        JL is max(0, J0-K), JH is min(NY, J0+K),
        between(IL, IH, I),
        between(JL, JH, J),
        c_cell(R, I, J),
        CX is X0 + I*G, CY is Y0 + J*G,
        visible(Obs, p(CX, CY), P)
    ),
    !.

% ---------------------------------------------------------------------------
% Objects
% ---------------------------------------------------------------------------

obj_point(R, E, P) :-
    (   is_list(E) -> pos_xy(E, P) ; rpoint(R, E, P) ).

centroid(R, Pts, p(CX, CY)) :-
    is_list(Pts),
    findall(X-Y, ( member(E, Pts), obj_point(R, E, p(X, Y)) ), XY),
    XY \== [],
    pairs_keys_values(XY, Xs, Ys),
    length(XY, N),
    sum_list(Xs, SX), sum_list(Ys, SY),
    CX is SX/N, CY is SY/N.

object_at(R, C, P) :-
    kb(R, object(_, Pts, C)),
    centroid(R, Pts, Cen),
    dist(P, Cen, D),
    obj_tol(Tol),
    D =< Tol.

% The scope knows that an object of color C is at P
object_known(Scope, C, P) :-
    scope_robot(Scope, R),
    (   object_at(R, C, P)
    ;   kb(R, nodeObject(_, C, L)),
        pos_xy(L, N),
        dist(P, N, D),
        sight(S),
        D =< S + 1.0e-9
    ),
    !.

% An object was picked up at P
picked_at(Scope, P) :-
    scope_robot(Scope, R),
    kb(R, objectPickedUp(_, _, L)),
    pos_xy(L, Q),
    dist(P, Q, D),
    obj_tol(Tol),
    D =< Tol,
    !.

% A different object is known to be at P
other_object_at(Scope, C, P) :-
    scope_robot(Scope, R),
    kb(R, object(_, Pts, C2)),
    C2 \== C,
    centroid(R, Pts, Cen),
    dist(P, Cen, D),
    obj_tol(Tol),
    D =< Tol,
    !.

% ---------------------------------------------------------------------------
% Exported predicates
% ---------------------------------------------------------------------------

current_can_retrieve(R, C, P, Cond) :-
    ensure_kb,
    kb_robot(R), !,
    pos_xy(P, T),
    retrieve_cond(R, C, T, Cond0), !,
    Cond = Cond0.

retrieve_cond(R, _, T, not_reachable) :-
    \+ known_reach(robot(R), T),
    \+ maybe_reach(R, T), !.
retrieve_cond(R, _, T, not_object) :-
    picked_at(robot(R), T), !.
retrieve_cond(R, C, T, Cond) :-
    object_known(robot(R), C, T), !,
    (   known_reach(robot(R), T) -> Cond = none ; Cond = explore ).
retrieve_cond(R, C, T, not_object) :-
    other_object_at(robot(R), C, T), !.
retrieve_cond(_, _, _, explore).

current_can_reach(R, P, Cond) :-
    ensure_kb,
    kb_robot(R), !,
    pos_xy(P, T),
    (   known_reach(robot(R), T)
    ->  Cond0 = none
    ;   maybe_reach(R, T)
    ->  Cond0 = explore
    ;   Cond0 = unreachable
    ),
    Cond = Cond0.

new_can_retrieve(C, P) :-
    ensure_kb,
    pos_xy(P, T),
    object_known(all, C, T),
    \+ picked_at(all, T),
    known_reach(all, T),
    !.

new_can_reach(Start, P) :-
    ensure_kb,
    pos_xy(Start, S),
    pos_xy(P, T),
    obstacles(all, Obs),
    (   visible(Obs, S, T)
    ->  true
    ;   scope_graph(all, Vs, VA, LA, _),
        sight(Sg),
        findall(K, ( member(K-V, Vs),
                     dist(S, V, D), D =< Sg,
                     passable(Obs, S, V) ),
                Seeds0),
        sort(Seeds0, Seeds),
        Seeds \== [],
        empty_assoc(E),
        foldl(mark, Seeds, E, V0),
        bfs(Seeds, Vs, VA, LA, Obs, V0, Vis),
        assoc_to_keys(Vis, Ks),
        member(K, Ks),
        get_assoc(K, VA, W),
        visible(Obs, W, T)
    ),
    !.

new_can_reach(P) :-
    ensure_kb,
    pos_xy(P, T),
    known_reach(all, T),
    !.

:- initialization(catch(ensure_kb, _, true)).
