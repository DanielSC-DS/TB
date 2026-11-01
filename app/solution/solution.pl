/*  solution.pl

    Queries over the knowledge bases built by a team of exploration robots.

    The knowledge base in /app/KB_test/ is read (never modified) the first
    time a query is asked.  Every robot X owns two files:
        agent_memoryX.txt     (agent_ID, point, edge, outline, wall, object,
                               objectPickedUp, ...)
        agent_navMemoryX.txt  (nodeNavStart, nodeNav, pathNav, nodeObject, ...)
    The facts are read as plain terms and stored in private predicates
    (named r_...) so that the files of the different robots do not clash.

    Exported queries
        current_can_retrieve(+R, +C, +P, ?Cond)
        new_can_retrieve(+C, +P)
        new_can_reach(+Start, +P)
        current_can_reach(+R, +P, ?Cond)
        has_preference(+R, +C)

    World model
      * Robots have a diameter of 0.15 m (radius 0.075 m) and a line of
        sight of 0.5 m: a point is reached directly when it is within 0.5 m
        of a known position and the straight segment towards it does not
        cross any known wall/edge.
      * Known positions of a robot: the nodes of its experience graph
        (nodeNav) and its starting point (all of them are connected).
      * A location is known to be unreachable when, treating every unknown
        place as free, no place the robot can drive to (free space
        flood-filled from its known positions with the robot radius as
        clearance) has the location on sight.  This detects regions closed
        by walls or by openings narrower than the robot.
*/

:- use_module(library(lists)).

:- dynamic kb_loaded/0.
:- dynamic r_robot/1.            % r_robot(R)
:- dynamic r_point/4.            % r_point(R, Id, X, Y)
:- dynamic r_seg/5.              % r_seg(R, AX, AY, BX, BY)
:- dynamic r_node/3.             % r_node(R, X, Y)
:- dynamic r_object/5.           % r_object(R, Id, C, X, Y)   (centroid)
:- dynamic r_picked/6.           % r_picked(R, Order, Id, C, X, Y)
:- dynamic r_nodeobj/4.          % r_nodeobj(R, C, X, Y)
:- dynamic flood_cache/2.        % flood_cache(Ctx, Grid)

/* ------------------------------------------------------------------ */
/* Parameters                                                          */
/* ------------------------------------------------------------------ */

kb_dir('/app/KB_test/').

sight(0.5).                 % line of sight
robot_radius(0.075).        % diameter 0.15
cell_size(0.025).           % grid resolution for the free-space flood fill
max_cells(250000).          % upper bound on the number of grid cells
grid_margin(0.6).           % free margin around the known map
object_tol(0.1).            % "same position" tolerance for objects
nodeobj_tol(0.6).           % max distance object <-> closest node
closest_tol(0.05).          % slack when checking the closest node
target_ignore(0.08).        % crossings this close to the target are the
                            % target itself (e.g. the outline of an object)

/* ------------------------------------------------------------------ */
/* Loading the knowledge base                                          */
/* ------------------------------------------------------------------ */

ensure_kb :-
    kb_loaded, !.
ensure_kb :-
    catch(load_kb, _, true),
    assertz(kb_loaded).

load_kb :-
    kb_dir(Dir),
    atom_concat(Dir, 'agent_memory*.txt', Pattern),
    expand_file_name(Pattern, Files),
    forall(member(F, Files), catch(load_robot(Dir, F), _, true)).

load_robot(Dir, MemFile) :-
    file_base_name(MemFile, Base),
    atom_concat(agent_memory, Rest, Base),
    atom_concat(Suffix, '.txt', Rest),
    read_terms(MemFile, MemTerms),
    (   memberchk(agent_ID(R), MemTerms), ground(R)
    ->  true
    ;   catch(atom_number(Suffix, R), _, fail)
    ->  true
    ;   R = Suffix
    ),
    atomic_list_concat([Dir, agent_navMemory, Suffix, '.txt'], NavFile),
    (   exists_file(NavFile)
    ->  read_terms(NavFile, NavTerms)
    ;   NavTerms = []
    ),
    (   r_robot(R) -> true ; assertz(r_robot(R)) ),
    store_points(R, MemTerms),
    store_segments(R, MemTerms),
    store_objects(R, MemTerms),
    store_picked(R, MemTerms, 1),
    store_nav(R, NavTerms).

%   read_terms(+File, -Terms): all clauses of File as terms.
read_terms(File, Terms) :-
    catch(setup_call_cleanup(open(File, read, S),
                             read_all(S, Terms),
                             close(S)),
          _, Terms = []).

read_all(S, Terms) :-
    char_pos(S, P0),
    catch(read_term(S, T, []), _, T = '$read_error'),
    (   T == end_of_file
    ->  Terms = []
    ;   T == '$read_error'
    ->  char_pos(S, P1),
        (   P1 > P0 -> read_all(S, Terms) ; Terms = [] )
    ;   Terms = [T|More],
        read_all(S, More)
    ).

char_pos(S, C) :-
    (   catch(( stream_property(S, position(Pos)),
                stream_position_data(char_count, Pos, C0) ), _, fail)
    ->  C = C0
    ;   C = 0
    ).

xy([X0, Y0|_], X, Y) :-
    number(X0), number(Y0),
    X is float(X0),
    Y is float(Y0).

store_points(R, Terms) :-
    forall(( member(point(Id, P), Terms), xy(P, X, Y) ),
           assertz(r_point(R, Id, X, Y))).

store_segments(R, Terms) :-
    forall(( member(T, Terms), term_pairs(T, A, B) ),
           add_segment(R, A, B)).

term_pairs(edge(_, [A, B|_]), A, B).
term_pairs(outline(_, Pts), A, B) :- is_list(Pts), consecutive(Pts, A, B).
term_pairs(wall(_, Pts), A, B)    :- is_list(Pts), consecutive(Pts, A, B).

consecutive([A, B|_], A, B).
consecutive([_|T], A, B) :- consecutive(T, A, B).

add_segment(R, A, B) :-
    A \== B,
    r_point(R, A, AX, AY),
    r_point(R, B, BX, BY),
    !,
    assertz(r_seg(R, AX, AY, BX, BY)).
add_segment(_, _, _).

store_objects(R, Terms) :-
    forall(( member(object(Id, Pts, C), Terms),
             is_list(Pts),
             centroid(R, Pts, X, Y) ),
           assertz(r_object(R, Id, C, X, Y))).

centroid(R, Pts, X, Y) :-
    findall(PX-PY, ( member(P, Pts), r_point(R, P, PX, PY) ), L0),
    sort(L0, L),                % closed outlines repeat the first point
    L \== [],
    sum_xy(L, 0.0, 0.0, SX, SY),
    length(L, N),
    X is SX / N,
    Y is SY / N.

sum_xy([], SX, SY, SX, SY).
sum_xy([X-Y|T], AX, AY, SX, SY) :-
    BX is AX + X,
    BY is AY + Y,
    sum_xy(T, BX, BY, SX, SY).

store_picked(_, [], _).
store_picked(R, [T|Ts], N) :-
    (   T = objectPickedUp(Id, C, P), xy(P, X, Y)
    ->  assertz(r_picked(R, N, Id, C, X, Y)),
        N1 is N + 1
    ;   N1 = N
    ),
    store_picked(R, Ts, N1).

store_nav(R, Terms) :-
    forall(( member(nodeNavStart(P), Terms), xy(P, X, Y) ),
           assertz(r_node(R, X, Y))),
    forall(( member(nodeNav(_, P, _, _, _), Terms), xy(P, X, Y) ),
           assertz(r_node(R, X, Y))),
    forall(( member(nodeObject(_, C, P), Terms), xy(P, X, Y) ),
           assertz(r_nodeobj(R, C, X, Y))).

/* ------------------------------------------------------------------ */
/* Geometry                                                            */
/* ------------------------------------------------------------------ */

dist(X1, Y1, X2, Y2, D) :-
    DX is X1 - X2,
    DY is Y1 - Y2,
    D is sqrt(DX*DX + DY*DY).

%   distance from point (PX,PY) to segment A-B
point_seg_dist(PX, PY, AX, AY, BX, BY, D) :-
    DX is BX - AX,
    DY is BY - AY,
    L is DX*DX + DY*DY,
    (   L =:= 0
    ->  T = 0.0
    ;   T0 is ((PX - AX)*DX + (PY - AY)*DY) / L,
        T is max(0.0, min(1.0, T0))
    ),
    EX is PX - AX - T*DX,
    EY is PY - AY - T*DY,
    D is sqrt(EX*EX + EY*EY).

%   intersection point of segments P-Q and A-B
seg_inter(PX, PY, QX, QY, AX, AY, BX, BY, IX, IY) :-
    RX is QX - PX, RY is QY - PY,
    SX is BX - AX, SY is BY - AY,
    Den is RX*SY - RY*SX,
    abs(Den) > 1.0e-12,
    T is ((AX - PX)*SY - (AY - PY)*SX) / Den,
    T >= 0.0, T =< 1.0,
    U is ((AX - PX)*RY - (AY - PY)*RX) / Den,
    U >= 0.0, U =< 1.0,
    IX is PX + T*RX,
    IY is PY + T*RY.

%   Contexts: robot(R) uses the knowledge of robot R only,
%             team uses the knowledge of every robot.
ctx_seg(robot(R), AX, AY, BX, BY) :- r_seg(R, AX, AY, BX, BY).
ctx_seg(team, AX, AY, BX, BY)     :- r_seg(_, AX, AY, BX, BY).

ctx_node(robot(R), X, Y) :- r_node(R, X, Y).
ctx_node(team, X, Y)     :- r_node(_, X, Y).

%   line of sight from Q to target P (no known wall in between)
los(Ctx, QX, QY, PX, PY) :-
    target_ignore(Ign),
    \+ ( ctx_seg(Ctx, AX, AY, BX, BY),
         seg_inter(QX, QY, PX, PY, AX, AY, BX, BY, IX, IY),
         dist(IX, IY, PX, PY, D),
         D > Ign ).

%   P is on sight from (QX,QY)
on_sight(Ctx, QX, QY, PX, PY) :-
    sight(S),
    dist(QX, QY, PX, PY, D),
    D =< S,
    los(Ctx, QX, QY, PX, PY).

%   P is on sight from one of the known positions of the context
sees_from_node(Ctx, PX, PY) :-
    ctx_node(Ctx, X, Y),
    on_sight(Ctx, X, Y, PX, PY),
    !.

/* ------------------------------------------------------------------ */
/* Free space flood fill (known unreachability)                        */
/* ------------------------------------------------------------------ */

flood(Ctx, Grid) :-
    flood_cache(Ctx, Grid0),
    !,
    Grid = Grid0.
flood(Ctx, Grid) :-
    build_flood(Ctx, Grid),
    assertz(flood_cache(Ctx, Grid)).

build_flood(Ctx, grid(X0, Y0, Cs, NX, NY, Vis)) :-
    findall(X-Y, ctx_map_point(Ctx, X, Y), Pts),
    Pts = [_|_],
    bbox(Pts, MinX, MinY, MaxX, MaxY),
    grid_margin(M),
    cell_size(Cs0),
    max_cells(MaxCells),
    Area is (MaxX - MinX + 2*M) * (MaxY - MinY + 2*M),
    Cs is max(Cs0, sqrt(Area / MaxCells)),
    X0 is MinX - M,
    Y0 is MinY - M,
    NX is integer(ceiling((MaxX + M - X0) / Cs)) + 1,
    NY is integer(ceiling((MaxY + M - Y0) / Cs)) + 1,
    N is NX * NY,
    functor(Blk, b, N),
    functor(Vis, v, N),
    findall(I, blocked_cell(Ctx, X0, Y0, Cs, NX, NY, I), BIs),
    mark_all(BIs, Blk),
    findall(I, ( ctx_node(Ctx, X, Y),
                 cell_index(X, Y, X0, Y0, Cs, NX, NY, I) ), Seeds0),
    sort(Seeds0, Seeds),
    seed(Seeds, Vis, Queue, Tail),
    bfs(Queue, Tail, Blk, Vis, NX, NY).

ctx_map_point(Ctx, X, Y) :-
    ctx_seg(Ctx, AX, AY, BX, BY),
    ( X = AX, Y = AY ; X = BX, Y = BY ).
ctx_map_point(Ctx, X, Y) :-
    ctx_node(Ctx, X, Y).

bbox([X-Y|T], MinX, MinY, MaxX, MaxY) :-
    bbox(T, X, Y, X, Y, MinX, MinY, MaxX, MaxY).
bbox([], A, B, C, D, A, B, C, D).
bbox([X-Y|T], A0, B0, C0, D0, A, B, C, D) :-
    A1 is min(A0, X), B1 is min(B0, Y),
    C1 is max(C0, X), D1 is max(D0, Y),
    bbox(T, A1, B1, C1, D1, A, B, C, D).

cell_index(X, Y, X0, Y0, Cs, NX, NY, I) :-
    IX is round((X - X0) / Cs),
    IY is round((Y - Y0) / Cs),
    IX >= 0, IX < NX,
    IY >= 0, IY < NY,
    I is IX*NY + IY + 1.

%   cells whose centre is closer than the robot radius to a wall
blocked_cell(Ctx, X0, Y0, Cs, NX, NY, I) :-
    robot_radius(Rad),
    ctx_seg(Ctx, AX, AY, BX, BY),
    IX0 is max(0, integer(floor((min(AX, BX) - Rad - X0) / Cs))),
    IX1 is min(NX - 1, integer(ceiling((max(AX, BX) + Rad - X0) / Cs))),
    IY0 is max(0, integer(floor((min(AY, BY) - Rad - Y0) / Cs))),
    IY1 is min(NY - 1, integer(ceiling((max(AY, BY) + Rad - Y0) / Cs))),
    between(IX0, IX1, IX),
    between(IY0, IY1, IY),
    CX is X0 + IX*Cs,
    CY is Y0 + IY*Cs,
    point_seg_dist(CX, CY, AX, AY, BX, BY, D),
    D < Rad,
    I is IX*NY + IY + 1.

mark_all([], _).
mark_all([I|Is], T) :-
    setarg(I, T, 1),
    mark_all(Is, T).

%   seed(+Cells, +Vis, -Queue, -Tail): open-ended queue of the seeds
seed([], _, T, T).
seed([I|Is], Vis, [I|Q], T) :-
    setarg(I, Vis, 1),
    seed(Is, Vis, Q, T).

bfs(Q, T, _, _, _, _) :-
    Q == T,
    !.
bfs([I|Q], T, Blk, Vis, NX, NY) :-
    I0 is I - 1,
    IX is I0 // NY,
    IY is I0 mod NY,
    XP is IX + 1, XM is IX - 1,
    YP is IY + 1, YM is IY - 1,
    visit(XP, IY, NX, NY, Blk, Vis, T, T1),
    visit(XM, IY, NX, NY, Blk, Vis, T1, T2),
    visit(IX, YP, NX, NY, Blk, Vis, T2, T3),
    visit(IX, YM, NX, NY, Blk, Vis, T3, T4),
    bfs(Q, T4, Blk, Vis, NX, NY).

visit(IX, IY, NX, NY, Blk, Vis, T0, T) :-
    (   IX >= 0, IX < NX,
        IY >= 0, IY < NY,
        Idx is IX*NY + IY + 1,
        arg(Idx, Vis, V), var(V),
        arg(Idx, Blk, B), var(B)
    ->  setarg(Idx, Vis, 1),
        T0 = [Idx|T]
    ;   T0 = T
    ).

%   Some free place connected to the known positions has P on sight.
%   A location outside the known map can not be enclosed by known walls.
reachable_by_flood(Ctx, PX, PY) :-
    (   flood(Ctx, grid(X0, Y0, Cs, NX, NY, Vis))
    ->  flood_sees(X0, Y0, Cs, NX, NY, Vis, Ctx, PX, PY)
    ;   true
    ).

flood_sees(X0, Y0, Cs, NX, NY, Vis, Ctx, PX, PY) :-
    sight(S),
    CI is round((PX - X0) / Cs),
    CJ is round((PY - Y0) / Cs),
    (   ( CI =< 0 ; CI >= NX - 1 ; CJ =< 0 ; CJ >= NY - 1 )
    ->  true
    ;   K is integer(ceiling(S / Cs)) + 1,
        IA is max(0, CI - K), IB is min(NX - 1, CI + K),
        JA is max(0, CJ - K), JB is min(NY - 1, CJ + K),
        between(IA, IB, IX),
        between(JA, JB, IY),
        Idx is IX*NY + IY + 1,
        arg(Idx, Vis, V),
        V == 1,
        CX is X0 + IX*Cs,
        CY is Y0 + IY*Cs,
        on_sight(Ctx, CX, CY, PX, PY)
    ->  true
    ).

/* ------------------------------------------------------------------ */
/* Knowledge about objects                                             */
/* ------------------------------------------------------------------ */

near(X, Y, PX, PY) :-
    object_tol(T),
    dist(X, Y, PX, PY, D),
    D =< T.

%   R sensed (and still believes in) an object of colour C at P
sensed_object(R, C, PX, PY) :-
    r_object(R, _, C, X, Y),
    near(X, Y, PX, PY),
    !.

%   R picked up an object of colour C at P
picked_at(R, C, PX, PY) :-
    r_picked(R, _, _, C, X, Y),
    near(X, Y, PX, PY),
    !.

%   R knows the position of something (object or pick-up place) at P
known_spot(R, PX, PY) :-
    (   r_object(R, _, _, X, Y)
    ;   r_picked(R, _, _, _, X, Y)
    ),
    near(X, Y, PX, PY),
    !.

%   nodeObject: the closest node of R to an object of colour C that R
%   sensed.  P matches when that node is (one of) the closest to P.
nodeobj_at(R, C, PX, PY) :-
    nodeobj_tol(Tol),
    closest_tol(Slack),
    r_nodeobj(R, C, X, Y),
    dist(X, Y, PX, PY, D),
    D =< Tol,
    \+ ( r_node(R, NX, NY),
         dist(NX, NY, PX, PY, DN),
         DN + Slack < D ),
    !.

knows_object(R, C, PX, PY) :-
    (   sensed_object(R, C, PX, PY)
    ->  true
    ;   nodeobj_at(R, C, PX, PY)
    ).

/* ------------------------------------------------------------------ */
/* Reachability of a current robot                                     */
/* ------------------------------------------------------------------ */

reach_status(R, PX, PY, S) :-
    (   known_spot(R, PX, PY)
    ->  S = none
    ;   sees_from_node(robot(R), PX, PY)
    ->  S = none
    ;   \+ reachable_by_flood(robot(R), PX, PY)
    ->  S = unreachable
    ;   S = explore
    ).

/* ------------------------------------------------------------------ */
/* Queries                                                             */
/* ------------------------------------------------------------------ */

%   current_can_reach(+R, +P, ?Cond)
current_can_reach(R, P, Cond) :-
    catch(current_can_reach_(R, P, Cond0), _, fail),
    !,
    Cond = Cond0.

current_can_reach_(R, P, Cond) :-
    ensure_kb,
    r_robot(R),
    !,
    xy(P, PX, PY),
    reach_status(R, PX, PY, Cond).

%   current_can_retrieve(+R, +C, +P, ?Cond)
current_can_retrieve(R, C, P, Cond) :-
    catch(current_can_retrieve_(R, C, P, Cond0), _, fail),
    !,
    Cond = Cond0.

current_can_retrieve_(R, C, P, Cond) :-
    ensure_kb,
    r_robot(R),
    !,
    xy(P, PX, PY),
    reach_status(R, PX, PY, RS),
    (   RS == unreachable
    ->  Cond = not_reachable
    ;   picked_at(R, C, PX, PY)
    ->  Cond = not_object           % R itself removed that object
    ;   knows_object(R, C, PX, PY)
    ->  Cond = none
    ;   RS == none
    ->  Cond = not_object           % R sees P and there is no such object
    ;   Cond = explore
    ).

%   new_can_retrieve(+C, +P)
new_can_retrieve(C, P) :-
    catch(new_can_retrieve_(C, P), _, fail),
    !.

new_can_retrieve_(C, P) :-
    ensure_kb,
    xy(P, PX, PY),
    \+ picked_at(_, C, PX, PY),
    r_robot(R),
    knows_object(R, C, PX, PY),
    !.

%   new_can_reach(+Start, +P)
new_can_reach(Start, P) :-
    catch(new_can_reach_(Start, P), _, fail),
    !.

new_can_reach_(Start, P) :-
    ensure_kb,
    xy(Start, SX, SY),
    xy(P, PX, PY),
    (   on_sight(team, SX, SY, PX, PY)
    ->  true
    ;   findall(R, ( r_robot(R), robot_sees(R, SX, SY) ), R0),
        sort(R0, Init),
        Init \== [],
        robot_closure(Init, Init, Reached),
        member(R, Reached),
        robot_reaches(R, PX, PY)
    ->  true
    ).

%   one of the nodes of R has (X,Y) on sight, using the team knowledge
robot_sees(R, X, Y) :-
    r_node(R, NX, NY),
    on_sight(team, NX, NY, X, Y),
    !.

robot_reaches(R, X, Y) :-
    (   known_spot(R, X, Y)
    ->  true
    ;   robot_sees(R, X, Y)
    ).

%   robots whose explored graphs are connected (on sight) to the reached ones
robot_closure([], Acc, Acc).
robot_closure([R|Queue], Acc, Out) :-
    findall(R2, ( r_robot(R2),
                  \+ memberchk(R2, Acc),
                  robots_linked(R, R2) ), New0),
    sort(New0, New),
    append(Queue, New, Queue1),
    append(Acc, New, Acc1),
    robot_closure(Queue1, Acc1, Out).

robots_linked(R1, R2) :-
    r_node(R2, X, Y),
    robot_sees(R1, X, Y),
    !.

%   has_preference(+R, +C)
has_preference(R, C) :-
    catch(has_preference_(R, C), _, fail),
    !.

has_preference_(R, C) :-
    ensure_kb,
    r_picked(R, OrderY, IdY, C, _, _),
    r_picked(R, OrderX, IdX, _, _, _),
    OrderX > OrderY,
    sensed_before(IdX, IdY),
    !.

sensed_before(IdX, IdY) :-
    (   number(IdX), number(IdY)
    ->  IdX < IdY
    ;   IdX @< IdY
    ).
