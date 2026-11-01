/*  solution.pl

    Queries over the knowledge bases of a team of exploring robots.

    The knowledge bases are read (never modified) from /app/KB_test/.
    Each robot X has two files: agent_memoryX.txt and agent_navMemoryX.txt.
    Their clauses are read as plain terms and copied into private facts
    (kb_*), so robots never clash with each other and the KB files and
    their predicates are left untouched.

    World model (2D, z is ignored)
    ------------------------------
    * Obstacles: every segment of edge/2, outline/2, wall/2 and object/3
      point chains.  The robot has a diameter of 0.15 m, so a grid cell is
      blocked when its centre is closer than 0.075 m to any obstacle.
    * Observed space: cells within the 0.5 m line of sight of a place the
      robot has been at (experience graph nodes, points along the graph
      links, start and current position) and not hidden behind a segment.
    * Known to be reachable (none): connected through free observed cells
      to the places the robot has been at.
    * Known to be unreachable: not connected even when every unknown cell
      is assumed to be free (only known obstacles block the robot).
    * Otherwise the robot has to explore.

    Public predicates
    -----------------
    current_can_retrieve(R, C, P, Cond)
    new_can_retrieve(C, P)
    new_can_reach(Start, P)
    current_can_reach(R, P, Cond)
    has_preference(R, C)
*/

:- dynamic kb_loaded/0.
:- dynamic kb_robot/1.
:- dynamic kb_seg/5.        % kb_seg(R, X1, Y1, X2, Y2)
:- dynamic kb_object/5.     % kb_object(R, Id, Color, CX, CY)
:- dynamic kb_picked/6.     % kb_picked(R, Seq, Id, Color, X, Y)
:- dynamic kb_vp/3.         % kb_vp(R, X, Y)      viewpoints
:- dynamic kb_seed/3.       % kb_seed(R, X, Y)    places the robot has been at
:- dynamic kb_bounds/4.     % kb_bounds(XMin, YMin, XMax, YMax)

kb_dir('/app/KB_test/').

cell_size(0.025).
robot_radius(0.075).
sight(0.5).
grid_margin(0.6).
object_tol(0.1).            % P matches a known object (distance to centroid)
picked_tol(0.15).           % P matches a picked up object
point_target(0.1).          % free cells within this distance reach a point
object_target(0.2).         % free cells within this distance reach an object

% ---------------------------------------------------------------------------
% Public predicates
% ---------------------------------------------------------------------------

current_can_retrieve(R, C, P, Cond) :-
    ensure_kb,
    kb_robot(R), !,
    p_xy(P, X, Y),
    retrieve_cond(R, C, X, Y, Cond0), !,
    Cond = Cond0.

current_can_reach(R, P, Cond) :-
    ensure_kb,
    kb_robot(R), !,
    p_xy(P, X, Y),
    reach_radius(R, X, Y, Rad),
    robot_world(R, W),
    world_status(W, X, Y, Rad, S), !,
    Cond = S.

new_can_reach(Start, P) :-
    ensure_kb,
    p_xy(Start, SX, SY),
    p_xy(P, X, Y),
    reach_radius(_, X, Y, Rad),
    start_world(SX, SY, X, Y, W),
    world_status(W, X, Y, Rad, S), !,
    S == none.

new_can_retrieve(C, P) :-
    ensure_kb,
    p_xy(P, X, Y),
    object_tol(OT),
    once(( kb_object(_, _, C, OX, OY), kb_dist(OX, OY, X, Y, D), D =< OT )),
    picked_tol(PT),
    \+ ( kb_picked(_, _, _, C, PX, PY), kb_dist(PX, PY, X, Y, D2), D2 =< PT ),
    object_target(Rad),
    team_world(W),
    world_status(W, X, Y, Rad, S), !,
    S \== unreachable.

has_preference(R, C) :-
    ensure_kb,
    once(( kb_picked(R, SeqY, IdY, C, _, _),
           kb_picked(R, SeqX, IdX, CX, _, _),
           SeqX > SeqY,
           IdX < IdY,
           CX \== C )).

% ---------------------------------------------------------------------------
% Retrieval conditions
% ---------------------------------------------------------------------------

retrieve_cond(R, C, X, Y, Cond) :-
    object_target(Rad),
    robot_world(R, W),
    world_status(W, X, Y, Rad, S),
    (   knows_object(R, C, X, Y)
    ->  ( S == unreachable -> Cond = not_reachable ; Cond = none )
    ;   S == unreachable
    ->  Cond = not_reachable
    ;   S == none,
        ( picked_near(R, C, X, Y) ; observed(R, X, Y) )
    ->  Cond = not_object
    ;   Cond = explore
    ).

knows_object(R, C, X, Y) :-
    object_tol(T),
    kb_object(R, _, C, OX, OY),
    kb_dist(OX, OY, X, Y, D),
    D =< T, !.

picked_near(R, C, X, Y) :-
    picked_tol(T),
    kb_picked(R, _, _, C, PX, PY),
    kb_dist(PX, PY, X, Y, D),
    D =< T, !.

% The robot has seen P if P was within its line of sight from some place it
% has been at.  Segments very close to P (an object lying there) are ignored.
observed(R, X, Y) :-
    sight(Sg),
    findall(s(A, B, C, D),
            ( kb_seg(R, A, B, C, D),
              seg_dist(X, Y, A, B, C, D, Ds), Ds > 0.12 ),
            Segs),
    kb_vp(R, VX, VY),
    kb_dist(VX, VY, X, Y, Dv),
    Dv =< Sg,
    \+ ( member(s(A, B, C, D), Segs),
         seg_inter(VX, VY, X, Y, A, B, C, D) ), !.

% Target radius: a point close to a known object is reached like an object.
reach_radius(R, X, Y, Rad) :-
    object_tol(T),
    (   kb_object(R, _, _, OX, OY), kb_dist(OX, OY, X, Y, D), D =< T
    ->  object_target(Rad)
    ;   point_target(Rad)
    ).

% ---------------------------------------------------------------------------
% Worlds (cached in global variables)
% ---------------------------------------------------------------------------

robot_world(R, W) :-
    format(atom(Key), 'kb_world_robot_~w', [R]),
    (   nb_current(Key, W0)
    ->  W = W0
    ;   findall(s(A, B, C, D), kb_seg(R, A, B, C, D), Segs),
        findall(VX-VY, kb_vp(R, VX, VY), VPs),
        findall(SX-SY, kb_seed(R, SX, SY), Seeds),
        base_grid(G),
        build_world(G, Segs, VPs, Seeds, W),
        nb_setval(Key, W)
    ).

% Obstacles and observations of the whole team.
team_layers(G, Segs, Blk, Vis) :-
    (   nb_current(kb_team_layers, layers(G, Segs, Blk, Vis))
    ->  true
    ;   findall(s(A, B, C, D), kb_seg(_, A, B, C, D), Segs),
        findall(VX-VY, kb_vp(_, VX, VY), VPs),
        base_grid(G),
        make_layers(G, Segs, VPs, Blk, Vis),
        nb_setval(kb_team_layers, layers(G, Segs, Blk, Vis))
    ).

% The team world, flooded from every place a robot has been at.
team_world(W) :-
    (   nb_current(kb_team_world, W0)
    ->  W = W0
    ;   team_layers(G, _, Blk, Vis),
        findall(SX-SY, kb_seed(_, SX, SY), Seeds),
        floods(G, Blk, Vis, Seeds, W),
        nb_setval(kb_team_world, W)
    ).

% A new robot deployed at Start, using the knowledge of the whole team.
% It also sees its own surroundings at Start.
start_world(SX, SY, X, Y, W) :-
    format(atom(Key), 'kb_world_start_~w_~w_~w_~w', [SX, SY, X, Y]),
    (   nb_current(Key, W0)
    ->  W = W0
    ;   base_grid(G0),
        (   in_grid(G0, SX, SY), in_grid(G0, X, Y)
        ->  team_layers(G, Segs, Blk0, Vis0),
            duplicate_term(Blk0, Blk),
            duplicate_term(Vis0, Vis),
            mark_vis(G, Segs, Blk, Vis, SX-SY)
        ;   findall(s(A, B, C, D), kb_seg(_, A, B, C, D), Segs),
            findall(VX-VY, kb_vp(_, VX, VY), VPs),
            grid_with(G0, [SX-SY, X-Y], G),
            make_layers(G, Segs, [SX-SY|VPs], Blk, Vis)
        ),
        floods(G, Blk, Vis, [SX-SY], W),
        nb_setval(Key, W)
    ).

build_world(G, Segs, VPs, Seeds, W) :-
    make_layers(G, Segs, VPs, Blk, Vis),
    floods(G, Blk, Vis, Seeds, W).

make_layers(G, Segs, VPs, Blk, Vis) :-
    G = g(_, _, NX, NY, _),
    N is NX * NY,
    new_array(N, Blk),
    new_array(N, Vis),
    forall(member(S, Segs), mark_seg(G, Blk, S)),
    forall(member(V, VPs), mark_vis(G, Segs, Blk, Vis, V)).

floods(G, Blk, Vis, Seeds, w(G, Known, Opt)) :-
    kb_flood(G, Blk, Vis, known, Seeds, Known),
    kb_flood(G, Blk, Vis, opt, Seeds, Opt).

% ---------------------------------------------------------------------------
% Grid
% ---------------------------------------------------------------------------

% g(X0, Y0, NX, NY, Cell); cell (I, J) has centre (X0+I*Cell, Y0+J*Cell)
% and index K = I*NY + J + 1 (argument of the array terms).

base_grid(G) :-
    (   kb_bounds(XA, YA, XB, YB)
    ->  true
    ;   XA = -2.0, YA = -2.0, XB = 2.0, YB = 2.0
    ),
    make_grid(XA, YA, XB, YB, G).

grid_with(g(X0, Y0, NX, NY, Cell), Pts, G) :-
    XB0 is X0 + (NX - 1) * Cell,
    YB0 is Y0 + (NY - 1) * Cell,
    grid_margin(M),
    XA0 is X0 + M, YA0 is Y0 + M,
    XB1 is XB0 - M, YB1 is YB0 - M,
    findall(PX, member(PX-_, Pts), Xs),
    findall(PY, member(_-PY, Pts), Ys),
    min_list([XA0|Xs], XA), max_list([XB1|Xs], XB),
    min_list([YA0|Ys], YA), max_list([YB1|Ys], YB),
    make_grid(XA, YA, XB, YB, G).

make_grid(XA, YA, XB, YB, g(X0, Y0, NX, NY, Cell)) :-
    cell_size(Cell),
    grid_margin(M),
    X0 is XA - M,
    Y0 is YA - M,
    NX is integer(ceiling((XB - XA + 2*M) / Cell)) + 1,
    NY is integer(ceiling((YB - YA + 2*M) / Cell)) + 1.

in_grid(g(X0, Y0, NX, NY, Cell), X, Y) :-
    X >= X0, Y >= Y0,
    X =< X0 + (NX - 1) * Cell,
    Y =< Y0 + (NY - 1) * Cell.

new_array(N, A) :-
    length(L, N),
    maplist(=(0), L),
    A =.. [a|L].

cell_index(g(_, _, _, NY, _), I, J, K) :-
    K is I * NY + J + 1.

% Cells whose centre lies within distance Rad of (X, Y).
cell_in_disk(g(X0, Y0, NX, NY, Cell), X, Y, Rad, K) :-
    I0 is max(0, integer(ceiling((X - Rad - X0) / Cell))),
    I1 is min(NX - 1, integer(floor((X + Rad - X0) / Cell))),
    J0 is max(0, integer(ceiling((Y - Rad - Y0) / Cell))),
    J1 is min(NY - 1, integer(floor((Y + Rad - Y0) / Cell))),
    between(I0, I1, I),
    CX is X0 + I * Cell,
    between(J0, J1, J),
    CY is Y0 + J * Cell,
    DX is CX - X, DY is CY - Y,
    DX*DX + DY*DY =< Rad*Rad,
    K is I * NY + J + 1.

% Blocked cells: centre closer than the robot radius to a segment.
mark_seg(G, Blk, s(A, B, C, D)) :-
    G = g(X0, Y0, NX, NY, Cell),
    robot_radius(Rr),
    I0 is max(0, integer(floor((min(A, C) - Rr - X0) / Cell))),
    I1 is min(NX - 1, integer(ceiling((max(A, C) + Rr - X0) / Cell))),
    J0 is max(0, integer(floor((min(B, D) - Rr - Y0) / Cell))),
    J1 is min(NY - 1, integer(ceiling((max(B, D) + Rr - Y0) / Cell))),
    forall(( between(I0, I1, I),
             CX is X0 + I * Cell,
             between(J0, J1, J),
             K is I * NY + J + 1,
             arg(K, Blk, 0),
             CY is Y0 + J * Cell,
             seg_dist(CX, CY, A, B, C, D, Ds),
             Ds < Rr ),
           nb_setarg(K, Blk, 1)).

% Visible cells from viewpoint (VX, VY).
mark_vis(G, Segs, Blk, Vis, VX-VY) :-
    G = g(X0, Y0, NX, NY, Cell),
    sight(Sg),
    include(seg_near(VX, VY, Sg), Segs, Near),
    I0 is max(0, integer(ceiling((VX - Sg - X0) / Cell))),
    I1 is min(NX - 1, integer(floor((VX + Sg - X0) / Cell))),
    Sg2 is Sg * Sg,
    forall(( between(I0, I1, I),
             CX is X0 + I * Cell,
             Rem is Sg2 - (CX - VX) * (CX - VX),
             Rem >= 0,
             H is sqrt(Rem),
             J0 is max(0, integer(ceiling((VY - H - Y0) / Cell))),
             J1 is min(NY - 1, integer(floor((VY + H - Y0) / Cell))),
             between(J0, J1, J),
             K is I * NY + J + 1,
             arg(K, Vis, 0),
             arg(K, Blk, 0),
             CY is Y0 + J * Cell,
             \+ ( member(s(A, B, C, D), Near),
                  seg_inter(VX, VY, CX, CY, A, B, C, D) ) ),
           nb_setarg(K, Vis, 1)).

seg_near(VX, VY, Sg, s(A, B, C, D)) :-
    seg_dist(VX, VY, A, B, C, D, Ds),
    Ds =< Sg.

% Breadth first flood fill (4-connected).
% Mode known: free and observed cells; mode opt: free cells.
kb_flood(G, Blk, Vis, Mode, Seeds, Mark) :-
    G = g(_, _, NX, NY, _),
    N is NX * NY,
    new_array(N, Mark),
    robot_radius(Rr),
    findall(K,
            ( member(SX-SY, Seeds),
              cell_in_disk(G, SX, SY, Rr, K),
              cell_ok(Mode, K, Blk, Vis) ),
            Ks0),
    sort(Ks0, Ks),
    forall(member(K, Ks), nb_setarg(K, Mark, 1)),
    flood_loop(Ks, NX, NY, Mode, Blk, Vis, Mark).

flood_loop([], _, _, _, _, _, _) :- !.
flood_loop(Front, NX, NY, Mode, Blk, Vis, Mark) :-
    findall(K2,
            ( member(K, Front),
              neighbour(K, NX, NY, K2),
              arg(K2, Mark, 0),
              cell_ok(Mode, K2, Blk, Vis) ),
            L0),
    sort(L0, L),
    forall(member(K, L), nb_setarg(K, Mark, 1)),
    flood_loop(L, NX, NY, Mode, Blk, Vis, Mark).

cell_ok(known, K, Blk, Vis) :- arg(K, Blk, 0), arg(K, Vis, 1).
cell_ok(opt, K, Blk, _) :- arg(K, Blk, 0).

neighbour(K, NX, NY, K2) :-
    K1 is K - 1,
    I is K1 // NY,
    J is K1 mod NY,
    (   I + 1 < NX, K2 is K + NY
    ;   I > 0, K2 is K - NY
    ;   J + 1 < NY, K2 is K + 1
    ;   J > 0, K2 is K - 1
    ).

% Status of the target "free cells within Rad of (X, Y)".
world_status(w(G, Known, Opt), X, Y, Rad, S) :-
    findall(K, cell_in_disk(G, X, Y, Rad, K), Ks),
    (   member(K, Ks), arg(K, Known, 1)
    ->  S = none
    ;   member(K, Ks), arg(K, Opt, 1)
    ->  S = explore
    ;   Ks == [], border_open(G, Opt)
    ->  S = explore
    ;   S = unreachable
    ).

% The free space around the robot is not closed by known obstacles.
border_open(G, Opt) :-
    G = g(_, _, NX, NY, _),
    IM is NX - 1, JM is NY - 1,
    (   between(0, IM, I), member(J, [0, JM])
    ;   between(0, JM, J), member(I, [0, IM])
    ),
    cell_index(G, I, J, K),
    arg(K, Opt, 1), !.

% ---------------------------------------------------------------------------
% Geometry
% ---------------------------------------------------------------------------

kb_dist(X1, Y1, X2, Y2, D) :-
    D is sqrt((X1 - X2) * (X1 - X2) + (Y1 - Y2) * (Y1 - Y2)).

seg_dist(PX, PY, X1, Y1, X2, Y2, D) :-
    DX is X2 - X1,
    DY is Y2 - Y1,
    L2 is DX * DX + DY * DY,
    (   L2 =:= 0
    ->  T = 0.0
    ;   T0 is ((PX - X1) * DX + (PY - Y1) * DY) / L2,
        T is max(0.0, min(1.0, T0))
    ),
    EX is PX - X1 - T * DX,
    EY is PY - Y1 - T * DY,
    D is sqrt(EX * EX + EY * EY).

kb_cross(OX, OY, AX, AY, BX, BY, C) :-
    C is (AX - OX) * (BY - OY) - (AY - OY) * (BX - OX).

% Proper intersection of segments P1-P2 and Q1-Q2.
seg_inter(P1X, P1Y, P2X, P2Y, Q1X, Q1Y, Q2X, Q2Y) :-
    kb_cross(Q1X, Q1Y, Q2X, Q2Y, P1X, P1Y, D1),
    kb_cross(Q1X, Q1Y, Q2X, Q2Y, P2X, P2Y, D2),
    D1 * D2 < 0,
    kb_cross(P1X, P1Y, P2X, P2Y, Q1X, Q1Y, D3),
    kb_cross(P1X, P1Y, P2X, P2Y, Q2X, Q2Y, D4),
    D3 * D4 < 0.

p_xy([X0, Y0|_], X, Y) :-
    number(X0), number(Y0),
    X is float(X0),
    Y is float(Y0).

% ---------------------------------------------------------------------------
% Loading the knowledge bases
% ---------------------------------------------------------------------------

ensure_kb :-
    kb_loaded, !.
ensure_kb :-
    (   catch(load_kb, _, fail)
    ->  true
    ;   true
    ),
    assertz(kb_loaded).

load_kb :-
    kb_dir(Dir),
    (   exists_directory(Dir)
    ->  directory_files(Dir, Files)
    ;   Files = []
    ),
    msort(Files, Sorted),
    forall(( member(F, Sorted),
             atom(F),
             atom_concat(agent_memory, Rest, F),
             atom_concat(X, '.txt', Rest),
             X \== '' ),
           (   catch(load_robot(Dir, X), _, fail)
           ->  true
           ;   true
           )),
    compute_bounds.

load_robot(Dir, X) :-
    atomic_list_concat([Dir, agent_memory, X, '.txt'], MemFile),
    atomic_list_concat([Dir, agent_navMemory, X, '.txt'], NavFile),
    kb_read_terms(MemFile, Mem),
    (   exists_file(NavFile)
    ->  kb_read_terms(NavFile, Nav)
    ;   Nav = []
    ),
    (   member(agent_ID(R0), Mem)
    ->  R = R0
    ;   atom_number(X, R)
    ->  true
    ;   R = X
    ),
    \+ kb_robot(R),
    assertz(kb_robot(R)),
    load_memory(R, Mem),
    load_nav(R, Mem, Nav).

kb_read_terms(File, Terms) :-
    setup_call_cleanup(open(File, read, S),
                       kb_read_all(S, Terms),
                       close(S)).

kb_read_all(S, Terms) :-
    catch(read_term(S, T, []), _, T = '$kb_bad'),
    (   T == end_of_file
    ->  Terms = []
    ;   T == '$kb_bad'
    ->  (   at_end_of_stream(S)
        ->  Terms = []
        ;   kb_read_all(S, Terms)
        )
    ;   Terms = [T|Rest],
        kb_read_all(S, Rest)
    ).

load_memory(R, Mem) :-
    findall(Id-(PX-PY),
            ( member(point(Id, [PX0, PY0|_]), Mem),
              number(PX0), number(PY0),
              PX is float(PX0), PY is float(PY0) ),
            Pairs),
    empty_assoc(E),
    foldl(put_pt, Pairs, E, Pts),
    forall(member(edge(_, L), Mem), add_chain(R, Pts, L)),
    forall(member(outline(_, L), Mem), add_chain(R, Pts, L)),
    forall(member(wall(_, L), Mem), add_chain(R, Pts, L)),
    forall(member(object(_, L, _), Mem), add_chain(R, Pts, L)),
    forall(member(object(Id, L, C), Mem), add_object(R, Pts, Id, L, C)),
    findall(Id-C-PX-PY,
            ( member(objectPickedUp(Id, C, [PX0, PY0|_]), Mem),
              number(PX0), number(PY0),
              PX is float(PX0), PY is float(PY0) ),
            Picked),
    forall(nth1(N, Picked, Id-C-PX-PY),
           assertz(kb_picked(R, N, Id, C, PX, PY))).

put_pt(Id-P, A0, A) :-
    (   get_assoc(Id, A0, _)
    ->  A = A0
    ;   put_assoc(Id, A0, P, A)
    ).

add_chain(R, Pts, L) :-
    is_list(L),
    !,
    forall(nextto(A, B, L),
           (   get_assoc(A, Pts, X1-Y1),
               get_assoc(B, Pts, X2-Y2)
           ->  assertz(kb_seg(R, X1, Y1, X2, Y2))
           ;   true
           )).
add_chain(_, _, _).

add_object(R, Pts, Id, L, C) :-
    is_list(L),
    sort(L, U),
    findall(X-Y, ( member(P, U), get_assoc(P, Pts, X-Y) ), XY),
    XY \== [],
    !,
    length(XY, N),
    findall(X, member(X-_, XY), Xs),
    findall(Y, member(_-Y, XY), Ys),
    sum_list(Xs, SX),
    sum_list(Ys, SY),
    CX is SX / N,
    CY is SY / N,
    assertz(kb_object(R, Id, C, CX, CY)).
add_object(_, _, _, _, _).

load_nav(R, Mem, Nav) :-
    findall(Id-(X-Y),
            ( member(nodeNav(Id, [X0, Y0|_], _, _, _), Nav),
              number(X0), number(Y0),
              X is float(X0), Y is float(Y0) ),
            Nodes0),
    empty_assoc(E),
    foldl(put_pt, Nodes0, E, Nodes),
    assoc_to_values(Nodes, NodePts),
    findall(X-Y,
            ( ( member(nodeNavStart([X0, Y0|_]), Nav)
              ; member(agent_Pos([X0, Y0|_], _), Mem)
              ; member(agent_Pos([X0, Y0|_], _), Nav)
              ),
              number(X0), number(Y0),
              X is float(X0), Y is float(Y0) ),
            Extra),
    append(NodePts, Extra, Seeds0),
    sort(Seeds0, Seeds),
    forall(member(X-Y, Seeds),
           ( assertz(kb_seed(R, X, Y)), assertz(kb_vp(R, X, Y)) )),
    % points along the links of the experience graph
    findall(A-B,
            ( member(nodeNav(Id, _, Parent, Con, _), Nav),
              (   is_list(Con), member(O, Con)
              ;   integer(Parent), O = Parent
              ),
              number(O), number(Id),
              O \== Id,
              A is min(Id, O), B is max(Id, O) ),
            Links0),
    sort(Links0, Links),
    forall(( member(A-B, Links),
             get_assoc(A, Nodes, AX-AY),
             get_assoc(B, Nodes, BX-BY) ),
           add_link_points(R, AX, AY, BX, BY)).

add_link_points(R, AX, AY, BX, BY) :-
    kb_dist(AX, AY, BX, BY, D),
    K is integer(floor(D / 0.15)),
    forall(( K >= 1,
             between(1, K, I),
             T is I / (K + 1.0),
             X is AX + T * (BX - AX),
             Y is AY + T * (BY - AY) ),
           assertz(kb_vp(R, X, Y))).

compute_bounds :-
    findall(X, ( kb_seg(_, X, _, _, _) ; kb_seg(_, _, _, X, _) ; kb_vp(_, X, _) ), Xs),
    findall(Y, ( kb_seg(_, _, Y, _, _) ; kb_seg(_, _, _, _, Y) ; kb_vp(_, _, Y) ), Ys),
    retractall(kb_bounds(_, _, _, _)),
    (   Xs \== [], Ys \== []
    ->  min_list(Xs, XA), max_list(Xs, XB),
        min_list(Ys, YA), max_list(Ys, YB),
        assertz(kb_bounds(XA, YA, XB, YB))
    ;   true
    ).
