/*  solution.pl

    Queries over the knowledge bases built by a team of exploring robots.

    The knowledge base is read from /app/KB_test/ (files agent_memoryX.txt
    and agent_navMemoryX.txt).  The files are only read: their facts are
    copied into private predicates (prefixed with s_) the first time a
    query is asked, so the original knowledge base is never modified.

    Public predicates:
        current_can_retrieve(+R, +C, +P, ?Cond)
        new_can_retrieve(+C, +P)
        new_can_reach(+Start, +P)
        current_can_reach(+R, +P, ?Cond)
        has_preference(+R, +C)

    Geometry (all robots share one reference frame, only X and Y are used):
      * walls   : segments given by edge/2, outline/2 and the outlines of
                  sensed objects (object/3);
      * sight   : a robot reaches directly any point within 0.5 m that is
                  not hidden behind a wall;
      * body    : diameter 0.15 m, so the robot centre keeps 0.075 m away
                  from walls and cannot pass through narrower gaps.

    A robot knows how to reach P when P is on sight from a place it has
    been: its start, the nodes of its experience graph, the segments
    between connected nodes and the nodes of its computed paths (and the
    places where it sensed or picked up objects).  A robot knows that P
    cannot be reached when, assuming every unknown place is free, no
    position its body can reach from the places it has been has P on
    sight (e.g. P is inside a closed room, or behind a gap narrower than
    the robot).  This is computed with a flood fill on an occupancy grid.
*/

:- dynamic s_loaded/0.
:- dynamic s_robot/1.
:- dynamic s_point/3.
:- dynamic s_edge/3.
:- dynamic s_outline/3.
:- dynamic s_object/4.
:- dynamic s_picked/4.
:- dynamic s_start/2.
:- dynamic s_node/6.
:- dynamic s_path/5.
:- dynamic s_nodeobj/4.
:- dynamic s_walls_c/2.
:- dynamic s_vps_c/2.
:- dynamic s_all_walls_c/1.
:- dynamic s_center_c/3.

s_kb_dir('/app/KB_test/').

s_sight(0.5).          % line of sight of a robot
s_radius(0.075).       % half of the robot diameter
s_obj_tol(0.15).       % tolerance to identify a location with an object
s_node_obj_radius(0.6). % max. distance between an object and its nodeObject
s_cell(0.025).         % size of the cells of the occupancy grid

/* ------------------------------------------------------------------ */
/*  Public predicates                                                  */
/* ------------------------------------------------------------------ */

current_can_retrieve(R, C, P, Cond) :-
    once(s_current_can_retrieve(R, C, P, X)),
    Cond = X.

current_can_reach(R, P, Cond) :-
    once(s_current_can_reach(R, P, X)),
    Cond = X.

new_can_retrieve(C, P) :-
    once(s_new_can_retrieve(C, P)).

new_can_reach(Start, P) :-
    once(s_new_can_reach(Start, P)).

has_preference(R, C) :-
    once(s_has_preference(R, C)).

/* ------------------------------------------------------------------ */
/*  current_can_reach / current_can_retrieve                           */
/* ------------------------------------------------------------------ */

s_current_can_reach(R, P, Cond) :-
    s_ensure_loaded,
    s_robot(R),
    s_xy(P, Q),
    s_reach(R, Q, Cond).

s_reach(R, Q, Cond) :-
    s_walls(R, Ws),
    (   s_known_reach(R, Q, Ws)
    ->  Cond = none
    ;   s_known_unreach(R, Q)
    ->  Cond = unreachable
    ;   Cond = explore
    ).

s_current_can_retrieve(R, C, P, Cond) :-
    s_ensure_loaded,
    s_robot(R),
    s_xy(P, Q),
    s_reach(R, Q, S),
    (   S == unreachable
    ->  Cond = not_reachable
    ;   s_picked_at(R, Q)
    ->  Cond = not_object
    ;   s_knows_object(R, C, Q)
    ->  (   S == none
        ->  Cond = none
        ;   Cond = explore
        )
    ;   S == none
    ->  Cond = not_object
    ;   Cond = explore
    ).

/* ------------------------------------------------------------------ */
/*  new_can_reach / new_can_retrieve                                   */
/* ------------------------------------------------------------------ */

s_new_can_reach(Start, P) :-
    s_ensure_loaded,
    s_xy(Start, S),
    s_xy(P, Q),
    s_new_reach(S, Q).

% P on sight from the deployment location
s_new_reach(S, Q) :-
    s_sight(Sight),
    s_dist(S, Q, D),
    D =< Sight,
    s_all_walls(AW),
    s_walls_near(AW, Q, Ws),
    s_visible(S, Q, Ws),
    !.
% P known to be reachable by a robot whose explored area can be reached
% from the deployment location using the knowledge of all the robots
s_new_reach(S, Q) :-
    s_all_walls(AW),
    findall(R, ( s_robot(R), s_start_linked(S, R, AW) ), Rs0),
    sort(Rs0, Rs1),
    Rs1 \== [],
    s_robot_closure(Rs1, AW, Rs),
    member(R, Rs),
    s_known_reach(R, Q, AW),
    !.

s_start_linked(S, R, AW) :-
    s_sight(Sight),
    s_radius(Rad),
    s_vps(R, V),
    member(N, V),
    s_dist(S, N, D),
    D =< Sight,
    s_link_free(S, N, AW, Rad),
    !.

s_robot_closure(Set, AW, Final) :-
    (   s_robot(R),
        \+ memberchk(R, Set),
        member(R2, Set),
        s_robots_linked(R2, R, AW)
    ->  s_robot_closure([R|Set], AW, Final)
    ;   Final = Set
    ).

s_robots_linked(R1, R2, AW) :-
    s_sight(Sight),
    s_radius(Rad),
    s_vps(R1, V1),
    s_vps(R2, V2),
    member(A, V1),
    member(B, V2),
    s_dist(A, B, D),
    D =< Sight,
    s_link_free(A, B, AW, Rad),
    !.

% the robot body can move along the segment A-B
s_link_free(A, B, AW, Rad) :-
    Clear is Rad - 0.005,
    s_seg_clear(A, B, AW, Clear).

s_new_can_retrieve(C, P) :-
    s_ensure_loaded,
    s_xy(P, Q),
    s_robot(R),
    s_belief(R, C, Q, Ref, Tol),
    \+ ( s_picked(_, _, C, L),
         s_dist(L, Ref, DL),
         DL =< Tol
       ),
    s_all_walls(AW),
    s_robot(R3),
    s_known_reach(R3, Q, AW),
    !.

/* ------------------------------------------------------------------ */
/*  has_preference                                                     */
/* ------------------------------------------------------------------ */

s_has_preference(R, C) :-
    s_ensure_loaded,
    s_robot(R),
    findall(I-Col, s_picked(R, I, Col, _), L),
    append(_, [Iy-C|After], L),
    member(Ix-Cx, After),
    Cx \== C,
    Ix @< Iy,
    !.

/* ------------------------------------------------------------------ */
/*  Knowledge about objects                                            */
/* ------------------------------------------------------------------ */

% s_belief(+R, +C, +Q, -Ref, -Tol)
%   robot R believes there is an object of colour C at Q.  Ref is the
%   location the belief refers to and Tol how precise it is.
s_belief(R, C, Q, Cn, Tol) :-
    s_obj_tol(Tol),
    s_object(R, Id, _, C),
    s_obj_center(R, Id, Cn),
    s_dist(Cn, Q, D),
    D =< Tol.
s_belief(R, C, Q, NP, Tol) :-
    s_node_obj_radius(Rad),
    Tol is Rad + 0.1,
    s_nodeobj(R, _, C, NP),
    s_dist(NP, Q, D),
    D =< Rad,
    \+ s_nodeobj_picked(R, C, NP, Tol).

% a nodeObject fact that refers to an object the robot has picked up
s_nodeobj_picked(R, C, NP, Tol) :-
    s_picked(R, _, C, L),
    s_dist(L, NP, D),
    D =< Tol,
    !.

% the robot picked up the object that was at Q
s_picked_at(R, Q) :-
    s_obj_tol(Tol),
    s_picked(R, _, _, L),
    s_dist(L, Q, D),
    D =< Tol,
    !.

s_knows_object(R, C, Q) :-
    s_belief(R, C, Q, _, _),
    !.

% centre of a sensed object
s_obj_center(R, Id, Cn) :-
    s_center_c(R, Id, Cn0),
    !,
    Cn = Cn0.
s_obj_center(R, Id, Cn) :-
    s_object(R, Id, Ps, _),
    sort(Ps, Ids),
    findall(X-Y, ( member(I, Ids), once(s_point(R, I, [X, Y])) ), XYs),
    XYs \== [],
    s_mean(XYs, Cn),
    !,
    assertz(s_center_c(R, Id, Cn)).

s_mean(XYs, [CX, CY]) :-
    s_sum_xy(XYs, 0.0, 0.0, SX, SY),
    length(XYs, N),
    CX is SX / N,
    CY is SY / N.

s_sum_xy([], SX, SY, SX, SY).
s_sum_xy([X-Y|T], AX, AY, SX, SY) :-
    BX is AX + X,
    BY is AY + Y,
    s_sum_xy(T, BX, BY, SX, SY).

/* ------------------------------------------------------------------ */
/*  Known reachability                                                 */
/* ------------------------------------------------------------------ */

% s_known_reach(+R, +Q, +Walls)
%   robot R knows how to reach Q, given the known walls Walls.
% picked up an object there
s_known_reach(R, Q, _) :-
    s_obj_tol(Tol),
    s_picked(R, _, _, L),
    s_dist(L, Q, D),
    D =< Tol,
    !.
% sensed an object there
s_known_reach(R, Q, _) :-
    s_obj_tol(Tol),
    s_object(R, Id, _, _),
    s_obj_center(R, Id, Cn),
    s_dist(Cn, Q, D),
    D =< Tol,
    !.
% on sight from a place the robot has been
s_known_reach(R, Q, Ws0) :-
    s_sight(Sight),
    s_vps(R, V),
    s_walls_near(Ws0, Q, Ws),
    member(A, V),
    s_dist(A, Q, D),
    D =< Sight,
    s_visible(A, Q, Ws),
    !.

% places where the robot has been (viewpoints)
s_vps(R, V) :-
    s_vps_c(R, V0),
    !,
    V = V0.
s_vps(R, V) :-
    findall(Q, s_vp(R, Q), L),
    sort(L, V),
    assertz(s_vps_c(R, V)).

s_vp(R, Q) :-
    s_start(R, Q).
s_vp(R, Q) :-
    s_node(R, _, Q, _, _, _).
s_vp(R, Q) :-
    s_node(R, _, A, _, Con, _),
    is_list(Con),
    member(J, Con),
    s_node(R, J, B, _, _, _),
    s_sample(A, B, Q).
s_vp(R, Q) :-
    s_path(R, _, _, Path, Opt),
    (   is_list(Path), member(E, Path)
    ;   is_list(Opt), member(E, Opt)
    ),
    is_list(E),
    E = [_, _, P|_],
    s_xy(P, Q).

s_sample([AX, AY], [BX, BY], [X, Y]) :-
    DX is BX - AX,
    DY is BY - AY,
    D is sqrt(DX*DX + DY*DY),
    N is max(1, ceiling(D / 0.05)),
    between(0, N, K),
    T is float(K) / N,
    X is AX + T*DX,
    Y is AY + T*DY.

/* ------------------------------------------------------------------ */
/*  Known unreachability: flood fill on an occupancy grid              */
/* ------------------------------------------------------------------ */

s_known_unreach(R, Q) :-
    s_grid(R, Grid),
    Grid = grid(X0, Y0, H, NX, NY, G, Border, true),
    Q = [PX, PY],
    s_sight(Sight),
    XMax is X0 + (NX-1)*H,
    YMax is Y0 + (NY-1)*H,
    (   ( PX - Sight < X0 ; PX + Sight > XMax ;
          PY - Sight < Y0 ; PY + Sight > YMax )
    ->  Border == false
    ;   true
    ),
    s_walls(R, Ws0),
    s_walls_near(Ws0, Q, Ws),
    S2 is Sight*Sight,
    I0 is max(0, floor((PX - Sight - X0) / H)),
    I1 is min(NX-1, ceiling((PX + Sight - X0) / H)),
    J0 is max(0, floor((PY - Sight - Y0) / H)),
    J1 is min(NY-1, ceiling((PY + Sight - Y0) / H)),
    \+ ( between(I0, I1, I),
         between(J0, J1, J),
         CX is X0 + I*H,
         CY is Y0 + J*H,
         DX is CX - PX,
         DY is CY - PY,
         DX*DX + DY*DY =< S2,
         K is J*NX + I + 1,
         arg(K, G, Val),
         Val == r,
         s_visible([CX, CY], Q, Ws)
       ).

s_grid(R, Grid) :-
    atomic_list_concat([s_grid_, R], Key),
    (   nb_current(Key, G0)
    ->  true
    ;   (   s_build_grid(R, G1)
        ->  true
        ;   G1 = none
        ),
        nb_setval(Key, G1),
        nb_getval(Key, G0)
    ),
    Grid = G0.

s_build_grid(R, grid(X0, Y0, H, NX, NY, G, Border, Seeded)) :-
    s_walls(R, Ws),
    s_vps(R, V),
    V \== [],
    findall(X-Y, ( member(seg(A, B, C, D), Ws), ( X = A, Y = B ; X = C, Y = D ) ), L1),
    findall(X-Y, member([X, Y], V), L2),
    append(L1, L2, L),
    s_bbox(L, MinX, MaxX, MinY, MaxY),
    s_sight(Sight),
    s_radius(Rad),
    s_cell(H0),
    M is Sight + 0.1,
    W is MaxX - MinX + 2*M,
    Ht is MaxY - MinY + 2*M,
    MaxCells = 250000,
    (   W*Ht / (H0*H0) > MaxCells
    ->  H is sqrt(W*Ht / MaxCells)
    ;   H = H0
    ),
    NX is ceiling(W / H) + 1,
    NY is ceiling(Ht / H) + 1,
    X0 is MinX - M,
    Y0 is MinY - M,
    N is NX*NY,
    functor(G, g, N),
    forall(member(S, Ws), s_block_seg(S, X0, Y0, H, NX, NY, Rad, G)),
    findall(K, ( member([VX, VY], V),
                 I is round((VX - X0) / H),
                 J is round((VY - Y0) / H),
                 between(-1, 1, DI),
                 between(-1, 1, DJ),
                 I2 is I + DI,
                 J2 is J + DJ,
                 I2 >= 0, I2 < NX, J2 >= 0, J2 < NY,
                 K is J2*NX + I2 + 1,
                 arg(K, G, Val),
                 var(Val)
               ), Ks0),
    sort(Ks0, Ks),
    forall(member(K, Ks), nb_setarg(K, G, r)),
    (   Ks == []
    ->  Seeded = false
    ;   Seeded = true
    ),
    s_bfs(Ks, G, NX, NY, false, Border).

s_bbox([X-Y|T], MinX, MaxX, MinY, MaxY) :-
    s_bbox(T, X, X, Y, Y, MinX, MaxX, MinY, MaxY).

s_bbox([], A, B, C, D, A, B, C, D).
s_bbox([X-Y|T], A0, B0, C0, D0, A, B, C, D) :-
    A1 is min(A0, X),
    B1 is max(B0, X),
    C1 is min(C0, Y),
    D1 is max(D0, Y),
    s_bbox(T, A1, B1, C1, D1, A, B, C, D).

% mark the cells too close to a wall segment
s_block_seg(S, X0, Y0, H, NX, NY, Rad, G) :-
    S = seg(AX, AY, BX, BY),
    I0 is max(0, floor((min(AX, BX) - Rad - X0) / H)),
    I1 is min(NX-1, ceiling((max(AX, BX) + Rad - X0) / H)),
    J0 is max(0, floor((min(AY, BY) - Rad - Y0) / H)),
    J1 is min(NY-1, ceiling((max(AY, BY) + Rad - Y0) / H)),
    forall(( between(I0, I1, I),
             between(J0, J1, J),
             CX is X0 + I*H,
             CY is Y0 + J*H,
             s_pt_seg_dist(CX, CY, S, D),
             D < Rad
           ),
           ( K is J*NX + I + 1,
             nb_setarg(K, G, b)
           )).

s_bfs([], _, _, _, B, B) :- !.
s_bfs(Front, G, NX, NY, B0, B) :-
    s_bfs_level(Front, G, NX, NY, B0, B1, [], Next),
    s_bfs(Next, G, NX, NY, B1, B).

s_bfs_level([], _, _, _, B, B, Acc, Acc).
s_bfs_level([K|Ks], G, NX, NY, B0, B, Acc0, Acc) :-
    K0 is K - 1,
    I is K0 mod NX,
    J is K0 // NX,
    (   ( I =:= 0 ; J =:= 0 ; I =:= NX-1 ; J =:= NY-1 )
    ->  B1 = true
    ;   B1 = B0
    ),
    I1 is I - 1,
    I2 is I + 1,
    J1 is J - 1,
    J2 is J + 1,
    s_bfs_try(I1, J, NX, NY, G, Acc0, A1),
    s_bfs_try(I2, J, NX, NY, G, A1, A2),
    s_bfs_try(I, J1, NX, NY, G, A2, A3),
    s_bfs_try(I, J2, NX, NY, G, A3, A4),
    s_bfs_level(Ks, G, NX, NY, B1, B, A4, Acc).

s_bfs_try(I, J, NX, NY, G, A0, A) :-
    (   I >= 0, I < NX, J >= 0, J < NY,
        K is J*NX + I + 1,
        arg(K, G, V),
        var(V)
    ->  nb_setarg(K, G, r),
        A = [K|A0]
    ;   A = A0
    ).

/* ------------------------------------------------------------------ */
/*  Walls                                                              */
/* ------------------------------------------------------------------ */

s_walls(R, Ws) :-
    s_walls_c(R, Ws0),
    !,
    Ws = Ws0.
s_walls(R, Ws) :-
    findall(S, s_wall_seg(R, S), L),
    sort(L, Ws),
    assertz(s_walls_c(R, Ws)).

s_all_walls(Ws) :-
    s_all_walls_c(Ws0),
    !,
    Ws = Ws0.
s_all_walls(Ws) :-
    findall(S, ( s_robot(R), s_walls(R, L), member(S, L) ), L0),
    sort(L0, Ws),
    assertz(s_all_walls_c(Ws)).

s_wall_seg(R, S) :-
    s_edge(R, A, B),
    s_mkseg(R, A, B, S).
s_wall_seg(R, S) :-
    s_outline(R, _, Ps),
    s_consec(Ps, A, B),
    s_mkseg(R, A, B, S).
s_wall_seg(R, S) :-
    s_object(R, _, Ps, _),
    s_closed_consec(Ps, A, B),
    s_mkseg(R, A, B, S).

s_consec([A, B|T], X, Y) :-
    (   X = A, Y = B
    ;   s_consec([B|T], X, Y)
    ).

s_closed_consec(Ps, A, B) :-
    (   s_consec(Ps, A, B)
    ;   Ps = [F, _, _|_],
        last(Ps, Lst),
        Lst \== F,
        A = Lst,
        B = F
    ).

s_mkseg(R, A, B, seg(X1, Y1, X2, Y2)) :-
    once(s_point(R, A, [XA, YA])),
    once(s_point(R, B, [XB, YB])),
    (   XA - XB =:= 0.0, YA - YB =:= 0.0
    ->  fail
    ;   true
    ),
    (   [XA, YA] @=< [XB, YB]
    ->  X1 = XA, Y1 = YA, X2 = XB, Y2 = YB
    ;   X1 = XB, Y1 = YB, X2 = XA, Y2 = YA
    ).

% walls that may hide Q from a point at most the line of sight away;
% the outline of an object located at Q does not hide it.
s_walls_near(Ws0, [PX, PY], Ws) :-
    s_sight(Sight),
    Lim is Sight + 0.05,
    findall(W, ( member(W, Ws0), s_wall_relevant(W, PX, PY, Lim) ), Ws).

s_wall_relevant(W, PX, PY, Lim) :-
    W = seg(AX, AY, BX, BY),
    s_pt_seg_dist(PX, PY, W, D),
    D =< Lim,
    \+ ( s_hyp(AX - PX, AY - PY, DA), DA =< 0.2,
         s_hyp(BX - PX, BY - PY, DB), DB =< 0.2
       ).

/* ------------------------------------------------------------------ */
/*  Geometry                                                           */
/* ------------------------------------------------------------------ */

s_hyp(EX, EY, D) :-
    X is EX,
    Y is EY,
    D is sqrt(X*X + Y*Y).

s_dist([X1, Y1], [X2, Y2], D) :-
    DX is X1 - X2,
    DY is Y1 - Y2,
    D is sqrt(DX*DX + DY*DY).

s_pt_seg_dist(PX, PY, seg(AX, AY, BX, BY), D) :-
    DX is BX - AX,
    DY is BY - AY,
    L is DX*DX + DY*DY,
    (   L =< 1.0e-12
    ->  T = 0.0
    ;   T0 is ((PX - AX)*DX + (PY - AY)*DY) / L,
        T is max(0.0, min(1.0, T0))
    ),
    EX is PX - AX - T*DX,
    EY is PY - AY - T*DY,
    D is sqrt(EX*EX + EY*EY).

% the segment A-B properly crosses the wall segment
s_cross(AX, AY, BX, BY, seg(CX, CY, DX, DY)) :-
    O1 is (BX - AX)*(CY - AY) - (BY - AY)*(CX - AX),
    O2 is (BX - AX)*(DY - AY) - (BY - AY)*(DX - AX),
    O1*O2 < 0.0,
    O3 is (DX - CX)*(AY - CY) - (DY - CY)*(AX - CX),
    O4 is (DX - CX)*(BY - CY) - (DY - CY)*(BX - CX),
    O3*O4 < 0.0.

% nothing hides B from A
s_visible([AX, AY], [BX, BY], Ws) :-
    \+ ( member(W, Ws),
         s_cross(AX, AY, BX, BY, W)
       ).

% the segment A-B keeps at least Clear away from every wall
s_seg_clear([AX, AY], [BX, BY], Ws, Clear) :-
    \+ ( member(W, Ws),
         s_seg_close(AX, AY, BX, BY, W, Clear)
       ).

s_seg_close(AX, AY, BX, BY, W, _) :-
    s_cross(AX, AY, BX, BY, W),
    !.
s_seg_close(AX, AY, BX, BY, W, Clear) :-
    W = seg(CX, CY, DX, DY),
    (   s_pt_seg_dist(AX, AY, W, D)
    ;   s_pt_seg_dist(BX, BY, W, D)
    ;   s_pt_seg_dist(CX, CY, seg(AX, AY, BX, BY), D)
    ;   s_pt_seg_dist(DX, DY, seg(AX, AY, BX, BY), D)
    ),
    D < Clear,
    !.

% planar coordinates of a location [X,Y,Z]
s_xy([X0, Y0|_], [X, Y]) :-
    number(X0),
    number(Y0),
    X is float(X0),
    Y is float(Y0).

/* ------------------------------------------------------------------ */
/*  Loading the knowledge base                                         */
/* ------------------------------------------------------------------ */

s_ensure_loaded :-
    s_loaded,
    !.
s_ensure_loaded :-
    s_kb_dir(Dir),
    (   catch(directory_files(Dir, Fs), _, fail)
    ->  true
    ;   Fs = []
    ),
    msort(Fs, SFs),
    forall(( member(F, SFs),
             atom(F),
             atom_concat(agent_memory, Rest, F),
             atom_concat(XA, '.txt', Rest),
             catch(atom_number(XA, X), _, fail)
           ),
           s_load_robot(Dir, XA, X)),
    assertz(s_loaded).

s_load_robot(Dir, XA, X) :-
    atomic_list_concat([Dir, agent_memory, XA, '.txt'], MF),
    atomic_list_concat([Dir, agent_navMemory, XA, '.txt'], NF),
    s_read_terms(MF, T1),
    (   exists_file(NF)
    ->  s_read_terms(NF, T2)
    ;   T2 = []
    ),
    (   member(agent_ID(Id0), T1),
        atomic(Id0)
    ->  Id = Id0
    ;   Id = X
    ),
    (   s_robot(Id)
    ->  true
    ;   assertz(s_robot(Id))
    ),
    forall(member(T, T1), s_store(Id, T)),
    forall(member(T, T2), s_store(Id, T)).

s_read_terms(File, Terms) :-
    catch(setup_call_cleanup(open(File, read, S),
                             s_read_all(S, Terms),
                             close(S)),
          _,
          Terms = []),
    !.
s_read_terms(_, []).

s_read_all(S, Terms) :-
    character_count(S, C0),
    catch(read_term(S, T, []), _, T = '$s_syntax_error'),
    (   T == end_of_file
    ->  Terms = []
    ;   T == '$s_syntax_error'
    ->  character_count(S, C1),
        (   C1 > C0
        ->  s_read_all(S, Terms)
        ;   Terms = []
        )
    ;   Terms = [T|Rest],
        s_read_all(S, Rest)
    ).

s_store(R, T) :-
    catch(s_store_(R, T), _, true),
    !.
s_store(_, _).

s_store_(R, point(I, P)) :-
    !,
    s_xy(P, Q),
    assertz(s_point(R, I, Q)).
s_store_(R, edge(_, [A, B|_])) :-
    !,
    assertz(s_edge(R, A, B)).
s_store_(R, outline(I, Ps)) :-
    !,
    is_list(Ps),
    assertz(s_outline(R, I, Ps)).
s_store_(R, object(I, Ps, C)) :-
    !,
    is_list(Ps),
    assertz(s_object(R, I, Ps, C)).
s_store_(R, objectPickedUp(I, C, P)) :-
    !,
    s_xy(P, Q),
    assertz(s_picked(R, I, C, Q)).
s_store_(R, nodeNavStart(P)) :-
    !,
    s_xy(P, Q),
    assertz(s_start(R, Q)).
s_store_(R, nodeNav(I, P, Par, Con, V)) :-
    !,
    s_xy(P, Q),
    assertz(s_node(R, I, Q, Par, Con, V)).
s_store_(R, pathNav(I, O, Path, Opt)) :-
    !,
    assertz(s_path(R, I, O, Path, Opt)).
s_store_(R, nodeObject(I, C, P)) :-
    !,
    s_xy(P, Q),
    assertz(s_nodeobj(R, I, C, Q)).
s_store_(_, _).
