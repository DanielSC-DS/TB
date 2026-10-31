/*  solution.pl

    Queries over the knowledge bases built by a team of exploring robots.

    The knowledge bases are read (never modified) from /app/KB/:
        agent_memoryX.txt     geometric memory of robot X
        agent_navMemoryX.txt  navigation memory (experience graph) of robot X

    Public predicates:
        current_can_retrieve(+R, +C, +P, ?Cond)
        current_can_reach(+R, +P, ?Cond)
        new_can_retrieve(+C, +P)
        new_can_reach(+Start, +P)
        new_can_reach(+P)

    Model
    -----
    * Robots have a diameter of 0.15 m and a 0.5 m line of sight: a point or
      an object is reachable from a place the robot can be at when it lies
      within 0.5 m and the straight line towards it does not cross a known
      wall (edge).
    * The places a robot can be at are the nodes of its experience graph,
      the nodes of its computed paths (and their origins), its start and its
      current position.  They are connected through the graph/path links and
      through mutual line of sight.  Only places connected to the start (or
      current position) are considered reachable.
    * A robot knows an object of colour C at P when it has an object/3 fact
      of colour C whose outline is centred at P, it picked up an object of
      colour C at P, or it recorded (nodeObject/3) an object of colour C next
      to a node that has P within sight.
    * A robot knows that P cannot be reached when P lies beyond a boundary
      wall it has seen (the outermost known walls of the arena) or beyond its
      vertical sight.
    * A new robot uses the union of the knowledge of the whole team.

    Derived data is cached in private s_* predicates, the knowledge bases
    themselves are left untouched.
*/

:- use_module(library(lists)).
:- use_module(library(apply)).

:- dynamic
    s_loaded/0,
    s_robot/1,
    s_alias/2,
    s_point/3,
    s_edge/3,
    s_object/3,
    s_picked/3,
    s_pos/2,
    s_start/2,
    s_node/4,
    s_pathwp/5,
    s_origin/3,
    s_nodeobj/3,
    s_scope_ready/1,
    s_wall/5,
    s_wp/3,
    s_link/3,
    s_reach/2,
    s_bound/3.

s_kb_dir('/app/KB').

s_radius(0.075).          % robot diameter 0.15 m
s_sight(0.5).             % line of sight
s_object_tol(0.1).        % objects are ~0.1 m wide

s_all('$all').

%   Read the knowledge bases and build the derived data at load time, so
%   that queries stay fast.  Queries also do it lazily if needed.

:- initialization(s_preload).

s_preload :-
    catch(( s_ensure,
            forall(s_robot(R), s_ensure_scope(R)),
            s_all(S),
            s_ensure_scope(S)
          ),
          E,
          print_message(warning, E)),
    !.
s_preload.

                 /*******************************
                 *        PUBLIC QUERIES        *
                 *******************************/

%!  current_can_retrieve(+R, +C, +P, ?Cond)

current_can_retrieve(R, C, P, Cond) :-
    s_ensure,
    once(s_robot_key(R, K)),
    s_xyz(P, Q),
    s_ensure_scope(K),
    s_retrieve_cond(K, [K], C, Q, Cond0),
    Cond = Cond0.

%!  current_can_reach(+R, +P, ?Cond)

current_can_reach(R, P, Cond) :-
    s_ensure,
    once(s_robot_key(R, K)),
    s_xyz(P, Q),
    s_ensure_scope(K),
    s_reach_cond(K, [K], Q, Cond0),
    Cond = Cond0.

%!  new_can_retrieve(+C, +P)

new_can_retrieve(C, P) :-
    s_ensure,
    s_xyz(P, Q),
    s_all(S),
    s_ensure_scope(S),
    findall(R, s_robot(R), Rs),
    once(( member(R, Rs), s_knows_object(R, C, Q) )),
    \+ s_known_unreachable(S, Q),
    !.

%!  new_can_reach(+Start, +P)

new_can_reach(Start, P) :-
    s_ensure,
    s_xyz(Start, QS),
    s_xyz(P, Q),
    s_all(S),
    s_ensure_scope(S),
    \+ s_known_unreachable(S, QS),
    \+ s_known_unreachable(S, Q),
    (   s_sees(S, QS, Q)
    ->  true
    ;   findall(K, ( s_wp(S, K, QW), s_sees(S, QW, QS) ), Starts),
        Starts \== [],
        s_bfs(S, Starts, Reached),
        once(( member(K1, Reached), s_wp(S, K1, QW1), s_sees(S, QW1, Q) ))
    ),
    !.

%!  new_can_reach(+P)
%
%   A new robot deployed at any of the team's deployment locations can
%   reach P.

new_can_reach(P) :-
    s_ensure,
    s_xyz(P, Q),
    s_all(S),
    s_ensure_scope(S),
    findall(R, s_robot(R), Rs),
    \+ s_known_unreachable(S, Q),
    (   s_reachable_place(S, Q)
    ->  true
    ;   once(( member(R, Rs), s_knows_object(R, _, Q) ))
    ),
    !.

                 /*******************************
                 *          CONDITIONS          *
                 *******************************/

s_retrieve_cond(S, Rs, C, Q, Cond) :-
    (   once(( member(R, Rs), s_knows_object(R, C, Q) ))
    ->  Cond = none
    ;   s_known_unreachable(S, Q)
    ->  Cond = not_reachable
    ;   s_can_reach(S, Rs, Q)
    ->  Cond = not_object
    ;   Cond = explore
    ).

s_reach_cond(S, Rs, Q, Cond) :-
    (   s_known_unreachable(S, Q)
    ->  Cond = unreachable
    ;   s_can_reach(S, Rs, Q)
    ->  Cond = none
    ;   Cond = explore
    ).

%   P is reachable when it is in sight of a reachable place, or when the
%   robot already saw/picked an object there.

s_can_reach(S, _, Q) :-
    s_reachable_place(S, Q),
    !.
s_can_reach(_, Rs, Q) :-
    member(R, Rs),
    s_object_at(R, _, Q),
    !.

s_reachable_place(S, Q) :-
    s_reach(S, K),
    s_wp(S, K, QW),
    s_sees(S, QW, Q),
    !.

%   Object knowledge.

s_knows_object(R, C, Q) :-
    s_object_at(R, C, Q),
    !.
s_knows_object(R, C, Q) :-
    s_sight(Sight),
    s_nodeobj(R, C, QN),
    s_dist3(QN, Q, D),
    D =< Sight,
    !.

s_object_at(R, C, Q) :-
    s_object_tol(Tol),
    s_object_centre(R, C, QC),
    s_dist2(QC, Q, D),
    D =< Tol,
    !.
s_object_at(R, C, Q) :-
    s_object_tol(Tol),
    s_picked(R, C, QP),
    s_dist2(QP, Q, D),
    D =< Tol,
    !.

s_object_centre(R, C, p(X, Y, Z)) :-
    s_object(R, Ids, C),
    findall(Q, ( member(I, Ids), s_point(R, I, Q) ), Qs),
    Qs \== [],
    length(Qs, N),
    foldl(s_sum_xyz, Qs, 0.0-0.0-0.0, SX-SY-SZ),
    X is SX/N, Y is SY/N, Z is SZ/N.

s_sum_xyz(p(X, Y, Z), X0-Y0-Z0, X1-Y1-Z1) :-
    X1 is X0+X, Y1 is Y0+Y, Z1 is Z0+Z.

%   Known unreachable locations.

s_known_unreachable(_, p(_, _, Z)) :-
    s_sight(Sight),
    abs(Z) > Sight,
    !.
s_known_unreachable(S, p(X, Y, _)) :-
    Eps = 0.001,
    (   s_bound(S, xmax, V), X > V+Eps
    ;   s_bound(S, xmin, V), X < V-Eps
    ;   s_bound(S, ymax, V), Y > V+Eps
    ;   s_bound(S, ymin, V), Y < V-Eps
    ),
    !.

%   Line of sight from a place QA to a target QB.

s_sees(S, QA, QB) :-
    s_sight(Sight),
    s_dist3(QA, QB, D),
    D =< Sight,
    s_radius(Rad),
    \+ s_blocked(S, QA, QB, Rad).

%   A wall blocks the segment A-B when it crosses it.  Walls closer than
%   Excl to the target B (e.g. the outline of the target object itself) are
%   ignored.

s_blocked(S, p(AX, AY, _), p(BX, BY, _), Excl) :-
    s_wall(S, CX, CY, DX, DY),
    s_cross(AX, AY, BX, BY, CX, CY, DX, DY),
    (   Excl > 0
    ->  s_seg_dist(BX, BY, CX, CY, DX, DY, Dist),
        Dist >= Excl
    ;   true
    ),
    !.

                 /*******************************
                 *           GEOMETRY           *
                 *******************************/

s_dist2(p(X1, Y1, _), p(X2, Y2, _), D) :-
    D is sqrt((X1-X2)*(X1-X2) + (Y1-Y2)*(Y1-Y2)).

s_dist3(p(X1, Y1, Z1), p(X2, Y2, Z2), D) :-
    D is sqrt((X1-X2)*(X1-X2) + (Y1-Y2)*(Y1-Y2) + (Z1-Z2)*(Z1-Z2)).

s_cross(AX, AY, BX, BY, CX, CY, DX, DY) :-
    max(AX, BX) >= min(CX, DX),
    max(CX, DX) >= min(AX, BX),
    max(AY, BY) >= min(CY, DY),
    max(CY, DY) >= min(AY, BY),
    D1 is (BX-AX)*(CY-AY) - (BY-AY)*(CX-AX),
    D2 is (BX-AX)*(DY-AY) - (BY-AY)*(DX-AX),
    D1*D2 < 0,
    D3 is (DX-CX)*(AY-CY) - (DY-CY)*(AX-CX),
    D4 is (DX-CX)*(BY-CY) - (DY-CY)*(BX-CX),
    D3*D4 < 0.

s_seg_dist(PX, PY, X1, Y1, X2, Y2, D) :-
    VX is X2-X1,
    VY is Y2-Y1,
    L2 is VX*VX + VY*VY,
    (   L2 =:= 0
    ->  T = 0.0
    ;   T0 is ((PX-X1)*VX + (PY-Y1)*VY)/L2,
        T is max(0.0, min(1.0, T0))
    ),
    CX is X1 + T*VX,
    CY is Y1 + T*VY,
    D is sqrt((PX-CX)*(PX-CX) + (PY-CY)*(PY-CY)).

s_xyz([X, Y, Z|_], p(X1, Y1, Z1)) :-
    number(X), number(Y), number(Z),
    !,
    X1 is float(X), Y1 is float(Y), Z1 is float(Z).
s_xyz([X, Y], p(X1, Y1, 0.0)) :-
    number(X), number(Y),
    X1 is float(X), Y1 is float(Y).

                 /*******************************
                 *        DERIVED SCOPES        *
                 *******************************/

%   A scope is either a robot id (that robot's own knowledge) or '$all'
%   (the union of the knowledge of the team).

s_ensure_scope(S) :-
    s_scope_ready(S),
    !.
s_ensure_scope(S) :-
    with_mutex(s_kb_scope,
               (   s_scope_ready(S)
               ->  true
               ;   s_build_scope(S),
                   assertz(s_scope_ready(S))
               )).

s_scope_robots(S, Rs) :-
    s_all(S),
    !,
    findall(R, s_robot(R), Rs).
s_scope_robots(S, [S]).

s_build_scope(S) :-
    s_scope_robots(S, Rs),
    s_build_walls(S, Rs),
    s_build_bounds(S),
    s_build_waypoints(S, Rs),
    s_build_links(S, Rs),
    findall(K, ( member(R, Rs),
                 ( K = st(R) ; K = pos(R) ),
                 s_wp(S, K, _) ),
            Starts),
    s_bfs(S, Starts, Reached),
    forall(member(K, Reached), assertz(s_reach(S, K))).

s_build_walls(S, Rs) :-
    forall(( member(R, Rs),
             s_edge(R, _, A-B),
             s_point(R, A, p(X1, Y1, _)),
             s_point(R, B, p(X2, Y2, _))
           ),
           assertz(s_wall(S, X1, Y1, X2, Y2))).

%   Outer walls: the extreme coordinates of the known walls, when a wall
%   lies along that extreme line.

s_build_bounds(S) :-
    Tol = 0.01,
    findall(X, ( s_wall(S, X1, _, X2, _), member(X, [X1, X2]) ), Xs),
    findall(Y, ( s_wall(S, _, Y1, _, Y2), member(Y, [Y1, Y2]) ), Ys),
    (   Xs \== []
    ->  max_list(Xs, XMax), min_list(Xs, XMin),
        max_list(Ys, YMax), min_list(Ys, YMin),
        (   s_wall(S, A, _, B, _), abs(A-XMax) =< Tol, abs(B-XMax) =< Tol
        ->  assertz(s_bound(S, xmax, XMax)) ; true ),
        (   s_wall(S, A1, _, B1, _), abs(A1-XMin) =< Tol, abs(B1-XMin) =< Tol
        ->  assertz(s_bound(S, xmin, XMin)) ; true ),
        (   s_wall(S, _, C, _, D), abs(C-YMax) =< Tol, abs(D-YMax) =< Tol
        ->  assertz(s_bound(S, ymax, YMax)) ; true ),
        (   s_wall(S, _, C1, _, D1), abs(C1-YMin) =< Tol, abs(D1-YMin) =< Tol
        ->  assertz(s_bound(S, ymin, YMin)) ; true )
    ;   true
    ).

s_build_waypoints(S, Rs) :-
    forall(( member(R, Rs), s_robot_place(R, K, Q) ),
           assertz(s_wp(S, K, Q))).

s_robot_place(R, n(R, I), Q)          :- s_node(R, I, Q, _).
s_robot_place(R, pth(R, I, W, N), Q)  :- s_pathwp(R, I, W, N, Q).
s_robot_place(R, org(R, I), Q)        :- s_origin(R, I, Q).
s_robot_place(R, st(R), Q)            :- s_start(R, Q).
s_robot_place(R, pos(R), Q)           :- s_pos(R, Q).

%   Explicit links (experience graph and computed paths) plus links between
%   places in mutual line of sight.

s_build_links(S, Rs) :-
    forall(( member(R, Rs), s_explicit_link(R, K1, K2),
             s_wp(S, K1, _), s_wp(S, K2, _) ),
           s_add_link(S, K1, K2)),
    findall(K-Q, s_wp(S, K, Q), KQs),
    s_sight(Sight),
    forall(( append(_, [K1-Q1|Rest], KQs),
             member(K2-Q2, Rest),
             s_dist3(Q1, Q2, D),
             D =< Sight,
             \+ s_blocked(S, Q1, Q2, 0)
           ),
           s_add_link(S, K1, K2)).

s_add_link(S, K1, K2) :-
    (   s_link(S, K1, K2) -> true ; assertz(s_link(S, K1, K2)) ),
    (   s_link(S, K2, K1) -> true ; assertz(s_link(S, K2, K1)) ).

s_explicit_link(R, n(R, I), n(R, J)) :-
    s_node(R, I, _, Links),
    member(J, Links),
    J \== I.
s_explicit_link(R, org(R, I), pth(R, I, W, 1)) :-
    s_pathwp(R, I, W, 1, _).
s_explicit_link(R, pth(R, I, W, N), pth(R, I, W, N1)) :-
    s_pathwp(R, I, W, N, _),
    N1 is N+1,
    s_pathwp(R, I, W, N1, _).
s_explicit_link(R, st(R), K) :-
    s_start(R, QS),
    s_node(R, I, QN, _),
    s_dist3(QS, QN, D),
    D < 0.01,
    K = n(R, I).

%   Breadth-first search over s_link/3.

s_bfs(S, Starts, Reached) :-
    sort(Starts, Starts1),
    s_bfs_(Starts1, S, Starts1, Reached).

s_bfs_([], _, Seen, Seen).
s_bfs_([K|Queue], S, Seen0, Seen) :-
    findall(N, ( s_link(S, K, N), \+ memberchk(N, Seen0) ), Ns0),
    sort(Ns0, Ns),
    append(Seen0, Ns, Seen1),
    append(Queue, Ns, Queue1),
    s_bfs_(Queue1, S, Seen1, Seen).

                 /*******************************
                 *      LOADING THE KB FILES    *
                 *******************************/

s_ensure :-
    s_loaded,
    !.
s_ensure :-
    s_kb_dir(Dir),
    exists_directory(Dir),
    with_mutex(s_kb_load,
               (   s_loaded
               ->  true
               ;   s_load_all,
                   assertz(s_loaded)
               )).

s_robot_key(R, R) :-
    s_robot(R).
s_robot_key(R, K) :-
    s_alias(R, K).

s_load_all :-
    s_kb_dir(Dir),
    (   exists_directory(Dir)
    ->  directory_files(Dir, Files0)
    ;   Files0 = []
    ),
    msort(Files0, Files),
    forall(( member(F, Files),
             s_kb_file(F, Id)
           ),
           (   atomic_list_concat([Dir, '/', F], Path),
               s_load_file(Path, Id)
           )).

s_kb_file(F, Id) :-
    atom(F),
    (   atom_concat(agent_memory, Rest, F)
    ;   atom_concat(agent_navMemory, Rest, F)
    ),
    atom_concat(IdA, '.txt', Rest),
    IdA \== '',
    (   atom_number(IdA, N)
    ->  Id = N
    ;   Id = IdA
    ).

s_load_file(Path, Id) :-
    (   s_robot(Id) -> true ; assertz(s_robot(Id)) ),
    catch(setup_call_cleanup(open(Path, read, Stream),
                             s_read_terms(Stream, Id),
                             close(Stream)),
          E,
          print_message(warning, E)),
    !.
s_load_file(_, _).

s_read_terms(Stream, Id) :-
    catch(read_term(Stream, T, [syntax_errors(dec10)]), _, T = end_of_file),
    (   T == end_of_file
    ->  true
    ;   ( catch(s_store(T, Id), _, true) -> true ; true ),
        s_read_terms(Stream, Id)
    ).

s_store(T, _) :-
    var(T),
    !.
s_store((:- _), _) :-
    !.
s_store(agent_ID(A), R) :-
    !,
    (   A == R -> true
    ;   s_alias(A, R) -> true
    ;   assertz(s_alias(A, R))
    ).
s_store(point(I, P), R) :-
    !,
    s_xyz(P, Q),
    assertz(s_point(R, I, Q)).
s_store(edge(I, [A, B]), R) :-
    !,
    assertz(s_edge(R, I, A-B)).
s_store(object(_, Ids, C), R) :-
    !,
    is_list(Ids),
    assertz(s_object(R, Ids, C)).
s_store(objectPickedUp(_, C, P), R) :-
    !,
    s_xyz(P, Q),
    assertz(s_picked(R, C, Q)).
s_store(agent_Pos(P, _), R) :-
    !,
    s_xyz(P, Q),
    (   s_pos(R, _) -> true ; assertz(s_pos(R, Q)) ).
s_store(nodeNavStart(P), R) :-
    !,
    s_xyz(P, Q),
    (   s_start(R, _) -> true ; assertz(s_start(R, Q)) ).
s_store(nodeNav(I, P, Parent, Con, _), R) :-
    !,
    s_xyz(P, Q),
    (   is_list(Con) -> Con1 = Con ; Con1 = [] ),
    (   integer(Parent) -> Links = [Parent|Con1] ; Links = Con1 ),
    assertz(s_node(R, I, Q, Links)).
s_store(pathNav(I, O, Path, Opt), R) :-
    !,
    (   s_xyz(O, QO) -> assertz(s_origin(R, I, QO)) ; true ),
    s_store_path(R, I, path, Path),
    s_store_path(R, I, opt, Opt).
s_store(nodeObject(_, C, P), R) :-
    !,
    s_xyz(P, Q),
    assertz(s_nodeobj(R, C, Q)).
s_store(_, _).

s_store_path(R, I, W, Path) :-
    (   is_list(Path)
    ->  forall(nth1(N, Path, Entry),
               (   Entry = [_, _, P|_], s_xyz(P, Q)
               ->  assertz(s_pathwp(R, I, W, N, Q))
               ;   true
               ))
    ;   true
    ).
