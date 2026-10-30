/*  solution.pl -- querying the beliefs of a team of exploring robots.

    The knowledge bases are read (never modified) from /app/KB/:
        agent_memoryX.txt     point/2, edge/2, outline/2, object/3,
                              objectPickedUp/3, ...
        agent_navMemoryX.txt  nodeNavStart/1, nodeNav/5, pathNav/4,
                              nodeObject/3, ...
    X is the id of the robot.  The terms of every file are read into a
    private store (kb_fact/2, tagged with the robot id), from which a
    geometric model is derived for each robot:

      walls      segments given by edge/2 and consecutive outline/2
                 points, except the segments that belong to an object
                 (objects are things to retrieve, not obstacles);
      graph      positions the robot has been at or knows how to get to:
                 nodeNavStart, nodeNav nodes and their connections,
                 pathNav nodes and their connections, nodeObject nodes;
      objects    object/3 (located at the centroid of its points),
                 objectPickedUp/3 and nodeObject/3.

    Derived beliefs (all distances are measured in the XY plane):

      reach      a point on sight (=< 0.5 m) of the graph whose line of
                 sight is not crossed by a known wall, or the location
                 of a known object;
      unreach    a point closer than the robot radius (0.075 m) to a
                 known wall, or a point beyond a known wall that bounds
                 everything the robot knows (the arena border);
      object     an object of colour C is known at P when P is at the
                 location of an object/3 of colour C, or on sight of a
                 nodeObject of colour C that does not refer to an object
                 whose location is known (picked up or object/3);
      no object  the robot picked up the object of colour C at P (it is
                 no longer there), or another object is located at P.

    A new robot uses the combined knowledge of the whole team: objects
    picked up by any robot are gone.
*/

:- use_module(library(lists)).
:- use_module(library(pairs)).

:- dynamic kb_loaded/0.
:- dynamic kb_robot/1.
:- dynamic kb_fact/2.
:- dynamic kb_wall/5.
:- dynamic kb_gpt/3.
:- dynamic kb_gseg/5.
:- dynamic kb_kpt/3.
:- dynamic kb_objloc/4.
:- dynamic kb_pick/4.
:- dynamic kb_nobj/4.

kb_dir('/app/KB').

kb_param(sight,  0.5).      % line of sight
kb_param(radius, 0.075).    % half of the robot diameter
kb_param(objtol, 0.1).      % P designates an object located within objtol
kb_param(region, 0.6).      % sight plus robot radius plus half an object

                 /*******************************
                 *            QUERIES           *
                 *******************************/

%!  current_can_retrieve(+R, +C, +P, ?Cond) is semidet.

current_can_retrieve(R, C, P, Cond) :-
    kb_ensure,
    kb_robot_exists(R),
    kb_xy(P, PX, PY),
    once(kb_retrieve_cond(R, C, PX, PY, Cond0)),
    Cond = Cond0.

%!  new_can_retrieve(+C, +P) is semidet.

new_can_retrieve(C, P) :-
    kb_ensure,
    kb_xy(P, PX, PY),
    \+ kb_picked(all, C, PX, PY),
    \+ kb_unreachable(all, PX, PY),
    once(kb_available(C, PX, PY)).

%!  new_can_reach(+P) is semidet.

new_can_reach(P) :-
    kb_ensure,
    kb_xy(P, PX, PY),
    once(kb_reach_cond(all, PX, PY, Cond)),
    Cond == none.

%!  current_can_reach(+R, +P, ?Cond) is semidet.

current_can_reach(R, P, Cond) :-
    kb_ensure,
    kb_robot_exists(R),
    kb_xy(P, PX, PY),
    once(kb_reach_cond(R, PX, PY, Cond0)),
    Cond = Cond0.

kb_robot_exists(R) :-
    nonvar(R),
    kb_robot(R),
    !.

                 /*******************************
                 *           DECISIONS          *
                 *******************************/

kb_retrieve_cond(S, C, PX, PY, not_object) :-
    kb_picked(S, C, PX, PY).
kb_retrieve_cond(S, C, PX, PY, none) :-
    kb_known_precise(S, C, PX, PY).
kb_retrieve_cond(S, _, PX, PY, not_reachable) :-
    kb_unreachable(S, PX, PY).
kb_retrieve_cond(S, C, PX, PY, none) :-
    kb_known_region(S, C, PX, PY).
kb_retrieve_cond(S, C, PX, PY, not_object) :-
    kb_other_object(S, C, PX, PY).
kb_retrieve_cond(_, _, _, _, explore).

kb_reach_cond(S, PX, PY, none) :-
    kb_object_spot(S, PX, PY).
kb_reach_cond(S, PX, PY, unreachable) :-
    kb_unreachable(S, PX, PY).
kb_reach_cond(S, PX, PY, none) :-
    kb_sight(S, PX, PY).
kb_reach_cond(_, _, _, explore).

%   A scope is either a robot id or `all` (the whole team).

kb_in_scope(all, R) :-
    !,
    kb_robot(R).
kb_in_scope(R, R).

                 /*******************************
                 *            OBJECTS           *
                 *******************************/

kb_known_precise(S, C, PX, PY) :-
    kb_param(objtol, T),
    kb_in_scope(S, R),
    kb_objloc(R, C, X, Y),
    kb_dist(PX, PY, X, Y, D),
    D =< T,
    !.

kb_picked(S, C, PX, PY) :-
    kb_param(objtol, T),
    kb_in_scope(S, R),
    kb_pick(R, C, X, Y),
    kb_dist(PX, PY, X, Y, D),
    D =< T,
    !.

kb_known_region(S, C, PX, PY) :-
    kb_param(region, T),
    kb_in_scope(S, R),
    kb_nobj(R, C, NX, NY),
    kb_dist(PX, PY, NX, NY, D),
    D =< T,
    \+ kb_nobj_resolved(S, C, NX, NY),
    !.

%   The node of a nodeObject is close to an object of the same colour
%   whose location is known: it refers to that object.

kb_nobj_resolved(S, C, NX, NY) :-
    kb_param(region, T),
    kb_in_scope(S, R),
    kb_located(R, C, X, Y),
    kb_dist(NX, NY, X, Y, D),
    D =< T,
    !.

kb_located(R, C, X, Y) :-
    kb_pick(R, C, X, Y).
kb_located(R, C, X, Y) :-
    kb_objloc(R, C, X, Y).

kb_other_object(S, C, PX, PY) :-
    kb_param(objtol, T),
    kb_in_scope(S, R),
    kb_located(R, C1, X, Y),
    C1 \== C,
    kb_dist(PX, PY, X, Y, D),
    D =< T,
    !.

kb_object_spot(S, PX, PY) :-
    kb_param(objtol, T),
    kb_in_scope(S, R),
    kb_located(R, _, X, Y),
    kb_dist(PX, PY, X, Y, D),
    D =< T,
    !.

%   An object known by some robot that no robot has picked up.

kb_available(C, PX, PY) :-
    kb_param(objtol, T),
    kb_objloc(_, C, X, Y),
    kb_dist(PX, PY, X, Y, D),
    D =< T,
    \+ ( kb_pick(_, C, X2, Y2),
         kb_dist(X, Y, X2, Y2, D2),
         D2 =< T
       ).
kb_available(C, PX, PY) :-
    kb_param(region, T),
    kb_nobj(_, C, NX, NY),
    kb_dist(PX, PY, NX, NY, D),
    D =< T,
    \+ kb_nobj_resolved(all, C, NX, NY).

                 /*******************************
                 *         REACHABILITY         *
                 *******************************/

kb_unreachable(S, PX, PY) :-
    kb_param(radius, Rad),
    kb_in_scope(S, R),
    kb_wall(R, X1, Y1, X2, Y2),
    kb_seg_closest(PX, PY, X1, Y1, X2, Y2, _, _, D),
    D < Rad,
    !.
kb_unreachable(S, PX, PY) :-
    kb_in_scope(S, R),
    kb_beyond_wall(R, PX, PY),
    !.

%   P lies beyond the line of a wall and everything robot R knows lies
%   on the other side of (or on) that line.

kb_beyond_wall(R, PX, PY) :-
    kb_param(radius, Rad),
    kb_wall(R, X1, Y1, X2, Y2),
    L is sqrt((X2-X1)*(X2-X1) + (Y2-Y1)*(Y2-Y1)),
    L > 1.0e-9,
    kb_cross(X1, Y1, X2, Y2, PX, PY, CP),
    SP is CP / L,
    abs(SP) > Rad,
    Sign is sign(SP),
    \+ ( kb_kpt(R, QX, QY),
         kb_cross(X1, Y1, X2, Y2, QX, QY, CQ),
         CQ / L * Sign > 0.01
       ),
    !.

kb_sight(S, PX, PY) :-
    kb_in_scope(S, R),
    kb_view_point(R, PX, PY, QX, QY),
    \+ kb_blocked(S, QX, QY, PX, PY),
    !.

kb_view_point(R, PX, PY, QX, QY) :-
    kb_param(sight, Sight),
    kb_gpt(R, QX, QY),
    kb_dist(PX, PY, QX, QY, D),
    D =< Sight.
kb_view_point(R, PX, PY, QX, QY) :-
    kb_param(sight, Sight),
    kb_gseg(R, X1, Y1, X2, Y2),
    kb_seg_closest(PX, PY, X1, Y1, X2, Y2, QX, QY, D),
    D =< Sight.

kb_blocked(S, AX, AY, BX, BY) :-
    kb_in_scope(S, R),
    kb_wall(R, X1, Y1, X2, Y2),
    kb_segments_cross(AX, AY, BX, BY, X1, Y1, X2, Y2),
    !.

                 /*******************************
                 *           GEOMETRY           *
                 *******************************/

kb_xy(P, X, Y) :-
    nonvar(P),
    P = [X0, Y0|_],
    number(X0),
    number(Y0),
    X is float(X0),
    Y is float(Y0).

kb_dist(X1, Y1, X2, Y2, D) :-
    D is sqrt((X1-X2)*(X1-X2) + (Y1-Y2)*(Y1-Y2)).

%   kb_cross(+AX,+AY, +BX,+BY, +PX,+PY, -V): cross product (B-A)x(P-A).

kb_cross(AX, AY, BX, BY, PX, PY, V) :-
    V is (BX-AX)*(PY-AY) - (BY-AY)*(PX-AX).

kb_seg_closest(PX, PY, X1, Y1, X2, Y2, QX, QY, D) :-
    DX is X2 - X1,
    DY is Y2 - Y1,
    L2 is DX*DX + DY*DY,
    (   L2 =< 1.0e-12
    ->  T = 0.0
    ;   T0 is ((PX-X1)*DX + (PY-Y1)*DY) / L2,
        T is max(0.0, min(1.0, T0))
    ),
    QX is X1 + T*DX,
    QY is Y1 + T*DY,
    kb_dist(PX, PY, QX, QY, D).

%   Proper intersection of segments A-B and C-D.

kb_segments_cross(AX, AY, BX, BY, CX, CY, DX, DY) :-
    kb_cross(CX, CY, DX, DY, AX, AY, D1),
    kb_cross(CX, CY, DX, DY, BX, BY, D2),
    D1*D2 < 0,
    kb_cross(AX, AY, BX, BY, CX, CY, D3),
    kb_cross(AX, AY, BX, BY, DX, DY, D4),
    D3*D4 < 0.

                 /*******************************
                 *            LOADING           *
                 *******************************/

kb_ensure :-
    kb_loaded,
    !.
kb_ensure :-
    (   catch(kb_load, _, fail)
    ->  assertz(kb_loaded)
    ;   true
    ).

kb_clear :-
    retractall(kb_robot(_)),
    retractall(kb_fact(_, _)),
    retractall(kb_wall(_, _, _, _, _)),
    retractall(kb_gpt(_, _, _)),
    retractall(kb_gseg(_, _, _, _, _)),
    retractall(kb_kpt(_, _, _)),
    retractall(kb_objloc(_, _, _, _)),
    retractall(kb_pick(_, _, _, _)),
    retractall(kb_nobj(_, _, _, _)).

kb_load :-
    kb_clear,
    kb_dir(Dir),
    exists_directory(Dir),
    directory_files(Dir, Files),
    msort(Files, Sorted),
    forall(( member(F, Sorted),
             kb_file_robot(F, Id)
           ),
           ( atomic_list_concat([Dir, '/', F], Path),
             kb_load_file(Path, Id)
           )),
    forall(kb_robot(R), kb_build(R)).

kb_file_robot(F, Id) :-
    atom(F),
    (   atom_concat(agent_memory, Rest, F)
    ;   atom_concat(agent_navMemory, Rest, F)
    ),
    atom_concat(IdA, '.txt', Rest),
    IdA \== '',
    catch(atom_number(IdA, Id), _, fail),
    integer(Id),
    !.

kb_load_file(Path, Id) :-
    (   kb_robot(Id)
    ->  true
    ;   assertz(kb_robot(Id))
    ),
    catch(setup_call_cleanup(open(Path, read, S),
                             kb_read_terms(S, Id),
                             close(S)),
          _, true).

kb_read_terms(S, Id) :-
    catch(read_term(S, T, []), _, T = '$kb_error'),
    (   T == end_of_file
    ->  true
    ;   T == '$kb_error', at_end_of_stream(S)
    ->  true
    ;   kb_store(T, Id),
        kb_read_terms(S, Id)
    ).

kb_store(T, _) :-
    var(T),
    !.
kb_store((:- _), _) :-
    !.
kb_store((_ :- _), _) :-
    !.
kb_store('$kb_error', _) :-
    !.
kb_store(T, Id) :-
    assertz(kb_fact(Id, T)).

                 /*******************************
                 *        DERIVED MODEL         *
                 *******************************/

kb_build(R) :-
    forall(kb_fact(R, point(_, P)), kb_add_kpt(R, P)),
    kb_build_walls(R),
    kb_build_graph(R),
    kb_build_objects(R).

kb_point_xy(_, P, X, Y) :-
    is_list(P),
    !,
    kb_xy(P, X, Y).
kb_point_xy(R, Id, X, Y) :-
    kb_fact(R, point(Id, P)),
    kb_xy(P, X, Y),
    !.

kb_raw_segment(R, A, B) :-
    kb_fact(R, edge(_, [A, B|_])).
kb_raw_segment(R, A, B) :-
    kb_fact(R, outline(_, Ps)),
    is_list(Ps),
    nextto(A, B, Ps).

kb_object_segment(R, A, B) :-
    kb_fact(R, object(_, Ps, _)),
    is_list(Ps),
    memberchk(A, Ps),
    memberchk(B, Ps),
    !.

kb_build_walls(R) :-
    findall(A-B,
            ( kb_raw_segment(R, A, B),
              \+ kb_object_segment(R, A, B)
            ),
            L0),
    sort(L0, L),
    forall(member(A-B, L),
           (   kb_point_xy(R, A, X1, Y1),
               kb_point_xy(R, B, X2, Y2)
           ->  assertz(kb_wall(R, X1, Y1, X2, Y2))
           ;   true
           )).

kb_build_graph(R) :-
    forall(kb_fact(R, nodeNavStart(P)), kb_add_gpt(R, P)),
    forall(kb_fact(R, nodeNav(_, P, _, _, _)), kb_add_gpt(R, P)),
    forall(( kb_fact(R, nodeNav(_, P1, _, Con, _)),
             is_list(Con),
             member(J, Con),
             kb_fact(R, nodeNav(J, P2, _, _, _))
           ),
           kb_add_gseg(R, P1, P2)),
    forall(( kb_fact(R, nodeNav(_, P1, Par, _, _)),
             integer(Par),
             kb_fact(R, nodeNav(Par, P2, _, _, _))
           ),
           kb_add_gseg(R, P1, P2)),
    forall(kb_fact(R, pathNav(_, O, _, _)), kb_add_gpt(R, O)),
    forall(( kb_fact(R, pathNav(_, _, Path, Opt)),
             ( L = Path ; L = Opt ),
             is_list(L)
           ),
           kb_add_path(R, L)),
    forall(kb_fact(R, nodeObject(_, _, P)), kb_add_gpt(R, P)),
    forall(kb_fact(R, agent_Pos(P, _)), kb_add_gpt(R, P)).

kb_add_path(R, L) :-
    forall(( member(E, L),
             nonvar(E),
             E = [_, _, P|_]
           ),
           kb_add_gpt(R, P)),
    forall(( member(E1, L),
             nonvar(E1),
             E1 = [_, _, P1, Con|_],
             is_list(Con),
             member(J, Con),
             member(E2, L),
             nonvar(E2),
             E2 = [J0, _, P2|_],
             J0 == J
           ),
           kb_add_gseg(R, P1, P2)).

kb_build_objects(R) :-
    forall(kb_fact(R, object(_, Ps, C)), kb_add_object(R, Ps, C)),
    forall(kb_fact(R, objectPickedUp(_, C, P)),
           (   kb_xy(P, X, Y)
           ->  assertz(kb_pick(R, C, X, Y))
           ;   true
           )),
    forall(kb_fact(R, nodeObject(_, C, P)),
           (   kb_xy(P, X, Y)
           ->  assertz(kb_nobj(R, C, X, Y))
           ;   true
           )).

kb_add_object(R, Ps, C) :-
    (   is_list(Ps),
        findall(X-Y, ( member(Id, Ps), kb_point_xy(R, Id, X, Y) ), XYs),
        XYs \== []
    ->  length(XYs, N),
        pairs_keys_values(XYs, Xs, Ys),
        sum_list(Xs, SX),
        sum_list(Ys, SY),
        CX is SX / N,
        CY is SY / N,
        assertz(kb_objloc(R, C, CX, CY))
    ;   true
    ).

kb_add_kpt(R, P) :-
    (   kb_xy(P, X, Y)
    ->  assertz(kb_kpt(R, X, Y))
    ;   true
    ).

kb_add_gpt(R, P) :-
    (   kb_xy(P, X, Y)
    ->  assertz(kb_gpt(R, X, Y)),
        assertz(kb_kpt(R, X, Y))
    ;   true
    ).

kb_add_gseg(R, P1, P2) :-
    (   kb_xy(P1, X1, Y1),
        kb_xy(P2, X2, Y2)
    ->  assertz(kb_gseg(R, X1, Y1, X2, Y2))
    ;   true
    ).

:- initialization(catch(kb_ensure, _, true)).
