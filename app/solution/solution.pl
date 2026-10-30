/*  solution.pl

    Query the beliefs of a team of exploring robots.

    The knowledge base of every robot is read directly from the files
    /app/KB/agent_memoryX.txt and /app/KB/agent_navMemoryX.txt (X is the
    robot id).  Reading the files ourselves keeps the facts of each robot
    apart, since all robots use the same predicate names (point/2, edge/2,
    nodeNav/5, ...) with independent ids.  The knowledge base is never
    modified: every derived fact is stored in sol_* predicates.  If the
    files cannot be found, the facts already loaded in module user are
    used instead.

    Exported predicates:
      current_can_retrieve(+R, +C, +P, ?Cond)
      new_can_retrieve(+C, +P)
      new_can_reach(+P)
      current_can_reach(+R, +P, ?Cond)

    Geometry is evaluated on the X-Y plane (robots move on the floor).
    A robot reaches what it has in sight: any point within 0.5m of the
    places it knows how to get to (its experience graph nodes and the
    connections between them, its start, its current position, its
    computed paths and the places where it picked up objects).
    A robot knows a point cannot be reached when the point lies beyond a
    boundary wall it has seen (the outermost walls of its map) or inside
    a closed obstacle outline.
*/

:- dynamic sol_loaded/0.
:- dynamic sol_robot/1.
:- dynamic sol_kb/2.
:- dynamic sol_pt/3.
:- dynamic sol_wall/3.
:- dynamic sol_seg/3.
:- dynamic sol_object/3.
:- dynamic sol_picked/3.
:- dynamic sol_nodeobj/3.
:- dynamic sol_obstacle/2.
:- dynamic sol_side/3.

sol_kb_dir('/app/KB').

sol_sight_radius(0.5).
sol_eps(1.0e-6).

%   ---------------------------------------------------------------
%   Public predicates
%   ---------------------------------------------------------------

current_can_retrieve(R, C, P, Cond) :-
    sol_ensure_loaded,
    sol_valid(P),
    sol_robot(R),
    sol_retrieve_cond(R, C, P, Cond0),
    Cond = Cond0.

current_can_reach(R, P, Cond) :-
    sol_ensure_loaded,
    sol_valid(P),
    sol_robot(R),
    sol_reach_cond(R, P, Cond0),
    Cond = Cond0.

new_can_reach(P) :-
    sol_ensure_loaded,
    sol_valid(P),
    \+ sol_team_unreachable(P),
    once(( sol_robot(R), sol_in_sight(R, P) )).

new_can_retrieve(C, P) :-
    sol_ensure_loaded,
    sol_valid(P),
    \+ sol_team_unreachable(P),
    once(( sol_robot(R), sol_knows_object(R, C, P) )).

%   ---------------------------------------------------------------
%   Conditions
%   ---------------------------------------------------------------

sol_retrieve_cond(R, C, P, Cond) :-
    (   sol_knows_object(R, C, P)
    ->  Cond = none
    ;   sol_unreachable(R, P)
    ->  Cond = not_reachable
    ;   sol_in_sight(R, P)
    ->  Cond = not_object
    ;   Cond = explore
    ).

sol_reach_cond(R, P, Cond) :-
    (   sol_unreachable(R, P)
    ->  Cond = unreachable
    ;   sol_in_sight(R, P)
    ->  Cond = none
    ;   Cond = explore
    ).

%   The robot knows an object of colour C at P.
sol_knows_object(R, C, P) :-
    sol_sight_radius(Rad),
    (   sol_object(R, C, Pts),
        sol_geom_dist(P, Pts, D),
        D =< Rad + 1.0e-9
    ;   sol_picked(R, C, Q),
        sol_dist(P, Q, D),
        D =< Rad + 1.0e-9
    ;   sol_nodeobj(R, C, Q),
        sol_dist(P, Q, D),
        D =< 0.01
    ),
    !.

sol_in_sight(R, P) :-
    sol_sight_radius(Rad),
    sol_seg(R, A, B),
    sol_seg_dist(P, A, B, D),
    D =< Rad + 1.0e-9,
    !.

sol_unreachable(R, P) :-
    sol_beyond_boundary(R, P),
    !.
sol_unreachable(R, P) :-
    sol_obstacle(R, Poly),
    sol_in_poly(P, Poly),
    !.

sol_team_unreachable(P) :-
    sol_robot(R),
    sol_unreachable(R, P),
    !.

sol_beyond_boundary(R, P) :-
    sol_eps(E),
    sol_xy(P, X, Y),
    (   sol_side(R, xmax, V), X > V + E
    ;   sol_side(R, xmin, V), X < V - E
    ;   sol_side(R, ymax, V), Y > V + E
    ;   sol_side(R, ymin, V), Y < V - E
    ),
    !.

%   ---------------------------------------------------------------
%   Geometry
%   ---------------------------------------------------------------

sol_valid(P) :-
    is_list(P),
    P = [X, Y|_],
    number(X),
    number(Y).

sol_xy([X, Y|_], X, Y).

sol_dist(P, Q, D) :-
    sol_xy(P, X1, Y1),
    sol_xy(Q, X2, Y2),
    DX is X1 - X2,
    DY is Y1 - Y2,
    D is sqrt(DX*DX + DY*DY).

%   Distance from P to the segment A-B.
sol_seg_dist(P, A, B, D) :-
    sol_xy(P, PX, PY),
    sol_xy(A, AX, AY),
    sol_xy(B, BX, BY),
    DX is BX - AX,
    DY is BY - AY,
    L is DX*DX + DY*DY,
    (   L =:= 0
    ->  T = 0.0
    ;   T0 is ((PX - AX)*DX + (PY - AY)*DY) / L,
        T is max(0.0, min(1.0, T0))
    ),
    CX is AX + T*DX - PX,
    CY is AY + T*DY - PY,
    D is sqrt(CX*CX + CY*CY).

%   Distance from P to the polyline through Pts.
sol_geom_dist(P, [Q], D) :-
    !,
    sol_dist(P, Q, D).
sol_geom_dist(P, Pts, D) :-
    sol_pairs(Pts, Pairs),
    findall(D1, ( member(A-B, Pairs), sol_seg_dist(P, A, B, D1) ), Ds),
    Ds \== [],
    min_list(Ds, D).

sol_pairs([A, B|T], [A-B|R]) :-
    !,
    sol_pairs([B|T], R).
sol_pairs(_, []).

%   Ray casting point-in-polygon test.
sol_in_poly(P, Poly) :-
    sol_xy(P, PX, PY),
    Poly = [F|_],
    append(Poly, [F], Closed),
    sol_pairs(Closed, Edges),
    aggregate_all(count,
                  ( member(A-B, Edges), sol_ray_cross(PX, PY, A, B) ),
                  N),
    N mod 2 =:= 1.

sol_ray_cross(PX, PY, A, B) :-
    sol_xy(A, XI, YI),
    sol_xy(B, XJ, YJ),
    ( YI > PY -> AI = 1 ; AI = 0 ),
    ( YJ > PY -> AJ = 1 ; AJ = 0 ),
    AI =\= AJ,
    XC is (XJ - XI)*(PY - YI)/(YJ - YI) + XI,
    PX < XC.

%   ---------------------------------------------------------------
%   Loading
%   ---------------------------------------------------------------

sol_ensure_loaded :-
    sol_loaded,
    !.
sol_ensure_loaded :-
    sol_reset,
    catch(sol_load_files, _, true),
    (   sol_robot(_)
    ->  true
    ;   catch(sol_load_from_user, _, true)
    ),
    forall(sol_robot(R), catch(sol_build(R), _, true)),
    assertz(sol_loaded).

sol_reset :-
    retractall(sol_robot(_)),
    retractall(sol_kb(_, _)),
    retractall(sol_pt(_, _, _)),
    retractall(sol_wall(_, _, _)),
    retractall(sol_seg(_, _, _)),
    retractall(sol_object(_, _, _)),
    retractall(sol_picked(_, _, _)),
    retractall(sol_nodeobj(_, _, _)),
    retractall(sol_obstacle(_, _)),
    retractall(sol_side(_, _, _)).

sol_load_files :-
    sol_kb_dir(Dir),
    exists_directory(Dir),
    directory_files(Dir, Files0),
    msort(Files0, Files),
    forall(( member(F, Files),
             atom(F),
             atom_concat(agent_memory, Rest, F),
             atom_concat(XA, '.txt', Rest),
             XA \== ''
           ),
           catch(sol_load_robot(Dir, XA), _, true)).

sol_load_robot(Dir, XA) :-
    format(atom(MF), '~w/agent_memory~w.txt', [Dir, XA]),
    format(atom(NF), '~w/agent_navMemory~w.txt', [Dir, XA]),
    sol_read_file(MF, MT),
    sol_read_file(NF, NT),
    (   member(agent_ID(ID0), MT), ground(ID0)
    ->  ID = ID0
    ;   catch(atom_number(XA, ID1), _, fail)
    ->  ID = ID1
    ;   ID = XA
    ),
    append(MT, NT, All),
    forall(( member(T, All), nonvar(T), T \= (:- _) ),
           assertz(sol_kb(ID, T))),
    (   sol_robot(ID)
    ->  true
    ;   assertz(sol_robot(ID))
    ).

sol_read_file(File, Terms) :-
    exists_file(File),
    !,
    setup_call_cleanup(open(File, read, S),
                       sol_read_terms(S, 0, Terms),
                       close(S)).
sol_read_file(_, []).

sol_read_terms(S, Errs, Terms) :-
    catch(read_term(S, T0, []), _, T0 = '$sol_read_error'),
    (   T0 == end_of_file
    ->  Terms = []
    ;   T0 == '$sol_read_error'
    ->  (   Errs > 10000
        ->  Terms = []
        ;   Errs1 is Errs + 1,
            sol_read_terms(S, Errs1, Terms)
        )
    ;   Terms = [T0|Rest],
        sol_read_terms(S, Errs, Rest)
    ).

%   Fallback: use the knowledge base loaded in module user.
sol_load_from_user :-
    findall(ID, catch(user:agent_ID(ID), _, fail), IDs0),
    sort(IDs0, IDs),
    IDs \== [],
    forall(member(ID, IDs),
           ( assertz(sol_robot(ID)),
             forall(( sol_kb_pred(Name/Arity),
                      functor(H, Name, Arity),
                      catch(user:H, _, fail)
                    ),
                    assertz(sol_kb(ID, H)))
           )).

sol_kb_pred(point/2).
sol_kb_pred(edge/2).
sol_kb_pred(outline/2).
sol_kb_pred(object/3).
sol_kb_pred(objectPickedUp/3).
sol_kb_pred(nodeNavStart/1).
sol_kb_pred(nodeNav/5).
sol_kb_pred(pathNav/4).
sol_kb_pred(nodeObject/3).
sol_kb_pred(agent_Pos/2).

%   ---------------------------------------------------------------
%   Derived (per robot) knowledge
%   ---------------------------------------------------------------

sol_build(R) :-
    sol_build_points(R),
    sol_build_walls(R),
    sol_build_reachable(R),
    sol_build_objects(R),
    sol_build_obstacles(R),
    sol_build_sides(R).

sol_build_points(R) :-
    forall(( sol_kb(R, point(Id, P)), sol_valid(P) ),
           assertz(sol_pt(R, Id, P))).

sol_coord(R, I, P) :-
    (   sol_valid(I)
    ->  P = I
    ;   sol_pt(R, I, P0)
    ->  P = P0
    ).

sol_coords(R, Ids, Pts) :-
    is_list(Ids),
    findall(P, ( member(I, Ids), sol_coord(R, I, P) ), Pts).

sol_build_walls(R) :-
    forall(sol_kb(R, edge(_, E)),
           (   E = [A, B], sol_coord(R, A, PA), sol_coord(R, B, PB)
           ->  assertz(sol_wall(R, PA, PB))
           ;   true
           )).

sol_add_seg(R, A, B) :-
    (   sol_valid(A), sol_valid(B)
    ->  assertz(sol_seg(R, A, B))
    ;   true
    ).

sol_add_chain(R, [A, B|T]) :-
    !,
    sol_add_seg(R, A, B),
    sol_add_chain(R, [B|T]).
sol_add_chain(R, [A]) :-
    !,
    sol_add_seg(R, A, A).
sol_add_chain(_, []).

sol_build_reachable(R) :-
    forall(sol_kb(R, nodeNav(_, P, _, Con, _)),
           ( sol_add_seg(R, P, P),
             (   is_list(Con)
             ->  forall(( member(N, Con), sol_kb(R, nodeNav(N, Q, _, _, _)) ),
                        sol_add_seg(R, P, Q))
             ;   true
             )
           )),
    forall(sol_kb(R, nodeNavStart(P)), sol_add_seg(R, P, P)),
    forall(sol_kb(R, agent_Pos(P, _)), sol_add_seg(R, P, P)),
    forall(sol_kb(R, objectPickedUp(_, _, P)), sol_add_seg(R, P, P)),
    forall(( sol_kb(R, pathNav(_, O, Path, Opt)),
             member(L, [Path, Opt]),
             sol_path_points(L, Qs),
             Qs \== []
           ),
           (   sol_valid(O)
           ->  sol_add_chain(R, [O|Qs])
           ;   sol_add_chain(R, Qs)
           )).

sol_path_points(L, Qs) :-
    is_list(L),
    findall(Q, ( member(E, L), is_list(E), E = [_, _, Q|_], sol_valid(Q) ), Qs).

sol_build_objects(R) :-
    forall(sol_kb(R, object(_, Ids, C)),
           (   sol_coords(R, Ids, Pts), Pts \== []
           ->  assertz(sol_object(R, C, Pts))
           ;   true
           )),
    forall(( sol_kb(R, objectPickedUp(_, C, P)), sol_valid(P) ),
           assertz(sol_picked(R, C, P))),
    forall(( sol_kb(R, nodeObject(_, C, P)), sol_valid(P) ),
           assertz(sol_nodeobj(R, C, P))).

%   Obstacles: closed outlines (last point next to the first one, or an
%   edge joining them) that do not describe an object.
sol_build_obstacles(R) :-
    forall(sol_kb(R, outline(_, Ids)),
           (   is_list(Ids),
               \+ sol_is_object_outline(R, Ids),
               sol_coords(R, Ids, Pts),
               length(Pts, N), N >= 3,
               sol_closed(R, Ids, Pts)
           ->  assertz(sol_obstacle(R, Pts))
           ;   true
           )).

sol_is_object_outline(R, Ids) :-
    msort(Ids, S),
    sol_kb(R, object(_, OIds, _)),
    is_list(OIds),
    msort(OIds, S),
    !.

sol_closed(_, _, Pts) :-
    Pts = [F|_],
    last(Pts, L),
    sol_dist(F, L, D),
    D =< 0.1,
    !.
sol_closed(R, Ids, _) :-
    Ids = [FI|_],
    last(Ids, LI),
    (   sol_kb(R, edge(_, [FI, LI]))
    ;   sol_kb(R, edge(_, [LI, FI]))
    ),
    !.

%   Boundary walls: a side of the map is known when a wall lies along
%   the outermost known coordinate of that side.
sol_build_sides(R) :-
    findall(X-Y, ( sol_pt(R, _, P), sol_xy(P, X, Y) ), XYs),
    XYs \== [],
    !,
    findall(X, member(X-_, XYs), Xs),
    findall(Y, member(_-Y, XYs), Ys),
    max_list(Xs, XMax),
    min_list(Xs, XMin),
    max_list(Ys, YMax),
    min_list(Ys, YMin),
    sol_side_check(R, xmax, XMax),
    sol_side_check(R, xmin, XMin),
    sol_side_check(R, ymax, YMax),
    sol_side_check(R, ymin, YMin).
sol_build_sides(_).

sol_side_check(R, Side, V) :-
    (   sol_wall(R, A, B),
        sol_side_coord(Side, A, VA),
        sol_side_coord(Side, B, VB),
        abs(VA - V) =< 0.01,
        abs(VB - V) =< 0.01,
        sol_dist(A, B, L),
        L > 0.05
    ->  assertz(sol_side(R, Side, V))
    ;   true
    ).

sol_side_coord(xmax, P, X) :- sol_xy(P, X, _).
sol_side_coord(xmin, P, X) :- sol_xy(P, X, _).
sol_side_coord(ymax, P, Y) :- sol_xy(P, _, Y).
sol_side_coord(ymin, P, Y) :- sol_xy(P, _, Y).
