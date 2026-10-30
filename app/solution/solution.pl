/*  solution.pl

    Queries over the beliefs of a team of exploring robots.

    Public predicates:
        current_can_retrieve(+R, +C, +P, ?Cond)
        new_can_retrieve(+C, +P)
        new_can_reach(+P)
        current_can_reach(+R, +P, ?Cond)

    The knowledgebase of every robot lives in /app/KB/agent_memoryX.txt and
    /app/KB/agent_navMemoryX.txt.  All robots use the same predicate names
    (point/2, edge/2, nodeNav/5, ...), so the files are read term by term and
    the facts are stored per robot in private predicates (prefix rk_).  The
    knowledgebase files themselves are never modified.  If the files cannot
    be found, the facts already loaded in the user module are used instead.

    Geometric model (2D, the Z coordinate is ignored):
      - Walls are the known edges, except the edges that belong to the
        outline of a known object.
      - A robot sees (and therefore can reach directly) every location within
        its 0.5m line of sight from a place it has been, provided no known
        wall is in between.  The places a robot has been are the nodes of
        its experience graph (nodeNav/5, nodeNavStart/1, pathNav/4) and the
        straight connections between connected nodes.
      - A location is known to be unreachable when it lies beyond a known
        boundary wall of the explored area, inside a gap narrower than the
        robot diameter (0.15m, e.g. inside a wall) or inside a closed
        obstacle outline.
      - A known object is located at the centroid of its outline points.  An
        object picked up (objectPickedUp/3) is no longer at that location.
*/

:- use_module(library(lists)).

:- dynamic rk_loaded/0.
:- dynamic rk_robot/1.          % rk_robot(R)
:- dynamic rk_pt/3.             % rk_pt(R, X, Y)            known points
:- dynamic rk_wall/5.           % rk_wall(R, X1, Y1, X2, Y2) obstacle edges
:- dynamic rk_src/3.            % rk_src(R, X, Y)           places visited
:- dynamic rk_obj/4.            % rk_obj(R, C, X, Y)        known objects
:- dynamic rk_picked/4.         % rk_picked(R, C, X, Y)     picked up objects
:- dynamic rk_poly/2.           % rk_poly(R, [X-Y,...])     closed obstacles
:- dynamic rk_bound/3.          % rk_bound(R, Side, Value)  boundary walls
:- dynamic rk_home/1.

rk_sight(0.5).                  % line of sight (m)
rk_diameter(0.15).              % robot diameter (m)
rk_obj_tol(0.1).                % max distance between P and an object (m)
rk_step(0.05).                  % sampling step along graph connections (m)

:- prolog_load_context(directory, Dir) -> assertz(rk_home(Dir)) ; true.


/*  ------------------------------------------------------------------
    Public predicates
    ------------------------------------------------------------------ */

%   current_can_retrieve(R, C, P, Cond)
current_can_retrieve(R, C, P, Cond) :-
    rk_init,
    once(rk_robot(R)),
    rk_xy(P, X, Y),
    once(rk_retrieve_cond(R, C, X, Y, Cond0)),
    Cond = Cond0.

%   current_can_reach(R, P, Cond)
current_can_reach(R, P, Cond) :-
    rk_init,
    once(rk_robot(R)),
    rk_xy(P, X, Y),
    once(rk_reach_cond(R, X, Y, Cond0)),
    Cond = Cond0.

%   new_can_retrieve(C, P): some robot knows an object of colour C at P and
%   no robot has picked it up.
new_can_retrieve(C, P) :-
    rk_init,
    rk_xy(P, X, Y),
    rk_obj(_, C, OX, OY),
    rk_near(OX, OY, X, Y),
    \+ ( rk_picked(_, C, QX, QY),
         rk_near(QX, QY, X, Y)
       ),
    !.

%   new_can_reach(P): using the knowledge of the whole team, P is known to
%   be reachable.
new_can_reach(P) :-
    rk_init,
    rk_xy(P, X, Y),
    (   rk_near_object(all, X, Y)
    ->  true
    ;   \+ rk_unreachable_any(X, Y),
        rk_in_sight(all, X, Y)
    ),
    !.


/*  ------------------------------------------------------------------
    Conditions
    ------------------------------------------------------------------ */

rk_retrieve_cond(R, C, X, Y, none) :-
    rk_believed(R, C, X, Y), !.
rk_retrieve_cond(R, _, X, Y, not_reachable) :-
    rk_unreachable(R, X, Y), !.
rk_retrieve_cond(R, _, X, Y, not_object) :-
    rk_reachable(R, X, Y), !.
rk_retrieve_cond(_, _, _, _, explore).

rk_reach_cond(R, X, Y, none) :-
    rk_near_object(R, X, Y), !.
rk_reach_cond(R, X, Y, unreachable) :-
    rk_unreachable(R, X, Y), !.
rk_reach_cond(R, X, Y, none) :-
    rk_in_sight(R, X, Y), !.
rk_reach_cond(_, _, _, explore).

%   The robot believes there is an object of colour C at (X,Y): it knows the
%   object and it did not pick it up itself.
rk_believed(R, C, X, Y) :-
    rk_obj(R, C, OX, OY),
    rk_near(OX, OY, X, Y),
    \+ ( rk_picked(R, C, QX, QY),
         rk_near(QX, QY, X, Y)
       ),
    !.

rk_reachable(R, X, Y) :-
    rk_near_object(R, X, Y), !.
rk_reachable(R, X, Y) :-
    \+ rk_unreachable(R, X, Y),
    rk_in_sight(R, X, Y), !.

%   (X,Y) is the location of an object known or picked up by the robot.
rk_near_object(Scope, X, Y) :-
    rk_scope_obj(Scope, OX, OY),
    rk_near(OX, OY, X, Y), !.

rk_scope_obj(Scope, X, Y) :- rk_in_scope(Scope, R), rk_obj(R, _, X, Y).
rk_scope_obj(Scope, X, Y) :- rk_in_scope(Scope, R), rk_picked(R, _, X, Y).

rk_in_scope(all, R) :- !, rk_robot(R).
rk_in_scope(R, R).

rk_near(X1, Y1, X2, Y2) :-
    rk_obj_tol(T),
    DX is X1 - X2, DY is Y1 - Y2,
    DX*DX + DY*DY =< T*T.


/*  ------------------------------------------------------------------
    Line of sight
    ------------------------------------------------------------------ */

rk_in_sight(Scope, X, Y) :-
    rk_sight(S),
    S2 is S*S,
    rk_in_scope(Scope, R),
    rk_src(R, SX, SY),
    DX is SX - X, DY is SY - Y,
    DX*DX + DY*DY =< S2,
    \+ rk_blocked(Scope, SX, SY, X, Y),
    !.

%   The segment A-P crosses a known wall before reaching P.
rk_blocked(Scope, AX, AY, PX, PY) :-
    DX is PX - AX, DY is PY - AY,
    rk_in_scope(Scope, R),
    rk_wall(R, W1X, W1Y, W2X, W2Y),
    EX is W2X - W1X, EY is W2Y - W1Y,
    Den is DX*EY - DY*EX,
    abs(Den) > 1.0e-12,
    T is ((W1X-AX)*EY - (W1Y-AY)*EX) / Den,
    T >= 0.0, T =< 1.0,
    U is ((W1X-AX)*DY - (W1Y-AY)*DX) / Den,
    U >= 0.0, U =< 1.0,
    IX is AX + T*DX - PX,
    IY is AY + T*DY - PY,
    IX*IX + IY*IY > 0.0001,
    !.


/*  ------------------------------------------------------------------
    Known unreachable locations
    ------------------------------------------------------------------ */

rk_unreachable_any(X, Y) :-
    rk_robot(R),
    rk_unreachable(R, X, Y), !.

rk_unreachable(R, X, Y) :-
    rk_beyond_bounds(R, X, Y), !.
rk_unreachable(R, X, Y) :-
    rk_in_gap(R, X, Y), !.
rk_unreachable(R, X, Y) :-
    rk_poly(R, Poly),
    rk_in_polygon(Poly, X, Y), !.

%   Beyond a boundary wall: the outermost known points in one direction
%   form a wall and the location lies further out.
rk_beyond_bounds(R, X, _) :- rk_bound(R, max_x, V), X > V + 0.001, !.
rk_beyond_bounds(R, X, _) :- rk_bound(R, min_x, V), X < V - 0.001, !.
rk_beyond_bounds(R, _, Y) :- rk_bound(R, max_y, V), Y > V + 0.001, !.
rk_beyond_bounds(R, _, Y) :- rk_bound(R, min_y, V), Y < V - 0.001, !.

%   The location lies between walls on opposite sides closer than the robot
%   diameter (e.g. inside a wall).
rk_in_gap(R, X, Y) :-
    rk_diameter(D),
    findall(v(VX, VY, L),
            ( rk_wall(R, X1, Y1, X2, Y2),
              rk_closest(X, Y, X1, Y1, X2, Y2, CX, CY),
              VX is CX - X, VY is CY - Y,
              L is sqrt(VX*VX + VY*VY),
              L > 0.001, L < D
            ),
            Vs),
    append(_, [v(AX, AY, LA)|Rest], Vs),
    member(v(BX, BY, LB), Rest),
    LA + LB < D,
    (AX*BX + AY*BY) / (LA*LB) < -0.7,
    !.

rk_closest(PX, PY, AX, AY, BX, BY, CX, CY) :-
    EX is BX - AX, EY is BY - AY,
    L2 is EX*EX + EY*EY,
    (   L2 =:= 0
    ->  T = 0.0
    ;   T0 is ((PX-AX)*EX + (PY-AY)*EY) / L2,
        T is max(0.0, min(1.0, T0))
    ),
    CX is AX + T*EX,
    CY is AY + T*EY.

rk_in_polygon(Poly, X, Y) :-
    last(Poly, Last),
    rk_crossings(Poly, Last, X, Y, 0, N),
    N mod 2 =:= 1.

rk_crossings([], _, _, _, N, N).
rk_crossings([AX-AY|T], BX-BY, X, Y, N0, N) :-
    (   ( AY > Y, BY =< Y ; AY =< Y, BY > Y ),
        X < (BX-AX)*(Y-AY)/(BY-AY) + AX
    ->  N1 is N0 + 1
    ;   N1 = N0
    ),
    rk_crossings(T, AX-AY, X, Y, N1, N).


/*  ------------------------------------------------------------------
    Loading the knowledgebases
    ------------------------------------------------------------------ */

rk_init :-
    rk_loaded, !.
rk_init :-
    retractall(rk_robot(_)),
    retractall(rk_pt(_, _, _)),
    retractall(rk_wall(_, _, _, _, _)),
    retractall(rk_src(_, _, _)),
    retractall(rk_obj(_, _, _, _)),
    retractall(rk_picked(_, _, _, _)),
    retractall(rk_poly(_, _)),
    retractall(rk_bound(_, _, _)),
    (   rk_kb_dir(Dir)
    ->  catch(rk_load_dir(Dir), _, true)
    ;   true
    ),
    (   rk_robot(_)
    ->  true
    ;   catch(rk_load_user, _, true)
    ),
    assertz(rk_loaded).

rk_kb_dir(Dir) :-
    rk_candidate_dir(Dir),
    exists_directory(Dir),
    rk_memory_file(Dir, _),
    !.

rk_candidate_dir('/app/KB').
rk_candidate_dir(Dir) :-
    rk_home(Home),
    member(Rel, ['../KB', 'KB', '.']),
    atomic_list_concat([Home, '/', Rel], Dir).
rk_candidate_dir('KB').
rk_candidate_dir('../KB').
rk_candidate_dir('.').

%   rk_memory_file(+Dir, -X): Dir contains agent_memoryX.txt
rk_memory_file(Dir, X) :-
    directory_files(Dir, Files),
    member(F, Files),
    atom_concat(agent_memory, Rest, F),
    atom_concat(X, '.txt', Rest),
    X \== ''.

rk_load_dir(Dir) :-
    findall(X, rk_memory_file(Dir, X), Xs0),
    sort(Xs0, Xs),
    forall(member(X, Xs), catch(rk_load_agent(Dir, X), _, true)).

rk_load_agent(Dir, X) :-
    atomic_list_concat([Dir, '/agent_memory', X, '.txt'], MemFile),
    atomic_list_concat([Dir, '/agent_navMemory', X, '.txt'], NavFile),
    rk_read_terms(MemFile, MemTerms),
    (   exists_file(NavFile)
    ->  rk_read_terms(NavFile, NavTerms)
    ;   NavTerms = []
    ),
    (   memberchk(agent_ID(Id), MemTerms), nonvar(Id)
    ->  true
    ;   atom_number(X, Id)
    ->  true
    ;   Id = X
    ),
    append(MemTerms, NavTerms, Terms),
    rk_build_robot(Id, Terms).

rk_read_terms(File, Terms) :-
    setup_call_cleanup(open(File, read, S),
                       rk_read_stream(S, Terms),
                       close(S)).

rk_read_stream(S, Terms) :-
    character_count(S, C0),
    catch(read_term(S, T, []), _, T = '$rk_error'),
    (   T == end_of_file
    ->  Terms = []
    ;   T == '$rk_error'
    ->  character_count(S, C1),
        (   C1 > C0
        ->  rk_read_stream(S, Terms)
        ;   Terms = []
        )
    ;   Terms = [T|More],
        rk_read_stream(S, More)
    ).

%   Fallback: knowledgebase of a single robot loaded in the user module.
rk_load_user :-
    findall(T,
            ( member(T, [ agent_ID(_), point(_, _), edge(_, _), outline(_, _),
                          object(_, _, _), objectPickedUp(_, _, _),
                          nodeNavStart(_), nodeNav(_, _, _, _, _),
                          pathNav(_, _, _, _), nodeObject(_, _, _)
                        ]),
              functor(T, Name, Arity),
              current_predicate(user:Name/Arity),
              call(user:T)
            ),
            Terms),
    findall(Id, member(agent_ID(Id), Terms), Ids0),
    sort(Ids0, Ids),
    (   Ids = [Id]
    ->  rk_build_robot(Id, Terms)
    ;   true
    ).

%   rk_build_robot(+R, +Terms)
rk_build_robot(R, Terms) :-
    assertz(rk_robot(R)),
    findall(Id-(X-Y),
            ( member(point(Id, P), Terms), rk_xy(P, X, Y) ),
            Points),
    forall(member(_-(X-Y), Points), assertz(rk_pt(R, X, Y))),
    findall(Ps-C,
            ( member(object(_, Ps, C), Terms), is_list(Ps) ),
            Objects),
    % objects
    forall(member(Ps-C, Objects),
           (   rk_centroid(Ps, Points, X, Y)
           ->  assertz(rk_obj(R, C, X, Y))
           ;   true
           )),
    forall(( member(objectPickedUp(_, C, P), Terms), rk_xy(P, X, Y) ),
           assertz(rk_picked(R, C, X, Y))),
    % walls
    forall(( member(edge(_, [A, B|_]), Terms),
             memberchk(A-(X1-Y1), Points),
             memberchk(B-(X2-Y2), Points),
             \+ ( member(Ps-_, Objects),
                  memberchk(A, Ps),
                  memberchk(B, Ps)
                )
           ),
           assertz(rk_wall(R, X1, Y1, X2, Y2))),
    % closed obstacle outlines
    forall(( member(outline(_, Ps), Terms),
             rk_closed_outline(Ps, Terms, Objects),
             rk_coords(Ps, Points, Poly)
           ),
           assertz(rk_poly(R, Poly))),
    % visited places
    forall(( member(nodeNav(_, P, _, _, _), Terms), rk_xy(P, X, Y) ),
           assertz(rk_src(R, X, Y))),
    forall(( member(nodeNavStart(P), Terms), rk_xy(P, X, Y) ),
           assertz(rk_src(R, X, Y))),
    forall(( member(pathNav(_, O, Path, Opt), Terms),
             rk_path_location(O, Path, Opt, P),
             rk_xy(P, X, Y)
           ),
           assertz(rk_src(R, X, Y))),
    forall(( member(nodeNav(I, P, _, Con, _), Terms),
             is_list(Con),
             member(J, Con),
             I @< J,
             memberchk(nodeNav(J, Q, _, _, _), Terms),
             rk_xy(P, AX, AY),
             rk_xy(Q, BX, BY)
           ),
           rk_add_connection(R, AX, AY, BX, BY)),
    rk_compute_bounds(R).

rk_path_location(O, _, _, O).
rk_path_location(_, Path, _, P) :-
    is_list(Path), member([_, _, P|_], Path).
rk_path_location(_, _, Opt, P) :-
    is_list(Opt), member([_, _, P|_], Opt).

rk_add_connection(R, AX, AY, BX, BY) :-
    rk_step(Step),
    L is sqrt((BX-AX)*(BX-AX) + (BY-AY)*(BY-AY)),
    N is max(1, ceiling(L / Step)),
    forall(between(1, N, I),
           ( SX is AX + (BX-AX)*I/N,
             SY is AY + (BY-AY)*I/N,
             assertz(rk_src(R, SX, SY))
           )).

rk_centroid(Ps, Points, X, Y) :-
    rk_coords(Ps, Points, Coords),
    Coords \== [],
    rk_sum(Coords, 0.0, 0.0, SX, SY),
    length(Coords, N),
    X is SX / N,
    Y is SY / N.

%   rk_coords(+PointsOrIds, +Points, -Coords): coordinates X-Y of a list
%   of point ids (or of explicit positions); unknown ids are skipped.
rk_coords([], _, []).
rk_coords([P|Ps], Points, Coords) :-
    (   rk_xy(P, X, Y)
    ->  Coords = [X-Y|More]
    ;   memberchk(P-(X-Y), Points)
    ->  Coords = [X-Y|More]
    ;   Coords = More
    ),
    rk_coords(Ps, Points, More).

rk_sum([], SX, SY, SX, SY).
rk_sum([X-Y|T], SX0, SY0, SX, SY) :-
    SX1 is SX0 + X,
    SY1 is SY0 + Y,
    rk_sum(T, SX1, SY1, SX, SY).

rk_closed_outline(Ps, Terms, Objects) :-
    is_list(Ps),
    length(Ps, Len), Len > 2,
    Ps = [First|_],
    last(Ps, Last),
    (   memberchk(edge(_, [First, Last|_]), Terms)
    ->  true
    ;   memberchk(edge(_, [Last, First|_]), Terms)
    ),
    msort(Ps, Sorted),
    \+ ( member(OPs-_, Objects), msort(OPs, Sorted) ).

rk_compute_bounds(R) :-
    findall(X, rk_pt(R, X, _), Xs),
    findall(Y, rk_pt(R, _, Y), Ys),
    (   Xs == []
    ->  true
    ;   max_list(Xs, MaxX), min_list(Xs, MinX),
        max_list(Ys, MaxY), min_list(Ys, MinY),
        rk_add_bound(R, max_x, x, MaxX),
        rk_add_bound(R, min_x, x, MinX),
        rk_add_bound(R, max_y, y, MaxY),
        rk_add_bound(R, min_y, y, MinY)
    ).

rk_add_bound(R, Side, Axis, V) :-
    (   rk_wall_on(R, Axis, V)
    ->  assertz(rk_bound(R, Side, V))
    ;   true
    ).

rk_wall_on(R, x, V) :-
    rk_wall(R, X1, _, X2, _),
    abs(X1 - V) =< 0.01, abs(X2 - V) =< 0.01, !.
rk_wall_on(R, y, V) :-
    rk_wall(R, _, Y1, _, Y2),
    abs(Y1 - V) =< 0.01, abs(Y2 - V) =< 0.01, !.

%   rk_xy(+P, -X, -Y): planar coordinates of a position [X,Y,Z]
rk_xy(P, X, Y) :-
    nonvar(P),
    P = [X0, Y0|_],
    number(X0), number(Y0),
    X is float(X0),
    Y is float(Y0).
