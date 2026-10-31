/*  solution.pl

    Queries over the knowledge bases built by a team of exploring robots.

    The knowledge bases are read (never modified) from /app/KB/:
        agent_memoryX.txt     map knowledge (points, edges, outlines, objects)
        agent_navMemoryX.txt  navigation knowledge (experience graph, paths)

    Public predicates:
        current_can_retrieve(+R, +C, +P, ?Cond)
        current_can_reach(+R, +P, ?Cond)
        new_can_retrieve(+C, +P)
        new_can_reach(+Start, +P)

    Geometry is evaluated on the XY plane (all robots share the same frame).
    Robots have a diameter of 0.15 m and a 0.5 m line of sight, so they can
    move directly to any point (or object) they can see.
*/

:- use_module(library(lists)).
:- use_module(library(apply)).

:- dynamic
    kb_loaded/0,
    r_robot/1,          % r_robot(R)              robot R exists
    r_pos/2,            % r_pos(R, [X,Y])         position robot R has been at
    r_wall/4,           % r_wall(R, A, B, Kind)   sensed segment, Kind = env | obj
    r_obj/3,            % r_obj(R, C, Centroid)   object of colour C sensed by R
    r_pick/3,           % r_pick(R, C, P)         object of colour C picked up at P
    r_nodeobj/3,        % r_nodeobj(R, C, P)      node closest to a sensed object
    r_bound/2,          % r_bound(R, Bounds)      known arena boundary of R
    t_bound/1,          % t_bound(Bounds)         known arena boundary of the team
    r_link/2.           % r_link(R1, R2)          experience graphs of R1, R2 touch

kb_dir('/app/KB').

% Tolerances (metres)
sight(0.5).             % line of sight
radius(0.075).          % robot radius
same_place(0.1).        % two locations denote the same object / place
visited(0.05).          % a location the robot has actually been at
object_size(0.15).      % object outlines closer than this to a target are ignored

		 /*******************************
		 *            LOADING           *
		 *******************************/

ensure_kb :-
    kb_loaded,
    !.
ensure_kb :-
    load_kbs.

load_kbs :-
    retractall(kb_loaded),
    retractall(r_robot(_)),
    retractall(r_pos(_, _)),
    retractall(r_wall(_, _, _, _)),
    retractall(r_obj(_, _, _)),
    retractall(r_pick(_, _, _)),
    retractall(r_nodeobj(_, _, _)),
    retractall(r_bound(_, _)),
    retractall(t_bound(_)),
    retractall(r_link(_, _)),
    kb_dir(Dir),
    (   exists_directory(Dir)
    ->  directory_files(Dir, Files0)
    ;   Files0 = []
    ),
    msort(Files0, Files),
    forall(( member(F, Files),
             atom(F),
             atom_concat(agent_memory, Rest, F),
             atom_concat(X, '.txt', Rest),
             X \== ''
           ),
           (   catch(load_robot(Dir, X), _, fail)
           ->  true
           ;   true
           )),
    compute_team,
    assertz(kb_loaded).

load_robot(Dir, X) :-
    atomic_list_concat([Dir, '/agent_memory', X, '.txt'], MemFile),
    atomic_list_concat([Dir, '/agent_navMemory', X, '.txt'], NavFile),
    read_terms(MemFile, Mem),
    (   exists_file(NavFile)
    ->  read_terms(NavFile, Nav)
    ;   Nav = []
    ),
    robot_id(Mem, X, R),
    \+ r_robot(R),
    append(Mem, Nav, Terms),
    assertz(r_robot(R)),
    store_robot(R, Terms).

robot_id(Terms, _, R) :-
    memberchk(agent_ID(R), Terms),
    nonvar(R),
    !.
robot_id(_, X, R) :-
    catch(atom_number(X, R), _, fail),
    !.
robot_id(_, X, X).

%   read_terms(+File, -Terms): all clauses of File, directives skipped.

read_terms(File, Terms) :-
    catch(setup_call_cleanup(open(File, read, S),
                             read_stream_terms(S, Terms),
                             close(S)),
          _,
          Terms = []).

read_stream_terms(S, Terms) :-
    catch(read_term(S, T, []), _, T = '$skip'),
    (   T == end_of_file
    ->  Terms = []
    ;   ( var(T) ; T == '$skip' ; T = (:- _) )
    ->  read_stream_terms(S, Terms)
    ;   Terms = [T|Rest],
        read_stream_terms(S, Rest)
    ).

store_robot(R, T) :-
    findall(Id-P2, ( member(point(Id, P), T), xy(P, P2) ), Pts),
    % walls: sensed edges; those that belong to an object outline are marked
    forall(( member(edge(_, [I, J]), T),
             memberchk(I-A, Pts),
             memberchk(J-B, Pts)
           ),
           (   (   member(object(_, Ids, _), T),
                   is_list(Ids),
                   memberchk(I, Ids),
                   memberchk(J, Ids)
               )
           ->  assertz(r_wall(R, A, B, obj))
           ;   assertz(r_wall(R, A, B, env))
           )),
    % objects directly sensed (located by the centroid of their outline)
    forall(( member(object(_, Ids, C), T),
             is_list(Ids),
             findall(Q, ( member(Id, Ids), memberchk(Id-Q, Pts) ), Qs),
             Qs \== []
           ),
           (   centroid(Qs, Cen),
               assertz(r_obj(R, C, Cen))
           )),
    forall(( member(objectPickedUp(_, C, P), T), xy(P, P2) ),
           assertz(r_pick(R, C, P2))),
    forall(( member(nodeObject(_, C, P), T), xy(P, P2) ),
           assertz(r_nodeobj(R, C, P2))),
    % places the robot has been at
    findall(P2, ( (   member(nodeNav(_, P, _, _, _), T)
                  ;   member(nodeNavStart(P), T)
                  ;   member(agent_Pos(P, _), T)
                  ;   member(agent_checkpointExplored(L), T),
                      is_list(L),
                      member(P, L)
                  ),
                  xy(P, P2)
                ),
            Pos0),
    sort(Pos0, Pos),
    forall(member(P, Pos), assertz(r_pos(R, P))),
    % arena boundary as known by the robot
    findall(A-B, r_wall(R, A, B, env), EnvW),
    findall(Q, ( member(A-B, EnvW), ( Q = A ; Q = B ) ), WPts),
    append(WPts, Pos, Known),
    bounds(EnvW, Known, Bs),
    assertz(r_bound(R, Bs)).

compute_team :-
    forall(( r_robot(I),
             r_robot(J),
             I @< J,
             once(( r_pos(I, P), r_pos(J, Q), los(team, P, Q) ))
           ),
           (   assertz(r_link(I, J)),
               assertz(r_link(J, I))
           )),
    findall(A-B, r_wall(_, A, B, env), EnvW),
    findall(Q, ( member(A-B, EnvW), ( Q = A ; Q = B ) ), WPts),
    findall(Q, r_pos(_, Q), Pos),
    append(WPts, Pos, Known),
    bounds(EnvW, Known, Bs),
    assertz(t_bound(Bs)).

		 /*******************************
		 *           GEOMETRY           *
		 *******************************/

xy(P, [X, Y]) :-
    is_list(P),
    P = [X0, Y0|_],
    number(X0),
    number(Y0),
    X is float(X0),
    Y is float(Y0).

dist([X1, Y1], [X2, Y2], D) :-
    D is sqrt((X1-X2)*(X1-X2) + (Y1-Y2)*(Y1-Y2)).

centroid(Ps, [X, Y]) :-
    length(Ps, N),
    foldl(add_xy, Ps, 0.0-0.0, SX-SY),
    X is SX/N,
    Y is SY/N.

add_xy([X, Y], SX0-SY0, SX-SY) :-
    SX is SX0+X,
    SY is SY0+Y.

%   seg_dist(+P, +A, +B, -D): distance from point P to segment AB.

seg_dist([PX, PY], [AX, AY], [BX, BY], D) :-
    DX is BX-AX,
    DY is BY-AY,
    L is DX*DX + DY*DY,
    (   L =:= 0
    ->  T = 0.0
    ;   T0 is ((PX-AX)*DX + (PY-AY)*DY)/L,
        T is max(0.0, min(1.0, T0))
    ),
    CX is AX + T*DX,
    CY is AY + T*DY,
    D is sqrt((PX-CX)*(PX-CX) + (PY-CY)*(PY-CY)).

orient([OX, OY], [AX, AY], [BX, BY], V) :-
    V is (AX-OX)*(BY-OY) - (AY-OY)*(BX-OX).

%   segs_cross(+P1, +P2, +Q1, +Q2): segments P1P2 and Q1Q2 properly cross.

segs_cross(P1, P2, Q1, Q2) :-
    orient(Q1, Q2, P1, D1),
    orient(Q1, Q2, P2, D2),
    D1*D2 < 0,
    orient(P1, P2, Q1, D3),
    orient(P1, P2, Q2, D4),
    D3*D4 < 0.

%   bounds(+Walls, +KnownPoints, -Bounds)
%   A side of the arena is bounded by the outermost long axis-aligned wall
%   when nothing known lies beyond it.

bounds(Walls, Pts, Bounds) :-
    findall(Side-V, side_bound(Side, Walls, Pts, V), Bounds).

side_bound(xmax, W, Pts, V) :-
    long_walls(v, W, Vs), Vs \== [], max_list(Vs, V),
    \+ ( member([X, _], Pts), X > V + 0.01 ).
side_bound(xmin, W, Pts, V) :-
    long_walls(v, W, Vs), Vs \== [], min_list(Vs, V),
    \+ ( member([X, _], Pts), X < V - 0.01 ).
side_bound(ymax, W, Pts, V) :-
    long_walls(h, W, Vs), Vs \== [], max_list(Vs, V),
    \+ ( member([_, Y], Pts), Y > V + 0.01 ).
side_bound(ymin, W, Pts, V) :-
    long_walls(h, W, Vs), Vs \== [], min_list(Vs, V),
    \+ ( member([_, Y], Pts), Y < V - 0.01 ).

long_walls(v, W, Vs) :-
    findall(V, ( member([AX, AY]-[BX, BY], W),
                 abs(AX-BX) < 0.02,
                 abs(AY-BY) >= 0.2,
                 V is (AX+BX)/2
               ),
            Vs).
long_walls(h, W, Vs) :-
    findall(V, ( member([AX, AY]-[BX, BY], W),
                 abs(AY-BY) < 0.02,
                 abs(AX-BX) >= 0.2,
                 V is (AY+BY)/2
               ),
            Vs).

outside_bounds(Bs, [X, Y]) :-
    (   memberchk(xmax-V1, Bs), X > V1
    ;   memberchk(xmin-V2, Bs), X < V2
    ;   memberchk(ymax-V3, Bs), Y > V3
    ;   memberchk(ymin-V4, Bs), Y < V4
    ),
    !.

%   Walls as known by one robot (robot(R)) or by the whole team (team).

wall_in(robot(R), A, B, K) :-
    r_wall(R, A, B, K).
wall_in(team, A, B, K) :-
    r_wall(_, A, B, K).

%   los(+Scope, +From, +To): To is in sight from From. The outline of an
%   object being approached does not block the view of that object.

los(Scope, From, To) :-
    dist(From, To, D),
    sight(S),
    D =< S,
    \+ blocked(Scope, From, To).

blocked(Scope, From, To) :-
    wall_in(Scope, A, B, K),
    \+ ignorable(K, From, To, A, B),
    segs_cross(From, To, A, B),
    !.

ignorable(obj, From, To, A, B) :-
    object_size(L),
    (   seg_dist(To, A, B, D)
    ;   seg_dist(From, A, B, D)
    ),
    D < L,
    !.

%   in_obstacle(+Scope, +P): a robot cannot stand at P (inside or touching a
%   known wall) or P lies outside the known arena boundary.

in_obstacle(Scope, P) :-
    radius(Rad),
    wall_in(Scope, A, B, env),
    seg_dist(P, A, B, D),
    D < Rad,
    !.
in_obstacle(robot(R), P) :-
    r_bound(R, Bs),
    outside_bounds(Bs, P),
    !.
in_obstacle(team, P) :-
    t_bound(Bs),
    outside_bounds(Bs, P),
    !.

		 /*******************************
		 *         REACHABILITY         *
		 *******************************/

%   current_can_reach(+R, +P, ?Cond)

current_can_reach(R, P, Cond) :-
    ensure_kb,
    r_robot(R),
    !,
    xy(P, P2),
    reach_cond(R, P2, Cond0),
    !,
    Cond = Cond0.

reach_cond(R, P, none) :-
    been_at(R, P),
    !.
reach_cond(R, P, unreachable) :-
    in_obstacle(robot(R), P),
    !.
reach_cond(R, P, none) :-
    r_pos(R, Q),
    los(robot(R), Q, P),
    !.
reach_cond(_, _, explore).

been_at(R, P) :-
    visited(V),
    r_pos(R, Q),
    dist(P, Q, D),
    D =< V,
    !.

%   new_can_reach(+Start, +P): a new robot deployed at Start can reach P
%   following the experience graphs of the deployed robots.

new_can_reach(Start, P) :-
    ensure_kb,
    xy(Start, S),
    xy(P, T),
    \+ in_obstacle(team, S),
    \+ in_obstacle(team, T),
    (   los(team, S, T)
    ->  true
    ;   findall(R, ( r_robot(R), touches(R, S) ), Rs),
        findall(R, ( r_robot(R), touches(R, T) ), Rt),
        member(A, Rs),
        member(B, Rt),
        robots_connected(A, B)
    ->  true
    ),
    !.

%   touches(+R, +P): P is a place where robot R has been, or is in sight of
%   one of them.

touches(R, P) :-
    been_at(R, P),
    !.
touches(R, P) :-
    r_pos(R, Q),
    los(team, Q, P),
    !.

robots_connected(A, A) :-
    !.
robots_connected(A, B) :-
    connected_set([A], [A], Set),
    memberchk(B, Set).

connected_set([], Vis, Vis).
connected_set([X|Xs], Vis, Set) :-
    findall(Y, ( r_link(X, Y), \+ memberchk(Y, Vis) ), Ys0),
    sort(Ys0, Ys),
    append(Vis, Ys, Vis1),
    append(Xs, Ys, Queue),
    connected_set(Queue, Vis1, Set).

		 /*******************************
		 *           RETRIEVAL          *
		 *******************************/

%   current_can_retrieve(+R, +C, +P, ?Cond)

current_can_retrieve(R, C, P, Cond) :-
    ensure_kb,
    r_robot(R),
    !,
    xy(P, P2),
    retrieve_cond(R, C, P2, Cond0),
    !,
    Cond = Cond0.

retrieve_cond(R, C, P, Cond) :-
    reach_cond(R, P, RC),
    (   RC == unreachable
    ->  Cond = not_reachable
    ;   knows_object(R, C, P)
    ->  (   ( RC == none ; near_nodeobj(R, C, P) )
        ->  Cond = none
        ;   Cond = explore
        )
    ;   knows_no_object(R, C, P)
    ->  Cond = not_object
    ;   Cond = explore
    ).

knows_object(R, C, P) :-
    same_place(Tol),
    r_obj(R, C, Q),
    dist(P, Q, D),
    D =< Tol,
    !.

%   The robot knows there is no object of colour C at P when it picked up
%   the object that was there, or it knows the object there has another
%   colour.

knows_no_object(R, _, P) :-
    same_place(Tol),
    r_pick(R, _, Q),
    dist(P, Q, D),
    D =< Tol,
    !.
knows_no_object(R, C, P) :-
    same_place(Tol),
    r_obj(R, C2, Q),
    C2 \== C,
    dist(P, Q, D),
    D =< Tol,
    !.

near_nodeobj(R, C, P) :-
    sight(S),
    r_nodeobj(R, C, Q),
    dist(P, Q, D),
    D =< S,
    !.

%   new_can_retrieve(+C, +P): some deployed robot knows an object of colour C
%   at P that it can retrieve, and no robot of the team has picked it up.

new_can_retrieve(C, P) :-
    ensure_kb,
    xy(P, P2),
    \+ team_picked(P2),
    r_robot(R),
    retrieve_cond(R, C, P2, none),
    !.

team_picked(P) :-
    same_place(Tol),
    r_pick(_, _, Q),
    dist(P, Q, D),
    D =< Tol,
    !.

:- initialization(( catch(ensure_kb, _, fail) -> true ; true )).
