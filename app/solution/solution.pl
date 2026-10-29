% solution.pl -- query the belief knowledge base of an exploring robot.
%
% The knowledge base (bfact/4, rule/3, world/3) is loaded alongside this file.
%
%   holds_at(+A, +T)                 A is valid or derivable at time T.
%   supports(+A, +T, -S)             S: all minimal supports (sets of fact ids) of A at T.
%   stale_basis(+A, +T, -Ids)        Ids: facts of some minimal support of A at T whose
%                                    atom is false in the world at T.
%   would_hold_without(+A, +T, +Id)  A holds at T if fact Id did not exist.
%   min_repair(+A, +T, -Repairs)     Repairs: all smallest sets of facts valid at T whose
%                                    removal makes A not derivable ([[]] if A does not hold).
%
% Semantics: the facts valid at T plus the rules form a stratified program (negation is
% never circular, positive cycles are allowed). Its model is computed with the alternating
% fixpoint, which for stratified programs yields the perfect model. For supports, negated
% literals are judged against the model built from all facts valid at T.

:- use_module(library(lists)).
:- use_module(library(apply)).
:- use_module(library(assoc)).
:- use_module(library(ordsets)).

% ---------------------------------------------------------------------------
% Public predicates
% ---------------------------------------------------------------------------

holds_at(A, T) :-
    once(kbq_holds(A, T, [])).

would_hold_without(A, T, Id) :-
    once(kbq_holds(A, T, [Id])).

supports(A, T, S) :-
    once(kbq_supports(A, T, S0, _)),
    S = S0.

stale_basis(A, T, Ids) :-
    once(kbq_supports(A, T, Sups, Facts)),
    ord_union(Sups, Used),
    include(kbq_stale_fact(T, Facts), Used, Ids0),
    sort(Ids0, Ids1),
    Ids = Ids1.

min_repair(A, T, Repairs) :-
    once(kbq_repairs(A, T, R0)),
    Repairs = R0.

% ---------------------------------------------------------------------------
% Knowledge base access (guarded, so a missing predicate raises no error)
% ---------------------------------------------------------------------------

kbq_bfact(Id, A, F, To) :- current_predicate(bfact/4), bfact(Id, A, F, To).
kbq_rule(H, Body)       :- current_predicate(rule/3), rule(_, H, Body).
kbq_world(A, F, To)     :- current_predicate(world/3), world(A, F, To).

% F =< T < To, with To possibly inf.
kbq_valid_at(T, F, To) :-
    catch(kbq_valid_at_(T, F, To), _, fail).

kbq_valid_at_(T, F, To) :-
    (   F == -inf -> true
    ;   F \== inf, F =< T
    ),
    (   To == inf -> true
    ;   To \== -inf, T < To
    ).

kbq_world_true(A, T) :-
    kbq_world(A, F, To),
    kbq_valid_at(T, F, To),
    !.

kbq_stale_fact(T, Facts, Id) :-
    memberchk(Id-A, Facts),
    \+ kbq_world_true(A, T).

kbq_lit_atom(not(X), X) :- !.
kbq_lit_atom(\+(X), X) :- !.
kbq_lit_atom(X, X).

kbq_split_body([], [], []).
kbq_split_body([L|Ls], Pos, Neg) :-
    (   L = not(X) -> Pos = Pos1, Neg = [X|Neg1]
    ;   L = \+(X)  -> Pos = Pos1, Neg = [X|Neg1]
    ;   Pos = [L|Pos1], Neg = Neg1
    ),
    kbq_split_body(Ls, Pos1, Neg1).

% ---------------------------------------------------------------------------
% Context of a query: the rules and the facts valid at T that A depends on.
%   Rules : list of r(Head, SortedPos, SortedNeg)
%   Facts : sorted list of Id-Atom
% ---------------------------------------------------------------------------

kbq_context(A, T, Rules, Facts) :-
    kbq_relevant([A], [A], Atoms),
    findall(r(H, Pos, Neg),
            ( member(H, Atoms),
              kbq_rule(H, Body),
              kbq_split_body(Body, Pos0, Neg0),
              sort(Pos0, Pos),
              sort(Neg0, Neg) ),
            Rules0),
    sort(Rules0, Rules),
    findall(Id-X,
            ( member(X, Atoms),
              kbq_bfact(Id, X, F, To),
              kbq_valid_at(T, F, To) ),
            Facts0),
    sort(Facts0, Facts).

% Atoms reachable backwards through rule bodies (positive and negated literals).
kbq_relevant([], S, S).
kbq_relevant([X|Xs], S0, S) :-
    findall(B, ( kbq_rule(X, Body), member(L, Body), kbq_lit_atom(L, B) ), Bs0),
    sort(Bs0, Bs),
    ord_subtract(Bs, S0, New),
    ord_union(S0, New, S1),
    append(Xs, New, Queue),
    kbq_relevant(Queue, S1, S).

kbq_fact_atoms(Facts, Atoms) :-
    findall(X, member(_-X, Facts), Xs),
    sort(Xs, Atoms).

kbq_without(Ids, Facts, Rest) :-
    exclude(kbq_fact_in(Ids), Facts, Rest).

kbq_fact_in(Ids, Id-_) :-
    memberchk(Id, Ids).

% ---------------------------------------------------------------------------
% Model: alternating fixpoint. Gamma(J) is the least model of the program in
% which not(X) is true iff X is not in J.
% ---------------------------------------------------------------------------

kbq_model(Rules, Base, Model) :-
    kbq_gamma(Rules, Base, [], Over),
    kbq_alternate(Rules, Base, Over, Model).

kbq_alternate(Rules, Base, Over, Model) :-
    kbq_gamma(Rules, Base, Over, Under),
    kbq_gamma(Rules, Base, Under, Over1),
    (   Over1 == Over
    ->  Model = Under
    ;   kbq_alternate(Rules, Base, Over1, Model)
    ).

kbq_gamma(Rules, Base, J, M) :-
    include(kbq_neg_ok(J), Rules, Rs),
    kbq_horn(Rs, Base, M).

kbq_neg_ok(J, r(_, _, Neg)) :-
    \+ ( member(X, Neg), ord_memberchk(X, J) ).

kbq_horn(Rules, M0, M) :-
    findall(H,
            ( member(r(H, Pos, _), Rules),
              \+ ord_memberchk(H, M0),
              ord_subset(Pos, M0) ),
            Hs0),
    (   Hs0 == []
    ->  M = M0
    ;   sort(Hs0, Hs),
        ord_union(M0, Hs, M1),
        kbq_horn(Rules, M1, M)
    ).

kbq_holds_in(Rules, Facts, A) :-
    kbq_fact_atoms(Facts, Base),
    kbq_model(Rules, Base, Model),
    ord_memberchk(A, Model).

kbq_holds(A, T, Excl) :-
    kbq_context(A, T, Rules, Facts0),
    kbq_without(Excl, Facts0, Facts),
    kbq_holds_in(Rules, Facts, A).

% ---------------------------------------------------------------------------
% Minimal supports: fixpoint over antichains of fact-id sets, using only the
% rules whose negated literals hold in the model of all facts valid at T.
% ---------------------------------------------------------------------------

kbq_supports(A, T, Sups, Facts) :-
    kbq_context(A, T, Rules, Facts),
    kbq_fact_atoms(Facts, Base),
    kbq_model(Rules, Base, Model),
    (   ord_memberchk(A, Model)
    ->  include(kbq_enabled(Model), Rules, Active),
        findall(X-L,
                ( member(X, Model),
                  findall([Id], member(Id-X, Facts), L0),
                  sort(L0, L) ),
                Init),
        list_to_assoc(Init, S0),
        kbq_sup_fix(Active, S0, S),
        get_assoc(A, S, Sups)
    ;   Sups = []
    ).

kbq_enabled(Model, r(H, Pos, Neg)) :-
    ord_memberchk(H, Model),
    ord_subset(Pos, Model),
    \+ ( member(X, Neg), ord_memberchk(X, Model) ).

kbq_sup_fix(Rules, S0, S) :-
    foldl(kbq_sup_rule, Rules, S0-false, S1-Changed),
    (   Changed == true
    ->  kbq_sup_fix(Rules, S1, S)
    ;   S = S1
    ).

kbq_sup_rule(r(H, Pos, _), S0-C0, S-C) :-
    kbq_cross(Pos, S0, [[]], Prod),
    (   Prod \== []
    ->  get_assoc(H, S0, Old),
        append(Old, Prod, All),
        kbq_minimize(All, New),
        (   New == Old
        ->  S = S0, C = C0
        ;   put_assoc(H, S0, New, S), C = true
        )
    ;   S = S0, C = C0
    ).

kbq_cross([], _, Acc, Acc).
kbq_cross([X|Xs], S, Acc, R) :-
    get_assoc(X, S, L),
    findall(U, ( member(P, Acc), member(Q, L), ord_union(P, Q, U) ), Us),
    kbq_minimize(Us, Acc1),
    kbq_cross(Xs, S, Acc1, R).

% Subset-minimal sets only, in standard order without duplicates.
kbq_minimize(Sets, Min) :-
    sort(Sets, Sorted),
    exclude(kbq_has_proper_subset(Sorted), Sorted, Min).

kbq_has_proper_subset(Sets, X) :-
    member(Y, Sets),
    Y \== X,
    ord_subset(Y, X),
    !.

% ---------------------------------------------------------------------------
% Minimum repairs: smallest sets of facts valid at T (only facts A depends on
% can matter) whose removal makes A not derivable, re-evaluating negation.
% ---------------------------------------------------------------------------

kbq_repairs(A, T, Repairs) :-
    kbq_context(A, T, Rules, Facts),
    (   kbq_holds_in(Rules, Facts, A)
    ->  findall(Id, member(Id-_, Facts), Ids0),
        sort(Ids0, Ids),
        length(Ids, N),
        once(( between(1, N, K),
               findall(C,
                       ( kbq_comb(K, Ids, C),
                         kbq_without(C, Facts, Rest),
                         \+ kbq_holds_in(Rules, Rest, A) ),
                       Cs),
               Cs \== [] )),
        sort(Cs, Repairs)
    ;   Repairs = [[]]
    ).

% Subsets of size K of a sorted list, each in sorted order.
kbq_comb(0, _, []) :- !.
kbq_comb(K, [X|Xs], [X|C]) :-
    K1 is K - 1,
    kbq_comb(K1, Xs, C).
kbq_comb(K, [_|Xs], C) :-
    length(Xs, L),
    L >= K,
    kbq_comb(K, Xs, C).
