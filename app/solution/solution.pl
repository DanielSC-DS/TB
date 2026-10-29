% solution.pl -- belief analysis over robot knowledge bases.
%
% The knowledge base (bfact/4, rule/3, world/3) is loaded alongside this file.
%
%   holds_at(+A, +T)                 A is a valid or derivable atom at time T.
%   supports(+A, +T, -S)             S: sorted list of all minimal supports of A at T.
%   stale_basis(+A, +T, -Ids)        Ids: facts in some minimal support of A at T that
%                                    are false in the world at T.
%   would_hold_without(+A, +T, +Id)  A holds at T when fact Id does not exist.
%
% Negated literals are judged against the model obtained from all facts valid at T
% (stratified evaluation; negation is never circular, positive cycles are allowed).

:- use_module(library(lists)).
:- use_module(library(apply)).
:- use_module(library(assoc)).
:- use_module(library(ordsets)).
:- use_module(library(pairs)).

% ---------------------------------------------------------------------------
% Public predicates
% ---------------------------------------------------------------------------

holds_at(A, T) :-
    once(analyse(A, T, [], _, _, _, Model)),
    ord_memberchk(A, Model).

supports(A, T, S) :-
    once(minimal_supports(A, T, S0)),
    S = S0.

stale_basis(A, T, Ids) :-
    once(minimal_supports(A, T, Sups)),
    ord_union(Sups, Used),
    include(stale_fact(T), Used, Ids0),
    Ids = Ids0.

% Negations stay judged against all facts valid at T, so A still holds without
% Id exactly when some minimal support of A does not use Id.
would_hold_without(A, T, Id) :-
    once(minimal_supports(A, T, Sups)),
    member(Sup, Sups),
    \+ ord_memberchk(Id, Sup),
    !.

% ---------------------------------------------------------------------------
% Knowledge base access (guarded, so a KB lacking a predicate does not raise)
% ---------------------------------------------------------------------------

kb_bfact(Id, A, F, To) :-
    current_predicate(bfact/4),
    bfact(Id, A, F, To).

kb_rule(Id, H, Body) :-
    current_predicate(rule/3),
    rule(Id, H, Body).

kb_world(A, F, To) :-
    current_predicate(world/3),
    world(A, F, To).

in_interval(T, F, To) :-
    F =< T,
    ( To == inf -> true ; T < To ).

world_true(A, T) :-
    kb_world(A, F, To),
    in_interval(T, F, To),
    !.

% A fact is stale at T when its atom is false in the world at T.
stale_fact(T, Id) :-
    kb_bfact(Id, A, _, _),
    \+ world_true(A, T),
    !.

lit_atom(not(X), X) :- !.
lit_atom(X, X).

% ---------------------------------------------------------------------------
% Analysis of the part of the KB relevant to atom A at time T
% ---------------------------------------------------------------------------

% analyse(+A, +T, +Excluded, -Atoms, -Rules, -Facts, -Model)
%   Atoms : ordset of atoms A depends on (including A)
%   Rules : relevant rules as r(Head, PosAtoms, NegAtoms)
%   Facts : Id-Atom pairs of facts valid at T (minus Excluded) on relevant atoms
%   Model : ordset of relevant atoms that hold at T
analyse(A, T, Excl, Atoms, Rules, Facts, Model) :-
    reach([A], [A], Atoms),
    findall(r(H, Pos, Neg),
            ( member(H, Atoms),
              kb_rule(_, H, Body),
              split_body(Body, Pos, Neg)
            ),
            Rules),
    findall(Id-FA,
            ( kb_bfact(Id, FA, F, To),
              \+ memberchk(Id, Excl),
              in_interval(T, F, To),
              ord_memberchk(FA, Atoms)
            ),
            Facts0),
    sort(Facts0, Facts),
    levels(Atoms, Rules, Lv),
    pairs_values(Facts, FactAtoms),
    sort(FactAtoms, M0),
    assoc_to_values(Lv, LvValues),
    max_list(LvValues, MaxL),
    numlist(0, MaxL, Ks),
    foldl(stratum(Rules, Lv), Ks, M0, Model).

% Backward reachability over rule bodies (positive and negated literals).
reach([], S, S).
reach([X|Xs], S0, S) :-
    findall(B, ( kb_rule(_, X, Body), member(L, Body), lit_atom(L, B) ), Bs0),
    sort(Bs0, Bs),
    ord_subtract(Bs, S0, New),
    ord_union(S0, New, S1),
    append(Xs, New, Queue),
    reach(Queue, S1, S).

split_body([], [], []).
split_body([L|Ls], Pos, Neg) :-
    (   L = not(X)
    ->  Neg = [X|Neg1], Pos = Pos1
    ;   Pos = [L|Pos1], Neg = Neg1
    ),
    split_body(Ls, Pos1, Neg1).

% Stratification: level(H) >= level(P) for positive P, level(H) > level(N) for not(N).
levels(Atoms, Rules, Lv) :-
    empty_assoc(E),
    foldl(init_level, Atoms, E, L0),
    length(Atoms, N),
    Cap is N + 1,
    level_iter(Rules, Cap, L0, Lv).

init_level(X, L0, L) :-
    put_assoc(X, L0, 0, L).

level_iter(Rules, Cap, L0, L) :-
    foldl(level_rule(Cap), Rules, L0-false, L1-Changed),
    (   Changed == true
    ->  level_iter(Rules, Cap, L1, L)
    ;   L = L1
    ).

level_rule(Cap, r(H, Pos, Neg), L0-C0, L-C) :-
    get_assoc(H, L0, LH),
    foldl(max_level(L0, 0), Pos, LH, M1),
    foldl(max_level(L0, 1), Neg, M1, M2),
    M is min(M2, Cap),               % guard against (excluded) negative cycles
    (   M > LH
    ->  put_assoc(H, L0, M, L), C = true
    ;   L = L0, C = C0
    ).

max_level(L, Off, X, M0, M) :-
    get_assoc(X, L, V),
    M is max(M0, V + Off).

% Saturate the rules whose head is at level =< K; all negated atoms of those
% rules are at a lower level and therefore already final.
stratum(Rules, Lv, K, M0, M) :-
    include(rule_upto(Lv, K), Rules, Rk),
    saturate(Rk, M0, M).

rule_upto(Lv, K, r(H, _, _)) :-
    get_assoc(H, Lv, LH),
    LH =< K.

saturate(Rules, M0, M) :-
    findall(H,
            ( member(r(H, Pos, Neg), Rules),
              \+ ord_memberchk(H, M0),
              all_in(Pos, M0),
              none_in(Neg, M0)
            ),
            Hs0),
    sort(Hs0, Hs),
    (   Hs == []
    ->  M = M0
    ;   ord_union(M0, Hs, M1),
        saturate(Rules, M1, M)
    ).

all_in([], _).
all_in([X|Xs], M) :- ord_memberchk(X, M), all_in(Xs, M).

none_in([], _).
none_in([X|Xs], M) :- \+ ord_memberchk(X, M), none_in(Xs, M).

% ---------------------------------------------------------------------------
% Minimal supports
% ---------------------------------------------------------------------------

minimal_supports(A, T, Sups) :-
    analyse(A, T, [], Atoms, Rules, Facts, Model),
    (   ord_memberchk(A, Model)
    ->  include(active_rule(Model), Rules, Active),
        empty_assoc(E),
        foldl(init_support(Facts), Atoms, E, S0),
        support_iter(Active, S0, S),
        get_assoc(A, S, Sups)
    ;   Sups = []
    ).

% A rule can contribute only if its negations hold in the full model at T
% (and then its head and positive body atoms are in the model as well).
active_rule(Model, r(H, Pos, Neg)) :-
    ord_memberchk(H, Model),
    all_in(Pos, Model),
    none_in(Neg, Model).

init_support(Facts, X, S0, S) :-
    findall([Id], member(Id-X, Facts), L0),
    sort(L0, L),
    put_assoc(X, S0, L, S).

support_iter(Rules, S0, S) :-
    foldl(support_rule, Rules, S0-false, S1-Changed),
    (   Changed == true
    ->  support_iter(Rules, S1, S)
    ;   S = S1
    ).

support_rule(r(H, Pos, _), S0-C0, S-C) :-
    (   product(Pos, S0, [[]], Prod)
    ->  get_assoc(H, S0, Old),
        append(Old, Prod, All),
        minimize(All, New),
        (   New == Old
        ->  S = S0, C = C0
        ;   put_assoc(H, S0, New, S), C = true
        )
    ;   S = S0, C = C0
    ).

% Fails if some positive body atom has no support yet.
product([], _, Acc, Acc).
product([X|Xs], S, Acc, R) :-
    get_assoc(X, S, L),
    L \== [],
    findall(U, ( member(P, Acc), member(Q, L), ord_union(P, Q, U) ), Us),
    minimize(Us, Acc1),
    product(Xs, S, Acc1, R).

% Keep only the subset-minimal sets; result in standard order.
minimize(Sets, Min) :-
    sort(Sets, Sets1),
    map_list_to_pairs(length, Sets1, Keyed),
    keysort(Keyed, ByLen0),
    pairs_values(ByLen0, ByLen),
    keep_minimal(ByLen, [], Kept),
    sort(Kept, Min).

keep_minimal([], K, K).
keep_minimal([X|Xs], K0, K) :-
    (   member(Y, K0), ord_subset(Y, X)
    ->  K1 = K0
    ;   K1 = [X|K0]
    ),
    keep_minimal(Xs, K1, K).
