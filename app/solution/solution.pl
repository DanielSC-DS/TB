% solution.pl -- query the belief knowledge base of a robot.
%
% The knowledge base (bfact/4, rule/3, world/3) is loaded alongside this file.
%
%   holds_at(+A, +T)                A is valid or derivable at time T.
%   supports(+A, +T, -S)            S: sorted list of all minimal supports of A at T.
%   stale_basis(+A, +T, -Ids)       Ids: facts of some minimal support of A at T
%                                   whose atom is false in the world at T.
%   would_hold_without(+A, +T, +Id) A holds at T if fact Id did not exist.
%
% Evaluation is stratified: negation is never circular, positive cycles are
% allowed. not(X) is judged against the model built from all facts valid at T.

:- use_module(library(lists)).
:- use_module(library(apply)).
:- use_module(library(assoc)).
:- use_module(library(ordsets)).
:- use_module(library(pairs)).

% ---------------------------------------------------------------------------
% Public predicates
% ---------------------------------------------------------------------------

holds_at(A, T) :-
    once(model(A, T, [], _, _, Model)),
    ord_memberchk(A, Model).

supports(A, T, S) :-
    once(minimal_supports(A, T, S0)),
    S = S0.

stale_basis(A, T, Ids) :-
    once(minimal_supports(A, T, Sups)),
    ord_union(Sups, Used),
    include(stale_fact(T), Used, Ids0),
    Ids = Ids0.

would_hold_without(A, T, Id) :-
    once(model(A, T, [Id], _, _, Model)),
    ord_memberchk(A, Model).

% ---------------------------------------------------------------------------
% Helpers
% ---------------------------------------------------------------------------

% KB access, guarded so a KB without some predicate raises no error.
kb_bfact(Id, A, F, To) :- current_predicate(bfact/4), bfact(Id, A, F, To).
kb_rule(Id, H, Body)   :- current_predicate(rule/3), rule(Id, H, Body).
kb_world(A, F, To)     :- current_predicate(world/3), world(A, F, To).

valid_in(T, F, To) :-
    F =< T,
    ( To == inf -> true ; T < To ).

world_true(A, T) :-
    kb_world(A, F, To),
    valid_in(T, F, To),
    !.

stale_fact(T, Id) :-
    kb_bfact(Id, A, _, _),
    \+ world_true(A, T),
    !.

lit_atom(not(X), X) :- !.
lit_atom(X, X).

split_body([], [], []).
split_body([not(X)|Ls], Pos, [X|Neg]) :- !,
    split_body(Ls, Pos, Neg).
split_body([X|Ls], [X|Pos], Neg) :-
    split_body(Ls, Pos, Neg).

all_in([], _).
all_in([X|Xs], M) :- ord_memberchk(X, M), all_in(Xs, M).

none_in([], _).
none_in([X|Xs], M) :- \+ ord_memberchk(X, M), none_in(Xs, M).

% ---------------------------------------------------------------------------
% Model of the part of the KB that A depends on, at time T, without the
% facts in Excl.
%   Rules : r(Head, PosAtoms, NegAtoms) for every relevant rule
%   Facts : sorted Id-Atom pairs of the relevant facts valid at T
%   Model : sorted list of relevant atoms that hold at T
% ---------------------------------------------------------------------------

model(A, T, Excl, Rules, Facts, Model) :-
    reach([A], [A], Atoms),
    findall(r(H, Pos, Neg),
            ( member(H, Atoms), kb_rule(_, H, Body), split_body(Body, Pos0, Neg0),
              sort(Pos0, Pos), sort(Neg0, Neg) ),
            Rules),
    findall(Id-FA,
            ( member(FA, Atoms), kb_bfact(Id, FA, F, To),
              \+ memberchk(Id, Excl), valid_in(T, F, To) ),
            Facts0),
    sort(Facts0, Facts),
    levels(Atoms, Rules, Lv),
    pairs_values(Facts, FAtoms),
    sort(FAtoms, M0),
    assoc_to_values(Lv, LvVals),
    max_list(LvVals, Max),
    numlist(0, Max, Ks),
    foldl(stratum(Rules, Lv), Ks, M0, Model).

% Atoms reachable backwards through rule bodies (positive and negated).
reach([], S, S).
reach([X|Xs], S0, S) :-
    findall(B, ( kb_rule(_, X, Body), member(L, Body), lit_atom(L, B) ), Bs0),
    sort(Bs0, Bs),
    ord_subtract(Bs, S0, New),
    ord_union(S0, New, S1),
    append(Xs, New, Queue),
    reach(Queue, S1, S).

% level(H) >= level(P) for positive P, level(H) > level(N) for not(N).
levels(Atoms, Rules, Lv) :-
    findall(X-0, member(X, Atoms), Ps),
    list_to_assoc(Ps, L0),
    length(Atoms, N),
    level_iter(Rules, N, L0, Lv).

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
    M is min(M2, Cap),
    (   M > LH
    ->  put_assoc(H, L0, M, L), C = true
    ;   L = L0, C = C0
    ).

max_level(L, Off, X, M0, M) :-
    get_assoc(X, L, V),
    M is max(M0, V + Off).

% Saturate the rules with head level =< K; their negated atoms are of a lower
% level and therefore already final.
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
              none_in(Neg, M0) ),
            Hs0),
    sort(Hs0, Hs),
    (   Hs == []
    ->  M = M0
    ;   ord_union(M0, Hs, M1),
        saturate(Rules, M1, M)
    ).

% ---------------------------------------------------------------------------
% Minimal supports: fixpoint over antichains of fact-id sets. Only rules whose
% negations hold in the full model at T can contribute.
% ---------------------------------------------------------------------------

minimal_supports(A, T, Sups) :-
    model(A, T, [], Rules, Facts, Model),
    (   ord_memberchk(A, Model)
    ->  include(active_rule(Model), Rules, Active),
        findall(X, ( member(r(H, Ps, _), Active), member(X, [H|Ps]) ), Xs0),
        pairs_values(Facts, FXs),
        append([A|FXs], Xs0, Xs1),
        sort(Xs1, Xs),
        findall(X-L, ( member(X, Xs),
                       findall([Id], member(Id-X, Facts), L0), sort(L0, L) ),
                Init),
        list_to_assoc(Init, S0),
        support_iter(Active, S0, S),
        get_assoc(A, S, Sups)
    ;   Sups = []
    ).

active_rule(Model, r(H, Pos, Neg)) :-
    ord_memberchk(H, Model),
    all_in(Pos, Model),
    none_in(Neg, Model).

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

% Fails when some positive body atom has no support yet.
product([], _, Acc, Acc).
product([X|Xs], S, Acc, R) :-
    get_assoc(X, S, L),
    L \== [],
    findall(U, ( member(P, Acc), member(Q, L), ord_union(P, Q, U) ), Us),
    minimize(Us, Acc1),
    product(Xs, S, Acc1, R).

% Subset-minimal sets only, in standard order.
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
