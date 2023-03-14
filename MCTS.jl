include("pypsa_πm.jl")

struct MDP
    γ # discount factor
    S # state space
    A # action space
    T # transition function
    R # reward function
    TR # sample transition and reward
end

struct MonteCarloTreeSearch
    P # problem where P::MDP
    N # visit counts
    Q # action value estimates
    d # depth
    m # number of simulations
    c # exploration constant
    U # value function estimate
end

function (πm::MonteCarloTreeSearch)(s)
    for k in 1:πm.m
        simulate!(πm, s)
    end
    return argmax(a->πm.Q[(s,a)], πm.P.A)
end


function simulate!(πm::MonteCarloTreeSearch, s, d=πm.d)
    if d ≤ 0
        return πm.U(s)
    end
    P, N, Q, c = πm.P, πm.N, πm.Q, πm.c
    A, TR, γ = P.A, P.TR, P.γ
    if !haskey(N, (s, first(A)))
        for a in A
            N[(s,a)] = 0
            Q[(s,a)] = 0.0
        end
        return πm.U(s)
    end
    a = explore(πm, s)
    sp, r = TR(s,a)
    q = r + γ*simulate!(πm, sp, d-1)
    N[(s,a)] += 1
    Q[(s,a)] += (q-Q[(s,a)])/N[(s,a)]
    return q
end

bonus(Nsa, Ns) = Nsa == 0 ? Inf : sqrt(log(Ns)/Nsa)

function explore(πm::MonteCarloTreeSearch, s)
    A, N, Q, c = πm.P.A, πm.N, πm.Q, πm.c
    Ns = sum(N[(s,a)] for a in A)
    return argmax(a->Q[(s,a)] + c*bonus(N[(s,a)], Ns), A)
end

γ = 0.95
S = # int, number of states year and what lines have been built  [2020 1 1]
A = ["build500kv", "buildHVDC"] # vector of actions
T = # function T(s,a,s′)
R = # function R(s,a)
TR = # function sp, r = TR(s,a)
# TR function calls on T and R function given (s,a) and yields next state and reward

P = MDP(γ,S,A,T,R,TR)
N = # vector of int, number of times explored, visit counts N[(s,a)] = 0 initialize as dict()
Q = # vector of float, Q value Q[(s,a)] = 0.0 intialize as dict()
d = 5
m = 5
c = # constant
U = # πm.U(s) intialize as dict()
# we initialize N(s, a) and Q(s, a) to zero for each action a

πm = MonteCarloTreeSearch(P,N,Q,d,m,c,U)