struct MDP
    γ # discount factor
    # S # state space
    A # action space
    # T # transition function
    # R # reward function
    TR # sample transition and reward
end

include("pypsa_mcts.jl")

struct MonteCarloTreeSearch
    P # problem where P::MDP
    N # visit counts
    Q # action value estimates
    d # depth
    m # number of simulations
    c # exploration constant
    # U # value function estimate
end

function (πm::MonteCarloTreeSearch)(s)
    for k in 1:πm.m
        simulate!(πm, s)
    end
    return argmax(a->πm.Q[(s,a)], πm.P.A)
end

function simulate!(πm::MonteCarloTreeSearch, s, d=πm.d)
    if d ≤ 0
        return 0 #πm.U(s)
    end
    P, N, Q = πm.P, πm.N, πm.Q
    A, γ , TR = P.A, P.γ, P.TR
    if !haskey(N, (s, first(A)))
        for a in A
            N[(s,a)] = 0
            Q[(s,a)] = 0.0
        end
        return 0 #πm.U(s)
    end
    a = explore(πm, s)
    sp, r = TR(s,a)
    prev_year = sp[1]
    sp[1] = prev_year + 5 #moving year forward 5
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

## INITIALIZE PARAMETERS

P = pypsa_MDP
N = Dict() # vector of int, number of times explored, visit counts N[(s,a)] = 0 initialize as dict()
Q = Dict() # vector of float, Q value Q[(s,a)] = 0.0 intialize as dict()
m = 6
c = 1000 # constant

## MAIN CODE

initial_state = [2020,0,0]
depth = [5, 5, 4, 3, 2, 1]
years = [2020, 2025, 2030, 2035, 2040, 2045]
for i = 1:length(years)
    println("Computing for year ", years[i])
    d = depth[i]
    πm = MonteCarloTreeSearch(P,N,Q,d,m,c)
    A = (πm)(initial_state)
    new_state, reward = P.TR(initial_state, A)
    initial_state = new_state
    println(Q)
    println(A)
end

# U = # πm.U(s) intialize as dict()
# S = # int, number of states year and what lines have been built  [2020 1 1]
# T = # function T(s,a,s′)
# R = # function R(s,a)
# TR = # function sp, r = TR(s,a) TR function calls on T and R function given (s,a) and yields next state and reward


# struct RolloutLookahead
# 	P # problem
# 	π # rollout policy
# 	d # depth
# end

# randstep(P::MDP, s, a) = P.TR(s, a)

# function rollout(P, s, π, d)
#     ret = 0.0
#     for t in 1:d
#         a = π(s)
#         s, r = randstep(P, s, a)
#         ret += P.γ^(t-1) * r
#     end
#     return ret
# end

# function (π::RolloutLookahead)(s)
# 	U(s) = rollout(π.P, s, π.π, π.d)
#     return greedy(π.P, U, s).a
# end