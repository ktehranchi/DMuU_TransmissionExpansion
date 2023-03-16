# using Pkg
# pkg"add POMDPs"
# pkg"add QuickPOMDPs"
# pkg"add POMDPTools"
# pkg"add PyCall"

ENV["PYTHON"] = "/Users/kamrantehranchi/opt/miniconda3/envs/pypsa-usa/bin/python"           # example for *nix
Pkg.build("PyCall")
using PyCall
so = pyimport("pypsa")
pyimport("pandas")
pyimport("numpy")
using QuickPOMDPs: QuickPOMDP
using Random, Distributions
Random.seed!(123)

# γ = 0.95
# S = # int, number of states year and what lines have been built  [2020 1 1]
# A = ["build500kv", "buildHVDC"] # vector of actions
# TR = # function sp, r = TR(s,a)
# TR function calls on T and R function given (s,a) and yields next state and reward

pypsa_MDP = MDP(
    0.95,
    # S_init = (2023,0,0) #Vector of interger state variables (year, HVAC, HVDC Undersea)
    ["build500kv", "buildHVDC"],
    function (state, action)
        py"""
        import pypsa
        import pandas as pd
        import numpy as np

        def add_hvdc(n,num):
            for i in range(0,num):
                n.add("Link", f'New_HVDC_{i}', bus0="CISO-PGAE0 13", bus1="CISO-PGAE0 8",
                    type="HVDC_VSC", carrier = "DC", efficiency=1, p_nom=100,p_min_pu=-1,)
            return n
        
        def add_hvac(n,num):

            n.line_types.loc["Rail"] = pd.Series(
                [60, 0.0683, 0.335, 15, 1.01],
                index=["f_nom", "r_per_length", "x_per_length", "c_per_length", "i_nom"],
                )
        

            for i in range(0,num):
                n.add("Line", f'New_HVAC_{i}', bus0="CISO-PGAE0 13", bus1="CISO-PGAE0 16",
                    r=0.815803, x=6.873208, b=0.000398 ,s_nom=536.51,length=400, 
                    type="Rail",v_nom=230, x_pu= 0.000725, r_pu= 0.000085, g_pu= 0.0,
                    b_pu= 7.557957, x_pu_eff= 0.000725, r_pu_eff= 0.000085, s_nom_opt= 2100.295202,)
            return n

        def add_ows(n,num):
            ows_cf = pd.read_csv('/Users/kamrantehranchi/Documents/GradSchool/Courses/Q5/Decision Making under Uncertainty/DMuU_TransmissionExpansion/OSW_CF_Humboldt.csv')

            for i in range(0,num):
                n.add("Generator", f'Humboldt_OSW_{i}', bus= "CISO-PGAE0 13", carrier="offwind",
                    p_nom=4000, marginal_cost=0, p_max_pu=np.tile(ows_cf.cf_humboldt.values,6),)
            return n

        def update_network(n, state, action):
            year = state[0]
            num_hvac = state[1]
            num_hvdc = state[2]

            if action == "build500kv":
                num_hvac += 1
            elif action == "buildHVDC":
                num_hvdc += 1
            
            add_hvac(n, num_hvac)
            add_hvdc(n, num_hvdc)
            add_ows(n, 1)
            
            new_state = (year, num_hvac, num_hvdc)
            return n , new_state

        def WECC_simulator(state, action):
            network_path='/Users/kamrantehranchi/Documents/GradSchool/Courses/Q5/Decision Making under Uncertainty/DMuU_TransmissionExpansion/pypsa_60_multiyear.nc'
            network = pypsa.Network(network_path)

            network, new_state= update_network(network, state, action)
            sim_year = new_state[0]
            simulation_hours = 168
            
            sim_snapshot = network.snapshots[network.snapshots.get_level_values(0) == sim_year][:simulation_hours]
            network.lopf(sim_snapshot, solver_name="gurobi")

            generation_ts = network.generators_t.p.loc[sim_snapshot]
            operation_costs = network.generators.marginal_cost * generation_ts
            print("Total Operational Cost:  " , operation_costs.sum(axis=1).sum())
            operating_cost = operation_costs.sum(axis=1).sum()
            return operating_cost, state
        
        """

        annualize = function (cost,capital_γ,lifetime)
            return (cost * capital_γ) / (1 - (1 + capital_γ)^-lifetime)
        end

        capital_discount_rate = 0.05
        lifetime = 20
        hvac_cost = [300e6, 400e6]
        hvdc_cost = [400e6, 500e6]

        hvac_cost_dist = Uniform(hvac_cost[1],hvac_cost[2])
        hvdc_cost_dist = Uniform(hvdc_cost[1],hvdc_cost[2])
        
        if action == "build500kv"
            random_cost_to_build = rand(hvac_cost_dist, 1)
        else
            random_cost_to_build = rand(hvdc_cost_dist, 1)
        end

        annualized_cost = annualize(random_cost_to_build,capital_discount_rate,lifetime) 

        operating_cost, new_state = py"WECC_simulator"(state,action)

        annual_reward = operating_cost + annualized_cost[1]

        return new_state, annual_reward
    end,
)


