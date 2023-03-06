using Pkg
pkg"add POMDPs"
pkg"add QuickPOMDPs"
pkg"add POMDPTools"
pkg"add PyCall"

ENV["PYTHON"] = "/Users/kamrantehranchi/opt/miniconda3/envs/pypsa-usa/bin/python"           # example for *nix
Pkg.build("PyCall")
using PyCall
so = pyimport("pypsa")

using QuickPOMDPs: QuickPOMDP
# using POMDPTools: ImplicitDistribution

pypsa_PCS = QuickPOMDP(
    actions = ['build500kv', 'no_build', 'buildHVDC'],
    discount = 0.95,

    gen = function (s, a, rng)
        random_cost_to_build = randn(rng)

        py"""
        import pypsa
        clustered_path='/Users/kamrantehranchi/Library/CloudStorage/OneDrive-Stanford/Kamran_OSW/PyPSA_Models/pypsa-breakthroughenergy-usa/workflow/resources/western/elec_s_50.nc'
        network = pypsa.Network(clustered_path)

        def WECC_simulator(random_cost_to_build,a):
            #use action and random_cost_to_build to modify the network by specified action and cost

            network.lopf(network.snapshots[0:48], solver_name="gurobi")
            return network.generators_t.p.sum().sum()
        """

        py"WECC_simulator"(s,a,random_cost_to_build)

        annualsed_cost = our own calculation based on the action and random_cost_to_build   
        
        operating_cost= py"WECC_simulator"(s,a,random_cost_to_build)
        return (sp=(xp, vp), r=r)
    end,

    initialstate = [0,0,0]
)



