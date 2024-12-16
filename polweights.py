import numpy as np

def polweights_optimal(B,M_max,tradeoff,truncate=0.01):
    """Finds optimal policy weights."""
    import gurobipy as gp

    ones = np.ones(M_max)
    increase = np.arange(M_max) + 1

    # Normalize objective
    if tradeoff > 0.0 and tradeoff < 1.0:
        # Recursively call without objective normalization or truncation
        weights_tv, _, _ = polweights_optimal(B,M_max,0.0,0.0)
        weights_ess, _, _ = polweights_optimal(B,M_max,1.0,0.0)

        tv_min = np.dot(increase,weights_tv)
        tv_max = B

        ess_min = np.sum(np.square(weights_ess))
        ess_max = 1/B

        objweight_ess = tradeoff / (ess_max - ess_min)
        objweight_tv = (1-tradeoff) / (tv_max - tv_min)
    else:
        objweight_ess = tradeoff
        objweight_tv = (1-tradeoff)

    # Optimize weights
    obj_tv = objweight_tv * increase
    obj_ess = np.identity(M_max) * objweight_ess

    model = gp.Model()
    model.Params.OutputFlag = 0

    p_opt = model.addMVar((M_max,),name='weights')

    model.setObjective(p_opt @ obj_ess @ p_opt + obj_tv @ p_opt)

    model.addConstr(ones @ p_opt == 1)
    model.addConstr(increase @ p_opt <= B)
    model.addConstr(p_opt @ p_opt <= (1/B))

    model.optimize()
    if model.Status != 2:
        raise ValueError(
            'Gurobi unable to find optimal weights (status %d)'%model.Status)

    weights = p_opt.X

    # Truncate weights and calculate relevant quantities
    active = weights > truncate
    weights = weights[active]
    weights = weights / np.sum(weights)

    eps_mult = 1 / np.dot(weights,increase[active])
        
    M = len(weights)

    return weights, M, eps_mult


B = 4
print(B, polweights_optimal(B, 10,0.5))