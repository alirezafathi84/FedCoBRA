import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fedcobra.fl import method_cfg
def T(c, tag): c = dict(c); c["tag"] = tag; return c
plan = []
MAIN = ["fedcobra", "masked", "oracle", "boxaspoly", "partial", "polyonly"]
for s in (0, 1):
    for m in MAIN:
        plan.append(T(method_cfg(m, seed=s), f"main_s{s}_{m}"))
for name, kw in [("noRADA", dict(agg="fedavg")), ("noShape", dict(shape=False)), ("noDistill", dict(distill=False)),
                 ("noAff", dict(aff=False)), ("dp1", dict(dp_eps=1.0))]:
    plan.append(T(method_cfg("fedcobra", seed=0, **kw), f"abl_s0_{name}"))
B, RATIO = 60, 79.2 / 7.0          # per-client budget = 60 polygon instances; box via extreme clicking
NB = int(round(B * RATIO))
for s in (0, 1):
    plan.append(T(method_cfg("oracle", seed=s, n_pool=NB, n_poly_img=B), f"budget_s{s}_allpoly"))
    plan.append(T(method_cfg("fedcobra", seed=s, rho=0.0, n_pool=NB, n_box_img=NB), f"budget_s{s}_allbox"))
    plan.append(T(method_cfg("fedcobra", seed=s, rho=0.3, n_pool=NB, n_poly_img=B, n_box_img=NB), f"budget_s{s}_mixed"))
for rho in (0.1, 0.5):
    for m in ("masked", "fedcobra"):
        plan.append(T(method_cfg(m, seed=0, rho=rho), f"rho_s0_{rho}_{m}"))
plan.append(T(method_cfg("oracle", seed=0, prox=0.01), "opt_s0_fedprox"))
plan.append(T(method_cfg("oracle", seed=0, server_opt="avgm", server_mom=0.5), "opt_s0_fedavgm"))
os.makedirs("configs", exist_ok=True)
json.dump(plan, open("configs/plan_all.json", "w"), indent=1)
json.dump([T(method_cfg("fedcobra", seed=0, rounds=2, eval_every=1), "smoke_fedcobra")], open("configs/plan_smoke.json", "w"), indent=1)
print(len(plan), "runs; box images per client in budget study:", NB)
