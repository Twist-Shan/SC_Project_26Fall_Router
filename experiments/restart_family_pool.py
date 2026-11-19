"""Fixed 50:50 blend of semantic restart-averaged EM and family-shared preferences."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from arena_pipeline import load_data
from lexical_refinement import bootstrap_contrasts, losses

ROOT = Path(__file__).resolve().parents[1]


def run(restarts, family):
    data = load_data()
    with np.load(restarts) as fits, np.load(family) as other:
        if not (np.array_equal(fits["ids"], data["ids"]) and
                np.array_equal(other["ids"], data["ids"]) and np.array_equal(fits["fold"], other["fold"])):
            raise ValueError("Inconsistent held-out battles")
        ensemble = fits["probabilities"][:,2]
        prob = (ensemble + other["semantic_only"]) / 2
        fold = fits["fold"]
        controls = {"previous family pool": other["pred"], "semantic EM": fits["probabilities"][:,0],
                    "family predictor": other["semantic_only"]}
    y = data["y"]
    new_correct = ((prob>.5)==y).astype(float)
    rows = []
    for name,p in controls.items():
        delta = np.column_stack([new_correct-((p>.5)==y),losses(prob,y)-losses(p,y)])
        lo,hi = bootstrap_contrasts(delta,data["groups"],1000)
        rows.append(dict(reference=name,accuracy=new_correct.mean(),logloss=losses(prob,y).mean(),
                         net_correct=int(delta[:,0].sum()),accuracy_change=delta[:,0].mean(),
                         accuracy_lo=lo[0],accuracy_hi=hi[0],loss_change=delta[:,1].mean(),
                         loss_lo=lo[1],loss_hi=hi[1],folds_with_lower_loss=sum(delta[fold==f,1].mean()<0 for f in np.unique(fold))))
    result = pd.DataFrame(rows)
    result.to_csv(restarts.parent/"family_pool.csv",index=False)
    print(result.to_string(index=False))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--restarts",type=Path,default=ROOT/"results/restart_offsets/local_oof.npz")
    parser.add_argument("--family",type=Path,required=True,help="Output of semantic_refinement.py --save-predictions")
    args=parser.parse_args()
    run(args.restarts,args.family)
