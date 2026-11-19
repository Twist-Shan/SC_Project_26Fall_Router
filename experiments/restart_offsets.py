"""Evaluate restart averaging and ridge-shrunk pair offsets on the existing Arena folds."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from arena_pipeline import categories, load_data
from lexical_refinement import bootstrap_contrasts, losses
from semantic_mixture import semantic_design
from semantic_refinement import check_inputs
from mixbt.restarts import RestartRouter


def run(reference):
    d=load_data()
    a,b,y,M=(d[k] for k in ("a","b","y","M"))
    with np.load(reference/"oof_predictions.npz") as base,np.load(ROOT/"data/semantic_minilm.npz") as cache:
        emb,fold,seed=check_inputs(d,base,cache,reference)
    with np.load(ROOT/"results/semantic_mixture_capacity/local_oof.npz") as ref:
        assert np.array_equal(ref["ids"],d["ids"]) and np.array_equal(ref["fold"],fold)
        old=ref["semantic_mixture"]
    methods=("best start","best + pair offset","three-start mean","mean + pair offset")
    preds=np.full((len(y),4),np.nan)
    attr=categories(d["text"]).to_numpy()
    for f in range(5):
        train,test=fold!=f,fold==f
        G=semantic_design(emb,attr,train,11,seed+f)
        cfg=json.loads((reference/f"fold_{f}.json").read_text())["chosen"]["mixture"]
        model=RestartRouter(cfg,seed+f).fit(G[train],a[train],b[train],y[train],M)
        for j,(average,correct) in enumerate(((False,False),(False,True),(True,False),(True,True))):
            preds[test,j]=model.predict_proba(G[test],a[test],b[test],average,correct)
        print("Finished fold",f,flush=True)
    if not np.allclose(preds[:,0],old,rtol=0,atol=1e-7):
        raise ValueError("Reference predictions changed")
    correct=((preds>.5)==y[:,None]).astype(float)
    nll=losses(preds,y[:,None])
    diff=np.column_stack([correct-correct[:,[0]],nll-nll[:,[0]]])
    lo,hi=bootstrap_contrasts(diff,d["groups"],1000)
    rows=[dict(method=name,accuracy=correct[:,j].mean(),logloss=nll[:,j].mean(),
               net_correct=int(diff[:,j].sum()),gain_lo=lo[j],gain_hi=hi[j],
               loss_change=diff[:,4+j].mean(),loss_lo=lo[4+j],loss_hi=hi[4+j]) for j,name in enumerate(methods)]
    out=ROOT/"results/restart_offsets";out.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(out/"scores.csv",index=False)
    np.savez_compressed(out/"local_oof.npz", ids=d["ids"], fold=fold, probabilities=preds)
    print(pd.DataFrame(rows).to_string(index=False))
    pd.DataFrame([dict(fold=f,method=name,accuracy=correct[fold==f,j].mean(),logloss=nll[fold==f,j].mean())
                  for f in range(5) for j,name in enumerate(methods)]).to_csv(out/"folds.csv",index=False)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference",type=Path,default=ROOT/"results/nested")
    with threadpool_limits(1):
        run(parser.parse_args().reference)
