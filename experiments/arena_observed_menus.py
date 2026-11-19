"""Fully observed human-vote triangles on exact first prompts, grouped outer holdout.

Compare the existing dimension-matched semantic mixture and scalar BT. Give
each exact prompt equal weight, each of its complete triangles equal weight
and each edge equal weight regardless of how often it was logged.
"""
from argparse import ArgumentParser
from collections import defaultdict
from itertools import combinations
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from semantic_link_comparison import ROOT, scalar_fit
from semantic_mixture import semantic_design
from semantic_refinement import check_inputs
from arena_pipeline import categories, load_data


def evaluate(pair_rows, tri, votes, mixture, scalar):
    edges = list(combinations(tri, 2))
    mats = []
    for arr in (votes, mixture, scalar):
        mat = np.full((3, 3), .5)
        for (u, v), edge in zip(combinations(range(3), 2), edges):
            ix, flip = zip(*pair_rows[edge])
            p = np.where(flip, 1-arr[list(ix)], arr[list(ix)])
            mat[u, v], mat[v, u] = p.mean(), 1-p.mean()
        mats.append(mat)
    truth, mix, sc = mats
    mw, sw = (int(np.argmax(m.sum(axis=1))) for m in (mix, sc))
    scores = (truth.sum(axis=1)-.5)/2
    res = dict(triple_mix=scores[mw], triple_scalar=scores[sw], disagree=int(mw!=sw))
    pair_scores = []
    reversals = [0, 0]
    for u, v in combinations(range(3), 2):
        picks = [u if m[u,v] >= .5 else v for m in (mix,sc)]
        pair_scores.append([truth[w,v if w==u else u] for w in picks])
        for j,(w,top) in enumerate(zip(picks,(mw,sw))):
            reversals[j] += int(top in (u,v) and top != w)
    res["pair_mix"], res["pair_scalar"] = np.mean(pair_scores,axis=0)
    res.update(mix_reversals=reversals[0], scalar_reversals=reversals[1],
               min_edge_votes=min(len(pair_rows[e]) for e in edges))
    return res


def main(reference):
    d=load_data()
    with np.load(reference/"oof_predictions.npz") as orig, np.load(ROOT/"data/semantic_minilm.npz") as cache:
        emb,folds,seed=check_inputs(d,orig,cache,reference)
    with np.load(ROOT/"results/semantic_mixture_capacity/local_oof.npz") as saved:
        assert np.array_equal(saved["ids"],d["ids"]) and np.array_equal(saved["fold"],folds)
        mix=saved["semantic_mixture"]
    by_prompt=defaultdict(lambda:defaultdict(list))
    for i,(x,a,b) in enumerate(zip(d["text"],d["a"],d["b"])):
        if a!=b:
            by_prompt[x][tuple(sorted((a,b)))].append((i,a>b))
    jobs=[]
    for edges in by_prompt.values():
        if len(edges)<3: continue
        nodes=sorted({m for edge in edges for m in edge})
        triangles=[t for t in combinations(nodes,3) if all(e in edges for e in combinations(t,2))]
        if triangles: jobs.append((edges,triangles))
    needed=sorted({i for edges,ts in jobs for t in ts for e in combinations(t,2) for i,_ in edges[e]})
    scalar=np.full(len(folds),np.nan)
    attrs=categories(d["text"]).to_numpy()
    for f in np.unique(folds[needed]):
        train=folds!=f
        ix=np.array([i for i in needed if folds[i]==f])
        G=semantic_design(emb,attrs,train,11,seed+int(f))
        cfg=json.loads((reference/f"fold_{f}.json").read_text())["chosen"]["mixture"]
        model,_=scalar_fit(cfg,G[train],d["a"][train],d["b"][train],d["y"][train],d["M"],seed+int(f))
        scalar[ix]=model.predict_proba(G[ix],d["a"][ix],d["b"][ix])
        print("Completed fold",f,flush=True)
    assert np.isfinite(scalar[needed]).all()
    rows=[]
    for group,(edges,triangles) in enumerate(jobs):
        for t in triangles:
            rows.append(dict(prompt_index=group,**evaluate(edges,t,d["y"],mix,scalar)))
    out=ROOT/"results/arena_menus"
    out.mkdir(exist_ok=True)
    df=pd.DataFrame(rows)
    df.to_csv(out/"triangles.csv",index=False)
    print("Eligible votes",len(needed),"exact prompts",len(jobs),"triangles",len(df))
    print(df.groupby("prompt_index").mean().mean().to_string())
    print(df[["disagree","mix_reversals","scalar_reversals"]].sum().to_string())
    print("Minimum votes per edge:",df.min_edge_votes.value_counts().to_dict())


if __name__=="__main__":
    parser=ArgumentParser(description=__doc__)
    parser.add_argument("--reference",type=Path,default=ROOT/"results/nested")
    with threadpool_limits(1):
        main(parser.parse_args().reference)
