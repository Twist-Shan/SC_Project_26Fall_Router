"""Partially observed three-candidate menus sharing an exact first prompt.

A menu enters if at least two of three edges have decisive human votes.
For a missing edge use the full reciprocal [0,1] interval. Bound the paired
policy difference with that same completion, not separate bounds for each policy.
"""
from argparse import ArgumentParser
from collections import defaultdict
from itertools import combinations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from arena_pipeline import categories,load_data
from semantic_refinement import check_inputs
from semantic_mixture import semantic_design
from semantic_link_comparison import scalar_fit
from mixbt import StyleMixtureBT


def run(reference):
    d=load_data()
    with np.load(reference/"oof_predictions.npz") as base,np.load(ROOT/"data/semantic_minilm.npz") as cache:
        emb,fold,seed=check_inputs(d,base,cache,reference)
    groups=defaultdict(lambda:defaultdict(list))
    for i,(x,a,b,y) in enumerate(zip(d["text"],d["a"],d["b"],d["y"])):
        if a!=b:groups[x][tuple(sorted((a,b)))].append((i,y if a<b else 1-y))
    jobs=defaultdict(list)
    for edges in groups.values():
        if len(edges)<2:continue
        menus=set()
        neighbours=defaultdict(set)
        for a,b in edges:
            neighbours[a].add(b)
            neighbours[b].add(a)
        for a,ns in neighbours.items():
            menus.update(tuple(sorted((a,b,c))) for b,c in combinations(ns,2))
        if menus:
            i=next(iter(edges.values()))[0][0]
            jobs[fold[i]].append((i,edges,sorted(menus)))
    attrs=categories(d["text"]).to_numpy()
    out=[]
    group=0
    for f,tasks in sorted(jobs.items()):
        train=fold!=f
        G=semantic_design(emb,attrs,train,11,seed+int(f))
        cfg=json.loads((reference/f"fold_{f}.json").read_text())["chosen"]["mixture"]
        args=tuple(d[key][train] for key in ("a","b","y"))
        mix=StyleMixtureBT(**cfg,style=None,n_init=3,seed=seed+int(f),max_iter=30,polish_all=True)
        mix.fit(G[train],*args,M=d["M"])
        scalar,_=scalar_fit(cfg,G[train],*args,d["M"],seed+int(f))
        for i,edges,menus in tasks:
            group+=1
            weights=mix.gate(G[i:i+1])[0]
            rs=mix.R_
            probs=expit(rs[:,None,:]-rs[None,:,:])@weights
            utility=scalar.R_@scalar.gate(G[i:i+1])[0]
            for t in menus:
                ix=list(t)
                m=int(np.argmax(probs[np.ix_(ix,ix)].sum(axis=1)))
                s=int(np.argmax(utility[ix]))
                diff=[];count=sum(z in edges for z in combinations(t,2))
                for endpoint in (0.,1.):
                    points=np.zeros(3)
                    for a,b in combinations(range(3),2):
                        edge=(t[a],t[b])
                        p=np.mean([v for _,v in edges[edge]]) if edge in edges else endpoint
                        points[a]+=p/2
                        points[b]+=(1-p)/2
                    diff.append(points[m]-points[s])
                rev=any(m in (a,b) and (a if probs[t[a],t[b]]>=.5 else b)!=m for a,b in combinations(range(3),2))
                out.append(dict(group=group,edges=count,differ=int(m!=s),reversal=int(rev),lo=min(diff),hi=max(diff)))
        print("Completed fold",f,flush=True)
    df=pd.DataFrame(out)
    dest=ROOT/"results/arena_menus";dest.mkdir(exist_ok=True)
    df.to_csv(dest/"partial_menus.csv",index=False)
    for edges in (2,3,None):
        subset=df if edges is None else df[df.edges==edges]
        print(edges,len(subset),subset.group.nunique(),subset.differ.sum(),subset.reversal.sum(),
              subset.groupby("group")[["lo","hi"]].mean().mean().tolist())


if __name__=="__main__":
    parser=ArgumentParser(description=__doc__)
    parser.add_argument("--reference",type=Path,default=ROOT/"results/nested")
    with threadpool_limits(1):run(parser.parse_args().reference)
