#!/usr/bin/env python3
"""Evaluate ANN recall@K versus exact neighbors, latency, and throughput."""
from __future__ import annotations
import argparse
import statistics
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np
from qmed.retrieval.faiss_index import RetrievalIndex
from qmed.utils.config import write_json

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index-dir",required=True)
    p.add_argument("--k",type=int,default=10)
    p.add_argument("--queries",type=int,default=1000)
    p.add_argument("--seed",type=int,default=0)
    p.add_argument("--out",default="outputs/index_benchmark.json")
    args=p.parse_args()
    index=RetrievalIndex.load(args.index_dir)
    reference=RetrievalIndex('exact').build(index.vectors,index.metadata)
    rng=np.random.default_rng(args.seed)
    sample=rng.choice(len(index.vectors),min(args.queries,len(index.vectors)),replace=False)
    latencies=[]
    recalls=[]
    for i in sample:
        img_id=str(index.metadata.iloc[int(i)].image_id)
        q=index.vectors[int(i)]
        gold=reference.search(q,args.k,exclude_id=img_id)
        start=time.perf_counter()
        actual=index.search(q,args.k,exclude_id=img_id)
        latencies.append(1000*(time.perf_counter()-start))
        true_ids={r['image_id'] for r in gold}
        predicted={r['image_id'] for r in actual}
        recalls.append(len(true_ids & predicted)/len(true_ids) if true_ids else float('nan'))
    latency=np.array(latencies)
    output={"n_queries":len(sample),"index_engine":index.engine,
            "mean_ann_recall_at_k":float(np.nanmean(recalls)),"k":args.k,
            "mean_search_latency_ms":float(latency.mean()),"median_search_latency_ms":float(np.median(latency)),
            "p95_search_latency_ms":float(np.quantile(latency,.95)),
            "search_throughput_queries_per_second":float(1000*len(latency)/latency.sum()),
            "query_encoding_included":False,"index_construction_included":False,
            "note":"Only search, metadata filtering and reranking are timed. This is NOT full image-to-result latency."}
    write_json(args.out,output)
    print(output)

if __name__=='__main__':
    main()
