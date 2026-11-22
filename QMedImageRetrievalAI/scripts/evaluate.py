#!/usr/bin/env python3
"""Exact full-gallery mAP/MRR/P@K/R@K/NDCG, all 9 modality directions."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from qmed.evaluation.metrics import evaluate_all_directions, evaluate_embeddings
from qmed.evaluation.statistics import cluster_bootstrap_mean
from qmed.retrieval.faiss_index import RetrievalIndex
from qmed.utils.config import write_json

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index-dir",required=True)
    p.add_argument("--out",default="outputs/evaluation")
    p.add_argument("--k",type=int,nargs="+",default=[5,10,20])
    p.add_argument("--max-queries",type=int,default=0,help="0=all queries; nonzero for development only")
    p.add_argument("--bootstrap",type=int,default=0,help="Patient-cluster bootstrap resamples (10000 recommended)")
    p.add_argument("--include-same-patient",action="store_true",help="Not recommended: can inflate MRI slice retrieval")
    args=p.parse_args()
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    idx=RetrievalIndex.load(args.index_dir)
    summaries, per=evaluate_all_directions(idx.vectors,idx.metadata,k_values=args.k,
                        max_queries=args.max_queries,exclude_same_patient=not args.include_same_patient)
    summaries.to_csv(out/"retrieval_summary.csv",index=False)
    per.to_csv(out/"per_query.csv",index=False)
    summary,overall=evaluate_embeddings(idx.vectors,idx.metadata,k_values=args.k,
                        max_queries=args.max_queries,exclude_same_patient=not args.include_same_patient)
    overall.to_csv(out/"overall_per_query.csv",index=False)
    write_json(out/"overall.json",summary)
    if args.bootstrap and not overall.empty:
        write_json(out/"overall_bootstrap.json",cluster_bootstrap_mean(overall,repeats=args.bootstrap))
    print("SYNTHETIC INPUTS, IF USED, ARE NOT CLINICAL RESULTS.")
    print(summaries.to_string(index=False))
    print("Overall:",summary)
    print(f"Saved metric tables to {out.resolve()}")

if __name__=="__main__":
    main()
