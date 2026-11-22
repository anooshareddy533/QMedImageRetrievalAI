#!/usr/bin/env python3
"""One command: five-seed training, test embedding, exact metrics and index per seed."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import pandas as pd
from qmed.data.dataset import MedicalDataset
from qmed.evaluation.metrics import evaluate_all_directions,evaluate_embeddings
from qmed.retrieval.embedding import extract_embeddings
from qmed.retrieval.faiss_index import RetrievalIndex
from qmed.training.trainer import train_model,load_checkpoint
from qmed.utils.config import load_config,resolve_device,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/default.yaml')
    p.add_argument('--manifest')
    p.add_argument('--seeds',default='0,1,2,3,4')
    p.add_argument('--out',default='outputs/full')
    args=p.parse_args()
    cfg=load_config(args.config)
    if args.manifest:cfg['data']['manifest']=args.manifest
    root=Path(args.out)
    results=[]
    for seed in [int(i) for i in args.seeds.split(',')]:
        cfg['seed']=seed
        folder=root/f'seed_{seed}'
        path=train_model(cfg,folder)
        model,config,_=load_checkpoint(path,resolve_device(cfg['training']['device']))
        d=config['data']
        ds=MedicalDataset(d['manifest'],image_size=d['image_size'],split='test',
                          roi_policy=d.get('roi_policy','none'),quality_filter=d.get('quality_filter',False),
                          min_quality=d.get('min_quality',.35))
        emb,meta=extract_embeddings(model,ds,next(model.parameters()).device,d['batch_size'],d['workers'])
        idx=RetrievalIndex.from_config(config).build(emb,meta)
        idx.save(folder/'index')
        summary,per=evaluate_embeddings(emb,meta,k_values=config['retrieval']['k_values'])
        per.to_csv(folder/'overall_per_query.csv',index=False)
        by_dir,by_query=evaluate_all_directions(emb,meta,k_values=config['retrieval']['k_values'])
        by_dir.to_csv(folder/'retrieval_directions.csv',index=False)
        by_query.to_csv(folder/'per_query_by_direction.csv',index=False)
        write_json(folder/'metrics.json',summary)
        results.append({'seed':seed,**summary})
        root.mkdir(parents=True,exist_ok=True)
        pd.DataFrame(results).to_csv(root/'summary_by_seed.csv',index=False)
    frame=pd.DataFrame(results)
    metrics=['mAP','MRR']+[f'P@{k}' for k in cfg['retrieval']['k_values']]+[f'R@{k}' for k in cfg['retrieval']['k_values']]
    stats={metric:{'mean':float(frame[metric].mean()),
                   'sd':float(frame[metric].std(ddof=1)) if len(frame)>1 else float('nan')}
           for metric in metrics if metric in frame}
    write_json(root/'summary_mean_sd.json',{'seeds':len(frame),'metrics':stats})
    print('Complete. Genuine experiment outputs:',root.resolve())
    print(stats)

if __name__=='__main__':
    main()
