import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from make_synthetic_fixtures import generate
from qmed.utils.config import load_config
from qmed.training.trainer import train_model,load_checkpoint
from qmed.data.dataset import MedicalDataset
from qmed.retrieval.embedding import extract_embeddings
from qmed.retrieval.faiss_index import RetrievalIndex
from qmed.evaluation.metrics import evaluate_embeddings


def test_full_train_save_load_retrieve(tmp_path):
    path=generate(tmp_path/"fixtures",patients_per_category=6,size=56)
    config=load_config(Path(__file__).resolve().parents[1]/"configs"/"smoke.yaml")
    config['data']['manifest']=str(path)
    config['training']['max_batches_per_epoch']=1
    checkpoint=train_model(config,tmp_path/"trained",verbose=False)
    assert checkpoint.exists()
    model,cfg,_=load_checkpoint(checkpoint,"cpu")
    ds=MedicalDataset(path,image_size=48,split='test')
    vectors,meta=extract_embeddings(model,ds,'cpu',batch_size=4)
    assert vectors.shape==(len(ds),128)
    assert np.allclose(np.linalg.norm(vectors,axis=1),1,atol=1e-5)
    idx=RetrievalIndex('exact').build(vectors,meta)
    res=idx.search(vectors[0],k=2,exclude_id=str(meta.iloc[0].image_id))
    assert res and res[0]['image_id']!=meta.iloc[0].image_id
    m,_=evaluate_embeddings(vectors,meta,k_values=[5])
    assert 0<=m['mAP']<=1
