import numpy as np
import pandas as pd
import pytest
from qmed.evaluation.metrics import evaluate_embeddings
from qmed.retrieval.faiss_index import RetrievalIndex


def example_index():
    # 3 separate patients in each class. Correct results rank same label first.
    vec=np.array([[1,0],[.99,.01],[.98,.02],[0,1],[.01,.99],[.02,.98]],dtype=np.float32)
    meta=pd.DataFrame(dict(image_id=[f"im{i}" for i in range(6)],
          patient_id=[f"p{i}" for i in range(6)],
          modality=['CT']*3+['MRI']*3,
          uccs=[0]*3+[2]*3,
          image_path=[f"/none/{i}.png" for i in range(6)]))
    return vec,meta


def test_full_gallery_map_and_recall_denominator():
    vec,meta=example_index()
    m,per=evaluate_embeddings(vec,meta,k_values=[1,2])
    assert m['n_queries']==6
    assert m['mAP']==pytest.approx(1.0)
    assert m['R@1']==pytest.approx(.5)
    assert m['P@1']==pytest.approx(1.0)
    assert m['R@2']==pytest.approx(1.0)


def test_index_save_and_unknown_query_label_not_used(tmp_path):
    vec,meta=example_index()
    index=RetrievalIndex(requested_engine='exact').build(vec,meta)
    found=index.search(vec[0],k=2,exclude_id='im0')
    assert len(found)==2
    assert all(x['image_id']!='im0' for x in found)
    with pytest.raises(ValueError):
        index.search(vec[0],k=2,label_aware=True)
    index.save(tmp_path)
    loaded=RetrievalIndex.load(tmp_path)
    assert loaded.search(vec[0],k=2,exclude_id='im0')[0]['image_id'] in ('im1','im2')


def test_patients_can_be_excluded_without_affecting_label_definition():
    vec,meta=example_index()
    meta.loc[1,'patient_id']='p0'
    m,per=evaluate_embeddings(vec,meta,k_values=[1])
    row=per[per.query_id=='im0'].iloc[0]
    assert row['relevant_gallery']==1
    assert row['R@1']==pytest.approx(1.0)
