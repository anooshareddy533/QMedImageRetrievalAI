from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
from qmed.data.uccs import deep_lesion_category,brats_category,xray_category,shared_categories
from qmed.data.manifest import patient_level_split,validate_manifest,_read_mm
from qmed.data.preprocessing import normalize_image,quality_score


def test_uccs_rules():
    assert deep_lesion_category(30)==0
    assert deep_lesion_category(29.9)==1
    assert deep_lesion_category(None) is None
    assert brats_category(np.array([[0,2],[0,0]]))==2
    assert brats_category(np.array([[0,2],[4,0]]))==0
    assert brats_category(np.zeros((5,5)))==4
    assert xray_category("Nodule|Mass|Effusion")==0
    assert xray_category("Nodule")==1
    assert xray_category("Effusion|Atelectasis")==2
    assert xray_category("Cardiomegaly")==3
    assert xray_category("No Finding")==4
    assert shared_categories('CT','MRI')=={0}


def test_lesion_measurement_from_recist_endpoints():
    row=pd.Series({"Measurement_coordinates":"[0, 0, 20, 0, 0, 0, 0, 10]",
                   "Spacing_mm_px_":"[2,2]"})
    assert abs(_read_mm(row)-40.0)<.01


def test_normalization_and_quality():
    x=normalize_image(np.arange(64).reshape(8,8),32)
    assert x.shape==(1,32,32)
    assert 0 <= float(x.min()) <= float(x.max()) <= 1
    assert 0 <= quality_score(x) <= 1
    assert not torch.isnan(normalize_image(np.zeros((8,8)),32)).any()


def test_patient_disjoint_splitting():
    rows=[]
    for mod,classes in (("CT",[0,1]),("MRI",[0,2,4]),("XRAY",[0,1,2,3,4])):
        for cat in classes:
            for pat in range(8):
                for img in range(2):
                    rows.append(dict(image_id=f"{mod}-{cat}-{pat}-{img}",patient_id=f"{mod}-{cat}-{pat}",
                                     modality=mod,uccs=cat,image_path="fake.png"))
    manifest=patient_level_split(pd.DataFrame(rows),seed=10)
    assert all(manifest.groupby('patient_id').split.nunique()==1)
    assert set(manifest.split.unique())=={'train','val','test'}


def test_official_deeplesion_csv_adapter(tmp_path):
    from qmed.data.manifest import prepare_deeplesion
    Image.fromarray(np.ones((16,16),dtype=np.uint8)*60).save(tmp_path/'a.png')
    Image.fromarray(np.ones((16,16),dtype=np.uint8)*90).save(tmp_path/'b.png')
    df=pd.DataFrame({
       'File_name':['a.png','b.png'],'Patient_index':[1,2],
       'Lesion_index':[1,2],
       'Measurement_coordinates':['[0,0,20,0,0,0,0,0]','[0,0,10,0,0,0,0,0]'],
       'Spacing_mm_px_':['[2,2]','[2,2]']})
    df.to_csv(tmp_path/'DL_info.csv',index=False)
    loaded=prepare_deeplesion(tmp_path)
    assert loaded.uccs.tolist()==[0,1]
    assert loaded.patient_id.nunique()==2


def test_official_chestxray_csv_adapter(tmp_path):
    from qmed.data.manifest import prepare_chestxray
    for name in ('xr1.png','xr2.png'):
        Image.fromarray(np.zeros((16,16),dtype=np.uint8)).save(tmp_path/name)
    pd.DataFrame({'Image Index':['xr1.png','xr2.png'],
                  'Patient ID':[1,2],
                  'Finding Labels':['Mass|Nodule','No Finding']}).to_csv(tmp_path/'Data_Entry_2017_v2020.csv',index=False)
    loaded=prepare_chestxray(tmp_path)
    assert loaded.uccs.tolist()==[0,4]


def test_precomputed_quality_filter_requires_column(tmp_path):
    from qmed.data.dataset import MedicalDataset
    Image.fromarray(np.ones((12,12),dtype=np.uint8)*123).save(tmp_path/'img.png')
    frame=pd.DataFrame({'image_id':['a'],'patient_id':['x'],'modality':['CT'],'uccs':[0],
                        'image_path':[str(tmp_path/'img.png')]})
    import pytest
    with pytest.raises(ValueError):
        MedicalDataset(frame,quality_filter=True)
    frame['quality']=.9
    assert len(MedicalDataset(frame,quality_filter=True,min_quality=.35))==1
