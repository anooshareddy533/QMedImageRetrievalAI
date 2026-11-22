#!/usr/bin/env python3
"""Print installed optional dependencies and quantum backend readiness."""
import importlib.util
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import torch

print('Python:',sys.version.split()[0])
print('PyTorch:',torch.__version__)
print('CUDA available:',torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:',torch.cuda.get_device_name(0))
for dep in ('pennylane','faiss','nibabel','pydicom','qiskit','qiskit_aer','open_clip','transformers','gradio'):
    print(f'{dep}:', 'INSTALLED' if importlib.util.find_spec(dep) else 'not installed (optional)')
from qmed.models.quantum import TorchStatevectorVQC
q=TorchStatevectorVQC(8,4)
assert q.num_observables==36
print('Torch VQC: READY (8 qubits, 4 layers, 96 parameters, 36 readouts)')
