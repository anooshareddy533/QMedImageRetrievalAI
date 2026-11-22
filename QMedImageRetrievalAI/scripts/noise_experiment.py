#!/usr/bin/env python3
"""Optional Qiskit Aer noisy-shot simulation of paper's RX-RY-RZ ring-CNOT circuit.

This does not train under noise. It compares readout deviations for explicitly
specified 1-qubit and 2-qubit depolarizing errors versus exact Torch readouts.
"""
from __future__ import annotations
import argparse
import itertools
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np
import torch
from qmed.models.quantum import TorchStatevectorVQC
from qmed.training.trainer import load_checkpoint
from qmed.utils.config import write_json


def noisy_readout(angles, theta, shots=4096, one_qubit_p=.001, two_qubit_p=.01, seed=0):
    try:
        from qiskit import QuantumCircuit, transpile
        from qiskit_aer import AerSimulator
        from qiskit_aer.noise import NoiseModel, depolarizing_error
    except ImportError as e:
        raise ImportError("Install optional qiskit/qiskit-aer: pip install 'qiskit>=1.0' 'qiskit-aer>=0.14'") from e
    n_layers,n,_=theta.shape
    angles=np.asarray(angles).reshape(n_layers,n,2)
    circuit=QuantumCircuit(n,n)
    for wire in range(n): circuit.h(wire)
    for layer in range(n_layers):
        for wire in range(n):
            circuit.ry(float(angles[layer,wire,0]),wire)
            circuit.rz(float(angles[layer,wire,1]),wire)
            for gate_idx,fn in enumerate((circuit.rx,circuit.ry,circuit.rz)):
                fn(float(theta[layer,wire,gate_idx]),wire)
        for wire in range(n): circuit.cx(wire,(wire+1)%n)
    circuit.measure(list(range(n)),list(range(n)))
    noise=NoiseModel()
    if one_qubit_p:
        noise.add_all_qubit_quantum_error(depolarizing_error(one_qubit_p,1),["h","ry","rz","rx"])
    if two_qubit_p:
        noise.add_all_qubit_quantum_error(depolarizing_error(two_qubit_p,2),["cx"])
    simulator=AerSimulator(noise_model=noise)
    transpiled=transpile(circuit,simulator,optimization_level=0)
    result=simulator.run(transpiled,shots=shots,seed_simulator=seed).result()
    counts=result.get_counts()
    singles=np.zeros(n,dtype=float)
    pairs=list(itertools.combinations(range(n),2))
    doubles=np.zeros(len(pairs),dtype=float)
    for bits,count in counts.items():
        bits=bits.replace(" ","")
        signs=np.array([1 if bits[-1-wire]=='0' else -1 for wire in range(n)])
        singles+=signs*count
        doubles+=np.array([signs[i]*signs[j] for i,j in pairs])*count
    return np.r_[singles,doubles]/shots


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",required=True)
    p.add_argument("--out",default="outputs/noise.json")
    p.add_argument("--shots",type=int,default=4096)
    p.add_argument("--samples",type=int,default=5)
    p.add_argument("--p1",type=float,default=.001)
    p.add_argument("--p2",type=float,default=.01)
    args=p.parse_args()
    model,cfg,_=load_checkpoint(args.checkpoint,"cpu")
    head=model.quantum
    if not isinstance(head.vqc,TorchStatevectorVQC):
        raise ValueError("Noise experiment requires torch state-vector circuit weights")
    rng=np.random.default_rng(0)
    inputs=torch.tensor(rng.normal(size=(args.samples,head.angle_dim)),dtype=torch.float32)
    with torch.no_grad():
        clean=head.vqc(inputs).numpy()
    theta=head.vqc.theta.detach().cpu().numpy()
    noisy=np.stack([noisy_readout(inputs[j].numpy(),theta,args.shots,args.p1,args.p2,j)
                    for j in range(args.samples)])
    result={"n_samples":args.samples,"shots":args.shots,"one_qubit_error":args.p1,
            "two_qubit_error":args.p2,"mean_absolute_readout_deviation":float(np.abs(noisy-clean).mean()),
            "clean_readouts":clean.tolist(),"noisy_readouts":noisy.tolist(),
            "limitation":"synthetic feature angles; tests readout noise, not medical retrieval mAP under noise"}
    write_json(args.out,result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('clean_readouts','noisy_readouts')},indent=2))

if __name__=="__main__":
    main()
