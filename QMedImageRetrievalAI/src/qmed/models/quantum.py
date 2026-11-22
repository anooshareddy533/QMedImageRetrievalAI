"""True differentiable 8-qubit statevector circuit, plus optional PennyLane backend.

Circuit: |+>^8, then 4 data-reuploading layers of RY/RZ, Rot(RX,RY,RZ)
and 8 CNOTs (ring). 8 Z and 28 ZZ Pauli expectations are returned. Torch
backend computes amplitudes and expectations exactly, with complex autodiff.
"""
from __future__ import annotations
import itertools
import math
import torch
import torch.nn as nn


def _rot(axis: str, angle: torch.Tensor) -> torch.Tensor:
    """Create batched [B,2,2] single-qubit unitaries with complex autograd."""
    half = angle * .5
    c, s = torch.cos(half).to(torch.complex64), torch.sin(half).to(torch.complex64)
    zero = torch.zeros_like(c)
    if axis == "Y":
        return torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2)
    if axis == "X":
        return torch.stack([torch.stack([c, -1j*s], -1), torch.stack([-1j*s, c], -1)], -2)
    if axis == "Z":
        return torch.stack([torch.stack([torch.exp(-.5j*angle), zero], -1),
                            torch.stack([zero, torch.exp(.5j*angle)], -1)], -2)
    raise ValueError(axis)


def _single(psi: torch.Tensor, unitary: torch.Tensor, wire: int, n: int) -> torch.Tensor:
    b = psi.shape[0]
    x = psi.reshape(b, 2**wire, 2, 2**(n-wire-1))
    # U_{i,j} psi_{j}; preserves gradients through inputs and trainable angles.
    x = torch.einsum("bij,bpjq->bpiq", unitary, x)
    return x.reshape(b, 2**n)


class TorchStatevectorVQC(nn.Module):
    def __init__(self, qubits=8, layers=4):
        super().__init__()
        if qubits < 2 or qubits > 12:
            raise ValueError("torch simulator supports 2..12 qubits; use 8 for the paper")
        self.n, self.layers = int(qubits), int(layers)
        self.theta = nn.Parameter(torch.randn(layers, qubits, 3) * .03)
        states = torch.arange(2**qubits, dtype=torch.long)
        perm = []
        for control in range(qubits):
            target = (control+1) % qubits
            ctrl_bit = (states >> (qubits-1-control)) & 1
            perm.append(states ^ (ctrl_bit << (qubits-1-target)))
        self.register_buffer("cnot_perms", torch.stack(perm), persistent=False)
        pairs = list(itertools.combinations(range(qubits), 2))
        signs = []
        for i in range(qubits):
            signs.append(1 - 2 * ((states >> (qubits-1-i)) & 1))
        for i,j in pairs:
            parity = ((states >> (qubits-1-i)) & 1) ^ ((states >> (qubits-1-j)) & 1)
            signs.append(1 - 2 * parity)
        self.register_buffer("observable_signs", torch.stack(signs, dim=1).float(), persistent=False)
        self.num_observables = len(signs)

    def forward(self, angles: torch.Tensor):
        b = angles.shape[0]
        if angles.shape[1] != self.n*self.layers*2:
            raise ValueError(f"VQC needs {self.n*self.layers*2} angles, got {angles.shape[1]}")
        if angles.device.type == "cuda":
            # The torch backend works on GPU too; always execute at FP32 precision.
            pass
        uploaded = angles.reshape(b, self.layers, self.n, 2).float()
        initial_amp = 2 ** (-self.n / 2)
        psi = torch.full((b, 2**self.n), initial_amp, dtype=torch.complex64, device=angles.device)
        for layer in range(self.layers):
            for wire in range(self.n):
                psi = _single(psi, _rot("Y", uploaded[:, layer, wire, 0]), wire, self.n)
                psi = _single(psi, _rot("Z", uploaded[:, layer, wire, 1]), wire, self.n)
                for axis_idx, axis in enumerate(("X", "Y", "Z")):
                    theta = self.theta[layer, wire, axis_idx].expand(b)
                    psi = _single(psi, _rot(axis, theta), wire, self.n)
            for wire in range(self.n):
                psi = psi.index_select(1, self.cnot_perms[wire])
        probs = psi.real.square() + psi.imag.square()
        return probs @ self.observable_signs


class PennyLaneVQC(nn.Module):
    """Same circuit in PennyLane; optional extra, differentiable using torch interface."""
    def __init__(self, qubits=8, layers=4, device_name="auto"):
        super().__init__()
        try:
            import pennylane as qml
        except ImportError as error:
            raise ImportError("PennyLane backend requires: pip install -e '.[quantum]'") from error
        self.qml = qml
        self.n, self.layers = int(qubits), int(layers)
        self.theta = nn.Parameter(torch.randn(layers, qubits, 3) * .03)
        if device_name == "auto":
            candidates = ("lightning.gpu", "lightning.qubit", "default.qubit") if torch.cuda.is_available() else ("lightning.qubit", "default.qubit")
        else:
            candidates = (device_name,)
        dev = None
        for name in candidates:
            try:
                dev = qml.device(name, wires=qubits, shots=None)
                self.device_name = name
                break
            except Exception:
                continue
        if dev is None:
            raise RuntimeError(f"No compatible PennyLane device among {candidates}")
        pairs = list(itertools.combinations(range(qubits), 2))
        def circuit(angles, theta):
            for wire in range(qubits):
                qml.Hadamard(wires=wire)
            reshaped = angles.reshape(layers, qubits, 2)
            for layer in range(layers):
                for wire in range(qubits):
                    qml.RY(reshaped[layer, wire, 0], wires=wire)
                    qml.RZ(reshaped[layer, wire, 1], wires=wire)
                    # Paper specifies RX -> RY -> RZ. qml.Rot is RZ->RY->RZ,
                    # therefore explicit gates are required for backend parity.
                    qml.RX(theta[layer, wire, 0], wires=wire)
                    qml.RY(theta[layer, wire, 1], wires=wire)
                    qml.RZ(theta[layer, wire, 2], wires=wire)
                for wire in range(qubits):
                    qml.CNOT(wires=[wire, (wire+1)%qubits])
            return [qml.expval(qml.PauliZ(wire)) for wire in range(qubits)] + [
                qml.expval(qml.PauliZ(i) @ qml.PauliZ(j)) for i,j in pairs]
        self.qnode = qml.QNode(circuit, dev, interface="torch", diff_method="best")
        self.num_observables = qubits + qubits*(qubits-1)//2

    def forward(self, angles):
        # Unrolled rather than qml.batch_input, which disables gradients through trainable inputs.
        results = []
        for sample in angles:
            outs = self.qnode(sample, self.theta)
            results.append(torch.stack(list(outs)) if isinstance(outs, (list, tuple)) else outs)
        return torch.stack(results).to(dtype=torch.float32)


class MatchedClassicalHead(nn.Module):
    """96 trainable parameters when n=8,L=4; nonlinear, differentiable classical control."""
    def __init__(self, qubits=8, layers=4):
        super().__init__()
        self.n, self.layers = qubits, layers
        self.theta = nn.Parameter(torch.randn(layers, qubits, 3) * .03)
        self.num_observables = qubits + qubits*(qubits-1)//2
        self.pairs = list(itertools.combinations(range(qubits), 2))

    def forward(self, angles):
        angles = angles.reshape(-1, self.layers, self.n, 2)
        state = torch.zeros_like(angles[:,0,:,0])
        for i in range(self.layers):
            t = self.theta[i]
            updated = torch.sin(angles[:,i,:,0] + t[:,0]) + torch.cos(angles[:,i,:,1] + t[:,1]) * torch.sin(t[:,2])
            state = torch.tanh(.5 * state + .35 * updated + .15 * torch.roll(state, 1, 1))
        features = [state] + [state[:,i:i+1] * state[:,j:j+1] for i,j in self.pairs]
        return torch.cat(features, dim=1)


class QuantumProjection(nn.Module):
    def __init__(self, input_dim=512, qubits=8, layers=4, output_dim=128,
                 backend="torch", device_name="auto"):
        super().__init__()
        self.n, self.layers = int(qubits), int(layers)
        self.angle_dim = 2*self.n*self.layers
        self.projection = nn.Linear(input_dim, self.angle_dim)
        self.norm = nn.LayerNorm(self.angle_dim)
        if backend == "torch":
            self.vqc = TorchStatevectorVQC(qubits, layers)
        elif backend == "pennylane":
            self.vqc = PennyLaneVQC(qubits, layers, device_name)
        elif backend == "classical_control":
            self.vqc = MatchedClassicalHead(qubits, layers)
        elif backend == "none":
            self.vqc = None
        else:
            raise ValueError(f"unknown quantum backend {backend!r}")
        readout_dim = self.n + self.n*(self.n-1)//2
        self.readout = nn.Linear(self.angle_dim + readout_dim, output_dim)
        self.backend = backend

    def forward(self, features, return_aux=False):
        projected = self.projection(features)
        scaled = math.pi * torch.tanh(self.norm(projected))
        if self.vqc is None:
            quantum = torch.zeros((features.shape[0], self.n + self.n*(self.n-1)//2),
                                  device=features.device, dtype=scaled.dtype)
        else:
            quantum = self.vqc(scaled)
        z = self.readout(torch.cat([projected, quantum], dim=1))
        if return_aux:
            return z, {"angles": scaled, "readout": quantum, "projected": projected}
        return z
