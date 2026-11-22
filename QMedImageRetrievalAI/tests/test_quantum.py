import math
import torch
from qmed.models.quantum import TorchStatevectorVQC, MatchedClassicalHead,QuantumProjection


def test_eight_qubits_four_layers_dimensions_and_trainable_parameters():
    vqc=TorchStatevectorVQC(8,4)
    assert sum(p.numel() for p in vqc.parameters()) == 96
    assert vqc.num_observables == 36
    x=torch.randn(2,64,requires_grad=True)
    result=vqc(x)
    assert result.shape==(2,36)
    assert torch.isfinite(result).all()
    assert result.abs().max() <= 1.00001
    result.square().sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert vqc.theta.grad is not None and torch.isfinite(vqc.theta.grad).all()


def test_zero_angles_zero_parameters_yield_plus_state_zero_z_expectations():
    vqc=TorchStatevectorVQC(8,4)
    with torch.no_grad():
        vqc.theta.zero_()
    z=vqc(torch.zeros(1,64))
    assert torch.allclose(z,torch.zeros_like(z),atol=2e-5)


def test_projection_residual_and_equal_parameter_control():
    q=QuantumProjection(input_dim=512,qubits=8,layers=4,output_dim=128)
    emb,aux=q(torch.randn(2,512),return_aux=True)
    assert emb.shape == (2,128)
    assert aux["angles"].shape == (2,64)
    assert aux["readout"].shape == (2,36)
    c=MatchedClassicalHead(8,4)
    assert sum(p.numel() for p in c.parameters())==96
    assert c(torch.randn(2,64)).shape==(2,36)


def test_width_depth_variants_retain_expected_shapes():
    for q,l in ((4,2),(6,4),(10,2)):
        v=TorchStatevectorVQC(q,l)
        assert v(torch.randn(1,2*q*l)).shape==(1,q+q*(q-1)//2)
