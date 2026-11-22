"""End-to-end image-only / four-sequence MRI hybrid retrieval network."""
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F
from .backbone import MultiScaleCNNTransformer
from .quantum import QuantumProjection


class QMedImageRetrievalAI(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        c = config["model"]
        self.feature_dim = c.get("feature_dim", 512)
        self.backbone = MultiScaleCNNTransformer(
            backbone=c.get("backbone", "resnet50"), pretrained=c.get("pretrained", False),
            feature_dim=self.feature_dim, transformer_layers=c.get("transformer_layers", 2),
            heads=c.get("transformer_heads", 8), tokens_per_scale=c.get("transformer_tokens_per_scale", 16),
            dropout=c.get("dropout", .3))
        self.mri_sequence_attention = nn.Sequential(nn.Linear(self.feature_dim, 128), nn.Tanh(), nn.Linear(128, 1))
        self.use_mri_attention = c.get("use_mri_attention", True)
        self.quantum = QuantumProjection(input_dim=self.feature_dim,
                                        qubits=c.get("qubits", 8), layers=c.get("quantum_layers", 4),
                                        output_dim=c.get("embedding_dim", 128),
                                        backend=c.get("quantum_backend", "torch"),
                                        device_name=c.get("quantum_device", "auto"))
        dim = c.get("embedding_dim", 128)
        self.modality_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, 256), nn.GELU(), nn.Dropout(c.get("dropout", .3)), nn.Linear(256, dim))
            for _ in range(3)])
        self.use_alignment = c.get("use_alignment", True)
        self.modality_embedding = nn.Embedding(3, 16)
        self.fusion_gate = nn.Sequential(nn.Linear(dim*2+16, dim), nn.GELU(), nn.Linear(dim, dim), nn.Sigmoid())
        self.refine = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.LayerNorm(dim), nn.Linear(dim, dim))

    def extract_classical(self, images: torch.Tensor, modalities: torch.Tensor):
        if images.ndim != 4 or images.shape[1] != 4:
            raise ValueError("image tensors must have shape [B,4,H,W] with missing channels zero-filled")
        b = len(modalities)
        features = images.new_zeros((b, self.feature_dim))
        seq_attn = None
        other_idxs = torch.where(modalities != 1)[0]
        if len(other_idxs):
            single = self.backbone(images.index_select(0, other_idxs)[:, :1])
            features = features.index_copy(0, other_idxs, single)
        mri_idxs = torch.where(modalities == 1)[0]
        if len(mri_idxs):
            mri = images.index_select(0, mri_idxs)
            bs, nseq, _, h, w = mri.shape[0], mri.shape[1], 1, mri.shape[-2], mri.shape[-1]
            seq_features = self.backbone(mri.reshape(bs*nseq, 1, h, w)).reshape(bs,nseq,-1)
            if self.use_mri_attention:
                weights = torch.softmax(self.mri_sequence_attention(seq_features).squeeze(-1), dim=1)
            else:
                weights = torch.full((bs, nseq), 1.0/nseq, device=seq_features.device, dtype=seq_features.dtype)
            fused_seq = (weights.unsqueeze(-1) * seq_features).sum(dim=1)
            features = features.index_copy(0, mri_idxs, fused_seq)
            seq_attn = weights
        return features, seq_attn

    def forward(self, images, modalities, return_aux=False):
        features, mri_weights = self.extract_classical(images, modalities)
        zq, aux = self.quantum(features, return_aux=True)
        zq = F.normalize(zq, dim=1)
        if self.use_alignment:
            aligned = torch.zeros_like(zq)
            for mod_id, head in enumerate(self.modality_heads):
                inds = torch.where(modalities == mod_id)[0]
                if len(inds):
                    aligned = aligned.index_copy(0, inds, F.normalize(head(zq.index_select(0, inds)), dim=1))
        else:
            aligned = zq
        mod_context = self.modality_embedding(modalities)
        weights = self.fusion_gate(torch.cat([zq, aligned, mod_context], dim=1))
        fused = weights*aligned + (1-weights)*zq
        final = F.normalize(self.refine(fused) + fused, dim=1)
        if return_aux:
            return final, {"classical": features, "quantum": zq, "aligned": aligned,
                           "fusion_gate": weights, "mri_sequence_weights": mri_weights,
                           **aux}
        return final
