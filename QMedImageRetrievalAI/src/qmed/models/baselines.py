"""Trainable classical torchvision retrieval baselines on identical patient splits/losses.

Foundation models can be added through supported optional OpenCLIP or Hugging Face
vision-only adapters; network and weights must be user-provided/authorized.
"""
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F


class ClassicalRetrievalBaseline(nn.Module):
    def __init__(self, config):
        super().__init__()
        cfg=config["model"]
        name=cfg.get("baseline_backbone","resnet50")
        pretrained=bool(cfg.get("pretrained",False))
        self.name=name
        self.input_size=224 if name in ('vit_b_16','dinov2','biomedclip','clip','medclip') else None
        if name in ("resnet18","resnet50","densenet121","vit_b_16"):
            from torchvision import models
            model_constructors={
                "resnet18":(models.resnet18,models.ResNet18_Weights.DEFAULT),
                "resnet50":(models.resnet50,models.ResNet50_Weights.DEFAULT),
                "densenet121":(models.densenet121,models.DenseNet121_Weights.DEFAULT),
                "vit_b_16":(models.vit_b_16,models.ViT_B_16_Weights.DEFAULT),
            }
            constructor,weights=model_constructors[name]
            self.encoder=constructor(weights=weights if pretrained else None)
            if name.startswith('resnet'):
                dim=self.encoder.fc.in_features
                self.encoder.fc=nn.Identity()
            elif name=='densenet121':
                dim=self.encoder.classifier.in_features
                self.encoder.classifier=nn.Identity()
            else:
                dim=768
                self.encoder.heads=nn.Identity()
        elif name in ("clip","biomedclip"):
            try:
                import open_clip
            except ImportError as error:
                raise ImportError("OpenCLIP models require 'pip install open_clip_torch' and available model weights") from error
            identifier=cfg.get("openclip_model", "ViT-B-32" if name=="clip" else "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
            weights=cfg.get("openclip_pretrained", "openai" if name=="clip" else None)
            if identifier.startswith("hf-hub:"):
                self.encoder,_=open_clip.create_model_from_pretrained(identifier)
            else:
                self.encoder,_,_=open_clip.create_model_and_transforms(identifier,pretrained=weights)
            dim=int(self.encoder.visual.output_dim)
        elif name in ("dinov2","hf_vision"):
            try:
                from transformers import AutoModel
            except ImportError as error:
                raise ImportError("DINOv2/HF vision baseline requires 'pip install transformers'") from error
            repo=cfg.get("hf_vision_id", "facebook/dinov2-base")
            self.encoder=AutoModel.from_pretrained(repo,trust_remote_code=False)
            dim=int(self.encoder.config.hidden_size)
        else:
            raise ValueError(f"Unknown baseline model: {name}")
        self.project=nn.Linear(dim,int(cfg.get("embedding_dim",128)))
        self.modality_heads=nn.ModuleList([nn.Sequential(nn.Linear(128,256),nn.ReLU(),nn.Linear(256,128)) for _ in range(3)])
        self.mri_attention=nn.Sequential(nn.Linear(dim,64),nn.Tanh(),nn.Linear(64,1))
        self.use_alignment=cfg.get("use_alignment",True)

    def _encode(self,image):
        # 1 grayscale image to RGB, with modality-normalized intensity already [0,1].
        if self.input_size and image.shape[-1] != self.input_size:
            image=F.interpolate(image, size=(self.input_size,self.input_size),mode='bilinear',align_corners=False)
        rgb=image.expand(-1,3,-1,-1)
        if self.name in ('clip','biomedclip'):
            mean=torch.tensor([.48145466,.4578275,.40821073],device=rgb.device).view(1,3,1,1)
            std=torch.tensor([.26862954,.26130258,.27577711],device=rgb.device).view(1,3,1,1)
            features=self.encoder.encode_image((rgb-mean)/std)
        elif self.name in ("dinov2","hf_vision"):
            mean=torch.tensor([.485,.456,.406],device=rgb.device).view(1,3,1,1)
            std=torch.tensor([.229,.224,.225],device=rgb.device).view(1,3,1,1)
            features=self.encoder(pixel_values=(rgb-mean)/std).last_hidden_state[:,0]
        else:
            mean=torch.tensor([.485,.456,.406],device=rgb.device).view(1,3,1,1)
            std=torch.tensor([.229,.224,.225],device=rgb.device).view(1,3,1,1)
            features=self.encoder((rgb-mean)/std)
        return features.float()

    def forward(self,images,modalities):
        count=len(images)
        d=self.project.in_features
        collected=images.new_zeros((count,d))
        non_mri=(modalities!=1).nonzero(as_tuple=True)[0]
        if len(non_mri):
            z=self._encode(images.index_select(0,non_mri)[:,:1])
            collected=collected.index_copy(0,non_mri,z)
        mri=(modalities==1).nonzero(as_tuple=True)[0]
        if len(mri):
            inp=images.index_select(0,mri)
            b,c,h,w=inp.shape
            seq=self._encode(inp.reshape(b*c,1,h,w)).reshape(b,c,-1)
            weights=torch.softmax(self.mri_attention(seq).squeeze(-1),dim=1)
            collected=collected.index_copy(0,mri,(weights.unsqueeze(-1)*seq).sum(dim=1))
        z=F.normalize(self.project(collected),dim=1)
        if self.use_alignment:
            aligned=torch.zeros_like(z)
            for mod,head in enumerate(self.modality_heads):
                indices=(modalities==mod).nonzero(as_tuple=True)[0]
                if len(indices):
                    aligned=aligned.index_copy(0,indices,head(z.index_select(0,indices)))
            z=F.normalize(.5*z+.5*aligned,dim=1)
        return z
