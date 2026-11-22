"""FAISS IVF-HNSW / Flat, with explicit exact-search fallback when FAISS absent."""
from __future__ import annotations
import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from qmed.utils.config import write_json

try:
    import faiss
except ImportError:
    faiss = None


def _normalize(x):
    x = np.ascontiguousarray(np.asarray(x, dtype=np.float32))
    if x.ndim == 1:
        x = x[None]
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


class RetrievalIndex:
    def __init__(self, requested_engine="faiss_ivf", nlist=1024, hnsw_m=32, nprobe=16):
        self.requested_engine = requested_engine
        self.nlist, self.hnsw_m, self.nprobe = int(nlist),int(hnsw_m),int(nprobe)
        self.engine = "exact"
        self.index = None
        self.vectors = None
        self.metadata = None

    @classmethod
    def from_config(cls, config):
        c = config["retrieval"]
        return cls(c.get("index", "faiss_ivf"), c.get("nlist", 1024),
                   c.get("hnsw_m", 32), c.get("nprobe", 16))

    def build(self, vectors: np.ndarray, metadata: pd.DataFrame):
        vectors = _normalize(vectors)
        if len(vectors) == 0 or len(vectors) != len(metadata):
            raise ValueError("index requires nonempty aligned vectors and metadata")
        self.vectors = vectors
        self.metadata = metadata.reset_index(drop=True).copy()
        self.engine = "exact"
        self.index = None
        if self.requested_engine != "exact" and faiss is None:
            warnings.warn("FAISS not installed. Falling back to exact cosine search; install faiss-cpu for ANN.")
        if faiss is not None and self.requested_engine in ("faiss_flat", "faiss_ivf"):
            d, n = vectors.shape[1], len(vectors)
            if self.requested_engine == "faiss_flat" or n < 100:
                index = faiss.IndexFlatIP(d)
                self.engine = "faiss_flat"
            else:
                # Standard FAISS training should have ~39 samples per centroid.
                nlist = max(1, min(self.nlist, n//39))
                try:
                    index = faiss.index_factory(d, f"IVF{nlist}_HNSW{self.hnsw_m},Flat", faiss.METRIC_INNER_PRODUCT)
                    self.engine = "faiss_ivf_hnsw"
                except (RuntimeError, AttributeError):
                    quantizer = faiss.IndexFlatIP(d)
                    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
                    self.engine = "faiss_ivf_flat"
                index.train(vectors)
                index.nprobe = min(self.nprobe, nlist)
            index.add(vectors)
            self.index = index
        return self

    def search(self, query_vec, k=10, candidate_multiplier=20, target_modality=None,
               exclude_id=None, exclude_patient=None, query_category=None,
               label_aware=False, alpha=.7, beta=.1, gamma=.2):
        if self.metadata is None:
            raise RuntimeError("build() or load() an index before searching")
        if label_aware and query_category is None:
            raise ValueError("Label-aware reranking requires an EXPLICIT query_category; never use withheld ground truth")
        q = _normalize(query_vec)
        if len(q) != 1:
            raise ValueError("single-query search expects [128] or [1,128]")
        # Request extra candidates to allow filtering and reranking.
        ask = min(len(self.vectors), max(int(k)*int(candidate_multiplier), int(k)))
        if self.index is None:
            scores = self.vectors @ q[0]
            order = np.argsort(-scores, kind="stable")[:ask]
        else:
            _, ids = self.index.search(q, ask)
            order = ids[0][ids[0] >= 0]
            scores = None
        def valid(i):
            row = self.metadata.iloc[int(i)]
            if target_modality and row.modality != target_modality:
                return False
            if exclude_id and str(row.image_id) == str(exclude_id):
                return False
            if exclude_patient and str(row.patient_id) == str(exclude_patient):
                return False
            return True
        selected = [int(i) for i in order if valid(i)]
        if len(selected) < int(k):
            # Filters sometimes remove most approximate candidates. Exact fallback
            # retrieves enough valid neighbors without misreporting ANN recall.
            scores = self.vectors @ q[0]
            order = np.argsort(-scores, kind="stable")
            selected = [int(i) for i in order if valid(i)][:max(ask, int(k))]
        if not selected:
            return []
        scores = self.vectors[selected] @ q[0]
        rows = []
        for i, score in zip(selected, scores):
            row = self.metadata.iloc[i]
            modality_score = float(row.modality == target_modality) if target_modality else 0.0
            relevance_score = float(int(row.uccs) == int(query_category)) if label_aware else 0.0
            ranked = alpha*float(score)+beta*modality_score+gamma*relevance_score if label_aware else float(score)
            rows.append({"image_id":str(row.image_id), "patient_id":str(row.patient_id),
                         "modality":str(row.modality), "uccs":int(row.uccs),
                         "image_path":str(row.get("image_path", "")),
                         "cosine_similarity":float(score), "ranking_score":ranked})
        rows.sort(key=lambda x:x["ranking_score"], reverse=True)
        return rows[:k]

    def save(self, directory):
        folder = Path(directory)
        folder.mkdir(parents=True, exist_ok=True)
        np.save(folder / "embeddings.npy", self.vectors)
        self.metadata.to_csv(folder / "metadata.csv", index=False)
        if self.index is not None:
            faiss.write_index(self.index, str(folder / "index.faiss"))
        write_json(folder / "index_config.json", {
            "requested_engine":self.requested_engine, "actual_engine":self.engine,
            "nlist":self.nlist, "nprobe":self.nprobe, "hnsw_m":self.hnsw_m,
            "rows":len(self.vectors), "dimension":self.vectors.shape[1]})
        return folder

    @classmethod
    def load(cls, folder):
        folder = Path(folder)
        with open(folder / "index_config.json", encoding="utf-8") as f:
            config = json.load(f)
        obj = cls(config["requested_engine"], config["nlist"], config["hnsw_m"], config["nprobe"])
        obj.vectors = _normalize(np.load(folder / "embeddings.npy", allow_pickle=False))
        obj.metadata = pd.read_csv(folder / "metadata.csv", keep_default_na=False)
        obj.engine = config["actual_engine"]
        if (folder / "index.faiss").is_file():
            if faiss is None:
                warnings.warn("FAISS index exists but FAISS is not installed: exact-search fallback.")
                obj.engine = "exact"
            else:
                obj.index = faiss.read_index(str(folder / "index.faiss"))
        return obj
