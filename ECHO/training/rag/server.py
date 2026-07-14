"""ECHO RAG sidecar: FAISS over search-cache keys; /retrieve, /add (durable), /stats."""
import argparse
import fcntl
import json
import os
import threading
import time
from typing import List, Optional

import faiss
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer


def load_cache(corpus_path: str) -> dict:
    with open(corpus_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_model(model_path: str, use_fp16: bool = False):
    AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    model.eval()
    model.cuda()
    if use_fp16:
        model = model.half()
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
    return model, tokenizer


def pooling(pooler_output, last_hidden_state, attention_mask=None, pooling_method="mean"):
    if pooling_method == "mean":
        last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    if pooling_method == "cls":
        return last_hidden_state[:, 0]
    if pooling_method == "pooler":
        return pooler_output
    raise NotImplementedError(f"Pooling method not implemented: {pooling_method}")


class Encoder:
    def __init__(self, model_name, model_path, pooling_method, max_length, use_fp16):
        self.model_name = model_name
        self.model_path = model_path
        self.pooling_method = pooling_method
        self.max_length = max_length
        self.use_fp16 = use_fp16
        self.model, self.tokenizer = load_model(model_path=model_path, use_fp16=use_fp16)
        self.model.eval()

    @torch.no_grad()
    def encode(self, query_list: List[str], is_query=True) -> np.ndarray:
        if isinstance(query_list, str):
            query_list = [query_list]

        if "e5" in self.model_name.lower():
            if is_query:
                query_list = [f"query: {q}" for q in query_list]
            else:
                query_list = [f"passage: {q}" for q in query_list]

        if "bge" in self.model_name.lower() and is_query:
            query_list = [f"Represent this sentence for searching relevant passages: {q}" for q in query_list]

        inputs = self.tokenizer(
            query_list,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        inputs = {k: v.cuda() for k, v in inputs.items()}

        if "T5" in type(self.model).__name__:
            decoder_input_ids = torch.zeros((inputs["input_ids"].shape[0], 1), dtype=torch.long).to(
                inputs["input_ids"].device
            )
            output = self.model(**inputs, decoder_input_ids=decoder_input_ids, return_dict=True)
            query_emb = output.last_hidden_state[:, 0, :]
        else:
            output = self.model(**inputs, return_dict=True)
            query_emb = pooling(
                output.pooler_output,
                output.last_hidden_state,
                inputs["attention_mask"],
                self.pooling_method,
            )
            if "dpr" not in self.model_name.lower():
                query_emb = torch.nn.functional.normalize(query_emb, dim=-1)

        return query_emb.detach().cpu().numpy().astype(np.float32, order="C")


class CacheKeyRetriever:
    def __init__(self, corpus_path: str, encoder: Encoder, batch_size: int):
        self.corpus_path = corpus_path
        self.encoder = encoder
        self.batch_size = batch_size
        self._lock = threading.Lock()
        self._lock_file = corpus_path + ".lock"

        cache = load_cache(corpus_path)
        self.keys: List[str] = list(cache.keys())
        self.values: List[str] = [cache[k] for k in self.keys]
        print(f"Loaded {len(self.keys)} cache entries from {corpus_path}")

        emb = self._encode_corpus(self.keys)
        self.dim = emb.shape[1]
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(emb)
        print(f"FAISS IndexFlatIP built (dim={self.dim}, ntotal={self.index.ntotal})")

        self._n_requests = 0
        self._n_hits = 0
        self._log_every = 100

    def _encode_corpus(self, texts: List[str]) -> np.ndarray:
        out = []
        for start in tqdm(range(0, len(texts), self.batch_size), desc="Encoding cache keys"):
            batch = texts[start : start + self.batch_size]
            out.append(self.encoder.encode(batch, is_query=False))
        return np.concatenate(out, axis=0).astype(np.float32, order="C")

    def _acquire_file_lock(self, timeout: int = 30):
        start = time.time()
        lock_fd = open(self._lock_file, "w+")
        while True:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return lock_fd
            except OSError:
                if time.time() - start > timeout:
                    lock_fd.close()
                    raise TimeoutError(f"Failed to acquire {self._lock_file} within {timeout}s")
                time.sleep(0.05)

    def _release_file_lock(self, lock_fd) -> None:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        lock_fd.close()

    def _persist(self, query: str, value: str) -> None:
        lock_fd = self._acquire_file_lock()
        try:
            on_disk = load_cache(self.corpus_path) if os.path.exists(self.corpus_path) else {}
            on_disk[query] = value
            tmp_path = self.corpus_path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(on_disk, f, ensure_ascii=False)
            os.replace(tmp_path, self.corpus_path)
        finally:
            self._release_file_lock(lock_fd)

    def retrieve(self, queries: List[str], topk: int, threshold: Optional[float]):
        q_emb = self.encoder.encode(queries, is_query=True)
        with self._lock:
            scores, idxs = self.index.search(q_emb, k=topk)
            results = []
            for q_scores, q_idxs in zip(scores.tolist(), idxs.tolist()):
                hits = []
                for s, i in zip(q_scores, q_idxs):
                    if i < 0:
                        continue
                    hits.append({"key": self.keys[i], "value": self.values[i], "score": float(s)})
                results.append(hits)

            if threshold is not None:
                self._n_requests += len(queries)
                for r in results:
                    if r and r[0]["score"] >= threshold:
                        self._n_hits += 1
                if self._n_requests >= self._log_every:
                    rate = self._n_hits / max(1, self._n_requests)
                    print(f"rag/hit_rate threshold={threshold:.3f} rate={rate:.4f} ({self._n_hits}/{self._n_requests})")
                    self._n_requests = 0
                    self._n_hits = 0

        return results

    def add(self, query: str, value: str):
        emb = self.encoder.encode([query], is_query=False)
        with self._lock:
            self.index.add(emb)
            self.keys.append(query)
            self.values.append(value)
            self._persist(query, value)


class RetrieveRequest(BaseModel):
    queries: List[str]
    topk: int = 1
    return_scores: bool = True
    threshold: Optional[float] = None


class AddRequest(BaseModel):
    query: str
    value: str


def build_app(retriever: CacheKeyRetriever) -> FastAPI:
    app = FastAPI()

    @app.post("/retrieve")
    def retrieve_endpoint(req: RetrieveRequest):
        return {"result": retriever.retrieve(req.queries, topk=req.topk, threshold=req.threshold)}

    @app.post("/add")
    def add_endpoint(req: AddRequest):
        retriever.add(req.query, req.value)
        return {"ok": True, "ntotal": retriever.index.ntotal}

    @app.get("/stats")
    def stats():
        return {"ntotal": retriever.index.ntotal, "dim": retriever.dim}

    return app


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus_path", type=str, required=True)
    p.add_argument("--retriever_model", type=str, default="intfloat/e5-base-v2")
    p.add_argument("--retrieval_method", type=str, default="e5")
    p.add_argument("--pooling_method", type=str, default="mean")
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--use_fp16", action="store_true", default=True)
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=5003)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    encoder = Encoder(
        model_name=args.retrieval_method,
        model_path=args.retriever_model,
        pooling_method=args.pooling_method,
        max_length=args.max_length,
        use_fp16=args.use_fp16,
    )
    retriever = CacheKeyRetriever(
        corpus_path=args.corpus_path,
        encoder=encoder,
        batch_size=args.batch_size,
    )
    app = build_app(retriever)
    print(f"Starting RAG sidecar on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)
