"""Optional E5 retrieval experiment. Heavy dependencies are loaded only on use."""
import hashlib
import json
import os
import tempfile
from pathlib import Path
import numpy as np

MODEL = 'intfloat/multilingual-e5-small'
REVISION = '614241f622f53c4eeff9890bdc4f31cfecc418b3'
WINDOW = 384
STRIDE = 256


def normalized(values):
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] == 0:
        raise ValueError("embedding matrix must have two dimensions")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if not np.isfinite(matrix).all() or not np.isfinite(norms).all() or np.any(norms <= 0):
        raise ValueError('invalid embedding matrix')
    return matrix / norms


def page_scores(chunk_scores, owners, count):
    if len(chunk_scores) != len(owners) or any(type(i) is not int or not 0 <= i < count for i in owners):
        raise ValueError("chunk scores and page owners must align")
    result = np.full(count, -np.inf)
    for score, owner in zip(chunk_scores, owners):
        result[owner] = max(result[owner], float(score))
    return result


def ranking(scores):
    return sorted(range(len(scores)), key=lambda i: (-float(scores[i]), i))


def minmax(scores):
    values = np.asarray(scores)
    span = np.ptp(values)
    return (values - values.min()) / span if span > 0 else np.zeros_like(values)


def fuse(rankings, constant=60):
    if constant < 1:
        raise ValueError('RRF constant must be positive')
    scores = {}
    for order in rankings:
        for place, i in enumerate(dict.fromkeys(order), 1):
            scores[i] = scores.get(i, 0) + 1 / (constant + place)
    return sorted(scores, key=lambda i: (-scores[i], i))


def select(order, passages, k, section=None, boost=2, excluded=None):
    order = [i for i in order if not excluded or passages[i].get('section') not in excluded]
    picked = order[:k]
    if section and boost:
        wanted = [i for i in order if passages[i].get('section') == section][:boost]
        missing = [i for i in wanted if i not in picked]
        removable = [i for i in reversed(picked) if passages[i].get('section') != section]
        for add, drop in zip(missing, removable):
            picked[picked.index(drop)] = add
        picked.sort(key=order.index)
    return [passages[i] for i in picked]


class LocalE5:
    def __init__(self, cache_dir: Path, device=None):
        import torch
        from sentence_transformers import SentenceTransformer
        self.device = device or ('mps' if torch.backends.mps.is_available() else 'cpu')
        torch.set_num_threads(4)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model = SentenceTransformer(MODEL, revision=REVISION, device=self.device,
                                         cache_folder=str(self.cache_dir / 'models'), trust_remote_code=False,
                                         model_kwargs={'use_safetensors': True})
        self.model.max_seq_length = 512

    def encode(self, texts, kind):
        if kind not in ('query', 'passage'):
            raise ValueError('E5 input requires query/passage prefix')
        return normalized(self.model.encode([f'{kind}: {text}' for text in texts], batch_size=16,
            normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False))

    def document(self, passages):
        payload = json.dumps({'model': MODEL, 'revision': REVISION, 'window': WINDOW, 'stride': STRIDE,
                              'texts': [p['text'] for p in passages]}, ensure_ascii=False, sort_keys=True)
        digest = hashlib.sha256(payload.encode()).hexdigest()
        path = self.cache_dir / f'{digest}.npz'
        if path.exists():
            with np.load(path, allow_pickle=False) as cached:
                return normalized(cached['vectors']), cached['owners'].tolist(), 'disk'
        chunks, owners = [], []
        for i, passage in enumerate(passages):
            text = passage['text']
            offsets = self.model.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)['offset_mapping']
            for start in range(0, len(offsets), STRIDE):
                end = min(start + WINDOW, len(offsets))
                chunks.append(text[offsets[start][0]:offsets[end - 1][1]])
                owners.append(i)
                if end == len(offsets):
                    break
        if set(owners) != set(range(len(passages))):
            raise ValueError('empty pages require explicit handling')
        vectors = self.encode(chunks, 'passage')
        with tempfile.NamedTemporaryFile(dir=self.cache_dir, suffix=".npz", delete=False) as temp:
            temporary = Path(temp.name)
            np.savez_compressed(temp, vectors=vectors, owners=np.asarray(owners))
        try:
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        return vectors, owners, 'built'
