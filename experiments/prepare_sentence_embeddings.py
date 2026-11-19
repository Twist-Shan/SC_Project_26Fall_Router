"""Encode first prompts with a fixed public multilingual MiniLM checkpoint.

Optional encoder environment: PyTorch 2.5.1 and transformers 4.48.3.
Use --model-dir for an offline copy of the five files from the pinned revision.
The resulting NPZ is row-level local data. Publish only aggregate evaluation.
"""
import argparse
import hashlib
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import perf_counter

import numpy as np
import torch
from torch.nn.functional import normalize
from transformers import AutoModel, AutoTokenizer

from arena_pipeline import load_data

ROOT = Path(__file__).resolve().parents[1]
MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
REVISION = "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
MAX_TOKENS = 128
# Hugging Face revision API: sha256 for LFS payloads, Git blob IDs otherwise.
CHECKSUMS = {
    "model.safetensors": "eaa086f0ffee582aeb45b36e34cdd1fe2d6de2bef61f8a559a1bbc9bd955917b",
    "tokenizer.json": "2c3387be76557bd40970cec13153b3bbf80407865484b209e655e5e4729076b8",
    "config.json": "c06d5b49495f044e6380e68a60538be17a6bd5d1",
    "special_tokens_map.json": "2ea7ad0e45a9d1d1591782ba7e29a703d0758831",
    "tokenizer_config.json": "3c1b565ae10a15a1d0c31096f834af2fd9359e91",
}


def check_model(path):
    for name, expected in CHECKSUMS.items():
        source = path / name
        if len(expected) == 40:
            digest = hashlib.sha1(f"blob {source.stat().st_size}\0".encode())
        else:
            digest = hashlib.sha256()
        with source.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise ValueError(f"{name} differs from revision {REVISION}")


def encode(text, model_dir, device, batch_size):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
    model = AutoModel.from_pretrained(model_dir, use_safetensors=True, local_files_only=True,
                                      trust_remote_code=False).to(device).eval()
    unique = list(dict.fromkeys(text))
    positions = {prompt: i for i, prompt in enumerate(unique)}
    order = np.argsort([len(t) for t in unique], kind="stable")
    features = np.empty((len(unique), model.config.hidden_size), dtype=np.float32)
    started = perf_counter()
    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.float16,
                                                enabled=device.type == "cuda"):
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            batch = tokenizer([unique[i] for i in idx], padding=True, truncation=True,
                              max_length=MAX_TOKENS, return_tensors="pt").to(device)
            hidden = model(**batch).last_hidden_state.float()
            mask = batch.attention_mask.unsqueeze(-1).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            features[idx] = normalize(pooled, dim=1).cpu().numpy()
            if start % (batch_size * 40) == 0 or start + batch_size >= len(order):
                print(f"Encoded {min(start + batch_size, len(order))}/{len(order)} unique prompts", flush=True)
    elapsed = perf_counter() - started
    np.testing.assert_allclose(np.linalg.norm(features, axis=1), 1, rtol=1e-5)
    return features[[positions[t] for t in text]], len(unique), elapsed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, help="Local Arena CSV, as in arena_pipeline.load_data")
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "semantic_minilm.npz")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=96)
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("Batch size must be positive")
    model_dir = args.model_dir
    if model_dir is None:
        from huggingface_hub import snapshot_download
        model_dir = Path(snapshot_download(MODEL, revision=REVISION, allow_patterns=list(CHECKSUMS)))
    check_model(model_dir)
    data = load_data(args.data)
    text = data["text"].tolist()
    text_hash = hashlib.sha256(json.dumps(text, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
    device = torch.device(args.device)
    features, n_unique, elapsed = encode(text, model_dir, device, args.batch_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # A completed file becomes visible atomically to the evaluation process.
    temp = None
    try:
        with NamedTemporaryFile(dir=args.output.parent, suffix=".npz", delete=False) as out:
            temp = Path(out.name)
            np.savez(out, features=features, ids=data["ids"], exact_first_prompt_hash=text_hash,
                     model_id=MODEL, revision=REVISION, max_tokens=MAX_TOKENS, unique_prompts=n_unique,
                     precision="amp-fp16/fp32-pooling" if device.type == "cuda" else "fp32",
                     elapsed_seconds=elapsed)
        temp.replace(args.output)
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)
    print(f"{features.shape} embeddings, {n_unique} unique prompts, {elapsed:.1f} s on {device}.")


if __name__ == "__main__":
    main()
