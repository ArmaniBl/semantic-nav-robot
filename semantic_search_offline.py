#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any

import requests


MODEL_NAME = "ruclip-vit-base-patch32-224"
MODEL_REPO = "ai-forever/ruclip-vit-base-patch32-224"
MODEL_FILES = ("config.json", "bpe.model", "pytorch_model.bin")
MODEL_DIR = Path("/home/arman/test/diplom/.cache/ruclip") / MODEL_NAME
DEFAULT_EMBEDDINGS_PATH = Path("/home/arman/test/diplom/data/keyframes/embeddings.jsonl")


def download_file(session: requests.Session, url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return

    with session.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with destination.open("wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)


def ensure_model_files(model_dir: Path) -> Path:
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})
    base_url = f"https://huggingface.co/{MODEL_REPO}/resolve/main"
    for filename in MODEL_FILES:
        download_file(session, f"{base_url}/{filename}", model_dir / filename)
    return model_dir


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
    return records


def normalize(vector):
    return vector / vector.norm(dim=-1, keepdim=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search recorded keyframes with a Russian text query using RuCLIP."
    )
    parser.add_argument("query", help="Russian text query, for example: 'найди стену'")
    parser.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS_PATH)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--include-stale",
        action="store_true",
        help="Include records with payload.status == stale.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device. Default: cuda if available, otherwise cpu.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import torch
    from ruclip import CLIP, RuCLIPProcessor

    if not args.embeddings.exists():
        raise FileNotFoundError(
            f"Embeddings file not found: {args.embeddings}. "
            "Run ruclip_embed_keyframes.py first."
        )

    model_dir = ensure_model_files(args.model_dir)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    records = read_jsonl(args.embeddings)
    if not args.include_stale:
        records = [
            record
            for record in records
            if record.get("payload", {}).get("status", "active") != "stale"
        ]
    if not records:
        raise ValueError("No searchable embedding records found.")

    model = CLIP.from_pretrained(model_dir).eval().to(device)
    processor = RuCLIPProcessor.from_pretrained(model_dir)

    inputs = processor(text=[args.query], return_tensors="pt", padding=True)
    with torch.inference_mode():
        text_vector = model.encode_text(inputs["input_ids"].to(device))
        text_vector = normalize(text_vector).detach().cpu()

    image_vectors = torch.tensor(
        [record["vector"] for record in records],
        dtype=torch.float32,
    )
    scores = torch.matmul(image_vectors, text_vector.squeeze(0))
    top_k = min(args.top_k, len(records))
    values, indexes = torch.topk(scores, k=top_k)

    print(f"Query: {args.query}")
    print(f"Embeddings: {args.embeddings}")
    print(f"Device: {device}")
    print()

    for rank, (score, index) in enumerate(zip(values.tolist(), indexes.tolist()), start=1):
        record = records[index]
        payload = record.get("payload", {})
        pose = payload.get("pose", {})
        position = pose.get("position", {})
        print(
            f"{rank}. {record['memory_id']} score={score:.4f} "
            f"pose_frame={payload.get('pose_frame')} "
            f"x={position.get('x')} y={position.get('y')}"
        )
        print(f"   image: {record['image_path']}")


if __name__ == "__main__":
    main()
