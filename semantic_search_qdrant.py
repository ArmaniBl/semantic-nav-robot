#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import requests
from qdrant_client import QdrantClient
from qdrant_client.http import models


MODEL_NAME = "ruclip-vit-base-patch32-224"
MODEL_REPO = "ai-forever/ruclip-vit-base-patch32-224"
MODEL_FILES = ("config.json", "bpe.model", "pytorch_model.bin")
MODEL_DIR = Path("/home/arman/test/diplom/.cache/ruclip") / MODEL_NAME
DEFAULT_QDRANT_URL = "http://127.0.0.1:6333"
DEFAULT_COLLECTION = "semantic_visual_memory"


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


def normalize(vector):
    return vector / vector.norm(dim=-1, keepdim=True)


def build_filter(args: argparse.Namespace) -> models.Filter | None:
    must = []
    must_not = []

    if not args.include_stale:
        must_not.append(
            models.FieldCondition(
                key="status",
                match=models.MatchValue(value="stale"),
            )
        )
    if args.status:
        must.append(
            models.FieldCondition(
                key="status",
                match=models.MatchValue(value=args.status),
            )
        )
    if args.embedding_model:
        must.append(
            models.FieldCondition(
                key="embedding_model",
                match=models.MatchValue(value=args.embedding_model),
            )
        )
    if args.pose_frame:
        must.append(
            models.FieldCondition(
                key="pose_frame",
                match=models.MatchValue(value=args.pose_frame),
            )
        )

    if not must and not must_not:
        return None
    return models.Filter(must=must or None, must_not=must_not or None)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search Qdrant visual memory with a Russian text query using RuCLIP."
    )
    parser.add_argument("query", help="Russian text query, for example: 'найди стену'")
    parser.add_argument("--url", default=DEFAULT_QDRANT_URL)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--include-stale",
        action="store_true",
        help="Include records with payload.status == stale.",
    )
    parser.add_argument("--status", help="Optional exact payload.status filter.")
    parser.add_argument("--embedding-model", help="Optional exact embedding_model filter.")
    parser.add_argument("--pose-frame", help="Optional exact pose_frame filter.")
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device. Default: cuda if available, otherwise cpu.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON instead of text output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.top_k < 1:
        raise ValueError("--top-k must be >= 1")

    import torch
    from ruclip import CLIP, RuCLIPProcessor

    model_dir = ensure_model_files(args.model_dir)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    model = CLIP.from_pretrained(model_dir).eval().to(device)
    processor = RuCLIPProcessor.from_pretrained(model_dir)

    inputs = processor(text=[args.query], return_tensors="pt", padding=True)
    with torch.inference_mode():
        text_vector = model.encode_text(inputs["input_ids"].to(device))
        text_vector = normalize(text_vector).squeeze(0).detach().cpu().tolist()

    client = QdrantClient(url=args.url)
    results = client.query_points(
        collection_name=args.collection,
        query=text_vector,
        query_filter=build_filter(args),
        limit=args.top_k,
        with_payload=True,
        with_vectors=False,
    ).points

    if args.json:
        print(
            json.dumps(
                {
                    "query": args.query,
                    "collection": args.collection,
                    "qdrant_url": args.url,
                    "device": device,
                    "results": [
                        {
                            "id": str(point.id),
                            "score": point.score,
                            "payload": point.payload or {},
                        }
                        for point in results
                    ],
                },
                ensure_ascii=False,
            )
        )
        return

    print(f"Query: {args.query}")
    print(f"Collection: {args.collection}")
    print(f"Qdrant URL: {args.url}")
    print(f"Device: {device}")
    print()

    if not results:
        print("No results.")
        return

    for rank, point in enumerate(results, start=1):
        payload = point.payload or {}
        pose = payload.get("pose", {})
        position = pose.get("position", {})
        print(
            f"{rank}. {payload.get('memory_id')} score={point.score:.4f} "
            f"pose_frame={payload.get('pose_frame')} "
            f"x={position.get('x')} y={position.get('y')}"
        )
        print(f"   image: {payload.get('image_path')}")


if __name__ == "__main__":
    main()
