#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any

import requests
from PIL import Image


MODEL_NAME = "ruclip-vit-base-patch32-224"
MODEL_REPO = "ai-forever/ruclip-vit-base-patch32-224"
MODEL_FILES = ("config.json", "bpe.model", "pytorch_model.bin")
MODEL_DIR = Path("/home/arman/test/diplom/.cache/ruclip") / MODEL_NAME
DEFAULT_METADATA_PATH = Path("/home/arman/test/diplom/data/keyframes/metadata.jsonl")
DEFAULT_OUTPUT_PATH = Path("/home/arman/test/diplom/data/keyframes/embeddings.jsonl")


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


def load_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def normalize(vector):
    return vector / vector.norm(dim=-1, keepdim=True)


def build_payload(record: dict[str, Any], embedding_model: str) -> dict[str, Any]:
    return {
        "memory_id": record["memory_id"],
        "timestamp": record["timestamp"],
        "image_path": record["image_path"],
        "image_topic": record.get("image_topic"),
        "image_frame": record.get("image_frame"),
        "pose_topic": record.get("pose_topic"),
        "pose_frame": record.get("pose_frame"),
        "child_frame_id": record.get("child_frame_id"),
        "pose_age_sec": record.get("pose_age_sec"),
        "pose": record.get("pose"),
        "status": record.get("status", "active"),
        "embedding_model": embedding_model,
        "similarity_metric": "cosine",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute RuCLIP image embeddings for recorded keyframes."
    )
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
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

    if not args.metadata.exists():
        raise FileNotFoundError(f"Metadata file not found: {args.metadata}")

    model_dir = ensure_model_files(args.model_dir)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    records = read_jsonl(args.metadata)
    if not records:
        raise ValueError(f"No keyframe records found in {args.metadata}")

    model = CLIP.from_pretrained(model_dir).eval().to(device)
    processor = RuCLIPProcessor.from_pretrained(model_dir)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp_output = args.output.with_suffix(args.output.suffix + ".tmp")

    with temp_output.open("w", encoding="utf-8") as file:
        for index, record in enumerate(records, start=1):
            image_path = Path(record["image_path"])
            if not image_path.exists():
                raise FileNotFoundError(
                    f"Image for {record['memory_id']} not found: {image_path}"
                )

            image = load_image(image_path)
            inputs = processor(text="", images=[image], return_tensors="pt", padding=True)
            with torch.inference_mode():
                vector = model.encode_image(inputs["pixel_values"].to(device))
                vector = normalize(vector).squeeze(0).detach().cpu()

            embedding = {
                "memory_id": record["memory_id"],
                "image_path": str(image_path),
                "embedding_model": MODEL_REPO,
                "vector_dim": int(vector.numel()),
                "vector": vector.tolist(),
                "payload": build_payload(record, MODEL_REPO),
            }
            file.write(json.dumps(embedding, ensure_ascii=False) + "\n")
            print(f"[{index}/{len(records)}] embedded {record['memory_id']}")

    temp_output.replace(args.output)
    print(f"Saved embeddings: {args.output}")
    print(f"Records: {len(records)}")
    print(f"Device: {device}")


if __name__ == "__main__":
    main()
