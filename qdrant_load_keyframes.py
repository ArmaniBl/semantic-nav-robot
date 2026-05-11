#!/usr/bin/env python3
import argparse
import json
import uuid
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models


DEFAULT_EMBEDDINGS_PATH = Path("/home/arman/test/diplom/data/keyframes/embeddings.jsonl")
DEFAULT_QDRANT_URL = "http://127.0.0.1:6333"
DEFAULT_COLLECTION = "semantic_visual_memory"
POINT_ID_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "semantic_visual_memory")


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


def point_id(memory_id: str) -> str:
    return str(uuid.uuid5(POINT_ID_NAMESPACE, memory_id))


def detect_vector_size(records: list[dict[str, Any]]) -> int:
    sizes = {int(record["vector_dim"]) for record in records}
    sizes.update(len(record["vector"]) for record in records)
    if len(sizes) != 1:
        raise ValueError(f"Embedding records have inconsistent vector sizes: {sorted(sizes)}")
    return sizes.pop()


def ensure_collection(
    client: QdrantClient,
    collection_name: str,
    vector_size: int,
    recreate: bool,
) -> None:
    vectors_config = models.VectorParams(
        size=vector_size,
        distance=models.Distance.COSINE,
    )

    if recreate and client.collection_exists(collection_name):
        client.delete_collection(collection_name=collection_name)

    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=vectors_config,
        )
        return

    info = client.get_collection(collection_name=collection_name)
    existing_config = info.config.params.vectors
    if existing_config.size != vector_size:
        raise ValueError(
            f"Collection {collection_name!r} has vector size {existing_config.size}, "
            f"expected {vector_size}. Use --recreate to rebuild it."
        )
    if existing_config.distance != models.Distance.COSINE:
        raise ValueError(
            f"Collection {collection_name!r} uses distance {existing_config.distance}, "
            "expected Cosine. Use --recreate to rebuild it."
        )


def create_payload_indexes(client: QdrantClient, collection_name: str) -> None:
    for field in ("memory_id", "status", "embedding_model", "pose_frame"):
        client.create_payload_index(
            collection_name=collection_name,
            field_name=field,
            field_schema=models.PayloadSchemaType.KEYWORD,
        )


def iter_batches(records: list[dict[str, Any]], batch_size: int):
    for start in range(0, len(records), batch_size):
        yield records[start : start + batch_size]


def build_point(record: dict[str, Any]) -> models.PointStruct:
    memory_id = record["memory_id"]
    payload = dict(record.get("payload") or {})
    payload.setdefault("memory_id", memory_id)
    payload.setdefault("image_path", record.get("image_path"))
    payload.setdefault("embedding_model", record.get("embedding_model"))
    payload.setdefault("similarity_metric", "cosine")

    return models.PointStruct(
        id=point_id(memory_id),
        vector=record["vector"],
        payload=payload,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load RuCLIP keyframe embeddings into a Qdrant collection."
    )
    parser.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS_PATH)
    parser.add_argument("--url", default=DEFAULT_QDRANT_URL)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete and recreate the collection before upload.",
    )
    parser.add_argument(
        "--skip-indexes",
        action="store_true",
        help="Do not create payload indexes for common filters.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.embeddings.exists():
        raise FileNotFoundError(f"Embeddings file not found: {args.embeddings}")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")

    records = read_jsonl(args.embeddings)
    if not records:
        raise ValueError(f"No embedding records found in {args.embeddings}")

    vector_size = detect_vector_size(records)
    client = QdrantClient(url=args.url)
    ensure_collection(client, args.collection, vector_size, args.recreate)
    if not args.skip_indexes:
        create_payload_indexes(client, args.collection)

    uploaded = 0
    for batch in iter_batches(records, args.batch_size):
        points = [build_point(record) for record in batch]
        client.upsert(collection_name=args.collection, points=points, wait=True)
        uploaded += len(points)
        print(f"Uploaded {uploaded}/{len(records)} points")

    count = client.count(collection_name=args.collection, exact=True).count
    print(f"Collection: {args.collection}")
    print(f"Qdrant URL: {args.url}")
    print(f"Vector size: {vector_size}")
    print(f"Uploaded points: {uploaded}")
    print(f"Collection points: {count}")


if __name__ == "__main__":
    main()
