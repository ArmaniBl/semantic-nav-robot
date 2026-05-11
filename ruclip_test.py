from pathlib import Path

import requests
import torch
from PIL import Image
from ruclip import CLIP, RuCLIPProcessor


MODEL_NAME = "ruclip-vit-base-patch32-224"
MODEL_REPO = "ai-forever/ruclip-vit-base-patch32-224"
MODEL_FILES = ("config.json", "bpe.model", "pytorch_model.bin")
MODEL_DIR = Path("/home/arman/test/diplom/.cache/ruclip") / MODEL_NAME
IMAGES_DIR = Path("/home/arman/test/diplom/images")
SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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


def ensure_model_files(session: requests.Session) -> Path:
    base_url = f"https://huggingface.co/{MODEL_REPO}/resolve/main"
    for filename in MODEL_FILES:
        download_file(session, f"{base_url}/{filename}", MODEL_DIR / filename)
    return MODEL_DIR


def find_image_file(images_dir: Path) -> Path:
    if not images_dir.exists():
        raise FileNotFoundError(f"Папка не найдена: {images_dir}")

    candidates = sorted(
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )

    if not candidates:
        raise FileNotFoundError(f"В папке {images_dir} нет поддерживаемых изображений")

    return candidates[0]


def load_image(image_path: Path) -> Image.Image:
    return Image.open(image_path).convert("RGB")


def normalize(vector: torch.Tensor) -> torch.Tensor:
    return vector / vector.norm(dim=-1, keepdim=True)


@torch.no_grad()
def get_image_vector(
    model: CLIP,
    processor: RuCLIPProcessor,
    image: Image.Image,
) -> torch.Tensor:
    inputs = processor(text="", images=[image], return_tensors="pt", padding=True)
    image_vector = model.encode_image(inputs["pixel_values"].to(DEVICE))
    return normalize(image_vector)


@torch.no_grad()
def get_text_vector(
    model: CLIP,
    processor: RuCLIPProcessor,
    text: str,
) -> torch.Tensor:
    inputs = processor(text=[text], return_tensors="pt", padding=True)
    text_vector = model.encode_text(inputs["input_ids"].to(DEVICE))
    return normalize(text_vector)


def cosine_similarity(image_vector: torch.Tensor, text_vector: torch.Tensor) -> float:
    return torch.matmul(image_vector, text_vector.T).item()


def main() -> None:
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})

    model_dir = ensure_model_files(session)
    image_path = find_image_file(IMAGES_DIR)
    image = load_image(image_path)

    model = CLIP.from_pretrained(model_dir).eval().to(DEVICE)
    processor = RuCLIPProcessor.from_pretrained(model_dir)
    image_vector = get_image_vector(model, processor, image)

    print(f"Изображение: {image_path}")
    print(f"Устройство: {DEVICE}")
    print("Введите текст для сравнения с изображением. Пустая строка завершит программу.")

    while True:
        text = input("Текст: ").strip()
        if not text:
            break

        text_vector = get_text_vector(model, processor, text)
        similarity = cosine_similarity(image_vector, text_vector)
        print(f"Cosine similarity: {similarity:.4f}")


if __name__ == "__main__":
    main()
