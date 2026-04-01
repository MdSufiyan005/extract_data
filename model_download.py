"""
Download-only script for main BF16 GGUF models.
Downloads the selected model(s) from Hugging Face into benchmark_data/models.

Usage:
    python download_f16_models.py
    python download_f16_models.py --model llama_8b
    python download_f16_models.py --model qwen_2b
"""

import argparse
import logging
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    hf_hub_download = None

OUTPUT_DIR = Path("benchmark_data")
MODELS_DIR = OUTPUT_DIR / "models"

HF_NAMESPACE = "unsloth"

MODELS = {
    "llama_1b": {
        "display_name": "Llama-3.2-1B",
        "gguf_repo_name": "Llama-3.2-1B-Instruct-GGUF",
        "gguf_base_filename": "Llama-3.2-1B-Instruct",
        "params_b": 1.0,
    },
    "qwen_2b": {
        "display_name": "Qwen3.5-2B",
        "gguf_repo_name": "Qwen3.5-2B-GGUF",
        "gguf_base_filename": "Qwen3.5-2B",
        "params_b": 2.0,
    },
    "llama_3b": {
        "display_name": "Llama-3.2-3B",
        "gguf_repo_name": "Llama-3.2-3B-Instruct-GGUF",
        "gguf_base_filename": "Llama-3.2-3B-Instruct",
        "params_b": 3.0,
    },
    "gemma_4b": {
        "display_name": "Gemma-3-4B-Instruct",
        "gguf_repo_name": "gemma-3-4b-it-GGUF",
        "gguf_base_filename": "gemma-3-4b-it",
        "params_b": 4.0,
    },
    "llama_8b": {
        "display_name": "Llama-3.1-8B",
        "gguf_repo_name": "Llama-3.1-8B-Instruct-GGUF",
        "gguf_base_filename": "Llama-3.1-8B-Instruct",
        "params_b": 8.0,
    },
    "ministral_8b": {
        "display_name": "Ministral-3-8B-Instruct",
        "gguf_repo_name": "Ministral-3-8B-Instruct-2512-GGUF",
        "gguf_base_filename": "Ministral-3-8B-Instruct-2512",
        "params_b": 8.0,
    },
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
)
logger = logging.getLogger("model_downloader")


def download_model(model_key: str) -> Path | None:
    if model_key not in MODELS:
        raise ValueError(f"Unknown model key: {model_key}")

    if hf_hub_download is None:
        logger.error("huggingface_hub is not installed. Run: pip install huggingface_hub")
        return None

    cfg = MODELS[model_key]
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    repo_id = f"{HF_NAMESPACE}/{cfg['gguf_repo_name']}"
    hf_filename = f"{cfg['gguf_base_filename']}-BF16.gguf"
    dest = MODELS_DIR / hf_filename

    if dest.exists():
        logger.info(f"Already present: {dest}")
        return dest

    logger.info(f"Downloading {cfg['display_name']} from {repo_id} ...")

    try:
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=hf_filename,
            local_dir=str(MODELS_DIR),
            local_dir_use_symlinks=False,
        )

        downloaded_path = Path(downloaded)

        # Ensure final file path is exactly what we expect
        if downloaded_path != dest and downloaded_path.exists():
            downloaded_path.rename(dest)

        logger.info(f"Done: {dest}")
        return dest

    except Exception as exc:
        logger.error(f"Download failed for {model_key}: {exc}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Download main BF16 GGUF models only")
    parser.add_argument(
        "--model",
        choices=list(MODELS.keys()),
        default=None,
        help="Download only one model; default downloads all models",
    )
    args = parser.parse_args()

    targets = [args.model] if args.model else list(MODELS.keys())

    logger.info("=" * 70)
    logger.info("Downloading BF16 base GGUF models")
    logger.info(f"Target models: {', '.join(targets)}")
    logger.info("=" * 70)

    for model_key in targets:
        download_model(model_key)

    logger.info("=" * 70)
    logger.info("Finished downloading selected BF16 models.")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()