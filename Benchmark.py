"""
Edge Device Benchmark Collector (RPi 4 / Jetson Nano / Intel PC) - KL DIVERGENCE
=================================================================================
Benchmarks GGUF quantized models on edge devices and computes:
- prompt TPS
- generation TPS
- load time
- RAM usage
- approximate symmetric KL divergence vs original base model

Notes:
- KL divergence is computed using llama-cpp-python (not llama.cpp CLI).
- It compares next-token distributions on a small prompt set.
- Lower KL divergence is better (closer to original model).
"""

import argparse
import csv
import json
import logging
import math
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict,Optional
import gc
import json
import math
import subprocess
import time
import psutil

# tqdm fallback
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        desc = kwargs.get("desc", "")
        items = list(iterable)
        print(f"{desc}: {len(items)} items")
        return items

# huggingface_hub
try:
    from huggingface_hub import hf_hub_download
except ImportError:
    print("[ERROR] pip install huggingface_hub")
    sys.exit(1)

# llama-cpp-python
try:
    from llama_cpp import Llama
except ImportError:
    print("[ERROR] pip install llama-cpp-python")
    sys.exit(1)

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

OUTPUT_DIR = Path("Inference_collData")
QUANTS_DIR = OUTPUT_DIR / "quants"
RESULTS_FILE = OUTPUT_DIR / "benchmark_results.json"
CSV_FILE = OUTPUT_DIR / "benchmark_results.csv"
LOG_FILE = OUTPUT_DIR / "collection.log"

LLAMA_BENCH = Path("external/llama.cpp/build/bin/llama-bench")
LLAMA_CLI = Path("external/llama.cpp/build/bin/llama-cli")

# HF_NAMESPACE = "Haider584"
HF_NAMESPACE = "unsloth"

# Local base/original models for KL divergence
BASE_MODELS_DIR = Path("benchmark_data/models")
BASE_MODEL_PATHS = {
    "llama_1b": BASE_MODELS_DIR / "llama-3.2-1b-bf16.gguf",
    "llama_3b": BASE_MODELS_DIR / "llama-3.2-3b-bf16.gguf",
    "llama_8b": BASE_MODELS_DIR / "llama-3.1-8b-f16.gguf",
    "qwen_2b": BASE_MODELS_DIR / "qwen3.5-2b-bf16.gguf"
}

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
        "hf_repo":      "unsloth/gemma-3-4b-it-GGUF",
        "hf_filename":  "gemma-3-4b-it-BF16.gguf",         
        "local_name":   "gemma-3-4b-it-bf16.gguf",
        "params_b":     4.0,
    },    
    "llama_8b": {
        "display_name": "Llama-3.1-8B",
        "gguf_repo_name": "Llama-3.1-8B-Instruct-GGUF",
        "gguf_base_filename": "Llama-3.1-8B-Instruct",
        "params_b": 8.0,
    },
    "ministral_8b": {  
        "display_name": "Ministral-3-8B-Instruct",
        "hf_repo":      "unsloth/Ministral-3-8B-Instruct-2512-GGUF",
        "hf_filename":  "Ministral-3-8B-Instruct-2512-BF16.gguf",
        "local_name":   "ministral-3-8b-bf16.gguf",
        "params_b":     8.0,
    },
}

MODEL_METADATA = {
    "llama_1b": {
        "architecture_family": "Llama",
        "hidden_size": 2048,
        "n_layers": 16,
        "n_heads": 32,
        "n_kv_heads": 8,
        "vocab_size": 128256,
        "context_length": 131072,
        "gqa": True,
    },
    "llama_3b": {
        "architecture_family": "Llama",
        "hidden_size": 2560,
        "n_layers": 28,
        "n_heads": 32,
        "n_kv_heads": 8,
        "vocab_size": 128256,
        "context_length": 131072,
        "gqa": True,
    },
    "llama_8b": {
        "architecture_family": "Llama",
        "hidden_size": 4096,
        "n_layers": 32,
        "n_heads": 32,
        "n_kv_heads": 8,
        "vocab_size": 128256,
        "context_length": 131072,
        "gqa": True,
    },
    "qwen_2b": {
        "architecture_family": "Qwen3.5",
        "hidden_size": 2048,
        "n_layers": 24,
        "n_heads": 16,  
        "n_kv_heads": 2,
        "vocab_size": 248320,
        "context_length": 262144,
        "gqa": True,  
    },
    "gemma_3_4b": {
    "architecture_family": "Gemma3",
    "hidden_size": 2560,
    "n_layers": 34,
    "n_heads": 8,
    "n_kv_heads": 4,
    "vocab_size": 262208,
    "context_length": 131072,
    "gqa": True,
    },
    "ministral_8b": {
    "architecture_family": "Ministral",
    "hidden_size": 4096,
    "n_layers": 32,
    "n_heads": 32,
    "n_kv_heads": 8,
    "vocab_size": 131072,
    "context_length": 128000,
    "gqa": True,
    },

}

QUANT_TYPES = [
    "Q2_K", "Q3_K_M", "Q4_0", "Q4_K_M","Q4_K_S","Q5_K_M", "Q5_K_S", "Q6_K", "Q8_0", "IQ4_XS"
]

# ── approximate model sizes (MB) per quant — used for is_feasible() checks ─────
# Updated March 2026 with real Unsloth GGUF sizes (more accurate than before)
APPROX_SIZES_MB = {
    1.0: {   # Llama-3.2-1B
        "Q2_K": 580, "Q3_K_M": 720, "Q4_0": 773, "Q4_K_M": 808, "Q4_K_S": 776,
        "Q5_K_M": 912, "Q5_K_S": 893, "Q6_K": 1024, "Q8_0": 1320, "IQ4_XS": 743
    },
    2.0: {   # Qwen3.5-2B / Qwen2.5-3B class (your new small Qwen)
        "Q2_K": 950, "Q3_K_M": 1250, "Q4_0": 1480, "Q4_K_M": 1580, "Q4_K_S": 1520,
        "Q5_K_M": 1850, "Q5_K_S": 1780, "Q6_K": 2150, "Q8_0": 2850, "IQ4_XS": 1450
    },
    3.0: {   # Llama-3.2-3B
        "Q2_K": 1320, "Q3_K_M": 1720, "Q4_0": 2100, "Q4_K_M": 2200, "Q4_K_S": 2120,
        "Q5_K_M": 2520, "Q5_K_S": 2430, "Q6_K": 2920, "Q8_0": 3850, "IQ4_XS": 2080
    },
    4.0: {   # Gemma-3-4B
        "Q2_K": 1772,"Q3_K_M": 2150,"Q4_0": 2427,"Q4_K_S": 2437,"Q4_K_M": 2550,
        "Q5_K_S": 2826,"Q5_K_M": 2898,"Q6_K": 3267,"Q8_0": 4229,"IQ4_XS": 2314,
    },
    8.0: {   # Llama-3.1-8B + Ministral-3-8B (shared — very similar sizes)
        "Q2_K": 3180, "Q3_K_M": 4020, "Q4_0": 4940, "Q4_K_M": 5200, "Q4_K_S": 4950,
        "Q5_K_M": 6060, "Q5_K_S": 5920, "Q6_K": 6970, "Q8_0": 8700, "IQ4_XS": 4710
    },
}


DEVICES = {
    "rpi4": {
        "display_name": "Raspberry Pi 4 (4GB)",
        "ram_mb": 3900,
        "cpu_threads": 4,
        "has_gpu": False,
        "gpu_layers": 0,
    },
    "jetson_nano": {
        "display_name": "Jetson Nano (4GB shared)",
        "ram_mb": 3200,
        "cpu_threads": 4,
        "has_gpu": True,
        "gpu_layers": 16,
    },
    "intel_pc": {
        "display_name": "Intel PC (7.5GB RAM)",
        "ram_mb": 7500,
        "cpu_threads": 4,
        "has_gpu": False,
        "gpu_layers": 0,
    },
}

# EVAL_PROMPT = "Explain the difference between a transformer and an LSTM neural network in three concise paragraphs."
EVAL_TOKENS_PROMPT = 512
EVAL_TOKENS_GEN = 128

# KL settings
KL_CTX = 512
KL_TOP_K = 50
# KL_TOP_K = 100
KL_EPS = 1e-12
KL_PROMPTS = [
    "Explain the difference between a transformer and an LSTM in one paragraph.",
    "What is model quantization and why is it useful for edge deployment?",
    "Write a short Python function to reverse a list.",
    "Why does self-attention help language models capture long-range dependencies?",
    "Give two advantages of smaller models for on-device inference.",
    "Describe perplexity in simple terms.",
    "What is the main tradeoff between accuracy and speed in neural network compression?",
    "Summarize how quantization can reduce memory usage.",
]
# KL_PROMPTS = [
#     # Technical explanation
#     "Explain the difference between a transformer and an LSTM in one paragraph.",
#     "What is model quantization and why is it useful for edge deployment?",

#     # Structured output
#     "List three advantages of smaller models for on-device inference.",
#     "Return a JSON object with keys: model, quant, memory.",

#     # Math / exact-token
#     "What is 17 multiplied by 23?",
#     "Solve: 3x + 5 = 20. x =",

#     # Open continuation / reasoning
#     "The main tradeoff in neural network compression is",
#     "Self-attention helps language models because",
# ]

CSV_COLUMNS = [
    "device", "model_key", "model_display", "quant_type", "params_b", "model_size_mb",
    "prompt_tps", "gen_tps", "latency_first_token_ms",
    "peak_ram_mb", "load_time_s", "kl_divergence",
    "success", "oom", "error_msg", "timestamp",
    "architecture_family", "hidden_size", "n_layers", "n_heads", "n_kv_heads",
    "vocab_size", "context_length", "gqa",
]

# ══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════════════════════════════════════════

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger("edge_benchmark_collector")

# ══════════════════════════════════════════════════════════════════════════════
# DEVICE / SYSTEM HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def detect_device() -> str:
    machine = platform.machine().lower()

    try:
        if Path("/proc/device-tree/model").exists():
            dt_model = Path("/proc/device-tree/model").read_text(errors="ignore").lower()
            if "jetson" in dt_model:
                logger.info("Device detected: Jetson Nano")
                return "jetson_nano"
    except Exception:
        pass

    try:
        if Path("/proc/cpuinfo").exists():
            cpuinfo = Path("/proc/cpuinfo").read_text(errors="ignore").lower()
            if "raspberry pi" in cpuinfo or "bcm2" in cpuinfo:
                logger.info("Device detected: Raspberry Pi 4")
                return "rpi4"
    except Exception:
        pass

    if machine in ("x86_64", "amd64", "i686"):
        logger.info("Device detected: Intel PC")
        return "intel_pc"

    logger.warning("Unknown device → defaulting to intel_pc")
    return "intel_pc"


def resolve_bin(default_path: Path, name: str) -> Optional[str]:
    if default_path.exists():
        return str(default_path)
    return shutil.which(name)


def is_feasible(model_key: str, quant_type: str, device_key: str) -> tuple[bool, str]:
    model = MODELS[model_key]
    device = DEVICES[device_key]
    sizes = APPROX_SIZES_MB.get(model["params_b"], {})
    size_mb = sizes.get(quant_type)

    if size_mb is None:
        return False, "Unknown approximate size"

    if size_mb * 1.15 > device["ram_mb"]:
        return False, f"OOM expected (~{size_mb * 1.15:.0f}MB needed)"

    return True, "ok"

def cleanup_quant_model(model_path: Path):
    try:
        if model_path.exists():
            model_path.unlink()
            logger.info(f"🧹 Deleted quant model: {model_path.name}")
    except Exception as e:
        logger.warning(f"Could not delete {model_path.name}: {e}")
# ══════════════════════════════════════════════════════════════════════════════
# DOWNLOAD
# ══════════════════════════════════════════════════════════════════════════════

def download_quantized_model(model_key: str, quant_type: str) -> dict:
    cfg = MODELS[model_key]
    q_upper = quant_type.upper()
    clean_filename = f"{cfg['gguf_base_filename']}-{q_upper}.gguf"
    dest = QUANTS_DIR / clean_filename

    if dest.exists():
        size_mb = round(dest.stat().st_size / (1024 ** 2), 2)
        logger.info(f"✅ Already present: {clean_filename} ({size_mb} MB)")
        return {"success": True, "output_path": dest, "model_size_mb": size_mb}

    QUANTS_DIR.mkdir(parents=True, exist_ok=True)
    repo_id = f"{HF_NAMESPACE}/{cfg['gguf_repo_name']}"
    logger.info(f"Downloading {clean_filename} from {repo_id} ...")

    try:
        hf_hub_download(repo_id=repo_id, filename=clean_filename, local_dir=str(QUANTS_DIR))
        size_mb = round(dest.stat().st_size / (1024 ** 2), 2)
        logger.info(f"Download complete: {dest} ({size_mb} MB)")
        return {"success": True, "output_path": dest, "model_size_mb": size_mb}
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return {"success": False, "error": str(e)}

# ══════════════════════════════════════════════════════════════════════════════
# KL DIVERGENCE
# ══════════════════════════════════════════════════════════════════════════════

def get_base_model_path(model_key: str) -> Optional[Path]:
    path = BASE_MODEL_PATHS.get(model_key)
    if path and path.exists():
        return path
    return None




def load_llm_for_kl(model_path: Path, device_key: str) -> Llama:
    """
    Load model for KL divergence estimation.
    logits_all belongs here, not in llm(...)
    """
    device = DEVICES[device_key]

    return Llama(
        model_path=str(model_path),
        n_ctx=KL_CTX,
        n_threads=device["cpu_threads"],
        n_batch=512,
        n_gpu_layers=0,   # safer / more portable for KL pass
        logits_all=True,  # constructor arg
        verbose=False,
        seed=42,
    )

def _extract_top_probs(llm: Llama, prompt: str, top_k: int = KL_TOP_K) -> Dict[str, float]:
    """
    Extract probabilities for the first generated token after the prompt.
    Uses echo=True so the last token in top_logprobs is the generated token.
    """
    out = llm(
        prompt,
        max_tokens=1,
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        logprobs=top_k,
        echo=True,
    )

    if not isinstance(out, dict):
        raise RuntimeError(f"Unexpected response type: {type(out)}")

    choices = out.get("choices") or []
    if not choices:
        raise RuntimeError("No choices returned by llama.cpp")

    choice0 = choices[0]
    logprobs = choice0.get("logprobs") or {}

    top_logprobs = logprobs.get("top_logprobs")
    if not isinstance(top_logprobs, list) or not top_logprobs:
        raise RuntimeError(f"No top_logprobs list found. keys={list(logprobs.keys())}")

    # With echo=True, the last entry corresponds to the generated token.
    # The first entry may be None by API convention.
    top_logprob_dict = next(
        (d for d in reversed(top_logprobs) if isinstance(d, dict) and d),
        None,
    )

    if top_logprob_dict is None:
        raise RuntimeError(f"No usable top_logprobs found. keys={list(logprobs.keys())}")

    probs = {}
    for token, logprob in top_logprob_dict.items():
        if logprob is None:
            continue
        try:
            probs[str(token)] = math.exp(float(logprob))
        except (TypeError, ValueError):
            continue

    total = sum(probs.values())
    if total <= 0:
        raise RuntimeError("Parsed probabilities sum to zero")

    return {k: v / total for k, v in probs.items()}

def _smooth_and_normalize(dist: dict[str, float], support: list[str], eps: float = KL_EPS) -> dict[str, float]:
    """
    Build a smoothed probability distribution over shared support + __OTHER__ bucket.
    """
    other_key = "__OTHER__"

    clean = {tok: max(float(dist.get(tok, 0.0)), 0.0) for tok in support if tok != other_key}
    used_mass = sum(clean.values())
    clean[other_key] = max(1.0 - used_mass, 0.0)

    for tok in support:
        clean[tok] = clean.get(tok, 0.0) + eps

    total = sum(clean.values())
    return {tok: val / total for tok, val in clean.items()}


def _kl_divergence(p: dict[str, float], q: dict[str, float]) -> float:
    return sum(p[t] * math.log(p[t] / q[t]) for t in p.keys())


def measure_kl_divergence(candidate_model: Path, base_model: Path, device_key: str) -> Optional[float]:
    """
    Compute approximate symmetric KL divergence between base and candidate models.
    Returns None if the measurement cannot be completed.
    """
    if not base_model.exists():
        logger.warning(f"Base model missing for KL: {base_model}")
        return None

    logger.info("Measuring KL divergence:")
    logger.info(f"   Base      : {base_model.name}")
    logger.info(f"   Candidate : {candidate_model.name}")

    try:
        base_llm = load_llm_for_kl(base_model, device_key)
        cand_llm = load_llm_for_kl(candidate_model, device_key)
    except Exception as e:
        logger.warning(f"Failed to load models for KL: {e}")
        return None

    scores = []

    try:
        for idx, prompt in enumerate(KL_PROMPTS, start=1):
            try:
                base_dist = _extract_top_probs(base_llm, prompt)
                cand_dist = _extract_top_probs(cand_llm, prompt)

                support = sorted(set(base_dist) | set(cand_dist) | {"__OTHER__"})
                p = _smooth_and_normalize(base_dist, support)
                q = _smooth_and_normalize(cand_dist, support)

                kl_pq = _kl_divergence(p, q)
                kl_qp = _kl_divergence(q, p)
                sym = 0.5 * (kl_pq + kl_qp)
                scores.append(sym)

                logger.info(
                    f"   [{idx}/{len(KL_PROMPTS)}] "
                    f"KL(base||cand)={kl_pq:.6f} | "
                    f"KL(cand||base)={kl_qp:.6f} | "
                    f"sym={sym:.6f}"
                )

            except Exception as e:
                logger.warning(f"KL failed for prompt {idx}: {e}")
                continue

        if not scores:
            logger.warning("KL divergence failed for all prompts")
            return None

        avg_kl = round(sum(scores) / len(scores), 6)
        logger.info(f"✅ Average symmetric KL = {avg_kl:.6f}")
        return avg_kl

    finally:
        try:
            del base_llm
        except Exception:
            pass
        try:
            del cand_llm
        except Exception:
            pass
        gc.collect()
# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK (llama-bench)
# ══════════════════════════════════════════════════════════════════════════════

def benchmark_model(model_path: Path, device_key: str) -> dict:
    bench_bin = resolve_bin(LLAMA_BENCH, "llama-bench")
    if not bench_bin:
        return {"success": False, "error_msg": "llama-bench not found"}

    device = DEVICES[device_key]
    process = psutil.Process()

    ram_before = process.memory_info().rss / (1024 ** 2)
    load_start = time.perf_counter()

    cmd = [
        bench_bin,
        "-m", str(model_path),
        "-p", str(EVAL_TOKENS_PROMPT),
        "-n", str(EVAL_TOKENS_GEN),
        "-r", "1",
        "-t", str(device["cpu_threads"]),
        "-ngl", str(device["gpu_layers"]) if device["gpu_layers"] > 0 else "0",
        "-o", "json",
        "-b", "256",
    ]

    logger.info(f"Benchmarking: {model_path.name}")
    logger.info("Command: " + " ".join(cmd))

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    except subprocess.TimeoutExpired:
        return {"success": False, "error_msg": "timeout"}

    load_time_s = round(time.perf_counter() - load_start, 2)
    peak_ram_mb = round(process.memory_info().rss / (1024 ** 2), 1)

    if proc.returncode != 0:
        stderr_lower = (proc.stderr or "").lower()
        oom = any(k in stderr_lower for k in ["oom", "out of memory", "killed"])
        return {
            "success": False,
            "peak_ram_mb": peak_ram_mb,
            "load_time_s": load_time_s,
            "oom": oom,
            "error_msg": (proc.stderr or "")[:500],
        }

    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError:
        logger.warning("llama-bench JSON parse failed")
        logger.warning(f"Raw stdout:\n{proc.stdout[:1000]}")
        return {"success": False, "error_msg": "JSON parse failed"}

    prompt_tps = None
    gen_tps = None
    latency = None

    entries = data if isinstance(data, list) else [data]
    for entry in entries:
        n_prompt = entry.get("n_prompt", 0)
        n_gen = entry.get("n_gen", 0)
        avg_ts = entry.get("avg_ts", 0)

        if n_prompt > 0 and n_gen == 0:
            try:
                prompt_tps = float(avg_ts)
                if prompt_tps > 0:
                    latency = round((EVAL_TOKENS_PROMPT / prompt_tps) * 1000, 1)
            except Exception:
                pass
        elif n_gen > 0:
            try:
                gen_tps = float(avg_ts)
            except Exception:
                pass

    return {
        "success": True,
        "prompt_tps": prompt_tps,
        "gen_tps": gen_tps,
        "latency_first_token_ms": latency,
        "peak_ram_mb": peak_ram_mb,
        "load_time_s": load_time_s,
        "oom": False,
        "error_msg": None,
    }

# ══════════════════════════════════════════════════════════════════════════════
# RESULTS
# ══════════════════════════════════════════════════════════════════════════════

def load_existing_results():
    if RESULTS_FILE.exists():
        try:
            with open(RESULTS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_results(results):
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    logger.info(f"Saved {len(results)} records → {CSV_FILE}")


def already_collected(results, device_key, model_key, quant_type):
    return any(
        r.get("device") == device_key and
        r.get("model_key") == model_key and
        r.get("quant_type") == quant_type
        for r in results
    )

# ══════════════════════════════════════════════════════════════════════════════
# MAIN LOGIC
# ══════════════════════════════════════════════════════════════════════════════

def build_run_plan(device_key, model_filter):
    plan = []
    keys = [model_filter] if model_filter else list(MODELS.keys())

    for mk in keys:
        for q in QUANT_TYPES:
            feasible, reason = is_feasible(mk, q, device_key)
            plan.append({
                "model_key": mk,
                "quant_type": q,
                "feasible": feasible,
                "reason": reason,
            })

    return plan


def run_collection(device_key, model_filter, dry_run, resume):
    device = DEVICES[device_key]

    logger.info("=" * 80)
    logger.info("🚀 Edge Benchmark Collector (downloads from HF)")
    logger.info(f"Device : {device['display_name']} ({device_key})")
    logger.info(f"HF     : {HF_NAMESPACE}")
    logger.info("=" * 80)

    plan = build_run_plan(device_key, model_filter)
    results = load_existing_results() if resume else []

    if dry_run:
        for p in plan:
            status = "RUN" if p["feasible"] else f"SKIP ({p['reason']})"
            logger.info(f"{p['model_key']:12} {p['quant_type']:10} → {status}")
        return

    for item in tqdm(plan, desc="Benchmarking"):
        model_key = item["model_key"]
        quant_type = item["quant_type"]
        model_cfg = MODELS[model_key]
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")

        if not item["feasible"]:
            if resume and already_collected(results, device_key, model_key, quant_type):
                continue

            row = {col: None for col in CSV_COLUMNS}
            row.update({
                "device": device_key,
                "model_key": model_key,
                "model_display": model_cfg["display_name"],
                "quant_type": quant_type,
                "params_b": model_cfg["params_b"],
                "success": False,
                "oom": True,
                "error_msg": item["reason"],
                "timestamp": timestamp,
                **MODEL_METADATA.get(model_key, {}),
            })
            results.append(row)
            save_results(results)
            continue

        if resume and already_collected(results, device_key, model_key, quant_type):
            logger.info(f"Skipping already done: {model_key}/{quant_type}")
            continue

        # Download quantized model
        q_result = download_quantized_model(model_key, quant_type)
        if not q_result["success"]:
            row = {col: None for col in CSV_COLUMNS}
            row.update({
                "device": device_key,
                "model_key": model_key,
                "model_display": model_cfg["display_name"],
                "quant_type": quant_type,
                "params_b": model_cfg["params_b"],
                "success": False,
                "error_msg": f"Download failed: {q_result.get('error')}",
                "timestamp": timestamp,
                **MODEL_METADATA.get(model_key, {}),
            })
            results.append(row)
            save_results(results)
            continue

        # Benchmark
        bench = benchmark_model(q_result["output_path"], device_key)

        # KL divergence (only if benchmark succeeded)
        base_model_path = get_base_model_path(model_key)
        kl_divergence = None
        if bench["success"]:
            if base_model_path is None:
                logger.warning(f"No base model path configured/found for {model_key}; skipping KL")
            else:
                kl_divergence = measure_kl_divergence(
                    candidate_model=q_result["output_path"],
                    base_model=base_model_path,
                    device_key=device_key,
                )

        row = {
            "device": device_key,
            "model_key": model_key,
            "model_display": model_cfg["display_name"],
            "quant_type": quant_type,
            "params_b": model_cfg["params_b"],
            "model_size_mb": q_result["model_size_mb"],
            "prompt_tps": bench.get("prompt_tps"),
            "gen_tps": bench.get("gen_tps"),
            "latency_first_token_ms": bench.get("latency_first_token_ms"),
            "peak_ram_mb": bench.get("peak_ram_mb"),
            "load_time_s": bench.get("load_time_s"),
            "kl_divergence": kl_divergence,
            "success": bench["success"],
            "oom": bench.get("oom", False),
            "error_msg": bench.get("error_msg"),
            "timestamp": timestamp,
            **MODEL_METADATA.get(model_key, {}),
        }

        results.append(row)
        save_results(results)
        cleanup_quant_model(q_result["output_path"])
        status = "✓" if bench["success"] else "✗"
        gen_tps_str = f"{bench.get('gen_tps'):.2f}" if isinstance(bench.get("gen_tps"), (int, float)) else "N/A"
        kl_str = f"{kl_divergence:.6f}" if isinstance(kl_divergence, (int, float)) else "N/A"
        ram_str = f"{bench.get('peak_ram_mb'):.1f}" if isinstance(bench.get("peak_ram_mb"), (int, float)) else "N/A"

        logger.info(
            f"{status} {model_key:12} {quant_type:10} | "
            f"gen={gen_tps_str:>8} t/s | "
            f"kl={kl_str:>10} | "
            f"ram={ram_str:>8} MB"
        )

    logger.info("=" * 80)
    logger.info("🎉 Finished! Results saved to Inference_collData/benchmark_results.csv")
    logger.info("=" * 80)

# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Edge benchmark collector with KL divergence")
    parser.add_argument("--device", choices=list(DEVICES.keys()), default=None, help="Force device")
    parser.add_argument("--model", choices=list(MODELS.keys()), default=None, help="Single model")
    parser.add_argument("--dry-run", action="store_true", help="Preview only")
    parser.add_argument("--resume", action="store_true", help="Skip done rows")
    args = parser.parse_args()

    device_key = args.device if args.device else detect_device()

    if not args.dry_run:
        missing = []
        for p, name in [(LLAMA_BENCH, "llama-bench")]:
            if not resolve_bin(p, name):
                missing.append(name)

        if missing:
            logger.error(f"Missing binaries: {missing}. Compile llama.cpp first.")
            sys.exit(1)

    run_collection(device_key, args.model, args.dry_run, args.resume)


if __name__ == "__main__":
    main()