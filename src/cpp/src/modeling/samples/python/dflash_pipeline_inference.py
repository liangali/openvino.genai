#!/usr/bin/env python3
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""
DFlash speculative decoding via openvino_genai.dflash_model() + LLMPipeline.

Usage:
  python dflash_decoding_lm.py <target_model_dir> <draft_model_dir> "<prompt>"

Example:
  python dflash_decoding_lm.py D:\\models\\Qwen3.5-4B D:\\models\\Qwen3.5-4B-DFlash-b16 "What is AI?"
"""

import argparse
import importlib.util
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


_DLL_DIR_HANDLES = []
_BOOTSTRAP_DLL_DIRS: List[Path] = []
_BOOTSTRAP_GENAI_DIR: Optional[Path] = None
_BOOTSTRAP_OPENVINO_PY_DIR: Optional[Path] = None


def _find_build_genai_dir(start_dir: Path) -> Optional[Path]:
    candidates: List[Path] = []
    search = start_dir
    for _ in range(10):
        candidates.append(search / "build" / "openvino_genai")
        candidates.append(search / "openvino.genai" / "build" / "openvino_genai")
        if search.parent == search:
            break
        search = search.parent

    for c in candidates:
        if c.is_dir() and any(c.glob("py_openvino_genai*.pyd")):
            return c
    return None


def _find_runtime_dll_dirs(genai_dir: Path) -> List[Path]:
    entries: List[Path] = [genai_dir]

    search = genai_dir.parent
    for _ in range(8):
        if search.parent == search:
            break
        search = search.parent

        for build_type in ("Release", "RelWithDebInfo", "Debug"):
            runtime = search / "openvino" / "bin" / "intel64" / build_type
            if runtime.is_dir() and (runtime / "openvino.dll").is_file():
                entries.append(runtime)
                break

        tbb_root = search / "openvino" / "temp"
        if tbb_root.is_dir():
            for tbb_bin in sorted(tbb_root.glob("*/tbb/bin")):
                if (tbb_bin / "tbb12.dll").is_file():
                    entries.append(tbb_bin)
                    break

    unique_entries: List[Path] = []
    seen = set()
    for p in entries:
        key = str(p.resolve())
        if key not in seen:
            seen.add(key)
            unique_entries.append(p)
    return unique_entries


def _find_local_openvino_python_dir(start_dir: Path) -> Optional[Path]:
    search = start_dir
    for _ in range(10):
        for build_type in ("Release", "RelWithDebInfo", "Debug"):
            candidate = search / "openvino" / "bin" / "intel64" / build_type / "python"
            if (candidate / "openvino" / "__init__.py").is_file():
                return candidate
        if search.parent == search:
            break
        search = search.parent
    return None


def _bootstrap_openvino_genai() -> None:
    global _BOOTSTRAP_DLL_DIRS
    global _BOOTSTRAP_GENAI_DIR
    global _BOOTSTRAP_OPENVINO_PY_DIR

    script_dir = Path(__file__).resolve().parent
    genai_dir = _find_build_genai_dir(script_dir)
    if genai_dir is None:
        return

    _BOOTSTRAP_GENAI_DIR = genai_dir

    build_dir = genai_dir.parent
    build_dir_str = str(build_dir)
    if build_dir_str not in sys.path:
        sys.path.insert(0, build_dir_str)

    local_ov_py_dir = _find_local_openvino_python_dir(script_dir)
    if local_ov_py_dir is not None:
        _BOOTSTRAP_OPENVINO_PY_DIR = local_ov_py_dir
        local_ov_py_dir_str = str(local_ov_py_dir)
        if local_ov_py_dir_str not in sys.path:
            sys.path.insert(0, local_ov_py_dir_str)

    dll_dirs = _find_runtime_dll_dirs(genai_dir)

    # Only fallback to environment OpenVINO libs when no local OpenVINO runtime was found.
    has_local_ov_runtime = any((d / "openvino.dll").is_file() for d in dll_dirs)
    if not has_local_ov_runtime:
        ov_spec = importlib.util.find_spec("openvino")
        if ov_spec and ov_spec.origin:
            ov_pkg_dir = Path(ov_spec.origin).resolve().parent
            ov_candidates = [
                ov_pkg_dir / "libs",
                ov_pkg_dir.parent / "openvino" / "libs",
            ]
            for c in ov_candidates:
                if c.is_dir() and any(c.glob("*.dll")):
                    dll_dirs.append(c)

    # Remove duplicates while preserving order.
    deduped: List[Path] = []
    seen = set()
    for d in dll_dirs:
        key = str(Path(d).resolve())
        if key not in seen:
            seen.add(key)
            deduped.append(Path(d))
    dll_dirs = deduped
    _BOOTSTRAP_DLL_DIRS = dll_dirs

    # Local OpenVINO python package may require OPENVINO_LIB_PATHS on Windows.
    existing_ov_lib_paths = os.environ.get("OPENVINO_LIB_PATHS", "").strip()
    if not existing_ov_lib_paths and dll_dirs:
        os.environ["OPENVINO_LIB_PATHS"] = ";".join(str(d) for d in dll_dirs)

    if not dll_dirs:
        return

    if hasattr(os, "add_dll_directory"):
        for d in dll_dirs:
            try:
                handle = os.add_dll_directory(str(d))
                _DLL_DIR_HANDLES.append(handle)
            except OSError:
                pass

    existing_path = os.environ.get("PATH", "")
    prepend = ";".join(str(d) for d in dll_dirs)
    os.environ["PATH"] = f"{prepend};{existing_path}" if existing_path else prepend


_bootstrap_openvino_genai()

try:
    import openvino_genai
except ImportError as exc:
    print("[bootstrap] Failed to import openvino_genai:", exc, file=sys.stderr)
    if _BOOTSTRAP_GENAI_DIR is not None:
        print(f"[bootstrap] openvino_genai package dir: {_BOOTSTRAP_GENAI_DIR}", file=sys.stderr)
    if _BOOTSTRAP_OPENVINO_PY_DIR is not None:
        print(f"[bootstrap] preferred openvino python dir: {_BOOTSTRAP_OPENVINO_PY_DIR}", file=sys.stderr)
    ov_spec = importlib.util.find_spec("openvino")
    if ov_spec and ov_spec.origin:
        print(f"[bootstrap] openvino module resolved to: {ov_spec.origin}", file=sys.stderr)
    if _BOOTSTRAP_DLL_DIRS:
        print("[bootstrap] DLL search dirs:", file=sys.stderr)
        for d in _BOOTSTRAP_DLL_DIRS:
            print(f"  - {d}", file=sys.stderr)
    else:
        print("[bootstrap] No DLL dirs were discovered.", file=sys.stderr)
    for dll_name in ("openvino.dll", "tbb12.dll", "openvino_tokenizers.dll", "openvino_genai.dll"):
        found = False
        for d in _BOOTSTRAP_DLL_DIRS:
            if (d / dll_name).is_file():
                found = True
                break
        print(f"[bootstrap] {dll_name}: {'FOUND' if found else 'MISSING'}", file=sys.stderr)
    raise


@dataclass
class StreamMetrics:
    start_time: float = 0.0
    first_chunk_time: float = 0.0
    end_time: float = 0.0
    chunk_count: int = 0
    generated_tokens: int = 0

    def begin(self):
        self.start_time = time.perf_counter()
        self.first_chunk_time = 0.0
        self.end_time = 0.0
        self.chunk_count = 0
        self.generated_tokens = 0

    def on_stream(self, subword):
        now = time.perf_counter()
        if subword:
            if self.first_chunk_time == 0.0:
                self.first_chunk_time = now
            self.chunk_count += 1
        print(subword, end="", flush=True)
        return openvino_genai.StreamingStatus.RUNNING

    def finish(self, generated_tokens):
        self.end_time = time.perf_counter()
        self.generated_tokens = generated_tokens

    @property
    def e2e_ms(self):
        return max(0.0, (self.end_time - self.start_time) * 1000.0)

    @property
    def ttft_ms(self):
        if self.first_chunk_time == 0.0:
            return self.e2e_ms
        return max(0.0, (self.first_chunk_time - self.start_time) * 1000.0)

    @property
    def tpot_ms(self):
        if self.generated_tokens <= 1 or self.first_chunk_time == 0.0:
            return 0.0
        decode_ms = max(0.0, (self.end_time - self.first_chunk_time) * 1000.0)
        return decode_ms / float(self.generated_tokens - 1)


def build_prompt(prompt, no_think):
    if not no_think:
        return prompt
    return (
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n<|im_start|>think\n\n<|im_end|>\n"
    )


def run_with_streaming(pipe, full_prompt, config):
    metrics = StreamMetrics()
    metrics.begin()
    result = pipe.generate(full_prompt, config, metrics.on_stream)

    generated_tokens = 0
    if hasattr(result, "tokens") and result.tokens and result.tokens[0]:
        generated_tokens = len(result.tokens[0])
    elif metrics.chunk_count > 0:
        generated_tokens = metrics.chunk_count

    metrics.finish(generated_tokens)
    return result, metrics


def run_dflash(model_dir, draft_dir, prompt, device, max_tokens, no_think):
    print("=" * 60)
    print("DFlash speculative decoding")
    print("=" * 60)
    print(f"Target model : {model_dir}")
    print(f"Draft model  : {draft_dir}")
    print(f"Device       : {device}")
    print(f"Max tokens   : {max_tokens}")
    print()

    draft = openvino_genai.dflash_model(draft_dir, device)
    pipe = openvino_genai.LLMPipeline(model_dir, device, dflash_model=draft)

    config = openvino_genai.GenerationConfig()
    config.max_new_tokens = max_tokens
    full_prompt = build_prompt(prompt, no_think)

    print("--- DFlash output ---")
    result, metrics = run_with_streaming(pipe, full_prompt, config)
    print("\n--- end ---")
    return result, metrics


def run_baseline(model_dir, prompt, device, max_tokens, no_think):
    print()
    print("=" * 60)
    print("Baseline (pure LLMPipeline)")
    print("=" * 60)

    pipe = openvino_genai.LLMPipeline(model_dir, device)

    config = openvino_genai.GenerationConfig()
    config.max_new_tokens = max_tokens
    full_prompt = build_prompt(prompt, no_think)

    print("--- Baseline output ---")
    result, metrics = run_with_streaming(pipe, full_prompt, config)
    print("\n--- end ---")
    return result, metrics


def main():
    parser = argparse.ArgumentParser(description="DFlash vs baseline Qwen3.5 inference comparison")
    parser.add_argument("model_dir", nargs="?", default=r"D:\Data\models\Huggingface\Qwen3.5-4B",
                        help="Path to target Qwen3.5 model directory")
    parser.add_argument("draft_model_dir", nargs="?", default=r"D:\Data\models\Huggingface\Qwen3.5-4B-DFlash-b16",
                        help="Path to DFlash draft model directory")
    parser.add_argument("prompt", nargs="?", default="who are you?", help="Input prompt text")
    parser.add_argument("--device", default="GPU", help="OpenVINO device (default: GPU)")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Max new tokens (default: 1024)")
    parser.add_argument("--no-think", action="store_true",
                        help="Skip thinking for Qwen3.5 (add no-think template)")
    parser.add_argument("--skip-baseline", action="store_true",
                        help="Skip baseline comparison")
    args = parser.parse_args()

    dflash_result, dflash_metrics = run_dflash(
        args.model_dir, args.draft_model_dir, args.prompt,
        args.device, args.max_tokens, args.no_think)

    if not args.skip_baseline:
        baseline_result, baseline_metrics = run_baseline(
            args.model_dir, args.prompt,
            args.device, args.max_tokens, args.no_think)

        print()
        e2e_speedup = "N/A"
        if dflash_metrics.e2e_ms > 0.0:
            e2e_speedup = f"{baseline_metrics.e2e_ms / dflash_metrics.e2e_ms:.2f}x"
        print(
            "COMPARE "
            f"DFlash(ttft={dflash_metrics.ttft_ms:.2f}ms, tpot={dflash_metrics.tpot_ms:.2f}ms, e2e={dflash_metrics.e2e_ms:.2f}ms) | "
            f"Baseline(ttft={baseline_metrics.ttft_ms:.2f}ms, tpot={baseline_metrics.tpot_ms:.2f}ms, e2e={baseline_metrics.e2e_ms:.2f}ms) | "
            f"E2E Speedup={e2e_speedup}"
        )
    else:
        print()
        print(
            "DFlash "
            f"ttft={dflash_metrics.ttft_ms:.2f}ms "
            f"tpot={dflash_metrics.tpot_ms:.2f}ms "
            f"e2e={dflash_metrics.e2e_ms:.2f}ms"
        )


if __name__ == "__main__":
    main()
