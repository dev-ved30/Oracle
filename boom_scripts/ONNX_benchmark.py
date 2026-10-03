"""Benchmark all three ORACLE-2 models on the first visible NVIDIA GPU.

Run from the repo root: python boom_scripts/ONNX_benchmark.py --lc-lengths 50 100 200
Requires oracle, onnx, onnxruntime-gpu, and nvidia-ml-py.
VRAM includes the CUDA context, weights, and ONNX Runtime arena; peak is sampled
every 5 ms during warmup and inference. Latency includes host/device transfers.
"""

import argparse
import csv
from itertools import product
import multiprocessing as mp
import os
from pathlib import Path
import threading
import time

import numpy as np
import onnxruntime as ort
import pynvml
import torch

from oracle.presets import get_model


CHECKPOINTS = {
    "BTSv2-lite": "models/BTSv2-lite/fancy-elevator-567/best_model_f1.pth",
    "BTSv2": "models/BTSv2/stilted-elevator-551/best_model_f1.pth",
    "BTSv2-pro": "models/BTSv2-pro/morning-feather-572/best_model_f1.pth",
}
BATCH_SIZES = range(100, 1001, 100)
IMAGE_SIZE = 63
WARMUP = 3
REPEATS = 10
OUTPUT_DIR = Path("analysis_outputs/onnx")
FIELDS = ["model", "lc_length", "batch_size", "peak_vram_mib", "latency_ms", "sources_per_second"]


class ExportModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, ts, length, static=None, postage_stamp=None):
        return self.model(dict(ts=ts, length=length, static=static, postage_stamp=postage_stamp))


def make_inputs(name, batch_size, lc_length):
    rng = np.random.default_rng(0)
    inputs = {
        "ts": rng.standard_normal((batch_size, lc_length, 5), dtype=np.float32),
        "length": np.full(batch_size, lc_length, dtype=np.int64),
    }
    if name != "BTSv2-lite":
        inputs["static"] = rng.standard_normal((batch_size, 30), dtype=np.float32)
    if name == "BTSv2-pro":
        inputs["postage_stamp"] = rng.standard_normal((batch_size, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    return inputs


def export_model(name, lc_length):
    print(f"Exporting {name}...", flush=True)
    model = ExportModel(get_model(name).cpu()).eval()
    model.model.load_state_dict(torch.load(CHECKPOINTS[name], map_location="cpu", weights_only=True))
    inputs = make_inputs(name, 1, lc_length)
    axes = {key: {0: "batch"} for key in inputs}
    axes["ts"][1] = "lc_length"
    axes["logits"] = {0: "batch"}
    with torch.inference_mode():
        torch.onnx.export(
            model, tuple(torch.from_numpy(value) for value in inputs.values()), OUTPUT_DIR / f"{name}.onnx",
            input_names=list(inputs), output_names=["logits"], dynamic_axes=axes,
            opset_version=17, dynamo=False,
        )
    print(f"Exported {OUTPUT_DIR / f'{name}.onnx'}", flush=True)


def benchmark(case):
    name, lc_length, batch_size = case
    print(f"Benchmarking {name}: lc_length={lc_length}, batch_size={batch_size}", flush=True)
    pynvml.nvmlInit()
    gpus = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount())]
    session = ort.InferenceSession(str(OUTPUT_DIR / f"{name}.onnx"), providers=["CUDAExecutionProvider"])
    session.disable_fallback()
    inputs = make_inputs(name, batch_size, lc_length)
    samples = []
    stop = threading.Event()

    def sample_vram():
        processes = [p for gpu in gpus for p in pynvml.nvmlDeviceGetComputeRunningProcesses(gpu)]
        samples.append(sum(p.usedGpuMemory for p in processes if p.pid == os.getpid()) / 2**20)

    def monitor():
        while not stop.wait(0.005):
            sample_vram()

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    for _ in range(WARMUP):
        session.run(None, inputs)
    start = time.perf_counter()
    for _ in range(REPEATS):
        session.run(None, inputs)
    latency = (time.perf_counter() - start) / REPEATS
    sample_vram()
    stop.set()
    thread.join()
    pynvml.nvmlShutdown()
    return dict(zip(FIELDS, (name, lc_length, batch_size, max(samples), latency * 1000, batch_size / latency)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lc-lengths", nargs="+", type=int, default=[100])
    args = parser.parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name in CHECKPOINTS:
        export_model(name, args.lc_lengths[0])

    # A fresh worker for each case prevents CUDA arena reuse across batch sizes.
    with mp.get_context("spawn").Pool(1, maxtasksperchild=1) as pool:
        with (OUTPUT_DIR / "benchmark.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=FIELDS)
            writer.writeheader()
            for result in pool.imap(benchmark, product(CHECKPOINTS, args.lc_lengths, BATCH_SIZES)):
                writer.writerow(result)
                file.flush()
                print(result, flush=True)
    print(f"Results saved to {OUTPUT_DIR / 'benchmark.csv'}", flush=True)


if __name__ == "__main__":
    main()
