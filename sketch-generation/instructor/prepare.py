#!/usr/bin/env python3
"""Prepare the approved sketch dataset and frozen evaluator artifacts.

Run each subcommand with --help for paths. Source indices, reference images and
classifier preparation data are written only below the private directory.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import urllib.request

import numpy as np

CHALLENGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CHALLENGE_ROOT))
from evaluator import (  # noqa: E402
    CHALLENGE_VERSION, CLASSES, IMAGE_SHAPE, METRIC_VERSION, REFERENCE_COUNT,
    SAMPLES_PER_CLASS, STANDARD_DEVIATION_FLOOR, SketchEvaluator,
    evaluator_fingerprint, sha256_file, squared_distances,
)

PARTITION_COUNTS = {
    "student_train": 5000,
    "public_validation": 512,
    "evaluator_train": 10000,
    "evaluator_validation": 2000,
    "calibration": 512,
    "private_reference": 512,
    "audit": 1000,
}
SOURCE_BASE = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/"
DEFAULT_SEED = 20261028


def report(event, **values):
    print(json.dumps({"event": event, **values}, sort_keys=True), flush=True)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as destination:
        np.savez_compressed(destination, **arrays)
    temporary.replace(path)


def file_entry(path, relative_path):
    return {"path": relative_path, "sha256": sha256_file(path),
            "size_bytes": Path(path).stat().st_size}


def dataset_fingerprint(train_sha256, validation_sha256):
    definition = {
        "challenge_version": CHALLENGE_VERSION, "classes": list(CLASSES),
        "image_shape": list(IMAGE_SHAPE), "train_sha256": train_sha256,
        "validation_sha256": validation_sha256,
    }
    return hashlib.sha256(json.dumps(definition, sort_keys=True,
                                    separators=(",", ":")).encode("utf-8")).hexdigest()


def refresh_manifest(public_dir):
    public_dir = Path(public_dir).resolve()
    files = {}
    for key, name in (("train", "train.npz"), ("validation", "validation.npz"),
                      ("evaluator", "evaluator.npz")):
        path = public_dir / name
        if path.is_file():
            files[key] = file_entry(path, f"data/{name}")
    for key, name in (("evaluator_module", "evaluator.py"), ("model_module", "model.py")):
        path = CHALLENGE_ROOT / name
        if path.is_file():
            files[key] = file_entry(path, name)
    if "train" not in files or "validation" not in files:
        raise ValueError("Prepare train.npz and validation.npz before generating the manifest.")
    manifest = {
        "challenge_version": CHALLENGE_VERSION, "classes": list(CLASSES),
        "image_shape": list(IMAGE_SHAPE), "pixel_convention": "0=background;255=ink",
        "samples_per_class": SAMPLES_PER_CLASS, "reference_count": REFERENCE_COUNT,
        "train_count_per_class": PARTITION_COUNTS["student_train"],
        "validation_count_per_class": PARTITION_COUNTS["public_validation"],
        "dataset_fingerprint": dataset_fingerprint(files["train"]["sha256"],
                                                   files["validation"]["sha256"]),
        "files": files,
        "source": {"name": "The Quick, Draw! Dataset", "creator": "Google, Inc.",
                   "url": "https://github.com/googlecreativelab/quickdraw-dataset",
                   "license": "CC BY 4.0", "modifications": "Selected, deduplicated bitmap subsets."},
        "metric": {"name": "Sketch distance", "version": METRIC_VERSION,
                   "direction": "minimize", "scale": 1000.0,
                   "estimator": "class-averaged biased empirical squared MMD",
                   "kernel": "equal mixture of RBF kernels at half, one and twice each class bandwidth"},
        "status": "data_prepared",
        "release_status": "pending_instructor_colab_rehearsal",
    }
    if "evaluator" in files:
        manifest["evaluator_fingerprint"] = evaluator_fingerprint(public_dir / "evaluator.npz")
        manifest["status"] = "evaluator_prepared"
        previous_path = public_dir / "manifest.json"
        if previous_path.is_file():
            previous = json.loads(previous_path.read_text(encoding="utf-8"))
            if (previous.get("dataset_fingerprint") == manifest["dataset_fingerprint"]
                    and previous.get("evaluator_fingerprint") == manifest["evaluator_fingerprint"]):
                for key in ("status", "evaluator_validation", "release_status"):
                    if key in previous:
                        manifest[key] = previous[key]
    atomic_json(public_dir / "manifest.json", manifest)
    report("public_manifest", path=str(public_dir / "manifest.json"),
           dataset_fingerprint=manifest["dataset_fingerprint"], status=manifest["status"])
    return manifest


def source_file(cache_dir, category):
    path = Path(cache_dir) / f"{category}.npy"
    path.parent.mkdir(parents=True, exist_ok=True)
    url = SOURCE_BASE + category + ".npy"
    if not path.is_file():
        temporary = path.with_name(path.name + ".part")
        try:
            with urllib.request.urlopen(url, timeout=60) as source, temporary.open("wb") as destination:
                shutil.copyfileobj(source, destination, length=1024 * 1024)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    images = np.load(path, mmap_mode="r", allow_pickle=False)
    if images.dtype != np.uint8 or images.ndim != 2 or images.shape[1] != 784:
        raise ValueError(f"Unexpected official bitmap format in {path}.")
    information = {"category": category, "url": url, "sha256": sha256_file(path),
                   "size_bytes": path.stat().st_size, "source_count": len(images)}
    report("source_ready", **information)
    return path, information


def prepare_data(args):
    public_dir, private_dir = Path(args.public_dir).resolve(), Path(args.private_dir).resolve()
    if public_dir == private_dir or public_dir in private_dir.parents or private_dir in public_dir.parents:
        raise ValueError("Use separate public and private output directories.")
    public_dir.mkdir(parents=True, exist_ok=True)
    private_dir.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=len(CLASSES)) as pool:
        sources = list(pool.map(lambda category: source_file(args.cache_dir, category), CLASSES))
    source_arrays = [np.load(path, mmap_mode="r", allow_pickle=False) for path, _ in sources]
    first_seen = {}
    conflicts = set()
    candidates = [[] for _ in CLASSES]
    counts = [{"empty": 0, "same_class_duplicates": 0} for _ in CLASSES]
    for label, images in enumerate(source_arrays):
        for index, image in enumerate(images):
            if not np.any(image):
                counts[label]["empty"] += 1
                continue
            digest = hashlib.sha256(image.tobytes()).digest()
            previous = first_seen.get(digest)
            if previous is None:
                first_seen[digest] = label
                candidates[label].append((index, digest))
            elif previous == label:
                counts[label]["same_class_duplicates"] += 1
            else:
                conflicts.add(digest)
        report("deduplicated_category", category=CLASSES[label],
               source_count=len(images), unique_candidates=len(candidates[label]), **counts[label])
    required = sum(PARTITION_COUNTS.values())
    indices = {key: [] for key in PARTITION_COUNTS}
    hashes = {key: [] for key in PARTITION_COUNTS}
    for label, category in enumerate(CLASSES):
        eligible = [(index, digest) for index, digest in candidates[label] if digest not in conflicts]
        if len(eligible) < required:
            raise ValueError(f"{category} has {len(eligible)} eligible drawings; need {required}.")
        order = np.random.default_rng(args.seed + label).permutation(len(eligible))[:required]
        chosen = [eligible[index] for index in order]
        offset = 0
        for partition, count in PARTITION_COUNTS.items():
            rows = chosen[offset:offset + count]
            indices[partition].append(np.array([row[0] for row in rows], dtype=np.int64))
            hashes[partition].append(np.frombuffer(b"".join(row[1] for row in rows), dtype=np.uint8).reshape(count, 32))
            offset += count
    source_index_arrays = {}
    for partition, count in PARTITION_COUNTS.items():
        selected_indices = np.stack(indices[partition])
        selected_hashes = np.stack(hashes[partition])
        source_index_arrays[partition + "_indices"] = selected_indices
        source_index_arrays[partition + "_hashes"] = selected_hashes
        images = np.concatenate([source_arrays[label][selected_indices[label]].reshape(count, *IMAGE_SHAPE)
                                 for label in range(len(CLASSES))], axis=0)
        labels = np.repeat(np.arange(len(CLASSES), dtype=np.int64), count)
        if partition == "student_train":
            path = public_dir / "train.npz"
        elif partition == "public_validation":
            path = public_dir / "validation.npz"
        else:
            path = private_dir / f"{partition}.npz"
        atomic_npz(path, images=images, labels=labels)
        report("partition_written", partition=partition, per_class=count, total=len(images), path=str(path))
    atomic_npz(private_dir / "split_indices.npz", **source_index_arrays)
    atomic_json(private_dir / "preparation.json", {
        "challenge_version": CHALLENGE_VERSION, "seed": args.seed,
        "classes": list(CLASSES), "partition_counts_per_class": PARTITION_COUNTS,
        "sources": [information for _, information in sources],
        "filter": "Discard empty bitmaps; keep first exact same-class bitmap; discard every cross-class conflicting bitmap.",
        "recognized_filter": False, "conflicting_bitmap_hashes": len(conflicts),
        "source_filter_counts": dict(zip(CLASSES, counts)),
        "status": "data_prepared",
    })
    refresh_manifest(public_dir)
    report("data_prepared", private_dir=str(private_dir), public_dir=str(public_dir))


def load_partition(private_dir, name, expected_per_class):
    with np.load(Path(private_dir) / f"{name}.npz", allow_pickle=False) as archive:
        if set(archive.files) != {"images", "labels"}:
            raise ValueError(f"Unexpected fields in {name}.npz.")
        images, labels = archive["images"], archive["labels"]
    if (images.dtype != np.uint8 or images.shape != (len(CLASSES) * expected_per_class, *IMAGE_SHAPE)
            or labels.dtype != np.int64
            or not np.array_equal(labels, np.repeat(np.arange(len(CLASSES)), expected_per_class))):
        raise ValueError(f"Invalid prepared partition: {name}.")
    return images, labels


def train_evaluator(args):
    if args.epochs < 1 or args.epochs > 20:
        raise ValueError("Train the evaluator for between 1 and 20 epochs.")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    torch.use_deterministic_algorithms(True)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable for evaluator preparation.")
    if device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable for evaluator preparation.")
    public_dir, private_dir = Path(args.public_dir).resolve(), Path(args.private_dir).resolve()
    training, training_labels = load_partition(private_dir, "evaluator_train", 10000)
    validation, validation_labels = load_partition(private_dir, "evaluator_validation", 2000)
    calibration, _ = load_partition(private_dir, "calibration", 512)
    classifier = nn.Sequential(nn.Flatten(), nn.Linear(784, 128), nn.ReLU(),
                               nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, len(CLASSES))).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3, weight_decay=1e-4)
    train_pixels = torch.from_numpy(training.astype(np.float32) / 255.0)
    validation_pixels = torch.from_numpy(validation.astype(np.float32) / 255.0)
    loader = DataLoader(TensorDataset(train_pixels, torch.from_numpy(training_labels)),
                        batch_size=256, shuffle=True, num_workers=0,
                        generator=torch.Generator().manual_seed(args.seed))
    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        classifier.train()
        training_sum = 0.0
        for pixels, labels in loader:
            pixels, labels = pixels.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(classifier(pixels), labels)
            loss.backward()
            optimizer.step()
            training_sum += float(loss.detach().cpu()) * len(labels)
        classifier.eval()
        validation_sum = 0.0
        correct = 0
        with torch.no_grad():
            for start in range(0, len(validation), 512):
                pixels = validation_pixels[start:start + 512].to(device)
                labels = torch.from_numpy(validation_labels[start:start + 512]).to(device)
                logits = classifier(pixels)
                validation_sum += float(nn.functional.cross_entropy(logits, labels, reduction="sum").cpu())
                correct += int((logits.argmax(1) == labels).sum().cpu())
        validation_loss = validation_sum / len(validation)
        row = {"epoch": epoch, "train_cross_entropy": training_sum / len(training),
               "validation_cross_entropy": validation_loss,
               "validation_accuracy": correct / len(validation)}
        history.append(row)
        report("evaluator_epoch", **row)
        if validation_loss < best_loss:
            best_loss, best_epoch = validation_loss, epoch
            best_state = {key: value.detach().cpu().clone() for key, value in classifier.state_dict().items()}
    classifier.load_state_dict(best_state)
    classifier.eval()
    arrays = {
        "format_version": np.array(1, dtype=np.int64),
        "challenge_version": np.array(CHALLENGE_VERSION), "metric_version": np.array(METRIC_VERSION),
        "classes": np.array(CLASSES),
        "w1": best_state["1.weight"].numpy().T.astype(np.float64),
        "b1": best_state["1.bias"].numpy().astype(np.float64),
        "w2": best_state["3.weight"].numpy().T.astype(np.float64),
        "b2": best_state["3.bias"].numpy().astype(np.float64),
        "w3": best_state["5.weight"].numpy().T.astype(np.float64),
        "b3": best_state["5.bias"].numpy().astype(np.float64),
        "mean_h": np.zeros(64), "std_h": np.ones(64),
        "mean_p": np.zeros(49), "std_p": np.ones(49), "bandwidths": np.ones(len(CLASSES)),
    }
    draft_path = private_dir / "evaluator_draft.npz"
    atomic_npz(draft_path, **arrays)
    draft = SketchEvaluator(draft_path)
    sum_h, square_h = np.zeros(64), np.zeros(64)
    sum_p, square_p = np.zeros(49), np.zeros(49)
    for start in range(0, len(training), 256):
        pixels = training[start:start + 256].astype(np.float64) / 255.0
        hidden = draft._hidden(pixels)
        pooled = pixels.reshape(-1, 7, 4, 7, 4).mean(axis=(2, 4)).reshape(-1, 49)
        sum_h += hidden.sum(axis=0)
        square_h += (hidden * hidden).sum(axis=0)
        sum_p += pooled.sum(axis=0)
        square_p += (pooled * pooled).sum(axis=0)
    for name, total, square in (("h", sum_h, square_h), ("p", sum_p, square_p)):
        mean = total / len(training)
        std = np.sqrt(np.maximum(square / len(training) - mean * mean, 0.0))
        arrays["mean_" + name] = mean
        arrays["std_" + name] = np.maximum(std, STANDARD_DEVIATION_FLOOR)
    atomic_npz(draft_path, **arrays)
    draft = SketchEvaluator(draft_path)
    calibration_features = draft.features(calibration).reshape(len(CLASSES), 512, -1)
    upper = np.triu_indices(512, k=1)
    for label, features in enumerate(calibration_features):
        distances = np.sqrt(squared_distances(features, features)[upper])
        nonzero = distances[distances > 0.0]
        if not len(nonzero):
            raise ValueError(f"Calibration features collapsed for {CLASSES[label]}.")
        arrays["bandwidths"][label] = np.median(nonzero)
    atomic_npz(public_dir / "evaluator.npz", **arrays)
    evaluator = SketchEvaluator(public_dir / "evaluator.npz")
    predictions = evaluator.classify(validation)
    recalls = {category: float(np.mean(predictions[validation_labels == label] == label))
               for label, category in enumerate(CLASSES)}
    balanced_accuracy = float(np.mean(list(recalls.values())))
    with torch.no_grad():
        torch_logits = classifier(validation_pixels[:512].to(device)).detach().cpu().numpy()
    pixels = validation[:512].astype(np.float64) / 255.0
    numpy_logits = evaluator._hidden(pixels) @ arrays["w3"] + arrays["b3"]
    parity_error = float(np.max(np.abs(numpy_logits - torch_logits)))
    draft_path.unlink(missing_ok=True)
    training_report = {
        "selected_epoch": best_epoch, "epochs": args.epochs, "seed": args.seed,
        "device": device, "torch_version": torch.__version__, "numpy_version": np.__version__,
        "maximum_numpy_torch_logit_error": parity_error,
        "weights_sha256": sha256_file(public_dir / "evaluator.npz"), "history": history,
    }
    finish_evaluator(public_dir, private_dir, training_report)


def finish_evaluator(public_dir, private_dir, training_report):
    evaluator = SketchEvaluator(public_dir / "evaluator.npz")
    validation, validation_labels = load_partition(private_dir, "evaluator_validation", 2000)
    reference_images, _ = load_partition(private_dir, "private_reference", REFERENCE_COUNT)
    predictions = evaluator.classify(validation)
    recalls = {category: float(np.mean(predictions[validation_labels == label] == label))
               for label, category in enumerate(CLASSES)}
    balanced_accuracy = float(np.mean(list(recalls.values())))
    references = evaluator.prepare_references(reference_images.reshape(len(CLASSES), REFERENCE_COUNT, *IMAGE_SHAPE))
    server_dir = private_dir / "server"
    server_dir.mkdir(parents=True, exist_ok=True)
    atomic_npz(server_dir / "private_reference.npz", **references)
    shutil.copyfile(public_dir / "evaluator.npz", server_dir / "evaluator.npz")
    sanity = metric_sanity(evaluator, references, public_dir, private_dir, training_report["seed"])
    parity_error = training_report["maximum_numpy_torch_logit_error"]
    gates = {
        "balanced_accuracy_at_least_90_percent": balanced_accuracy >= 0.90,
        "every_class_recall_at_least_85_percent": min(recalls.values()) >= 0.85,
        "numpy_torch_logits_agree": parity_error < 1e-4,
        "human_sketches_beat_degenerate_batches": sanity["passed"],
    }
    training_report = {**training_report,
        "balanced_accuracy": balanced_accuracy, "recall_by_class": recalls,
        "bandwidths": evaluator.arrays["bandwidths"].tolist(),
        "evaluator_fingerprint": evaluator.fingerprint,
        "metric_sanity": sanity, "gates": gates, "passed": all(gates.values()),
    }
    atomic_json(private_dir / "evaluator_report.json", training_report)
    manifest = refresh_manifest(public_dir)
    manifest["status"] = "evaluator_ready" if training_report["passed"] else "quality_gate_failed"
    manifest["evaluator_validation"] = {"balanced_accuracy": balanced_accuracy,
                                        "recall_by_class": recalls, "gates": gates}
    atomic_json(public_dir / "manifest.json", manifest)
    reference_path = server_dir / "private_reference.npz"
    server_manifest = {
        "challenge_version": CHALLENGE_VERSION, "classes": list(CLASSES),
        "image_shape": list(IMAGE_SHAPE), "samples_per_class": SAMPLES_PER_CLASS,
        "reference_count": REFERENCE_COUNT,
        "dataset_fingerprint": manifest["dataset_fingerprint"],
        "evaluator_fingerprint": evaluator.fingerprint,
        "reference_fingerprint": sha256_file(reference_path),
        "quality_gates_passed": training_report["passed"],
        "files": {
            "evaluator": file_entry(server_dir / "evaluator.npz", "evaluator.npz"),
            "references": file_entry(reference_path, "private_reference.npz"),
        },
    }
    atomic_json(server_dir / "servermanifest.json", server_manifest)
    report("evaluator_prepared", **{key: training_report[key] for key in
                                   ("selected_epoch", "balanced_accuracy", "recall_by_class", "passed", "gates")})
    if not training_report["passed"]:
        raise RuntimeError("Evaluator quality gates failed; inspect the private evaluator_report.json.")


def freeze_evaluator(args):
    """Refresh artifact fingerprints/references after scoring-code maintenance."""
    public_dir, private_dir = Path(args.public_dir).resolve(), Path(args.private_dir).resolve()
    report_path = private_dir / "evaluator_report.json"
    training_report = json.loads(report_path.read_text(encoding="utf-8"))
    if training_report.get("weights_sha256") != sha256_file(public_dir / "evaluator.npz"):
        raise ValueError("Evaluator weights changed; train-evaluator must prepare their training report.")
    finish_evaluator(public_dir, private_dir, training_report)


def metric_sanity(evaluator, references, public_dir, private_dir, seed):
    audit, _ = load_partition(private_dir, "audit", PARTITION_COUNTS["audit"])
    audit = audit.reshape(len(CLASSES), PARTITION_COUNTS["audit"], *IMAGE_SHAPE)
    with np.load(public_dir / "train.npz", allow_pickle=False) as archive:
        training = archive["images"].reshape(len(CLASSES), PARTITION_COUNTS["student_train"], *IMAGE_SHAPE)
    means = np.floor(training.astype(np.float64).mean(axis=1) + 0.5).astype(np.uint8)
    shape = (len(CLASSES), SAMPLES_PER_CLASS, *IMAGE_SHAPE)
    fixed_results = {
        "blank": evaluator.score(np.zeros(shape, dtype=np.uint8), references),
        "repeated_class_mean": evaluator.score(np.repeat(means[:, None], SAMPLES_PER_CLASS, axis=1), references),
    }
    degenerate = ("blank", "noise", "wrong_categories", "repeated_human", "repeated_class_mean")
    runs = []
    for iteration in range(20):
        batch_seed = seed + 100 + iteration
        rng = np.random.default_rng(batch_seed)
        human = np.stack([row[rng.choice(len(row), size=SAMPLES_PER_CLASS, replace=False)]
                          for row in audit])
        batches = {
            "held_out_human": human,
            "noise": rng.integers(0, 256, size=shape, dtype=np.uint8),
            "wrong_categories": np.roll(human, shift=1 + iteration % (len(CLASSES) - 1), axis=0),
            "repeated_human": np.repeat(human[:, :1], SAMPLES_PER_CLASS, axis=1),
            "resampled_training": np.stack([row[rng.integers(0, len(row), SAMPLES_PER_CLASS)]
                                             for row in training]),
        }
        results = {**fixed_results,
                   **{name: evaluator.score(batch, references) for name, batch in batches.items()}}
        runs.append({
            "seed": batch_seed,
            "scores": {name: value["score"] for name, value in results.items()},
            "scores_by_class": {name: value["scores_by_class"] for name, value in results.items()},
            "diagnostics": {name: value["diagnostics"] for name, value in results.items()},
        })

    def summarize(values):
        values = np.asarray(values, dtype=np.float64)
        return {"mean": float(values.mean()), "std": float(values.std(ddof=1)),
                "min": float(values.min()), "max": float(values.max())}

    names = tuple(runs[0]["scores"])
    summary = {name: summarize([run["scores"][name] for run in runs]) for name in names}
    comparisons = {}
    per_class_comparisons = {category: {} for category in CLASSES}
    for name in degenerate:
        margins = np.array([run["scores"][name] - run["scores"]["held_out_human"] for run in runs])
        comparisons[name] = {"human_win_count": int(np.sum(margins > 0.0)),
                             "worst_margin": float(margins.min()), "mean_margin": float(margins.mean())}
        for category in CLASSES:
            margins = np.array([run["scores_by_class"][name][category]
                                - run["scores_by_class"]["held_out_human"][category] for run in runs])
            per_class_comparisons[category][name] = {
                "human_win_count": int(np.sum(margins > 0.0)),
                "worst_margin": float(margins.min()), "mean_margin": float(margins.mean()),
            }
    passed = all(value["human_win_count"] == len(runs) for value in comparisons.values())
    report("metric_sanity", passed=passed, batch_count=len(runs), summary=summary,
           human_win_counts={name: value["human_win_count"] for name, value in comparisons.items()})
    return {
        "passed": passed, "batch_count": len(runs), "batch_seed_base": seed + 100,
        "sampling": "Each independently seeded batch samples 200 distinct images per class from the disjoint audit pool; images may recur between batches.",
        "scores": runs[0]["scores"], "summary": summary, "comparisons": comparisons,
        "per_class_comparisons": per_class_comparisons,
        "scores_by_class_summary": {
            category: {name: summarize([run["scores_by_class"][name][category] for run in runs])
                       for name in names} for category in CLASSES
        },
        "batches": runs,
        "note": "Training-image resampling is an instructor calibration baseline; distribution scoring assesses the submitted batch. Per-class comparisons are diagnostic; the preparation gate uses every aggregate batch comparison.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    data = subparsers.add_parser("prepare-data", help="Prepare disjoint public and private bitmap subsets.")
    data.add_argument("--cache-dir", required=True)
    data.add_argument("--private-dir", required=True)
    data.add_argument("--public-dir", default=str(CHALLENGE_ROOT / "data"))
    data.add_argument("--seed", type=int, default=DEFAULT_SEED)
    data.set_defaults(action=prepare_data)
    train = subparsers.add_parser("train-evaluator", help="Train and freeze the instructor's sketch evaluator.")
    train.add_argument("--private-dir", required=True)
    train.add_argument("--public-dir", default=str(CHALLENGE_ROOT / "data"))
    train.add_argument("--device", choices=("cpu", "cuda", "mps", "auto"), default="cpu")
    train.add_argument("--epochs", type=int, default=20)
    train.add_argument("--seed", type=int, default=DEFAULT_SEED)
    train.set_defaults(action=train_evaluator)
    freeze = subparsers.add_parser("freeze-evaluator", help="Refresh references after code changes that preserve features.")
    freeze.add_argument("--private-dir", required=True)
    freeze.add_argument("--public-dir", default=str(CHALLENGE_ROOT / "data"))
    freeze.set_defaults(action=freeze_evaluator)
    manifest = subparsers.add_parser("manifest", help="Refresh public file checksums after helper-module changes.")
    manifest.add_argument("--public-dir", default=str(CHALLENGE_ROOT / "data"))
    manifest.set_defaults(action=lambda args: refresh_manifest(args.public_dir))
    args = parser.parse_args()
    try:
        args.action(args)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Preparation failed: {error}\n")


if __name__ == "__main__":
    main()
