"""Frozen sketch features and the course's class-conditional MMD score.

This module requires NumPy. The notebook and submission server use identical
copies. Images contain ink intensities: 0 is background and 255 is full ink.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

CLASSES = ("apple", "house", "fish", "tree", "umbrella")
CHALLENGE_VERSION = "sketch-generation-v1"
METRIC_VERSION = "sketch-mmd-rbf-v1"
SAMPLES_PER_CLASS = 200
REFERENCE_COUNT = 512
IMAGE_SHAPE = (28, 28)
FEATURE_DIM = 113
STANDARD_DEVIATION_FLOOR = 0.05
NORM_FLOOR = 1e-8
SCORE_SCALE = 1000.0
KERNEL_MULTIPLIERS = (0.5, 1.0, 2.0)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluator_fingerprint(weights_path):
    """Bind scoring code, frozen weights, and the metric's definition."""
    definition = {
        "weights_sha256": sha256_file(weights_path),
        "evaluator_code_sha256": sha256_file(__file__),
        "metric_version": METRIC_VERSION,
    }
    encoded = json.dumps(definition, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _images(value, *, expected=None):
    value = np.asarray(value)
    if value.dtype != np.uint8 or value.ndim != 3 or value.shape[1:] != IMAGE_SHAPE:
        raise ValueError("Images must be a uint8 array with shape [N,28,28].")
    if expected is not None and len(value) != expected:
        raise ValueError(f"Expected exactly {expected} images.")
    return value


def _unit_vectors(values):
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), NORM_FLOOR)


def squared_distances(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    distances = ((left * left).sum(axis=1)[:, None]
                 + (right * right).sum(axis=1)[None, :] - 2.0 * left @ right.T)
    return np.maximum(distances, 0.0)


def kernel_matrix(left, right, bandwidth):
    distances = squared_distances(left, right)
    result = np.zeros_like(distances)
    for multiplier in KERNEL_MULTIPLIERS:
        result += np.exp(-distances / (2.0 * (bandwidth * multiplier) ** 2))
    return result / len(KERNEL_MULTIPLIERS)


class SketchEvaluator:
    """Evaluate quantized sketches with a frozen MLP and pooled image features."""

    def __init__(self, weights_path):
        self.weights_path = Path(weights_path)
        expected = {
            "format_version", "challenge_version", "metric_version", "classes",
            "w1", "b1", "w2", "b2", "w3", "b3", "mean_h", "std_h",
            "mean_p", "std_p", "bandwidths",
        }
        with np.load(self.weights_path, allow_pickle=False) as archive:
            if set(archive.files) != expected:
                raise ValueError("The evaluator archive has unexpected fields.")
            metadata = {key: archive[key] for key in
                        ("format_version", "challenge_version", "metric_version", "classes")}
            arrays = {key: np.asarray(archive[key], dtype=np.float64)
                      for key in expected - set(metadata)}
        if (metadata["format_version"].shape != () or int(metadata["format_version"]) != 1
                or metadata["challenge_version"].shape != ()
                or str(metadata["challenge_version"]) != CHALLENGE_VERSION
                or metadata["metric_version"].shape != ()
                or str(metadata["metric_version"]) != METRIC_VERSION
                or metadata["classes"].tolist() != list(CLASSES)):
            raise ValueError("The evaluator archive belongs to another challenge version.")
        shapes = {"w1": (784, 128), "b1": (128,), "w2": (128, 64), "b2": (64,),
                  "w3": (64, 5), "b3": (5,), "mean_h": (64,), "std_h": (64,),
                  "mean_p": (49,), "std_p": (49,), "bandwidths": (5,)}
        for key, shape in shapes.items():
            if arrays[key].shape != shape or not np.isfinite(arrays[key]).all():
                raise ValueError(f"Invalid evaluator array: {key}.")
        if ((arrays["std_h"] < STANDARD_DEVIATION_FLOOR).any()
                or (arrays["std_p"] < STANDARD_DEVIATION_FLOOR).any()
                or (arrays["bandwidths"] <= 1e-6).any()):
            raise ValueError("Invalid evaluator feature scales or kernel bandwidths.")
        self.arrays = arrays
        self.fingerprint = evaluator_fingerprint(self.weights_path)

    def _hidden(self, pixels):
        arrays = self.arrays
        first = np.maximum(pixels.reshape(-1, 784) @ arrays["w1"] + arrays["b1"], 0.0)
        return np.maximum(first @ arrays["w2"] + arrays["b2"], 0.0)

    def features(self, images):
        """Return [N,113] float64 features, batching to bound temporary memory."""
        images = _images(images)
        result = np.empty((len(images), FEATURE_DIM), dtype=np.float64)
        arrays = self.arrays
        for start in range(0, len(images), 256):
            pixels = images[start:start + 256].astype(np.float64) / 255.0
            hidden = (self._hidden(pixels) - arrays["mean_h"]) / arrays["std_h"]
            pooled = pixels.reshape(-1, 7, 4, 7, 4).mean(axis=(2, 4)).reshape(-1, 49)
            pooled = (pooled - arrays["mean_p"]) / arrays["std_p"]
            result[start:start + len(pixels)] = np.concatenate(
                (_unit_vectors(hidden), _unit_vectors(pooled)), axis=1) / np.sqrt(2.0)
        return result

    def classify(self, images):
        """Return predicted integer category labels for a batch of images."""
        images = _images(images)
        result = np.empty(len(images), dtype=np.int64)
        for start in range(0, len(images), 256):
            pixels = images[start:start + 256].astype(np.float64) / 255.0
            logits = self._hidden(pixels) @ self.arrays["w3"] + self.arrays["b3"]
            result[start:start + len(pixels)] = logits.argmax(axis=1)
        return result

    def prepare_references(self, images):
        """Return numeric arrays for a [5,512,28,28] reference archive.

        Keys: features[5,512,113], squared_norms[5,512], kernel_mean[5],
        evaluator_digest[32]. The digest contains the fingerprint's raw bytes.
        """
        images = np.asarray(images)
        if images.shape != (len(CLASSES), REFERENCE_COUNT, *IMAGE_SHAPE):
            raise ValueError("Reference images must have shape [5,512,28,28].")
        features = self.features(images.reshape(-1, *IMAGE_SHAPE)).reshape(
            len(CLASSES), REFERENCE_COUNT, FEATURE_DIM)
        return {
            "features": features,
            "squared_norms": (features * features).sum(axis=2),
            "kernel_mean": np.array([
                kernel_matrix(row, row, self.arrays["bandwidths"][index]).mean()
                for index, row in enumerate(features)
            ], dtype=np.float64),
            "evaluator_digest": np.frombuffer(bytes.fromhex(self.fingerprint), dtype=np.uint8).copy(),
        }

    def _references(self, references):
        expected = {"features", "squared_norms", "kernel_mean", "evaluator_digest"}
        if set(references) != expected:
            raise ValueError("Unexpected reference archive fields.")
        shapes = {"features": (len(CLASSES), REFERENCE_COUNT, FEATURE_DIM),
                  "squared_norms": (len(CLASSES), REFERENCE_COUNT), "kernel_mean": (len(CLASSES),),
                  "evaluator_digest": (32,)}
        for key, shape in shapes.items():
            array = np.asarray(references[key])
            if array.shape != shape or not np.isfinite(array).all():
                raise ValueError(f"Invalid reference array: {key}.")
        digest = np.asarray(references["evaluator_digest"])
        if digest.dtype != np.uint8 or digest.tobytes().hex() != self.fingerprint:
            raise ValueError("Reference features use a different evaluator.")
        features = np.asarray(references["features"], dtype=np.float64)
        if not np.allclose((features * features).sum(axis=2), references["squared_norms"],
                           rtol=1e-12, atol=1e-12):
            raise ValueError("Reference feature norms are inconsistent.")
        if ((np.asarray(references["kernel_mean"]) < 0.0).any()
                or (np.asarray(references["kernel_mean"]) > 1.0 + 1e-12).any()):
            raise ValueError("Invalid reference kernel mean.")
        return features

    def score(self, samples, references):
        """Score exactly 200 generated images for each of the five categories."""
        samples = np.asarray(samples)
        if samples.shape != (len(CLASSES), SAMPLES_PER_CLASS, *IMAGE_SHAPE):
            raise ValueError("Generated samples must have shape [5,200,28,28].")
        flat = _images(samples.reshape(-1, *IMAGE_SHAPE), expected=len(CLASSES) * SAMPLES_PER_CLASS)
        reference_features = self._references(references)
        generated = self.features(flat).reshape(len(CLASSES), SAMPLES_PER_CLASS, FEATURE_DIM)
        predictions = self.classify(flat).reshape(len(CLASSES), SAMPLES_PER_CLASS)
        scores = {}
        diagnostics = {}
        for index, category in enumerate(CLASSES):
            row = generated[index]
            bandwidth = self.arrays["bandwidths"][index]
            distance = (kernel_matrix(row, row, bandwidth).mean()
                        + float(references["kernel_mean"][index])
                        - 2.0 * kernel_matrix(row, reference_features[index], bandwidth).mean())
            if not np.isfinite(distance) or distance < -1e-12:
                raise ValueError("The MMD calculation produced an invalid distance.")
            scores[category] = SCORE_SCALE * max(float(distance), 0.0)
            images = samples[index]
            unique = len(np.unique(images.reshape(SAMPLES_PER_CLASS, -1), axis=0))
            diagnostics[category] = {
                "recognition_percent": 100.0 * float(np.mean(predictions[index] == index)),
                "duplicate_fraction": 1.0 - unique / SAMPLES_PER_CLASS,
                "mean_ink_intensity": float(images.mean() / 255.0),
                "blank_count": int(np.sum(np.count_nonzero(images > 32, axis=(1, 2)) < 5)),
            }
        return {"score": float(np.mean(list(scores.values()))),
                "scores_by_class": scores, "diagnostics": diagnostics}
