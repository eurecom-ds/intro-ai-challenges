# ---
# jupyter:
#   colab:
#     name: network-latency.ipynb
#     provenance: []
#   jupytext:
#     cell_metadata_filter: all
#     notebook_metadata_filter: colab,kernelspec
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Will this video call lag?
#
# [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eurecom-ds/intro-ai-challenges/blob/main/network-latency/notebook.ipynb)
#
# <div style="background:#0f2746;color:#f1f5f9;border:1px solid #234566;border-radius:14px;padding:18px 20px;margin:16px 0;line-height:1.55;"><div style="color:#93c5fd;font-size:11px;font-weight:700;letter-spacing:.1em;text-transform:uppercase;margin-bottom:6px;">Hands-on challenge</div><div style="font-size:20px;font-weight:700;margin-bottom:5px;">Predict latency from network measurements</div><div style="font-size:14px;">45–60 minutes</div></div>
#
# During a video call, frames can arrive with different delays. Consider two equally sized frames sent under different traffic loads. Which would you expect to arrive later? What information would help you make that prediction?
#
# Each row in this challenge describes one frame. Its **inputs** are ten measurements, including frame size, bandwidth and traffic load. Its **target** is the observed delivery latency in milliseconds. The data are a fictional teaching example.
#
# Your task is **regression**: learn a relationship between inputs and a numerical target, then use it to predict latency for new frames.
#
# Fit a model on **training frames**, using their measurements and observed latencies. Then predict latencies for **validation frames**, whose observed latencies are held out during fitting. Compare predictions with observations using **RMSE**, which summarizes prediction error in milliseconds. A lower RMSE indicates better predictive accuracy on those frames.
#
# Start with a degree-0 baseline that predicts the same latency for every frame. Then compare higher-degree polynomial models and a neural network. Investigate how model flexibility and regularization affect training and validation error. Before each change, predict its effect, run the experiment, and explain what you observe.
#
# <div style="display:flex;flex-wrap:wrap;gap:10px;margin:20px 0;"><div style="flex:1 1 145px;min-width:0;background:#f8fafc;color:#334155;border:1px solid #e2e8f0;border-radius:12px;padding:14px;"><div style="color:#2563eb;font-size:11px;font-weight:700;letter-spacing:.06em;">01 · 10 min</div><div style="color:#0f172a;font-weight:700;font-size:14px;margin:6px 0;">Inspect the data</div><div style="font-size:12px;line-height:1.5;">Measurements and training/validation split</div></div><div style="flex:1 1 145px;min-width:0;background:#f8fafc;color:#334155;border:1px solid #e2e8f0;border-radius:12px;padding:14px;"><div style="color:#2563eb;font-size:11px;font-weight:700;letter-spacing:.06em;">02 · 10 min</div><div style="color:#0f172a;font-weight:700;font-size:14px;margin:6px 0;">Run the baseline</div><div style="font-size:12px;line-height:1.5;">Record the starting score</div></div><div style="flex:1 1 145px;min-width:0;background:#f8fafc;color:#334155;border:1px solid #e2e8f0;border-radius:12px;padding:14px;"><div style="color:#2563eb;font-size:11px;font-weight:700;letter-spacing:.06em;">03 · 15 min</div><div style="color:#0f172a;font-weight:700;font-size:14px;margin:6px 0;">Compare model settings</div><div style="font-size:12px;line-height:1.5;">Investigate degree and regularization</div></div><div style="flex:1 1 145px;min-width:0;background:#f8fafc;color:#334155;border:1px solid #e2e8f0;border-radius:12px;padding:14px;"><div style="color:#2563eb;font-size:11px;font-weight:700;letter-spacing:.06em;">04 · 10 min</div><div style="color:#0f172a;font-weight:700;font-size:14px;margin:6px 0;">Compare models</div><div style="font-size:12px;line-height:1.5;">Neural network and model choice</div></div><div style="flex:1 1 145px;min-width:0;background:#f8fafc;color:#334155;border:1px solid #e2e8f0;border-radius:12px;padding:14px;"><div style="color:#2563eb;font-size:11px;font-weight:700;letter-spacing:.06em;">05 · 5 min</div><div style="color:#0f172a;font-weight:700;font-size:14px;margin:6px 0;">Upload results</div><div style="font-size:12px;line-height:1.5;">Selected model and remaining errors</div></div></div>

# %% [markdown]
# ## Setup
# Use a CPU runtime in Colab and run the next three setup cells in order. Setup downloads the two public data files automatically.
#
# For each experiment, choose the settings and press ▶ to fit and evaluate the model. Keep the original CSV files unchanged. Rerunning Setup 3 clears the experiment table and current selection.

# %% cellView="form" jupyter={"source_hidden": true}
# @title Setup 1: load libraries { display-mode: "form" }
import copy
import csv
from datetime import datetime, timezone
import getpass
import hashlib
import html
import io
from itertools import combinations_with_replacement
import json
from pathlib import Path
import re
import unicodedata
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, HTML, FileLink

plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False,
                     "axes.spines.right": False, "text.parse_math": False})

# Polynomial models use NumPy; neural-network fitting imports PyTorch on demand.
CHALLENGE_VERSION = "network-latency-v1"
FEATURE_NAMES = ("traffic_load", "link_quality", "frame_kb", "interference",
                 "bandwidth_mbps", "distance_km", "route_hops", "priority",
                 "diagnostic_a", "diagnostic_b")
TARGET_NAME = "latency_ms"
PUBLIC_SPLIT_SEED = 42
TRAIN_COUNT = 600
PUBLIC_COUNT = 1000
TEST_COUNT = 2400
BASELINE = (0, 0.0)
MAX_INPUT_FEATURES = 20
MAX_POLYNOMIAL_FEATURES = 2500
MODEL_FIELDS = frozenset({"model_family", "target_transform", "degree", "regularization",
                          "loss", "huber_delta",
                          "hidden_units", "learning_rate", "epochs", "weight_decay"})


def _number(value, name, *, maximum=None, positive=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"Set {name} to a finite {'positive' if positive else 'nonnegative'} number.")
    try:
        value = float(value)
    except (OverflowError, TypeError, ValueError):
        raise ValueError(f"Set {name} to a finite number.") from None
    if not np.isfinite(value) or (value <= 0 if positive else value < 0) or (maximum is not None and value > maximum):
        limit = f" no greater than {maximum:g}" if maximum is not None else ""
        raise ValueError(f"Set {name} to a finite {'positive' if positive else 'nonnegative'} number{limit}.")
    return value


def settings(degree=0, regularization=0.0, *, model_family="polynomial", target_transform="raw",
             loss="squared", huber_delta=1.0,
             hidden_units=32, learning_rate=0.01, epochs=1000, weight_decay=0.001):
    """Validate active controls; inactive controls do not identify another model."""
    if model_family not in ("polynomial", "neural_network"):
        raise ValueError("Set model_family to 'polynomial' or 'neural_network'.")
    if target_transform not in ("raw", "log1p"):
        raise ValueError("Set target_transform to 'raw' or 'log1p'.")
    config = {"model_family": model_family, "target_transform": target_transform}
    if model_family == "neural_network":
        if isinstance(hidden_units, (bool, np.bool_)) or not isinstance(hidden_units, (int, np.integer)) or not 4 <= hidden_units <= 128:
            raise ValueError("Set hidden_units to a whole number from 4 to 128.")
        if isinstance(epochs, (bool, np.bool_)) or not isinstance(epochs, (int, np.integer)) or not 1 <= epochs <= 5000:
            raise ValueError("Set epochs to a whole number from 1 to 5000.")
        config.update(hidden_units=int(hidden_units),
                      learning_rate=_number(learning_rate, "learning_rate", maximum=1, positive=True),
                      epochs=int(epochs),
                      weight_decay=_number(weight_decay, "weight_decay", maximum=10))
        return config
    if isinstance(degree, (bool, np.bool_)) or not isinstance(degree, (int, np.integer)) or not 0 <= degree <= 4:
        raise ValueError("Set degree to a whole number from 0 to 4.")
    regularization = _number(regularization, "regularization", maximum=1_000_000)
    if loss not in ("squared", "huber"):
        raise ValueError("Set loss to 'squared' or 'huber'.")
    config.update(degree=int(degree), regularization=regularization, loss=loss)
    if loss == "huber":
        config["huber_delta"] = _number(huber_delta, "huber_delta", positive=True)
    return config


def _record_settings(record):
    if not isinstance(record, dict):
        raise ValueError("A model configuration must be a dictionary.")
    return settings(**{key: record[key] for key in MODEL_FIELDS if key in record})


def configuration_key(record):
    return tuple(sorted(_record_settings(record).items()))


def describe_settings(config):
    config = _record_settings(config)
    if config["model_family"] == "neural_network":
        return (f"Neural network | hidden={config['hidden_units']} | epochs={config['epochs']} | "
                f"η={config['learning_rate']:g} | weight decay={config['weight_decay']:g} | "
                f"{config['target_transform']}")
    loss_label = "squared" if config["loss"] == "squared" else f"Huber δ={config['huber_delta']:g}"
    return f"Polynomial | d={config['degree']} | λ={config['regularization']:g} | {loss_label} | {config['target_transform']}"


def _vector(values, name):
    try:
        values = np.asarray(values, dtype=float)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{name} must contain finite numbers.") from None
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError(f"{name} must be a nonempty one-dimensional array of finite numbers.")
    return values


def _matrix(values, name, *, raw=False):
    try:
        values = np.asarray(values, dtype=float)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{name} must contain finite numbers.") from None
    if values.ndim != 2 or not values.shape[0] or not 1 <= values.shape[1] <= MAX_INPUT_FEATURES or not np.isfinite(values).all():
        raise ValueError(f"{name} must be a finite two-dimensional matrix with 1–{MAX_INPUT_FEATURES} feature columns.")
    if raw and values.shape[1] != len(FEATURE_NAMES):
        raise ValueError(f"{name} must have the {len(FEATURE_NAMES)} original feature columns in FEATURE_NAMES order.")
    return values


def arrays(rows, include_target=True):
    """Read CSV dictionaries in the documented, fixed feature order."""
    if not rows:
        raise ValueError("The data file is empty.")
    try:
        x = _matrix([[row[name] for name in FEATURE_NAMES] for row in rows], "Data features", raw=True)
        if include_target:
            return x, _vector([row[TARGET_NAME] for row in rows], "Data targets")
        return x
    except KeyError as error:
        raise ValueError(f"The data file is missing column {error.args[0]}.") from None


def _terms(feature_count, degree):
    terms = []
    for order in range(1, degree + 1):
        for term in combinations_with_replacement(range(feature_count), order):
            terms.append(term)
            if len(terms) > MAX_POLYNOMIAL_FEATURES:
                raise ValueError(f"This representation expands to more than {MAX_POLYNOMIAL_FEATURES:,} polynomial terms. Reduce degree or the number of input features.")
    return tuple(terms)


def _polynomial_features(x, terms):
    if not terms:
        return np.empty((len(x), 0))
    return np.column_stack([np.prod(x[:, term], axis=1) for term in terms])


def _design(model, x):
    if x.shape[1] != len(model["center"]):
        raise ValueError("Prediction features must have the same columns as the training representation.")
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            standardized = (x - model["center"]) / model["scale"]
            polynomial = _polynomial_features(standardized, model["terms"])
            features = (polynomial - model["feature_mean"]) / model["feature_scale"]
    except FloatingPointError:
        raise ValueError("Polynomial features overflowed. Reduce degree and inspect your feature ranges.") from None
    return np.column_stack([np.ones(len(x)), features])


def _least_squares(design, target, regularization, weights=None):
    """Solve ridge with an unpenalized intercept, using SVD (no normal equations).

    The objective is mean(weight_i * residual_i**2) + lambda * ||w||**2.
    Weighted centering removes the intercept before the singular-value solve.
    """
    n = len(target)
    weights = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    total_weight = weights.sum()
    x = design[:, 1:]
    x_mean = np.sum(weights[:, None] * x, axis=0) / total_weight
    y_mean = float(np.sum(weights * target) / total_weight)
    root_weights = np.sqrt(weights)
    centered_x = (x - x_mean) * root_weights[:, None]
    centered_y = (target - y_mean) * root_weights
    try:
        if not x.shape[1]:
            slope = np.empty(0)
        elif regularization == 0:
            slope = np.linalg.lstsq(centered_x, centered_y, rcond=None)[0]
        else:
            left, singular, right = np.linalg.svd(centered_x, full_matrices=False)
            shrinkage = singular / (singular ** 2 + n * regularization)
            slope = right.T @ (shrinkage * (left.T @ centered_y))
    except np.linalg.LinAlgError:
        raise ValueError("The least-squares solver did not converge. Reduce degree or inspect your features.") from None
    coefficients = np.concatenate([[y_mean - x_mean @ slope], slope])
    if not np.isfinite(coefficients).all():
        raise ValueError("The fit produced non-finite coefficients. Reduce degree or increase regularization.")
    return coefficients


def _huber_fit(design, target, regularization, delta):
    """IRLS for mean rho_delta(r) + lambda*||w||² in standardized target units.

    rho is r² inside the threshold and 2*delta*abs(r)-delta² outside,
    matching squared loss and ridge scaling on the quadratic branch.
    """
    coefficients = _least_squares(design, target, regularization)
    for iteration in range(1, 201):
        fitted = design @ coefficients
        residual = target - fitted
        absolute = np.abs(residual)
        weights = np.ones(len(target))
        outside = absolute > delta
        weights[outside] = delta / absolute[outside]
        updated = _least_squares(design, target, regularization, weights)
        updated_fit = design @ updated
        if not np.isfinite(updated_fit).all():
            raise ValueError("Huber fitting became numerically unstable. Reduce degree or increase regularization.")
        change = float(np.max(np.abs(updated_fit - fitted)))
        coefficient_change = float(np.linalg.norm(updated - coefficients))
        coefficient_size = float(np.linalg.norm(coefficients))
        coefficients = updated
        if change <= 1e-7 * (1 + float(np.max(np.abs(fitted)))) and coefficient_change <= 1e-7 * (1 + coefficient_size):
            return coefficients, iteration
    raise ValueError("Huber fitting did not converge. Reduce degree, increase regularization, or increase huber_delta.")


def _neural_fit(model, x, target):
    """Fit one ReLU hidden layer using only the supplied training examples.

    Features and transformed targets are standardized on these rows. Adam
    minimizes mean squared residual + weight_decay * sum(weight matrices²);
    neither bias is penalized. Training runs for exactly the selected number
    of full-batch updates, without consulting validation examples or labels.
    """
    try:
        import torch
    except ImportError:
        raise ValueError("Neural network fitting needs PyTorch. Run this notebook in Colab or the course compute environment.") from None
    target_mean, target_scale = float(target.mean()), float(target.std())
    if not np.isfinite(target_mean) or not np.isfinite(target_scale):
        raise ValueError("Training targets are too large to normalize safely.")
    target_scale = target_scale if target_scale > 0 else 1.0
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            inputs = (x - model["center"]) / model["scale"]
            targets = (target - target_mean) / target_scale
    except FloatingPointError:
        raise ValueError("Neural network inputs are too large to normalize safely. Inspect the feature ranges.") from None
    config = model["config"]
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        # Preserve the notebook's CPU random state; initialization is identical
        # for every rerun of a configuration, with no GPU or minibatch randomness.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(42)
            network = torch.nn.Sequential(
                torch.nn.Linear(x.shape[1], config["hidden_units"], dtype=torch.float64, device="cpu"),
                torch.nn.ReLU(),
                torch.nn.Linear(config["hidden_units"], 1, dtype=torch.float64, device="cpu"))
            input_tensor = torch.as_tensor(inputs, dtype=torch.float64, device="cpu")
            target_tensor = torch.as_tensor(targets, dtype=torch.float64, device="cpu")
            optimizer = torch.optim.Adam(network.parameters(), lr=config["learning_rate"], weight_decay=0)
            history = []
            for _ in range(config["epochs"]):
                optimizer.zero_grad(set_to_none=True)
                residual = network(input_tensor).squeeze(-1) - target_tensor
                mse = residual.square().mean()
                penalty = network[0].weight.square().sum() + network[2].weight.square().sum()
                objective = mse + config["weight_decay"] * penalty
                if not torch.isfinite(objective).item():
                    raise ValueError("Neural network training became non-finite. Reduce the learning rate or inspect the features.")
                history.append(float(mse.detach()))
                objective.backward()
                optimizer.step()
            with torch.no_grad():
                history.append(float((network(input_tensor).squeeze(-1) - target_tensor).square().mean()))
            arrays = {"hidden_weight": network[0].weight.detach().numpy().copy(),
                      "hidden_bias": network[0].bias.detach().numpy().copy(),
                      "output_weight": network[2].weight.detach().numpy().copy(),
                      "output_bias": network[2].bias.detach().numpy().copy()}
    finally:
        torch.set_num_threads(previous_threads)
    if not all(np.isfinite(value).all() for value in arrays.values()) or not np.isfinite(history).all():
        raise ValueError("Neural network training produced non-finite weights. Reduce the learning rate or inspect the features.")
    model.update(**arrays, target_mean=target_mean, target_scale=target_scale,
                 iterations=config["epochs"], training_history=history)
    return model


def fit_model(x, y, degree=0, regularization=0.0, **kwargs):
    """Fit the supplied measurements; all learned scaling is training-only."""
    config = settings(degree, regularization, **kwargs)
    x, y = _matrix(x, "Training features"), _vector(y, "Training targets")
    if len(x) != len(y):
        raise ValueError("Training features and targets must have the same number of rows.")
    if config["target_transform"] == "log1p":
        if np.any(y < 0):
            raise ValueError("The log1p target requires nonnegative training latencies.")
        target = np.log1p(y)
    else:
        target = y.copy()
    center, scale = x.mean(axis=0), x.std(axis=0)
    if not np.isfinite(center).all() or not np.isfinite(scale).all():
        raise ValueError("Training features are too large to normalize safely.")
    scale = np.where(scale > 0, scale, 1.0)
    model = dict(config, config=config.copy(), center=center, scale=scale)
    if config["model_family"] == "neural_network":
        return _neural_fit(model, x, target)
    terms = _terms(x.shape[1], config["degree"])
    model.update(terms=terms, feature_mean=np.zeros(len(terms)), feature_scale=np.ones(len(terms)))
    polynomial = _design(model, x)[:, 1:]
    model["feature_mean"] = polynomial.mean(axis=0)
    feature_scale = polynomial.std(axis=0)
    model["feature_scale"] = np.where(feature_scale > 0, feature_scale, 1.0)
    design = _design(model, x)
    target_mean, target_scale = float(target.mean()), float(target.std())
    if not np.isfinite(target_mean) or not np.isfinite(target_scale):
        raise ValueError("Training targets are too large to normalize safely.")
    target_scale = target_scale if target_scale > 0 else 1.0
    standardized_target = (target - target_mean) / target_scale
    if not terms and config["loss"] == "squared":
        # Zero in standardized target units is the training-target mean.
        coefficients, iterations = np.zeros(1), 1
    elif config["loss"] == "huber":
        coefficients, iterations = _huber_fit(design, standardized_target,
                                             config["regularization"], config["huber_delta"])
    else:
        coefficients = _least_squares(design, standardized_target, config["regularization"])
        iterations = 1
    model.update(coefficients=coefficients, target_mean=target_mean, target_scale=target_scale,
                 iterations=iterations, converged=True)
    return model


def _neural_predict(model, x):
    """Use saved NumPy weights; inference does not require importing PyTorch."""
    if x.shape[1] != len(model["center"]):
        raise ValueError("Prediction features must have the same columns as the training representation.")
    standardized = (x - model["center"]) / model["scale"]
    hidden = np.maximum(0, standardized @ model["hidden_weight"].T + model["hidden_bias"])
    transformed = (hidden @ model["output_weight"].T + model["output_bias"]).ravel()
    return transformed * model["target_scale"] + model["target_mean"]


def predict_model(model, x):
    x = _matrix(x, "Prediction features")
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            if model["model_family"] == "neural_network":
                prediction = _neural_predict(model, x)
            else:
                prediction = (_design(model, x) @ model["coefficients"]) * model["target_scale"] + model["target_mean"]
            if model["target_transform"] == "log1p":
                prediction = np.expm1(prediction)
    except FloatingPointError:
        raise ValueError("Predictions overflowed in the model or inverse log1p transform. Reduce complexity or increase regularization.") from None
    if not np.isfinite(prediction).all():
        raise ValueError("The model produced non-finite predictions. Choose a more stable configuration.")
    return prediction


def rmse(actual, predicted):
    actual, predicted = _vector(actual, "Observed latencies"), _vector(predicted, "Predicted latencies")
    if actual.shape != predicted.shape:
        raise ValueError("RMSE needs one prediction for each observed latency.")
    try:
        with np.errstate(over="raise", invalid="raise"):
            result = float(np.sqrt(np.mean((actual - predicted) ** 2)))
    except (FloatingPointError, OverflowError):
        raise ValueError("Prediction errors are too large for a finite RMSE. Choose a more stable configuration.") from None
    if not np.isfinite(result):
        raise ValueError("Prediction errors are too large for a finite RMSE. Choose a more stable configuration.")
    return result


def _regime_masks(raw_x):
    raw_x = _matrix(raw_x, "Original features", raw=True)
    load = raw_x[:, 0]
    return {"normal": load < 0.65, "busy": (load >= 0.65) & (load < 0.85),
            "congested": load >= 0.85}


def completion_issues(experiments, selected, student_name, student_surname):
    issues, by_key = [], {}
    for experiment in experiments:
        try:
            config = _record_settings(experiment)
            by_key[configuration_key(config)] = config
        except (KeyError, TypeError, ValueError):
            issues.append("An experiment has invalid settings; rerun that experiment.")
    configs = list(by_key.values())
    if configuration_key(settings(*BASELINE)) not in by_key:
        issues.append("Run the starting constant model: degree 0, regularization 0, squared loss and raw target.")
    if len(configs) < 5:
        issues.append("Run at least five distinct active configurations. Identical reruns do not count.")

    def comparison(field, *, polynomial_only=False):
        for a in configs:
            for b in configs:
                if a["model_family"] != b["model_family"] or (polynomial_only and a["model_family"] != "polynomial"):
                    continue
                if field not in a or field not in b or a[field] == b[field]:
                    continue
                if field == "regularization" and a.get("degree", 0) == 0:
                    continue
                if {key: value for key, value in a.items() if key != field} == {key: value for key, value in b.items() if key != field}:
                    return True
        return False

    if not comparison("degree", polynomial_only=True):
        issues.append("Compare polynomial degrees while holding the other active settings fixed.")
    if not comparison("regularization", polynomial_only=True):
        issues.append("Compare ridge strengths at degree 1 or higher while holding the other active polynomial settings fixed.")
    if not any(config["model_family"] == "neural_network" for config in configs):
        issues.append("Run at least one neural network configuration to compare learned nonlinear features with polynomial regression.")
    try:
        student_identity(student_name, student_surname)
    except ValueError as error:
        issues.append(str(error))
    try:
        chosen = configuration_key(selected)
        if chosen not in by_key:
            issues.append("Rerun your chosen configuration in the experiment cell before finalizing.")
    except (KeyError, TypeError, ValueError):
        issues.append("Run the experiment cell with your chosen settings before finalizing.")
    return list(dict.fromkeys(issues))


def student_identity(student_name, student_surname):
    """Normalize names while preserving their spelling, accents and casing."""
    identity = {}
    for field, value in (("student_name", student_name), ("student_surname", student_surname)):
        label = "name" if field == "student_name" else "surname"
        if not isinstance(value, str):
            raise ValueError(f"Enter your {label} as text.")
        if any(unicodedata.category(character).startswith("C")
               or unicodedata.category(character) in ("Zl", "Zp") for character in value):
            raise ValueError(f"Enter your {label} without control characters.")
        value = " ".join(unicodedata.normalize("NFC", value).split())
        if not 1 <= len(value) <= 100:
            raise ValueError(f"Enter your {label} using 1 to 100 characters.")
        allowed = all(unicodedata.category(character)[0] in "LM"
                      or unicodedata.category(character) == "Pd"
                      or character in " .'’‘" for character in value)
        if not allowed or value[0] in "=+-@" or not any(unicodedata.category(character).startswith("L") for character in value):
            raise ValueError(f"Enter your {label} using letters, spaces, apostrophes, hyphens or periods.")
        identity[field] = value
    return identity


def student_identity_key(student_name, student_surname):
    """Group the same normalized name and surname across repeated uploads."""
    identity = student_identity(student_name, student_surname)
    folded = {field: unicodedata.normalize("NFC", value.casefold()) for field, value in identity.items()}
    canonical = json.dumps(folded, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "name:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def submission_filename(student_name, student_surname):
    """Build a bounded ASCII filename without using names as path components."""
    identity = student_identity(student_name, student_surname)
    readable = unicodedata.normalize("NFKD", " ".join(identity.values()).casefold()).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-z0-9]+", "-", readable).strip("-")[:60].strip("-") or "student"
    digest = student_identity_key(**identity).split(":", 1)[1][:12]
    return f"network-latency_{slug}_{digest}.json"


def split_public(rows, train_percent=60):
    """Stratify labelled public examples; test rows have no role in this split.

    Each regime is permuted with a fixed seed, then divided using the chosen
    training percentage. Both parts preserve the public traffic mixture.
    """
    if isinstance(train_percent, (bool, np.bool_)) or not isinstance(train_percent, (int, np.integer)) or not 50 <= train_percent <= 90:
        raise ValueError("Set train_percent to a whole number from 50 to 90.")
    if len(rows) != PUBLIC_COUNT:
        raise ValueError(f"Expected the {PUBLIC_COUNT:,} labelled rows in public.csv.")
    masks = _regime_masks(arrays(rows, include_target=False))
    rng = np.random.default_rng(PUBLIC_SPLIT_SEED)
    train_indices, validation_indices = [], []
    for name, expected_count in (("normal", 500), ("busy", 300), ("congested", 200)):
        indices = np.flatnonzero(masks[name])
        if len(indices) != expected_count:
            raise ValueError(f"Expected {expected_count} {name} public rows; check the challenge data files.")
        indices = rng.permutation(indices)
        training_count = expected_count * int(train_percent) // 100
        train_indices.extend(indices[:training_count])
        validation_indices.extend(indices[training_count:])
    train_indices = rng.permutation(train_indices)
    validation_indices = rng.permutation(validation_indices)
    return ([rows[int(i)] for i in train_indices], [rows[int(i)] for i in validation_indices])


def dataset_fingerprint(public_bytes, test_bytes):
    return hashlib.sha256(b"public.csv\0" + public_bytes + b"\0test.csv\0" + test_bytes).hexdigest()


def submission_digest(payload, created_at):
    """Bind one final submission to its timestamp, independently of JSON formatting."""
    body = {"created_at": created_at,
            "submission": {key: value for key, value in payload.items() if key != "fingerprint"}}
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def make_fingerprint(payload, created_at=None):
    if created_at is None:
        created_at = datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")
    return {"created_at": created_at, "algorithm": "sha256", "value": submission_digest(payload, created_at)}


# %% cellView="form" jupyter={"source_hidden": true}
# @title Setup 2: load the data { display-mode: "form" }
DATA_DIRECTORY = Path("data")
DATA_BASE_URL = "https://raw.githubusercontent.com/eurecom-ds/intro-ai-challenges/network-latency-v1/network-latency/data/"
EXPECTED_FILE_HASHES = {'public.csv': 'b48db55b42bb589ec864884dcaaf883034ab616c90611cffc3b2fd9422bebb00', 'test.csv': 'c9a7bed4fcd197d2a90148db7bee82231a0c1d0119f119c83acad4576afd3936'}


def load_data_file(filename):
    candidates = [DATA_DIRECTORY / filename, Path(filename)]
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        try:
            with urllib.request.urlopen(DATA_BASE_URL + filename, timeout=30) as response:
                contents = response.read()
        except (OSError, urllib.error.URLError) as error:
            raise RuntimeError(
                f"Could not download {filename}. Check the internet connection and rerun Setup 2. "
                "You can also download both CSVs from the challenge repository and upload them "
                "through Colab's Files panel."
            ) from error
    else:
        contents = path.read_bytes()
    if hashlib.sha256(contents).hexdigest() != EXPECTED_FILE_HASHES[filename]:
        raise ValueError(
            f"{filename} does not match this notebook. Use the original notebook and both CSVs "
            "from the same challenge release; do not edit the data files."
        )
    if path is None:
        DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)
        (DATA_DIRECTORY / filename).write_bytes(contents)
    return contents, list(csv.DictReader(io.StringIO(contents.decode("utf-8"))))

public_bytes, public_rows = load_data_file("public.csv")
test_bytes, test_rows = load_data_file("test.csv")
data_fingerprint = dataset_fingerprint(public_bytes, test_bytes)
X_public, y_public = arrays(public_rows)
X_test = arrays(test_rows, include_target=False)
test_ids = [row["id"] for row in test_rows]
print(f"Loaded {len(public_rows):,} labelled frames. Choose the training/validation percentages below.")
print("A separate private test set is available for scoring your uploads.")

# %% cellView="form" jupyter={"source_hidden": true}
# @title Setup 3: prepare the experiment table and uploads { display-mode: "form" }
# Rerunning this cell clears the local comparison table and frozen submission.
experiments = []
last_result = None
finalized_submission = None
active_train_percent = None

HTML_FONT = "font-family:system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;"


def show_notice(title, message, *, tone="info", items=()):
    accent, background = {"info": ("#2563eb", "#eff6ff"),
                          "success": ("#15803d", "#f0fdf4"),
                          "attention": ("#a16207", "#fffbeb")}[tone]
    checklist = ""
    if items:
        checklist = '<ul style="margin:10px 0 0;padding-left:20px;">' + "".join(
            '<li style="margin:5px 0;">' + html.escape(str(item)) + '</li>' for item in items) + '</ul>'
    display(HTML('<div style="' + HTML_FONT + 'margin:12px 0;padding:16px 20px;border:1px solid ' + accent +
                 ';border-left:5px solid ' + accent + ';border-radius:10px;background:' + background +
                 ';color:#172c4d;line-height:1.6;overflow-wrap:anywhere;">' +
                 '<div style="font-weight:700;margin-bottom:3px;">' + html.escape(str(title)) + '</div>' +
                 '<div>' + html.escape(str(message)) + '</div>' + checklist + '</div>'))


def show_metrics(cards, *, title, detail=""):
    panels = []
    for label, value, note, accent in cards:
        panels.append('<div style="flex:1 1 170px;min-width:0;padding:16px 18px;border:1px solid #dce5ef;' +
                      'border-top:3px solid ' + accent + ';border-radius:10px;background:#ffffff;">' +
                      '<div style="font-size:11px;font-weight:700;letter-spacing:.06em;text-transform:uppercase;color:#52647c;">' +
                      html.escape(str(label)) + '</div><div style="margin:5px 0;font-size:28px;font-weight:750;' +
                      'font-variant-numeric:tabular-nums;line-height:1.25;color:#172c4d;overflow-wrap:anywhere;">' +
                      html.escape(str(value)) + '</div><div style="font-size:12px;line-height:1.5;color:#52647c;">' +
                      html.escape(str(note)) + '</div></div>')
    display(HTML('<div style="' + HTML_FONT + 'margin:16px 0;padding:16px;border:1px solid #e2e8f0;' +
                 'border-radius:12px;background:#f8fafc;color:#172c4d;">' +
                 '<div style="font-size:15px;font-weight:700;line-height:1.5;overflow-wrap:anywhere;">' +
                 html.escape(str(title)) + '</div><div style="margin:3px 0 12px;font-size:12px;color:#52647c;">' +
                 html.escape(str(detail)) + '</div><div style="display:flex;flex-wrap:wrap;gap:12px;">' +
                 ''.join(panels) + '</div></div>'))


def show_table(headers, rows, *, highlight=None):
    minimum_width = "min-width:640px;" if len(headers) > 3 else ""
    heading = ''.join('<th scope="col" style="padding:12px 14px;text-align:left;background:#eff4fa;color:#334d70;' +
                      'font-size:11px;letter-spacing:.04em;border-bottom:1px solid #dce5ef;">' +
                      html.escape(str(value)) + '</th>' for value in headers)
    body = []
    for index, row in enumerate(rows):
        background = "#eff6ff" if index == highlight else ("#ffffff" if index % 2 == 0 else "#f8fafc")
        weight = "650" if index == highlight else "400"
        body.append('<tr>' + ''.join('<td style="padding:11px 14px;vertical-align:top;text-align:left;color:#172c4d;' +
                    'border-bottom:1px solid #e6edf5;background:' + background + ';font-weight:' + weight +
                    ';font-variant-numeric:tabular-nums;">' + html.escape(str(value)) + '</td>' for value in row) + '</tr>')
    display(HTML('<div style="' + HTML_FONT + 'margin:14px 0;border:1px solid #dce5ef;border-radius:10px;' +
                 'overflow-x:auto;background:#ffffff;color:#172c4d;">' +
                 '<table style="border-collapse:collapse;width:100%;' + minimum_width +
                 'font-size:13px;line-height:1.5;color:#172c4d;">' +
                 '<thead><tr>' + heading + '</tr></thead><tbody>' + ''.join(body) + '</tbody></table></div>'))


def set_train_validation_split(train_percent):
    global train_indices, validation_indices, X_train, y_train, X_validation, y_validation
    global active_train_percent, last_result, finalized_submission
    try:
        training, validation = split_public(public_rows, train_percent)
    except (TypeError, ValueError) as error:
        print(str(error))
        return
    changed = active_train_percent is not None and active_train_percent != train_percent
    if active_train_percent != train_percent:
        experiments.clear()
        last_result = None
        finalized_submission = None
    position = {row["id"]: index for index, row in enumerate(public_rows)}
    train_indices = np.array([position[row["id"]] for row in training])
    validation_indices = np.array([position[row["id"]] for row in validation])
    X_train, y_train = X_public[train_indices], y_public[train_indices]
    X_validation, y_validation = X_public[validation_indices], y_public[validation_indices]
    active_train_percent = int(train_percent)
    show_metrics([("Training", f"{active_train_percent}%", f"{len(training)} frames · learn the model", "#2563eb"),
                  ("Validation", f"{100-active_train_percent}%", f"{len(validation)} frames · compare models", "#ea580c")],
                 title="Your data split", detail="Keep these percentages fixed for a fair comparison.")
    if changed:
        show_notice("Split changed", "Local comparisons and selection cleared. Run the baseline again before comparing models.", tone="attention")
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.1), layout="constrained")
    for count, start, name, color in [(len(training), 0, "Training", "#2563eb"),
                                      (len(validation), len(training), "Validation", "#ea580c")]:
        axes[0].barh([0], [count], left=start, color=color, height=.45,
                     label=f"{name} ({100*count/len(public_rows):.0f}%)")
        axes[0].text(start + count/2, 0, str(count), color="white", ha="center", va="center")
    axes[0].set(xlim=(0, len(public_rows)), ylim=(-.55, .55), yticks=[], xlabel="Labelled frames", title="Two separate roles")
    axes[0].legend(loc="upper center", ncol=2, frameon=False, fontsize=9)
    bins = np.linspace(y_public.min(), y_public.max(), 25)
    axes[1].hist(y_train, bins=bins, weights=np.ones(len(y_train))/len(y_train), histtype="step", linewidth=2, label="Training", color="#2563eb")
    axes[1].hist(y_validation, bins=bins, weights=np.ones(len(y_validation))/len(y_validation), histtype="step", linewidth=2, label="Validation", color="#ea580c")
    axes[1].set(xlabel="Observed latency (ms)", ylabel="Fraction of frames", title="Do both sets cover similar targets?")
    axes[1].legend(fontsize=8)
    plt.show(); plt.close(fig)


def inspect_data():
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), layout="constrained")
    points = axes[0].scatter(X_public[:, 0], y_public, c=X_public[:, 4], cmap="viridis", s=13, alpha=.6)
    axes[0].set(xlabel="Traffic load (fraction)", ylabel="Observed latency (ms)", title="How does latency vary with load?")
    fig.colorbar(points, ax=axes[0], label="Nominal bandwidth (Mbps)", shrink=.8)
    load_points = axes[1].scatter(X_public[:, 2], y_public, c=X_public[:, 0], cmap="plasma", s=13, alpha=.6)
    axes[1].set(xlabel="Frame size (decimal kB)", ylabel="Observed latency (ms)", title="A single measurement is insufficient")
    fig.colorbar(load_points, ax=axes[1], label="Traffic load (fraction)", shrink=.8)
    axes[2].hist(y_public, bins=25, color="#2563eb", alpha=.8)
    axes[2].set(xlabel="Observed latency (ms)", ylabel="Number of frames", title="Which delays are common?")
    for ax in axes:
        ax.grid(alpha=.15)
    plt.show(); plt.close(fig)


def experiment_table(records, current=None):
    rows = []
    highlight = None
    for number, record in enumerate(records, start=1):
        mark = " ← current" if current is not None and configuration_key(record) == configuration_key(current) else ""
        if mark:
            highlight = number - 1
        rows.append([number, describe_settings(record) + mark,
                     f"{record['train_rmse']:.1f}", f"{record['validation_rmse']:.1f}"])
    show_table(["Run", "Active configuration", "Training RMSE ↓", "Validation RMSE ↓"], rows, highlight=highlight)


def draw_comparisons(records):
    recent = records[-12:]
    numbers = np.arange(len(records) - len(recent) + 1, len(records) + 1)
    training = np.array([record["train_rmse"] for record in recent])
    validation = np.array([record["validation_rmse"] for record in recent])
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), layout="constrained")
    axes[0].plot(numbers, training, "o-", color="#2563eb", label="Training")
    axes[0].plot(numbers, validation, "o-", color="#ea580c", label="Validation")
    axes[0].set(xlabel="Run in the comparison table", ylabel="RMSE (ms)", title="Compare your experiments", xticks=numbers)
    axes[0].legend(fontsize=8)
    axes[1].scatter(training, validation, color="#ea580c", s=45)
    for number, x, y in zip(numbers, training, validation):
        axes[1].annotate(str(number), (x, y), xytext=(5, 4), textcoords="offset points", fontsize=8)
    upper = max(float(training.max()), float(validation.max()), 1) * 1.15
    axes[1].plot([0, upper], [0, upper], "--", color="#94a3b8", label="Equal training and validation error")
    axes[1].set(xlabel="Training RMSE (ms)", ylabel="Validation RMSE (ms)", title="Small training error can hide a large gap")
    axes[1].legend(fontsize=7, loc="upper left")
    for ax in axes:
        ax.grid(alpha=.15)
    print(f"Comparison plots show the latest {len(recent)} configurations at the same {active_train_percent}/{100-active_train_percent} split.")
    plt.show(); plt.close(fig)


def draw_diagnostics(training_predictions, validation_predictions):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    axes = axes.ravel()
    for target, predicted, label, color in [(y_train, training_predictions, "Training", "#2563eb"),
                                          (y_validation, validation_predictions, "Validation", "#ea580c")]:
        axes[0].scatter(target, predicted, s=12, alpha=.45, color=color, label=label)
    lower = min(float(training_predictions.min()), float(validation_predictions.min()), 0)
    upper = max(float(y_public.max()), float(training_predictions.max()), float(validation_predictions.max()))
    axes[0].plot([lower, upper], [lower, upper], ls="--", color="#64748b", lw=1)
    axes[0].set(xlabel="Observed latency (ms)", ylabel="Predicted latency (ms)", title="Same fitted model on two sets")
    axes[0].legend(fontsize=8)
    axes[1].scatter(X_validation[:, 0], validation_predictions - y_validation, s=15, alpha=.65, color="#ea580c")
    axes[1].axhline(0, ls="--", color="#64748b", lw=1)
    axes[1].set(xlabel="Validation traffic load", ylabel="Prediction − observation (ms)", title="Where does the model fail?")
    training_errors, validation_errors = training_predictions - y_train, validation_predictions - y_validation
    bins = np.histogram_bin_edges(np.r_[training_errors, validation_errors], bins=25)
    axes[2].hist(training_errors, bins=bins, weights=np.ones(len(training_errors))/len(training_errors), histtype="step", linewidth=2, label="Training", color="#2563eb")
    axes[2].hist(validation_errors, bins=bins, weights=np.ones(len(validation_errors))/len(validation_errors), histtype="step", linewidth=2, label="Validation", color="#ea580c")
    axes[2].axvline(0, color="#64748b", ls="--", lw=1)
    axes[2].set(xlabel="Prediction error (ms)", ylabel="Fraction of frames", title="Are errors centred near zero?")
    axes[2].legend(fontsize=8)
    scores = [rmse(y_train, training_predictions), rmse(y_validation, validation_predictions),
              rmse(y_validation, np.full(len(y_validation), y_train.mean()))]
    bars = axes[3].bar(["Training", "Validation", "Degree-0\nbaseline (val.)"], scores,
                       color=["#2563eb", "#ea580c", "#94a3b8"])
    axes[3].bar_label(bars, fmt="%.1f", padding=3, fontsize=9)
    axes[3].set(ylabel="RMSE (ms)", title="Does the model improve on degree 0?", ylim=(0, max(scores) * 1.2 + 1))
    for ax in axes:
        ax.grid(alpha=.15)
    plt.show(); plt.close(fig)


def draw_training_curve(model):
    history = model.get("training_history")
    if history is None:
        return
    fig, ax = plt.subplots(figsize=(9, 3), layout="constrained")
    ax.plot(np.arange(len(history)), history, color="#2563eb", lw=2)
    ax.set(xlabel="Training updates", ylabel="Training MSE (standardized target)",
           title="Is the neural network still improving its training fit?")
    ax.grid(alpha=.15)
    print("The curve shows training MSE on standardized targets. Training runs for the selected number of epochs.")
    plt.show(); plt.close(fig)


def run_experiment(model_family="polynomial", target_transform="raw", **parameters):
    global last_result
    try:
        if active_train_percent is None:
            raise ValueError("Run the training/validation split cell before running a model.")
        # Snapshot the measurements so this experiment and its final refit
        # use the same inputs.
        prepared_public = X_public.copy()
        prepared_test = X_test.copy()
        configuration = settings(model_family=model_family, target_transform=target_transform,
                                 **parameters)
        identity = configuration_key(configuration)
        existing = next((i for i, record in enumerate(experiments) if configuration_key(record) == identity), None)
        if len(experiments) >= 100 and existing is None:
            raise ValueError("The local table contains 100 configurations. Compare existing runs or restart setup for a new investigation.")
        model = fit_model(prepared_public[train_indices], y_train, **configuration)
        training_predictions = predict_model(model, prepared_public[train_indices])
        validation_predictions = predict_model(model, prepared_public[validation_indices])
        record = {**configuration, "train_percent": active_train_percent,
                  "train_rmse": rmse(y_train, training_predictions),
                  "validation_rmse": rmse(y_validation, validation_predictions)}
    except (TypeError, ValueError, FloatingPointError, np.linalg.LinAlgError, NameError) as error:
        last_result = None
        show_notice("Experiment not recorded", str(error), tone="attention")
        return
    if existing is None:
        experiments.append(record.copy())
    else:
        experiments[existing] = record.copy()
    last_result = {"selected": copy.deepcopy(record), "configuration": copy.deepcopy(configuration),
                   "prepared_public": prepared_public.copy(), "prepared_test": prepared_test.copy()}
    print(describe_settings(record))
    print(f"{len(experiments)} distinct configurations; selection-phase fit uses only {len(train_indices)} training labels.")
    show_metrics([("Training RMSE", f"{record['train_rmse']:,.1f} ms", "Error on the fitting examples", "#2563eb"),
                  ("Validation RMSE", f"{record['validation_rmse']:,.1f} ms", "Use this score to compare models", "#ea580c"),
                  ("Validation − training", f"{record['validation_rmse'] - record['train_rmse']:+,.1f} ms", "A large positive gap can indicate overfitting", "#64748b")],
                 title="Current experiment", detail=f"{active_train_percent}% training / {100-active_train_percent}% validation · RMSE: lower is better")
    draw_diagnostics(training_predictions, validation_predictions)
    draw_training_curve(model)
    experiment_table(experiments, current=record)
    draw_comparisons(experiments)
    if finalized_submission is not None:
        print("Your previous finalized submission remains frozen. Finalize again to replace it with this run.")


def finalize_submission(student_name, student_surname, *, baseline=False):
    global finalized_submission
    finalized_submission = None
    selected = last_result["selected"] if last_result else None
    if baseline:
        issues = []
        try:
            student_identity(student_name, student_surname)
        except ValueError as error:
            issues.append(str(error))
        if selected is None or configuration_key(selected) != configuration_key(settings(*BASELINE)):
            issues.append("Run the supplied baseline cell immediately above before uploading the baseline.")
    else:
        issues = completion_issues(experiments, selected, student_name, student_surname)
    if issues:
        show_notice("Before uploading", "Complete these steps, then run the upload cell again.", tone="attention", items=issues)
        return
    try:
        identity = student_identity(student_name, student_surname)
        snapshot = copy.deepcopy(last_result)
        final_model = fit_model(snapshot["prepared_public"], y_public, **snapshot["configuration"])
        predictions = predict_model(final_model, snapshot["prepared_test"])
        public_predictions = predict_model(final_model, snapshot["prepared_public"])
        selected = copy.deepcopy(snapshot["configuration"])
        for key in ("train_percent", "train_rmse", "validation_rmse"):
            selected[key] = snapshot["selected"][key]
        selected["public_refit_rmse"] = rmse(y_public, public_predictions)
        result = {"challenge_version": CHALLENGE_VERSION, **identity,
                  "dataset_fingerprint": data_fingerprint, "selected": selected,
                  "predictions": [{"id": point_id, "predicted_latency_ms": float(value)}
                                  for point_id, value in zip(test_ids, predictions)]}
        result["fingerprint"] = make_fingerprint(result)
        payload = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False).encode("utf-8")
        filename = submission_filename(**identity)
    except (TypeError, ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
        show_notice("Could not finalize this configuration", str(error) +
                    " Reduce complexity or increase regularization, evaluate it, and finalize again.", tone="attention")
        return
    Path(filename).write_bytes(payload)
    finalized_submission = (filename, payload)
    show_notice("Submission saved", f"{filename} contains {len(predictions):,} predictions after refitting on all {len(public_rows):,} public frames. Settings and predictions are frozen together.")
    show_metrics([("Validation RMSE", f"{selected['validation_rmse']:,.1f} ms", f"Selection score at the {selected['train_percent']}/{100-selected['train_percent']} split", "#ea580c"),
                  ("Public fitting RMSE", f"{selected['public_refit_rmse']:,.1f} ms", "Error on the public frames used for this refit", "#2563eb")],
                 title="Ready to upload", detail="Validation describes model selection; public fitting describes the refit.")


def export_submission():
    if finalized_submission is None:
        print("Run your chosen experiment, enter your name and surname, and finalize it first.")
        return
    filename, payload = finalized_submission
    Path(filename).write_bytes(payload)
    try:
        from google.colab import files
    except ImportError:
        display(FileLink(filename))
    else:
        files.download(filename)


def upload_submission(api_url, token=None):
    if finalized_submission is None:
        print("Return to the Experiment cell, run your chosen settings, then enter your name and surname and submit.")
        return
    filename, payload = finalized_submission
    try:
        Path(filename).write_bytes(payload)
    except OSError:
        print("The submission could not be saved here. Use a writable notebook folder and submit again.")
        return

    def saved_copy():
        show_notice("Your file is saved", f"Download {filename} below for manual upload if needed.")
        display(FileLink(filename))

    try:
        address = urllib.parse.urlsplit(api_url)
        hostname = address.hostname
    except (TypeError, ValueError):
        hostname = None
        address = None
    if address is None or address.scheme not in {"http", "https"} or not hostname:
        print("Set submission_url to the upload address supplied by your instructor, then upload again.")
        saved_copy()
        return
    if token is not None and (not isinstance(token, str) or not token.strip()):
        print("Use the access code supplied by your instructor, or leave token unset.")
        return

    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if token is not None:
        headers["Authorization"] = "Bearer " + token.strip()
    request = urllib.request.Request(api_url, data=payload, headers=headers, method="POST")

    class DirectUpload(urllib.request.HTTPRedirectHandler):
        # Use the supplied endpoint directly; do not forward an access code elsewhere.
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            return None

    try:
        with urllib.request.build_opener(DirectUpload()).open(request, timeout=20) as response:
            if response.status not in {200, 201}:
                print("The server did not confirm the upload. Check the address with your instructor.")
                saved_copy()
                return
            contents = response.read(65_537)
        if len(contents) > 65_536:
            raise ValueError("Unexpected response")
        result = json.loads(contents.decode("utf-8"))
        if not isinstance(result, dict):
            raise ValueError("Unexpected response")
        scores = [result["rmse"], result["best_rmse"]]
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               or not np.isfinite(value) or value < 0 for value in scores):
            raise ValueError("Unexpected scores")
    except urllib.error.HTTPError as error:
        if error.code in {401, 403}:
            print("The upload needs a valid access code from your instructor.")
            print("Enable server_requires_access_code in the upload cell and run it again; enter the instructor's code when prompted.")
        elif error.code in {301, 302, 303, 307, 308, 404}:
            print("Check the upload address with your instructor; use the challenge submission endpoint shown in Upload 1.")
        elif error.code >= 500:
            print("The upload service is unavailable. Keep your saved result and try uploading again later.")
        else:
            print(f"The upload was not accepted (HTTP {error.code}).")
            try:
                detail = json.loads(error.read(65_536).decode("utf-8")).get("detail")
                if isinstance(detail, dict):
                    message, issues = detail.get("message"), detail.get("issues", [])
                    if isinstance(message, str):
                        print(message[:400])
                    if isinstance(issues, list):
                        for issue in issues[:10]:
                            if isinstance(issue, str):
                                print("- " + issue[:300])
                elif isinstance(detail, str):
                    print(detail[:400])
                elif isinstance(detail, list):
                    for issue in detail[:10]:
                        if isinstance(issue, dict) and isinstance(issue.get("msg"), str):
                            print("- " + issue["msg"][:300])
            except (AttributeError, OSError, TypeError, ValueError, UnicodeError):
                pass
            print("Follow the feedback, rerun your chosen settings if needed, and submit again.")
        error.close()
        saved_copy()
        return
    except (urllib.error.URLError, TimeoutError):
        print("The upload service could not be reached. Check your connection and the instructor's address, then try uploading again.")
        saved_copy()
        return
    except (OSError, ValueError, KeyError, TypeError, UnicodeError):
        print("No readable score was received. Check the dashboard or ask your instructor before uploading again.")
        saved_copy()
        return

    print("Already uploaded: this result is saved." if result.get("duplicate") else "Uploaded.")
    show_metrics([("This upload", f"{result['rmse']:,.1f} ms", "Private test RMSE returned by the server", "#2563eb"),
                  ("Your best", f"{result['best_rmse']:,.1f} ms", "Best accepted upload so far", "#15803d")],
                 title="Upload confirmed", detail="RMSE: lower is better. Compare this score with your baseline.")
    dashboard = result.get("dashboard_url")
    if isinstance(dashboard, str):
        try:
            dashboard = urllib.parse.urljoin(api_url, dashboard)
            destination = urllib.parse.urlsplit(dashboard)
            if destination.scheme in {"http", "https"} and destination.hostname:
                display(HTML('<a href="' + html.escape(dashboard, quote=True) +
                             '" target="_blank" rel="noopener noreferrer" style="' + HTML_FONT +
                             'display:inline-block;margin:4px 0 16px;padding:11px 18px;border-radius:8px;' +
                             'background:#2563eb;color:#ffffff;text-decoration:none;font-size:14px;font-weight:650;">Open the challenge dashboard →</a>'))
        except ValueError:
            pass


# %% [markdown]
# <div style="display:inline-block;background:#eff6ff;color:#1d4ed8;border:1px solid #bfdbfe;border-radius:20px;padding:4px 10px;margin:20px 0 2px;font-size:11px;font-weight:700;letter-spacing:.07em;text-transform:uppercase;">01 · Explore the data</div>
#
# ## 1. Inspect the measurements
# Each row describes one video frame. The ten measurements below are the model inputs. The target, `latency_ms`, records the observed delivery latency in milliseconds.
#
# | Input | Interpretation | Units |
# |---|---|---|
# | `traffic_load` | Current traffic relative to capacity | Fraction between 0 and 1 |
# | `link_quality` | Link-quality indicator; larger is better | Fraction |
# | `frame_kb` | Frame size | Decimal kilobytes |
# | `interference` | Radio-interference indicator; larger is worse | Fraction |
# | `bandwidth_mbps` | Nominal bandwidth | Megabits per second |
# | `distance_km` | Route distance | Kilometres |
# | `route_hops` | Number of links along the route | Count |
# | `priority` | High-priority frame indicator | 0 or 1 |
# | `diagnostic_a`, `diagnostic_b` | Additional monitoring channels | Dimensionless |
#
# Each frame has an `id` so the server can match its prediction to the corresponding observation. Use the measurement columns to fit the models.
#
# <div style="background:#f8fafc;color:#334155;border:1px solid #cbd5e1;border-radius:14px;padding:18px 20px;margin:16px 0;line-height:1.55;"><div style="color:#475569;font-size:11px;font-weight:700;letter-spacing:.1em;text-transform:uppercase;margin-bottom:6px;">Before running</div><div style="font-size:20px;font-weight:700;margin-bottom:5px;">Relate measurements to latency</div><div style="font-size:14px;">How should frame size and traffic load affect latency? State a prediction, then compare it with the plots.</div></div>

# %% cellView="form" jupyter={"source_hidden": true}
# @title Explore the public measurements { display-mode: "form" }
inspect_data()

# %% [markdown]
# <div style="display:inline-block;background:#eff6ff;color:#1d4ed8;border:1px solid #bfdbfe;border-radius:20px;padding:4px 10px;margin:20px 0 2px;font-size:11px;font-weight:700;letter-spacing:.07em;text-transform:uppercase;">02 · Set up a fair comparison</div>
#
# ## 2. Split the public data
# `public.csv` contains **1,000 labelled frames**. Divide these frames into training and validation sets so you can evaluate predictions on examples held out during fitting. The server uses separate private test labels to score uploads.
#
# | Data | Purpose |
# |---|---|
# | Training | Learn model coefficients or weights and estimate input scaling. |
# | Validation | Compare fitted models and choose their settings. |
# | Private test | Score the uploaded predictions. |
#
# The default split gives **600 training frames and 400 validation frames**. Increasing the training percentage provides more examples for fitting and fewer for validation.
#
# Choose the percentage once and keep the split fixed across comparisons. The same percentage gives the same frames in each set. Changing it clears the experiment table and current selection, so rerun the baseline and investigations afterward.
#
# Compare the latency distributions in the split plot. Check how well each set covers the range of observed delays.
#
# During comparisons, each model estimates its input scaling from the training frames. Each upload refits its configuration on all public labels.

# %% cellView="form" jupyter={"source_hidden": true}
# @title Choose the training/validation split, then press ▶ { display-mode: "form" }
# @markdown The remaining frames form the validation set. Keep this split fixed across comparisons. Changing it clears the experiment table and current selection; rerun the baseline and investigations afterward.
train_percent = 60 # @param {type:"slider", min:50, max:90, step:5}
set_train_validation_split(train_percent)

# %% [markdown]
# ### Measure prediction error
# For a set of $n$ frames, let $y_i$ be the observed latency and $\hat y_i$ the predicted latency. The **root mean squared error (RMSE)** is
#
# $$\operatorname{RMSE}=\sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat y_i)^2}.$$
#
# Squaring gives larger errors more weight. Averaging and taking the square root expresses the result in milliseconds. A lower RMSE indicates more accurate predictions on the evaluated frames.
#
# Start with a constant prediction: the **average training latency**. This is a degree-0 polynomial model. Its validation RMSE gives a reference for assessing models that use the network measurements.

# %% [markdown]
# <div style="display:inline-block;background:#eff6ff;color:#1d4ed8;border:1px solid #bfdbfe;border-radius:20px;padding:4px 10px;margin:20px 0 2px;font-size:11px;font-weight:700;letter-spacing:.07em;text-transform:uppercase;">03 · Evaluate the baseline</div>
#
# ## 3. Evaluate the baseline model
# A **degree-0 polynomial** has one parameter, a constant $b$. It predicts $\hat y=b$ for every frame. With squared loss, fitting sets $b$ to the **average training latency**. This is the baseline against which you will compare the other models.
#
# Higher degrees introduce relationships with the measurements. Degree 1 uses a weighted sum of the inputs plus a constant. Degree 2 adds squares and products such as frame size × traffic load. Increasing the degree allows more complex relationships, which can improve validation predictions or fit accidental patterns in the training sample.
#
# **Overfitting** occurs when a model fits the training examples closely and has substantially larger errors on new examples. Compare training and validation RMSE to assess this behaviour as you increase the degree.
#
# **Ridge regularization** penalizes the squared nonconstant weights. The degree-0 baseline has only an unpenalized constant, so investigate regularization at a positive degree. A stronger penalty discourages large weights; use validation error to assess its effect on prediction.
#
# Run the baseline with **degree 0, zero regularization, the raw target and squared loss**. What shape should its predictions have across different traffic loads? Inspect its training and validation RMSE before adding input-dependent terms.

# %% cellView="form" jupyter={"source_hidden": true}
# @title Run the baseline configuration { display-mode: "form" }
run_experiment(degree=BASELINE[0], regularization=BASELINE[1])

# %% [markdown]
# <div style="background:#eff6ff;color:#1e3a8a;border:1px solid #bfdbfe;border-radius:14px;padding:18px 20px;margin:16px 0;line-height:1.55;"><div style="color:#1d4ed8;font-size:11px;font-weight:700;letter-spacing:.1em;text-transform:uppercase;margin-bottom:6px;">Upload 1 of 2</div><div style="font-size:20px;font-weight:700;margin-bottom:5px;">Record the baseline score</div><div style="font-size:14px;">Run the baseline, enter your name and surname, then upload its predictions before changing the settings.</div></div>

# %% cellView="form" jupyter={"source_hidden": true}
# @title Upload 1: refit and upload the baseline { display-mode: "form" }
# @markdown Enter your name and surname. Keep the supplied upload address unless your instructor gives you another one. Run this cell to refit the baseline on all public frames, save its JSON and upload its predictions.
name = "" # @param {type:"string"}
surname = "" # @param {type:"string"}
submission_url = "https://eurecom-probai-intro-ai-leaderboard.hf.space/api/challenges/network-latency/submissions" # @param {type:"string"}
# @markdown Enter your name and surname as you want them to appear on the dashboard.
# @markdown Enter the instructor's course upload code at the private prompt. Upload 2 reuses these names, upload address and checkbox setting.
server_requires_access_code = True # @param {type:"boolean"}
finalize_submission(name, surname, baseline=True)
if finalized_submission is not None:
    access_code = getpass.getpass("Access code: ") if server_requires_access_code else None
    try:
        upload_submission(submission_url, token=access_code)
    finally:
        access_code = None

# %% [markdown]
# ### Choose model settings
# After uploading the baseline, use the settings below for each investigation. Change **one modelling decision at a time** so you can relate the result to that change.
#
# The first upload returns a private test score and a dashboard link. Use **validation RMSE** to choose model settings during the investigations: it measures predictions on frames held out during fitting. If the baseline upload fails, download its JSON before continuing so you retain a copy when the final upload replaces the local file.
#
# Use the settings for the selected model family. Keep `loss="squared"` and `target_transform="raw"` during the main investigations; explore the optional choices afterward.
#
# In the predicted-versus-observed plot, points on the diagonal have exact predictions. A **residual** is predicted latency minus observed latency. Positive residuals are overestimates; negative residuals are underestimates. Inspect the residual plots for patterns and large errors.
#
# The histogram shows the residual distribution. The RMSE bars summarize training and validation error. After several runs, compare configurations in the experiment table and comparison plots. In the training-versus-validation plot, points above the diagonal have larger validation error than training error.

# %% cellView="form" jupyter={"source_hidden": true}
# @title Experiment: choose settings and run { display-mode: "form" }
# @markdown Predict what will change before running. Each run fits on training frames and evaluates on validation frames.
# @markdown **Model family and target scale.**
model_family = "polynomial" # @param ["polynomial", "neural_network"]
target_transform = "raw" # @param ["raw", "log1p"]
# @markdown **Polynomial settings.** Compare regularization strengths such as 0, 0.0001, 0.001, 0.01, 0.1 and 1. Use huber_delta when fitting with Huber loss.
degree = 0 # @param {type:"slider", min:0, max:4, step:1}
regularization = 0.0 # @param {type:"number"}
loss = "squared" # @param ["squared", "huber"]
huber_delta = 1.0 # @param {type:"number"}
# @markdown **Neural-network settings.** Start with the defaults. Change one setting at a time to compare width, training duration or the weight penalty.
hidden_units = 32 # @param {type:"slider", min:4, max:128, step:4}
learning_rate = 0.01 # @param {type:"number"}
epochs = 1000 # @param {type:"slider", min:100, max:5000, step:100}
weight_decay = 0.001 # @param {type:"number"}

parameters = (dict(degree=degree, regularization=regularization, loss=loss, huber_delta=huber_delta)
              if model_family == "polynomial" else dict(hidden_units=hidden_units, learning_rate=learning_rate,
                                                       epochs=epochs, weight_decay=weight_decay))
run_experiment(model_family, target_transform, **parameters)

# %% [markdown]
# <div style="background:#f8fafc;color:#334155;border:1px solid #cbd5e1;border-radius:14px;padding:18px 20px;margin:16px 0;line-height:1.55;"><div style="color:#475569;font-size:11px;font-weight:700;letter-spacing:.1em;text-transform:uppercase;margin-bottom:6px;">Investigation A</div><div style="font-size:20px;font-weight:700;margin-bottom:5px;">Compare degree and regularization</div><div style="font-size:14px;">Change one setting at a time. Explain how training and validation error respond.</div></div>
#
# ### Investigation A: degree and regularization
# 1. **Degree:** keep regularization at zero and compare degree 0, degree 1 and a higher degree. How do training and validation RMSE change? Which degree gives the lowest validation error?
# 2. **Regularization:** keep a positive degree fixed and compare zero regularization with strengths such as 0.001, 0.01 and 0.1. Compare how training and validation error respond, and explain the result.
# 3. **Residuals:** inspect where the model overestimates or underestimates latency. Check whether large errors occur at particular traffic loads.
#
# Keep the baseline in the experiment table, which records one entry per distinct configuration. Explain your conclusions through comparisons that hold the other settings fixed.

# %% [markdown]
# <div style="display:inline-block;background:#eff6ff;color:#1d4ed8;border:1px solid #bfdbfe;border-radius:20px;padding:4px 10px;margin:20px 0 2px;font-size:11px;font-weight:700;letter-spacing:.07em;text-transform:uppercase;">04 · Compare a neural network</div>
#
# ## 4. Compare a neural network
# This **neural network** has one hidden layer. Each hidden unit forms a weighted sum of the inputs, adds a bias and applies a **ReLU** function: positive values pass through, and negative values become zero. A linear output combines the hidden-unit responses to predict latency.
#
# Training learns the weights. You choose the **hyperparameters**, which control the network size and learning process:
#
# | Setting | Role | Values to compare |
# |---|---|---|
# | `hidden_units` | Number of hidden units; more units allow more input combinations | 8, 32 and 64 |
# | `learning_rate` | Scale of the weight updates | 0.001 and 0.01 |
# | `epochs` | Number of passes through the training frames | 300, 1000 and 3000 |
# | `weight_decay` | Strength of the penalty on squared weights | 0, 0.001 and 0.01 |
#
# Keep the other settings fixed for each comparison. A large learning rate can make training unstable; a small one may require more epochs. The network trains for the selected number of epochs using the training frames.
#
# <div style="background:#f8fafc;color:#334155;border:1px solid #cbd5e1;border-radius:14px;padding:18px 20px;margin:16px 0;line-height:1.55;"><div style="color:#475569;font-size:11px;font-weight:700;letter-spacing:.1em;text-transform:uppercase;margin-bottom:6px;">Investigation B</div><div style="font-size:20px;font-weight:700;margin-bottom:5px;">Compare model families</div><div style="font-size:14px;">Keep the data split and target scale fixed when comparing the two model families.</div></div>
#
# ### Investigation B: model family and training duration
# Select `model_family="neural_network"` and run the default neural-network settings. Compare its validation RMSE with a polynomial model using the **same data split and target scale**.
#
# If time permits, increase the number of epochs while keeping the other settings fixed. Then compare two widths or two regularization strengths. Use validation RMSE to assess whether the change improved predictions on held-out frames.
#
# The learning curve shows training MSE on targets standardized to zero mean and unit standard deviation using the training data. It measures squared prediction error; the fitting objective also includes the weight penalty. The score cards report RMSE in milliseconds, which you can compare across both model families.

# %% [markdown]
# ### Optional: fitting loss and target scale
# After the main investigations, compare these choices one at a time:
#
# - **Huber loss:** `loss="huber"` reduces the influence of large residuals during polynomial fitting. The threshold `huber_delta` is measured in standard deviations of the fitting target. A smaller threshold reduces their influence sooner.
# - **Logarithmic target:** `target_transform="log1p"` fits `log(1 + latency_ms)`. Both model families support this transformation. Predictions are converted back to milliseconds before evaluation.
#
# Compare each choice with its default using validation RMSE. Evaluation uses the same error measure in milliseconds for every configuration.

# %% [markdown]
# <div style="display:inline-block;background:#eff6ff;color:#1d4ed8;border:1px solid #bfdbfe;border-radius:20px;padding:4px 10px;margin:20px 0 2px;font-size:11px;font-weight:700;letter-spacing:.07em;text-transform:uppercase;">05 · Select a model</div>
#
# ## 5. Select a model using validation evidence
# Choose a model using **validation RMSE** and the diagnostic plots. Prepare to explain:
#
# - Which controlled comparison reduced validation RMSE most relative to the baseline.
# - How training and validation error support your interpretation of model flexibility.
# - Where the largest residuals occur and which patterns remain.
# - How the training percentage affects the numbers of fitting and evaluation examples.
#
# **Completion:** retain at least **five distinct configurations** in the experiment table. Include:
#
# - The degree-0 baseline with zero regularization, the raw target and squared loss.
# - A polynomial comparison in which only degree changes.
# - A polynomial comparison at a positive degree in which only regularization changes.
# - At least one neural-network configuration.
#
# Comparisons can share runs. Each distinct configuration counts once. Discuss your reasoning with the instructor; the upload records the selected model's predictions, settings and scores.
#
# Rerun the selected configuration before the final upload. The upload refits its settings on **all public labels**. Validation RMSE describes the earlier fit on the training split; public fitting RMSE describes the new fit on all public frames.

# %% [markdown]
# <div style="display:inline-block;background:#eff6ff;color:#1d4ed8;border:1px solid #bfdbfe;border-radius:20px;padding:4px 10px;margin:20px 0 2px;font-size:11px;font-weight:700;letter-spacing:.07em;text-transform:uppercase;">06 · Upload the selected model</div>
#
# ## 6. Refit and upload the selected model
# Run the selected configuration in the experiment cell, then run **Upload 2**. It reuses the name, surname, upload address and access-code choice from Upload 1. If the instructor provides an access code, enter it again at the private prompt.
#
# To correct your name, surname or upload address, update the fields in Upload 1. Run the baseline and Upload 1 again to save those values, then rerun the selected experiment and Upload 2.
#
# Upload 2 checks the required investigations and lists any remaining steps. Complete those steps and run it again. It then refits the model on all public frames, saves the predictions and uploads them. Compare the private test score with the baseline score on the dashboard.
#
# The JSON contains the selected model's predictions and metadata. The final upload replaces the local JSON file. The server retains both accepted uploads in its score history.

# %% [markdown]
# <div style="background:#0f2746;color:#f1f5f9;border:1px solid #234566;border-radius:14px;padding:18px 20px;margin:16px 0;line-height:1.55;"><div style="color:#93c5fd;font-size:11px;font-weight:700;letter-spacing:.1em;text-transform:uppercase;margin-bottom:6px;">Upload 2 of 2</div><div style="font-size:20px;font-weight:700;margin-bottom:5px;">Upload the selected model</div><div style="font-size:14px;">Run the selected configuration again, then upload its predictions using the name, surname and upload address from Upload 1.</div></div>

# %% cellView="form" jupyter={"source_hidden": true}
# @title Upload 2: refit and upload the selected model { display-mode: "form" }
# @markdown Run the selected configuration again before uploading. This cell checks the required investigations, then refits and uploads using the name, surname and upload address from Upload 1.
if not all(field in globals() for field in ("name", "surname", "submission_url", "server_requires_access_code")):
    print("Run Upload 1 after the baseline to set your name, surname and upload address first.")
else:
    finalize_submission(name, surname)
    if finalized_submission is not None:
        access_code = getpass.getpass("Access code: ") if server_requires_access_code else None
        try:
            upload_submission(submission_url, token=access_code)
        finally:
            access_code = None

# %% [markdown]
# If the server is unavailable, download the submission JSON below and keep it for manual upload. To submit a different configuration, run its experiment and Upload 2 again to save predictions with those settings.

# %% cellView="form" jupyter={"source_hidden": true}
# @title Download the submission JSON { display-mode: "form" }
export_submission()
