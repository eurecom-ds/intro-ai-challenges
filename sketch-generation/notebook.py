# ---
# jupyter:
#   colab:
#     name: sketch-generation.ipynb
#     provenance: []
#     accelerator: GPU
#   jupytext:
#     cell_metadata_filter: all
#     notebook_metadata_filter: colab,kernelspec,accelerator
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
#   accelerator: GPU
# ---

# %% [markdown]
# # Teach a model to draw
#
# [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eurecom-ds/intro-ai-challenges/blob/main/sketch-generation/notebook.ipynb)
#
# ![Teach a model to draw. 60 minutes. Colab GPU required. Five drawing categories.](https://raw.githubusercontent.com/eurecom-ds/intro-ai-challenges/main/sketch-generation/assets/overview.svg)
#
# Can a machine learn what makes a drawing look like a fish, then draw another one? People draw the same object in many ways. Your model will learn from drawings of **apples, houses, fish, trees and umbrellas**, then generate new images of a chosen category.
#
# Each image contains **28 × 28 pixels**. A **conditional variational autoencoder (VAE)** learns a compact, probabilistic representation of the drawings. The category is its **condition**: it tells the model which kind of object to generate.
#
# Use the supplied controls. Before each change, predict its effect. Then inspect the generated images and public validation score. Finish by choosing a recorded model and uploading **1,000 generated drawings** for server scoring.
#
# | Minutes | Activity |
# |---|---|
# | 0–8 | Connect a GPU, load the drawings and inspect their variation. |
# | 8–15 | Train the starting model and upload its drawings. |
# | 15–25 | Compare five and twenty training epochs. |
# | 25–38 | Compare weaker and stronger latent regularization. |
# | 38–47 | Compare latent dimensions and explore an interpolation. |
# | 47–55 | Choose a recorded model and upload its drawings. |
# | 55–60 | Discuss quality, variation and remaining failures. |

# %% [markdown]
# ## Setup
# In Colab, choose **Runtime → Change runtime type → T4 GPU** or another available GPU, then connect. Run the next **three setup cells** in order. The public files download automatically and are checked against their release hashes.
#
# After opening an updated notebook, rerun all three setup cells to load its current model helper. Setup refreshes outdated cached files after checking the downloaded replacements.
#
# Rerunning Setup 3 clears the experiment table, retained models and current selection. Keep this runtime connected while completing the activity.

# %% cellView="form" jupyter={"source_hidden": true}
# @title Setup 1: connect the GPU and load libraries { display-mode: "form" }
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import base64
from datetime import datetime, timezone
import hashlib
import html
import importlib.util
import json
from pathlib import Path
import re
import sys
import unicodedata
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
import matplotlib.pyplot as plt
import torch
from IPython.display import display, HTML, FileLink

if not torch.cuda.is_available():
    raise RuntimeError("This activity needs a GPU. Choose Runtime > Change runtime type > T4 GPU, reconnect, then rerun Setup 1.")
DEVICE = "cuda"
print("GPU connected:", torch.cuda.get_device_name(0))
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False,
                     "axes.spines.right": False, "text.parse_math": False})

CHALLENGE_VERSION = "sketch-generation-v1"
CLASSES = ("apple", "house", "fish", "tree", "umbrella")
MANIFEST_SHA256 = "56ce85a3b9fe181f486745f4f7d6427f5351e6e253b6af3a06fc71b5babfc2d2"
RELEASE_REVISION = "61fbd4797fe1be15769e126d60e1250dafc8a184"
RELEASE_BASE = f"https://raw.githubusercontent.com/eurecom-ds/intro-ai-challenges/{RELEASE_REVISION}/sketch-generation/"
ROOT = Path.cwd()
if not (ROOT / "data/manifest.json").is_file() and (ROOT / "sketch-generation/data/manifest.json").is_file():
    ROOT = ROOT / "sketch-generation"


def checked_file(relative, expected_hash, expected_size=None):
    destination = ROOT / relative
    contents = destination.read_bytes() if destination.is_file() else None
    if (contents is None or (expected_size is not None and len(contents) != expected_size)
            or hashlib.sha256(contents).hexdigest() != expected_hash):
        print("Refreshing" if contents is not None else "Downloading", relative, "…")
        try:
            with urllib.request.urlopen(RELEASE_BASE + relative, timeout=60) as response:
                limit = expected_size + 1 if expected_size is not None else 100_001
                contents = response.read(limit)
        except (OSError, urllib.error.URLError) as error:
            raise RuntimeError("Could not download " + relative + ". Check your connection and the challenge release. You can also upload the approved public files through Colab's Files panel.") from error
        if expected_size is not None and len(contents) != expected_size:
            raise ValueError("Unexpected file size for " + relative + ". Restore the approved release file.")
        if hashlib.sha256(contents).hexdigest() != expected_hash:
            raise ValueError("The release hash differs for " + relative + ". Restore the approved file before continuing.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(contents)
    return destination


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# %% cellView="form" jupyter={"source_hidden": true}
# @title Setup 2: load the public drawings and scoring helper { display-mode: "form" }
manifest_path = checked_file("data/manifest.json", MANIFEST_SHA256)
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
if (manifest["challenge_version"] != CHALLENGE_VERSION or tuple(manifest["classes"]) != CLASSES
        or manifest["image_shape"] != [28, 28] or manifest["samples_per_class"] != 200
        or manifest["reference_count"] != 512):
    raise ValueError("The public files use another challenge version. Restore the approved release.")
paths = {}
for key in ("train", "validation", "evaluator", "evaluator_module", "model_module"):
    entry = manifest["files"][key]
    paths[key] = checked_file(entry["path"], entry["sha256"], entry["size_bytes"])
sketch_model = load_module("sketch_challenge_model", paths["model_module"])
sketch_evaluation = load_module("sketch_challenge_evaluator", paths["evaluator_module"])
with np.load(paths["train"], allow_pickle=False) as archive:
    train_images, train_labels = archive["images"].copy(), archive["labels"].copy()
with np.load(paths["validation"], allow_pickle=False) as archive:
    validation_images, validation_labels = archive["images"].copy(), archive["labels"].copy()
for images, labels, per_class in ((train_images, train_labels, 5000), (validation_images, validation_labels, 512)):
    if (images.dtype != np.uint8 or images.shape != (5 * per_class, 28, 28)
            or labels.dtype != np.int64 or labels.shape != (len(images),)
            or not np.array_equal(np.bincount(labels, minlength=5), np.full(5, per_class))):
        raise ValueError("The drawing archive has unexpected dimensions or categories. Restore the approved public files.")
evaluator = sketch_evaluation.SketchEvaluator(paths["evaluator"])
if evaluator.fingerprint != manifest["evaluator_fingerprint"]:
    raise ValueError("The scoring helper differs from the approved release.")
reference_images = np.stack([validation_images[validation_labels == index] for index in range(5)])
public_references = evaluator.prepare_references(reference_images)
dataset_fingerprint = manifest["dataset_fingerprint"]
print(f"Loaded {len(train_images):,} training drawings and {len(validation_images):,} public validation drawings.")
print("Categories:", ", ".join(CLASSES) + ".")
print("Private reference drawings stay on the scoring server.")

# %% cellView="form" jupyter={"source_hidden": true}
# @title Setup 3: prepare the experiment table and uploads { display-mode: "form" }
if globals().get("configuration_selector") is not None:
    configuration_selector.disabled = True
    configuration_selector.close()
configuration_selector = None
experiments = []
last_result = None
finalized_submission = None
upload_identity = None
upload_address = None
upload_access_code = None
HTML_FONT = "font-family:system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;"


def show_notice(title, message, *, tone="info", items=()):
    accent, background = {"info": ("#2563eb", "#eff6ff"), "success": ("#15803d", "#f0fdf4"),
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
                      'font-variant-numeric:tabular-nums;color:#172c4d;">' + html.escape(str(value)) +
                      '</div><div style="font-size:12px;line-height:1.5;color:#52647c;">' + html.escape(str(note)) + '</div></div>')
    display(HTML('<div style="' + HTML_FONT + 'margin:16px 0;padding:16px;border:1px solid #e2e8f0;' +
                 'border-radius:12px;background:#f8fafc;color:#172c4d;">' +
                 '<div style="font-size:15px;font-weight:700;">' + html.escape(str(title)) +
                 '</div><div style="margin:3px 0 12px;font-size:12px;color:#52647c;">' + html.escape(str(detail)) +
                 '</div><div style="display:flex;flex-wrap:wrap;gap:12px;">' + ''.join(panels) + '</div></div>'))


def show_table(headers, rows):
    heading = ''.join('<th scope="col" style="padding:10px 12px;text-align:left;background:#eff4fa;color:#334d70;">' +
                      html.escape(str(value)) + '</th>' for value in headers)
    body = ''.join('<tr>' + ''.join('<td style="padding:10px 12px;border-bottom:1px solid #e6edf5;color:#172c4d;">' +
                                   html.escape(str(value)) + '</td>' for value in row) + '</tr>' for row in rows)
    display(HTML('<div style="' + HTML_FONT + 'margin:14px 0;border:1px solid #dce5ef;border-radius:10px;overflow-x:auto;">'
                 '<table style="border-collapse:collapse;width:100%;min-width:640px;font-size:13px;line-height:1.5;">'
                 '<thead><tr>' + heading + '</tr></thead><tbody>' + body + '</tbody></table></div>'))


def configuration_key(config):
    return json.dumps(sketch_model.settings(**config), sort_keys=True, separators=(",", ":"))


def describe_settings(config):
    return f"{config['likelihood'].capitalize()} · latent={config['latent_dim']} · β={config['beta']:g} · epochs={config['epochs']} · lr={config['learning_rate']:g}"


def recorded_run(key):
    return next((record for record in experiments if configuration_key(record["config"]) == key), None)


def draw_grid(images, *, title, columns=8, categories=CLASSES):
    fig, axes = plt.subplots(len(categories), columns, figsize=(10, 1.35 * len(categories)), squeeze=False,
                             layout="constrained")
    for row, category in enumerate(categories):
        for column in range(columns):
            axes[row, column].imshow(images[row, column], cmap="gray_r", vmin=0, vmax=255, interpolation="nearest")
            axes[row, column].set(xticks=[], yticks=[])
            for spine in axes[row, column].spines.values():
                spine.set_visible(False)
        axes[row, 0].set_ylabel(category, rotation=0, ha="right", va="center", fontsize=10)
    fig.suptitle(title, fontsize=12)
    plt.show()
    plt.close(fig)


def draw_reconstructions(record):
    indices = np.concatenate([np.flatnonzero(validation_labels == index)[:2] for index in range(5)])
    originals, labels = validation_images[indices], validation_labels[indices]
    rebuilt = sketch_model.reconstruct_images(record["checkpoint"], originals, labels, device=DEVICE)
    fig, axes = plt.subplots(2, len(originals), figsize=(12, 3.2), layout="constrained")
    for column in range(len(originals)):
        for row, images in enumerate((originals, rebuilt)):
            axes[row, column].imshow(images[column], cmap="gray_r", vmin=0, vmax=255, interpolation="nearest")
            axes[row, column].axis("off")
        axes[0, column].set_title(CLASSES[int(labels[column])], fontsize=8)
    fig.suptitle("Top: held-out drawings. Bottom: reconstructions using the encoder's mean latent code.", fontsize=11)
    plt.show()
    plt.close(fig)


def draw_training(record):
    history = record["checkpoint"]["history"]
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2), layout="constrained")
    for key, axis, title in (("reconstruction", axes[0], record["config"]["likelihood"].capitalize() + " reconstruction"),
                             ("kl", axes[1], "Latent regularization (KL)")):
        axis.plot(epochs, [row[key] for row in history], label="Training", color="#2563eb")
        axis.plot(epochs, [row["validation_" + key] for row in history], label="Validation", color="#ea580c")
        axis.set(xlabel="Training epoch", ylabel="Average per drawing", title=title)
        axis.legend(fontsize=8)
        axis.grid(alpha=0.15)
    plt.show()
    plt.close(fig)
    print("Training curves average minibatches during fitting; validation uses the epoch's completed model and fixed posterior noise.")


def refresh_configuration_selector(preferred=None):
    if configuration_selector is None:
        return
    choice = configuration_selector.value if preferred is None else preferred
    options = [(f"Run {index} · validation {record['validation_score']:.4f} · " + describe_settings(record["config"]),
                configuration_key(record["config"])) for index, record in enumerate(experiments, start=1)]
    configuration_selector.options = options
    configuration_selector.disabled = not options
    if options:
        keys = [key for _, key in options]
        best = min(experiments, key=lambda record: record["validation_score"])
        configuration_selector.value = choice if choice in keys else configuration_key(best["config"])


def show_configuration_selector():
    global configuration_selector
    previous = configuration_selector.value if configuration_selector is not None else None
    if configuration_selector is not None:
        configuration_selector.disabled = True
        configuration_selector.close()
    if not experiments:
        configuration_selector = None
        show_notice("Run an experiment first", "The dropdown lists the trained models retained in this runtime.", tone="attention")
        return
    import ipywidgets as widgets
    try:
        from google.colab import output
    except ImportError:
        pass
    else:
        output.enable_custom_widget_manager()
    configuration_selector = widgets.Dropdown(description="Configuration:",
                                              layout=widgets.Layout(width="100%"),
                                              style={"description_width": "initial"})
    refresh_configuration_selector(preferred=previous)
    display(configuration_selector)


def experiment_table():
    rows = []
    for index, record in enumerate(experiments, start=1):
        config = record["config"]
        rows.append((index, config["likelihood"].capitalize(), config["latent_dim"], f"{config['beta']:g}", config["epochs"],
                     f"{config['learning_rate']:g}", f"{record['validation_score']:.4f}",
                     f"{record['reconstruction_loss']:.2f}", f"{record['kl_loss']:.2f}"))
    show_table(("Run", "Likelihood", "Latent", "β", "Epochs", "Learning rate", "Validation score ↓", "Val. reconstruction", "Val. KL"), rows)


def run_experiment(latent_dim=16, beta=1.0, epochs=20, learning_rate=0.001, likelihood="binomial"):
    global last_result
    try:
        if not hasattr(sketch_model, "LIKELIHOODS"):
            raise ValueError("The runtime has an older model helper. Rerun Setup 1, Setup 2 and Setup 3, then train the starting model again.")
        config = sketch_model.settings(latent_dim, beta, epochs, learning_rate, likelihood=likelihood)
        key = configuration_key(config)
        existing = next((index for index, record in enumerate(experiments) if configuration_key(record["config"]) == key), None)
        if existing is None and len(experiments) >= 100:
            raise ValueError("The table contains 100 configurations. Compare retained models or rerun Setup 3 for a new investigation.")
        print("Training", describe_settings(config), "on", torch.cuda.get_device_name(0) if DEVICE == "cuda" else DEVICE)
        progress = display(HTML("Preparing the first epoch…"), display_id=True)

        def update(row):
            progress.update(HTML('<p style="' + HTML_FONT + 'color:#334155;">Epoch ' + str(row["epoch"]) +
                                 ' / ' + str(config["epochs"]) + ' · reconstruction ' + f"{row['reconstruction']:.2f}" +
                                 ' · KL ' + f"{row['kl']:.2f}" + '</p>'))

        checkpoint = sketch_model.train_cvae(train_images, train_labels, config, device=DEVICE,
                                             validation_images=validation_images, validation_labels=validation_labels,
                                             progress=update)
        samples = sketch_model.generate_samples(checkpoint, device=DEVICE, seed=2026, samples_per_class=200)
        validation_result = evaluator.score(samples, public_references)
        final_epoch = checkpoint["history"][-1]
        record = {"config": config, "checkpoint": checkpoint, "samples": samples.copy(),
                  "validation_score": float(validation_result["score"]),
                  "reconstruction_loss": float(final_epoch["validation_reconstruction"]),
                  "kl_loss": float(final_epoch["validation_kl"]), "validation_result": validation_result}
        if not np.isfinite([record["validation_score"], record["reconstruction_loss"], record["kl_loss"]]).all():
            raise ValueError("The experiment produced non-finite scores. Reduce the learning rate and rerun.")
    except (ValueError, TypeError, RuntimeError) as error:
        show_notice("Experiment not recorded", str(error), tone="attention")
        return
    except KeyboardInterrupt:
        show_notice("Training interrupted", "Earlier recorded models are retained. Run this configuration again to complete it.", tone="attention")
        return
    if existing is None:
        experiments.append(record)
    else:
        experiments[existing] = record
    last_result = record
    refresh_configuration_selector()
    show_metrics([("Validation score", f"{record['validation_score']:.4f}", "Sketch distance; lower is better", "#2563eb"),
                  ("Validation reconstruction", f"{record['reconstruction_loss']:.2f}", config["likelihood"].capitalize() + " loss per held-out drawing", "#ea580c"),
                  ("Validation KL", f"{record['kl_loss']:.2f}", "Latent penalty per held-out drawing", "#64748b")],
                 title="Final-epoch model retained", detail=f"{len(experiments)} distinct configurations · no early stopping or final refit")
    draw_training(record)
    draw_reconstructions(record)
    draw_grid(samples, title="Fresh drawings from sampled latent variables, grouped by the requested category")
    experiment_table()
    if finalized_submission is not None:
        print("Your saved upload stays frozen. Run an upload cell again to replace it.")


def completion_issues(selected=None):
    configs = [record["config"] for record in experiments]
    issues = []
    if recorded_run(configuration_key(sketch_model.BASELINE)) is None:
        issues.append("Train the supplied five-epoch starting model.")
    if len(configs) < 5:
        issues.append("Record at least five distinct configurations; identical reruns count once.")

    def same_except(left, right, field):
        return left[field] != right[field] and {k: v for k, v in left.items() if k != field} == {k: v for k, v in right.items() if k != field}

    for field, message in (("epochs", "Compare two epoch counts while holding the other settings fixed."),
                           ("latent_dim", "Compare two latent dimensions while holding the other settings fixed.")):
        if not any(same_except(a, b, field) for a in configs for b in configs):
            issues.append(message)
    reference_configs = [config for config in configs if config["beta"] == 1.0]
    if not any(any(other["beta"] < 1 and same_except(anchor, other, "beta") for other in configs)
               and any(other["beta"] > 1 and same_except(anchor, other, "beta") for other in configs)
               for anchor in reference_configs):
        issues.append("At fixed settings, compare β = 1 with both weaker (β < 1) and stronger (β > 1) regularization.")
    if selected is not None and not any(selected is record for record in experiments):
        issues.append("Choose a model retained in the experiment table.")
    return issues


def selected_record():
    if configuration_selector is None or configuration_selector.disabled:
        return None
    return recorded_run(configuration_selector.value)


def preview_record():
    return selected_record() if selected_record() is not None else last_result


def inspect_latents(category="apple", preview_seed=7):
    record = preview_record()
    if record is None:
        show_notice("Train a model first", "The preview uses a retained model from the experiment table.", tone="attention")
        return
    if category not in CLASSES or isinstance(preview_seed, bool) or not isinstance(preview_seed, int) or not 0 <= preview_seed <= 9999:
        show_notice("Check the preview controls", "Choose a category and a whole-number seed from 0 to 9999.", tone="attention")
        return
    index = CLASSES.index(category)
    print("Viewing", describe_settings(record["config"]), "· preview seed", preview_seed)
    samples = sketch_model.generate_samples(record["checkpoint"], device=DEVICE, seed=preview_seed, samples_per_class=8)
    draw_grid(samples[index:index+1], title="Fresh preview drawings (separate from the fixed scoring batch)", categories=(category,))
    path = sketch_model.interpolate_images(record["checkpoint"], index, device=DEVICE, seed=preview_seed, steps=9)
    draw_grid(path[None], title="A straight path between two random latent codes, with the category held fixed",
              columns=9, categories=(category,))
    print("Changing this preview leaves the recorded validation score and upload images unchanged.")


def inspect_nearest_training():
    record = preview_record()
    if record is None:
        show_notice("Train a model first", "Nearest-neighbor inspection uses a retained model.", tone="attention")
        return
    generated = record["samples"][:, 0]
    nearest = []
    for index in range(5):
        candidates = train_images[train_labels == index]
        residual = candidates.astype(np.float32) - generated[index].astype(np.float32)
        distances = np.mean(residual * residual, axis=(1, 2))
        nearest.append(candidates[int(np.argmin(distances))])
    paired = np.stack((generated, np.stack(nearest)), axis=1)
    draw_grid(paired, title="Left: generated drawing. Right: nearest same-category training drawing by pixel error.", columns=2)
    print("Compare outlines, position and distinctive details. Use this inspection to discuss copying and variation.")


def student_identity(student_name, student_surname):
    identity = {}
    for field, value in (("student_name", student_name), ("student_surname", student_surname)):
        label = "name" if field == "student_name" else "surname"
        if not isinstance(value, str):
            raise ValueError(f"Enter your {label} as text.")
        if any(unicodedata.category(character).startswith("C") or unicodedata.category(character) in ("Zl", "Zp") for character in value):
            raise ValueError(f"Enter your {label} without control characters.")
        value = " ".join(unicodedata.normalize("NFC", value).split())
        if not 1 <= len(value) <= 100:
            raise ValueError(f"Enter your {label} using 1 to 100 characters.")
        allowed = all(unicodedata.category(character)[0] in "LM" or unicodedata.category(character) == "Pd"
                      or character in " .'’‘" for character in value)
        if not allowed or value[0] in "=+-@" or not any(unicodedata.category(character).startswith("L") for character in value):
            raise ValueError(f"Enter your {label} using letters, spaces, apostrophes, hyphens or periods.")
        identity[field] = value
    return identity


def submission_filename(identity):
    readable = unicodedata.normalize("NFKD", " ".join(identity.values()).casefold()).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-z0-9]+", "-", readable).strip("-")[:60].strip("-") or "student"
    folded = {field: unicodedata.normalize("NFC", value.casefold()) for field, value in identity.items()}
    digest = hashlib.sha256(json.dumps(folded, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()[:12]
    return f"sketch-generation_{slug}_{digest}.json"


def make_fingerprint(payload, created_at=None):
    if created_at is None:
        created_at = datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")
    body = {"created_at": created_at, "submission": {key: value for key, value in payload.items() if key != "fingerprint"}}
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return {"created_at": created_at, "algorithm": "sha256", "value": hashlib.sha256(canonical.encode("utf-8")).hexdigest()}


def finalize_submission(record, identity, *, phase="final"):
    global finalized_submission
    if record is None:
        show_notice("Choose a trained model", "Run the baseline or select a configuration before uploading.", tone="attention")
        return False
    if phase == "baseline" and configuration_key(record["config"]) != configuration_key(sketch_model.BASELINE):
        show_notice("Run the starting model", "Upload 1 uses the supplied five-epoch baseline.", tone="attention")
        return False
    if phase == "final":
        issues = completion_issues(record)
        if issues:
            show_notice("Before uploading", "Complete the remaining investigations, then run Upload 2 again.", tone="attention", items=issues)
            return False
    selected = {**record["config"], "validation_score": record["validation_score"],
                "reconstruction_loss": record["reconstruction_loss"], "kl_loss": record["kl_loss"]}
    pixels = np.ascontiguousarray(record["samples"], dtype=np.uint8)
    if pixels.shape != (5, 200, 28, 28):
        raise ValueError("This model's saved image batch differs from the challenge format. Rerun the experiment.")
    payload = {"challenge_version": CHALLENGE_VERSION, "dataset_fingerprint": dataset_fingerprint,
               "evaluator_fingerprint": evaluator.fingerprint, **identity, "phase": phase, "selected": selected,
               "sampling": {"seed": 2026, "samples_per_class": 200},
               "images": {"encoding": "uint8-base64", "shape": [5, 200, 28, 28], "class_order": list(CLASSES),
                          "data": base64.b64encode(pixels.tobytes(order="C")).decode("ascii")}}
    if finalized_submission is not None:
        previous_destination, previous_contents = finalized_submission
        previous = json.loads(previous_contents.decode("utf-8"))
        if {key: value for key, value in previous.items() if key != "fingerprint"} == payload:
            previous_destination.write_bytes(previous_contents)
            show_notice("Saved submission reused", "The identity, model and images match your saved upload. Its timestamp and fingerprint stay fixed for this retry.", tone="success")
            return True
    payload["fingerprint"] = make_fingerprint(payload)
    contents = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
    filename = submission_filename(identity)
    destination = ROOT / filename
    destination.write_bytes(contents)
    finalized_submission = (destination, contents)
    show_notice("Submission saved", f"{filename} contains this checkpoint's 1,000 frozen drawings and settings. No further training was performed.", tone="success")
    return True


def upload_submission(api_url, token):
    if finalized_submission is None:
        show_notice("Save a submission first", "Run Upload 1 or Upload 2 to prepare its images.", tone="attention")
        return
    destination, contents = finalized_submission

    def saved_copy():
        show_notice("Your file is saved", "Download its JSON below for a manual upload if needed.")
        display(FileLink(str(destination)))

    try:
        address = urllib.parse.urlsplit(api_url)
        if address.scheme not in {"https", "http"} or not address.hostname:
            raise ValueError("Set submission_url to the instructor's challenge submission endpoint.")
        if not isinstance(token, str) or not token.strip():
            raise ValueError("Enter the current access code from your instructor in Upload 1.")
        request = urllib.request.Request(api_url, data=contents, method="POST",
                                         headers={"Content-Type": "application/json", "Accept": "application/json",
                                                  "Authorization": "Bearer " + token.strip()})

        class DirectUpload(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, req, fp, code, msg, headers, newurl):
                return None

        with urllib.request.build_opener(DirectUpload()).open(request, timeout=30) as response:
            if response.status not in {200, 201}:
                raise ValueError("The server did not confirm this upload.")
            body = response.read(65_537)
        if len(body) > 65_536:
            raise ValueError("Unexpected server response. Check the dashboard before uploading again.")
        result = json.loads(body.decode("utf-8"))
        if not isinstance(result, dict) or any(isinstance(result[field], bool) or not isinstance(result[field], (int, float))
                                              or not np.isfinite(result[field]) or result[field] < 0
                                              for field in ("score", "best_score")):
            raise ValueError("No readable server score was received. Check the dashboard before uploading again.")
    except urllib.error.HTTPError as error:
        if error.code in {401, 403}:
            message = "Enter the instructor's current access code in Upload 1 and upload again."
        elif error.code in {301, 302, 303, 307, 308, 404}:
            message = "Check the submission endpoint with your instructor."
        elif error.code >= 500:
            message = "The scoring service is unavailable. Keep your saved JSON and try again later."
        else:
            message = f"The upload was not accepted (HTTP {error.code})."
            try:
                detail = json.loads(error.read(65_536).decode("utf-8")).get("detail")
                if isinstance(detail, str):
                    message += " " + detail[:500]
                elif isinstance(detail, dict):
                    message += " " + str(detail.get("message", ""))[:500]
                    for issue in detail.get("issues", [])[:8]:
                        if isinstance(issue, str):
                            print(issue[:300])
            except (OSError, ValueError, TypeError, UnicodeError, AttributeError):
                pass
        error.close()
        show_notice("Upload needs attention", message, tone="attention")
        saved_copy()
        return
    except (urllib.error.URLError, TimeoutError, OSError, ValueError, TypeError, KeyError, UnicodeError) as error:
        show_notice("Upload needs attention", str(error) if isinstance(error, ValueError) else
                    "The service could not be reached or returned an unreadable result. Keep your saved JSON and check the dashboard.", tone="attention")
        saved_copy()
        return
    show_metrics([("This upload", f"{result['score']:.4f}", "Server-computed private Sketch distance", "#2563eb"),
                  ("Your best", f"{result['best_score']:.4f}", "Best accepted private score so far", "#15803d")],
                 title="Already uploaded" if result.get("duplicate") else "Upload confirmed", detail="Lower is better. Use public validation evidence to choose settings.")
    dashboard = result.get("dashboard_url")
    if isinstance(dashboard, str):
        url = urllib.parse.urljoin(api_url, dashboard)
        try:
            parsed = urllib.parse.urlsplit(url)
            if parsed.scheme in {"http", "https"} and parsed.hostname:
                display(HTML('<a href="' + html.escape(url, quote=True) + '" target="_blank" rel="noopener noreferrer">Open the challenge dashboard →</a>'))
        except ValueError:
            pass


def upload_baseline(student_name, student_surname, api_url, token):
    global upload_identity, upload_address, upload_access_code
    try:
        identity = student_identity(student_name, student_surname)
        record = recorded_run(configuration_key(sketch_model.BASELINE))
        if not finalize_submission(record, identity, phase="baseline"):
            return
    except (ValueError, TypeError, OSError) as error:
        show_notice("Before uploading", str(error), tone="attention")
        return
    upload_identity, upload_address, upload_access_code = identity, api_url, token
    upload_submission(api_url, token)


def upload_selected():
    if upload_identity is None:
        show_notice("Complete Upload 1 first", "Enter your name, surname and instructor's code in Upload 1, then run it.", tone="attention")
        return
    record = selected_record()
    try:
        if finalize_submission(record, upload_identity, phase="final"):
            upload_submission(upload_address, upload_access_code)
    except (ValueError, TypeError, OSError) as error:
        show_notice("Could not prepare the upload", str(error), tone="attention")


def export_submission():
    if finalized_submission is None:
        show_notice("Prepare an upload first", "Run Upload 1 or Upload 2 to save its JSON.", tone="attention")
        return
    destination, contents = finalized_submission
    destination.write_bytes(contents)
    try:
        from google.colab import files
    except ImportError:
        display(FileLink(str(destination)))
    else:
        files.download(str(destination))


show_notice("Ready to draw", "Inspect the data, train the starting model and record its score before changing settings.", tone="success")

# %% [markdown]
# ## 1. Inspect real drawings
# Each category has **5,000 training drawings** and **512 public validation drawings**. The split is fixed. The model fits its weights using training drawings; public validation drawings provide separate examples for comparing models.
#
# Look for differences in outline, size, position and detail within a category. Which variations should a useful generator reproduce?

# %% cellView="form" jupyter={"source_hidden": true}
# @title Inspect eight training drawings from each category { display-mode: "form" }
examples = np.stack([train_images[train_labels == index][:8] for index in range(5)])
draw_grid(examples, title="Human drawings: the same category has many valid appearances")

# %% [markdown]
# ## 2. Understand the generative model
# The **encoder** receives a drawing and its category. It predicts the mean and variance of a Gaussian distribution over a compact **latent code**. During training, we sample a code from that distribution. The **decoder** receives the code and category and predicts pixel intensities.
#
# The training objective combines two terms:
#
# $$\text{reconstruction loss} + \beta\,\text{latent regularization (KL)}.$$
#
# **Reconstruction loss** rewards reproducing input drawings. **KL regularization** encourages the encoder's latent distributions to stay close to a standard Gaussian. The control **β** sets the strength of that regularization. Both terms sum within one drawing and average over drawings in a batch.
#
# The **likelihood** control chooses how the model penalizes pixel errors. Pixel intensities are scaled from 0–255 to 0–1:
#
# - **Binomial** uses binary cross-entropy. Each original intensity is a count out of 255; the loss is its Binomial negative log likelihood divided by 255, with constant terms omitted.
# - **Gaussian** uses half the squared pixel error, corresponding to a Gaussian with fixed variance 1 on the 0–1 intensity scale, with its constant term omitted.
#
# These losses have different scales, so the same β gives a different balance with KL. Compare generated drawings using **validation Sketch distance** across likelihoods. Both options display the decoder's mean pixel intensities.
#
# For **fresh generation**, sample a new latent code from the standard Gaussian and choose a category. The decoder produces a drawing from that code. Inspect both reconstruction and generation: they ask the model to perform different tasks.
#
# The **starting model** uses the Binomial likelihood, 16 latent dimensions, β = 1, five training epochs and learning rate 0.001. An **epoch** is one pass through the training drawings.

# %% cellView="form" jupyter={"source_hidden": true}
# @title Train the five-epoch starting model { display-mode: "form" }
run_experiment(latent_dim=16, beta=1.0, epochs=5, learning_rate=0.001, likelihood="binomial")

# %% [markdown]
# ### How the score works
# The fixed scoring model maps each image to visual features. **Sketch distance** compares the distribution of generated features with reference features within each category and averages the five distances. **Lower is better.** The comparison considers where samples lie and how their distribution spreads across different appearances.
#
# The public validation score uses reference drawings supplied with the notebook. The server uses separate private reference drawings for the ranked score. Use public validation evidence to choose settings and inspect the generated grids to interpret the result.

# %% [markdown]
# ![Upload 1: train the five-epoch baseline, enter your details and upload its 1,000 generated drawings.](https://raw.githubusercontent.com/eurecom-ds/intro-ai-challenges/main/sketch-generation/assets/upload-baseline.svg)

# %% cellView="form" jupyter={"source_hidden": true}
# @title Upload 1: upload the starting model's drawings { display-mode: "form" }
# @markdown Enter your name, surname and the current code from your instructor. Keep the supplied address unless your instructor gives you another one. Upload 2 reuses these fields. Your name and surname appear on the course dashboard.
name = "" # @param {type:"string"}
surname = "" # @param {type:"string"}
submission_url = "https://eurecom-probai-intro-ai-leaderboard.hf.space/api/challenges/sketch-generation/submissions" # @param {type:"string"}
access_code = "" # @param {type:"string"}
upload_baseline(name, surname, submission_url, access_code)

# %% [markdown]
# ## 3. Investigate the training choices
# ![Change one setting, inspect the drawings. Compare training time, latent regularization and latent dimensions.](https://raw.githubusercontent.com/eurecom-ds/intro-ai-challenges/main/sketch-generation/assets/investigations.svg)
#
# Keep the likelihood at **Binomial** and the learning rate at **0.001** during the main comparisons. Before each run, predict what will change. Compare its public validation score, reconstructions, fresh drawings and training curves with the reference configuration.
#
# | Run | Latent dimensions | β | Epochs | Question |
# |---|---:|---:|---:|---|
# | Starting model | 16 | 1 | 5 | What can the model draw after a short training run? |
# | More training | 16 | 1 | 20 | Which details change with more training? |
# | Weaker regularization | 16 | 0.1 | 20 | How do reconstructions and fresh samples change? |
# | Stronger regularization | 16 | 4 | 20 | How does the range of generated drawings change? |
# | Smaller latent space | 2 | 1 | 20 | Which kinds of variation can the model represent? |
#
# For an additional comparison, repeat the twenty-epoch reference configuration with **Gaussian** likelihood. How do the outlines, gray pixels and variety change? Use validation Sketch distance to compare generation across the two loss scales.
#
# Each run starts from the same initialization seed and trains for the selected epoch count. Its **final-epoch model** stays in memory. The experiment table retains one entry per distinct configuration. The preview and upload cells use these retained models.

# %% cellView="form" jupyter={"source_hidden": true}
# @title Choose the settings and train an experiment { display-mode: "form" }
# @markdown Change one setting at a time for controlled comparisons. Binomial uses binary cross-entropy; Gaussian uses half the squared pixel error. Their scales differ, so compare generation using validation Sketch distance. Training retains the final model after exactly the chosen number of epochs.
latent_dim = 16 # @param [2, 4, 8, 16, 32] {type:"raw"}
beta = 1 # @param [0, 0.1, 0.5, 1, 2, 4, 8] {type:"raw"}
epochs = 20 # @param [5, 10, 20, 40] {type:"raw"}
learning_rate = 0.001 # @param [0.0003, 0.001, 0.003] {type:"raw"}
likelihood = "binomial" # @param ["binomial", "gaussian"]
run_experiment(latent_dim, beta, epochs, learning_rate, likelihood)

# %% [markdown]
# ## 4. Explore variation in latent space
# Use the most recent model below. After opening the configuration selector later in the notebook, these displays use its selected model.
#
# **Draw again:** choose a category and preview seed to see another set of drawings. The preview uses separate randomness; each recorded run keeps its fixed scoring batch.
#
# **Interpolation:** the second row follows a straight path between two random latent codes with the category fixed. Which visual properties change smoothly? Which drawings become ambiguous?

# %% cellView="form" jupyter={"source_hidden": true}
# @title Generate a preview and explore a latent interpolation { display-mode: "form" }
preview_category = "apple" # @param ["apple", "house", "fish", "tree", "umbrella"]
preview_seed = 7 # @param {type:"integer"}
inspect_latents(preview_category, preview_seed)

# %% [markdown]
# ### Compare with training drawings
# The display below pairs one generated image per category with its closest training image under pixel error. Compare the overall structure and distinctive details. What evidence would help you distinguish repeated training examples from new variations?

# %% cellView="form" jupyter={"source_hidden": true}
# @title Inspect the closest training drawings { display-mode: "form" }
inspect_nearest_training()

# %% [markdown]
# ## 5. Choose a generative model
# Use the experiment table, public validation score and image grids. Prepare to explain:
#
# - The effect of additional training, with other settings fixed.
# - How weaker and stronger β change reconstruction, latent regularization and generation.
# - Which differences you observe between two and sixteen latent dimensions.
# - The variety captured within each category and remaining failures.
#
# **Completion:** retain the starting model and at least **five distinct configurations**. Include an epoch-only comparison, a latent-dimension-only comparison, and β = 1 compared with both weaker and stronger regularization at fixed settings. Runs can serve more than one comparison.
#
# The dropdown initially selects the lowest public validation score. New experiments update the list while keeping your choice. Each option refers to an already trained model. Selecting and uploading a model performs **no additional training**.

# %% cellView="form" jupyter={"source_hidden": true}
# @title Review the experiment table and choose a recorded model { display-mode: "form" }
experiment_table()
show_configuration_selector()

# %% [markdown]
# ![Upload 2: choose a recorded generative model and upload its frozen drawings without retraining.](https://raw.githubusercontent.com/eurecom-ds/intro-ai-challenges/main/sketch-generation/assets/upload-selected.svg)
#
# ## 6. Upload the selected model's drawings
# Upload 2 checks the required comparisons, then saves and uploads the selected run's fixed batch: **200 drawings for each of the five categories**. The server computes the ranked Sketch distance from those images.
#
# Upload 2 reuses your details from Upload 1. To correct your name, surname, address or access code, edit the fields in Upload 1 and run it again. Your retained models stay available for selection.
#
# The final JSON replaces the local baseline file. The server keeps accepted score history. If a baseline upload failed, download its JSON before replacing it with a final submission.

# %% cellView="form" jupyter={"source_hidden": true}
# @title Upload 2: upload the selected model's drawings { display-mode: "form" }
upload_selected()

# %% [markdown]
# ### Keep a copy
# If the service is unavailable, download the saved JSON and use the dashboard's manual upload form later. To upload another retained model, change the dropdown and run Upload 2 again.

# %% cellView="form" jupyter={"source_hidden": true}
# @title Download the saved submission JSON { display-mode: "form" }
export_submission()

# %% [markdown]
# Drawings: Google's [Quick, Draw! dataset](https://github.com/googlecreativelab/quickdraw-dataset), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). This activity uses a fixed public subset. Version: **sketch-generation-v1**.
