# ---
# jupyter:
#   colab:
#     name: sketch-generation-diffusion.ipynb
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
# # Teach a diffusion model to draw
#
# Can a model turn random noise into a drawing of a fish? Train a small **conditional diffusion model from scratch** on drawings of apples, houses, fish, trees and umbrellas. Each drawing contains **28 × 28 grayscale pixels**. The category tells the model which kind of object to draw.
#
# Use the supplied controls. Predict the effect of each change, then inspect the drawings and **public validation Sketch distance**. Lower scores indicate closer generated and reference feature distributions.
#
# | Stage | Activity |
# |---|---|
# | Setup | Connect a Colab GPU and inspect the training drawings. |
# | Starting model | Train a small denoising network for five epochs. |
# | Investigation | Compare training duration, generation steps and model size. |
# | Selection | Choose a recorded result and download its 1,000 drawings. |
#
# This experimental version saves results locally. The course leaderboard currently accepts the VAE version's settings. Use the [VAE notebook](https://github.com/eurecom-ds/intro-ai-challenges/blob/main/sketch-generation/notebook.ipynb) for leaderboard uploads. GPU timing and drawing quality need an instructor rehearsal before scheduling this activity.

# %% [markdown]
# ## Setup
# Choose **Runtime → Change runtime type → T4 GPU** in Colab, then connect. Run the three setup cells in order. Setup installs the pinned Hugging Face Diffusers library and loads the same public drawings and fixed scoring model as the VAE activity.
#
# Setup 3 clears the experiment table, retained models and current selection. Keep this runtime connected during the activity.

# %% cellView="form" jupyter={"source_hidden": true}
# @title Setup 1: connect the GPU and load Diffusers { display-mode: "form" }
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import hashlib
import html
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython.display import display, HTML, FileLink

if not torch.cuda.is_available():
    raise RuntimeError("Choose Runtime > Change runtime type > T4 GPU, reconnect, then rerun Setup 1.")
DEVICE = "cuda"
DIFFUSERS_VERSION = "0.36.0"
try:
    installed_version = importlib.metadata.version("diffusers")
except importlib.metadata.PackageNotFoundError:
    installed_version = None
if installed_version != DIFFUSERS_VERSION:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "diffusers==" + DIFFUSERS_VERSION])
if "diffusers" in sys.modules and sys.modules["diffusers"].__version__ != DIFFUSERS_VERSION:
    raise RuntimeError("Diffusers was already loaded at another version. Restart the runtime and rerun setup.")

print("GPU connected:", torch.cuda.get_device_name(0))
print("Diffusers:", DIFFUSERS_VERSION)
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False,
                     "axes.spines.right": False, "text.parse_math": False})
CLASSES = ("apple", "house", "fish", "tree", "umbrella")
DATA_VERSION = "sketch-generation-v1"
EXPERIMENT_VERSION = "sketch-diffusion-experimental-v1"
MANIFEST_SHA256 = "26dfd447e5855a5e8bf1bb28a8c5ab3cd8f1108d461010871a7a545b05e4f178"
DIFFUSION_MODEL_SHA256 = "abb8314dede81977a2b04c4f645e15bc7f5ea4aa8a247b0f2a23fa91de4d3f97"
RELEASE_BASE = "https://raw.githubusercontent.com/eurecom-ds/intro-ai-challenges/sketch-generation-v1/sketch-generation/"
HELPER_REVISION = "e3d1b7c4412aca7defda73c09285f889d283a2fc"
HELPER_BASE = "https://raw.githubusercontent.com/eurecom-ds/intro-ai-challenges/" + HELPER_REVISION + "/sketch-generation/"
HELPER_CACHE = Path(tempfile.gettempdir()) / "intro-ai-sketch-diffusion" / DIFFUSION_MODEL_SHA256
ROOT = Path.cwd()
if not (ROOT / "data/manifest.json").is_file() and (ROOT / "sketch-generation/data/manifest.json").is_file():
    ROOT = ROOT / "sketch-generation"


def checked_file(relative, expected_hash, expected_size=None, *, base_url=RELEASE_BASE,
                 cache_root=None, refresh_mismatch=False):
    destination = (ROOT if cache_root is None else cache_root) / relative

    def matches(contents):
        return ((expected_size is None or len(contents) == expected_size)
                and hashlib.sha256(contents).hexdigest() == expected_hash)

    if destination.is_file():
        contents = destination.read_bytes()
        if matches(contents):
            return destination
        if not refresh_mismatch:
            raise ValueError("Unexpected release contents for " + relative + ". Restore the matching public file.")
        print("Refreshing cached", relative)
    else:
        print("Downloading", relative)
    try:
        with urllib.request.urlopen(base_url + relative, timeout=60) as response:
            contents = response.read(expected_size + 1 if expected_size is not None else 100_001)
    except (OSError, urllib.error.URLError) as error:
        raise RuntimeError("Could not download " + relative + ". Check your connection and rerun Setup 2. You can also upload the matching public files through Colab's Files panel.") from error
    if not matches(contents):
        raise ValueError("Downloaded " + relative + " uses another version. Reopen the latest notebook and rerun Setup 1 and Setup 2.")
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
# @title Setup 2: load public drawings, the diffusion helper and the scoring model { display-mode: "form" }
manifest_path = checked_file("data/manifest.json", MANIFEST_SHA256)
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
if (manifest["challenge_version"] != DATA_VERSION or tuple(manifest["classes"]) != CLASSES
        or manifest["image_shape"] != [28, 28] or manifest["samples_per_class"] != 200
        or manifest["reference_count"] != 512):
    raise ValueError("Restore the public files for this dataset version.")
paths = {}
for key in ("train", "validation", "evaluator", "evaluator_module"):
    entry = manifest["files"][key]
    paths[key] = checked_file(entry["path"], entry["sha256"], entry["size_bytes"])
local_model = ROOT / "diffusion_model.py"
if local_model.is_file() and hashlib.sha256(local_model.read_bytes()).hexdigest() == DIFFUSION_MODEL_SHA256:
    model_path = local_model
else:
    model_path = checked_file("diffusion_model.py", DIFFUSION_MODEL_SHA256, base_url=HELPER_BASE,
                              cache_root=HELPER_CACHE, refresh_mismatch=True)
diffusion_model = load_module("sketch_challenge_diffusion", model_path)
sketch_evaluation = load_module("sketch_challenge_evaluator", paths["evaluator_module"])
with np.load(paths["train"], allow_pickle=False) as archive:
    train_images, train_labels = archive["images"].copy(), archive["labels"].copy()
with np.load(paths["validation"], allow_pickle=False) as archive:
    validation_images, validation_labels = archive["images"].copy(), archive["labels"].copy()
for images, labels, per_class in ((train_images, train_labels, 5000), (validation_images, validation_labels, 512)):
    if (images.dtype != np.uint8 or images.shape != (5 * per_class, 28, 28)
            or labels.dtype != np.int64 or labels.shape != (len(images),)
            or np.any((labels < 0) | (labels >= 5))
            or not np.array_equal(np.bincount(labels, minlength=5), np.full(5, per_class))):
        raise ValueError("The drawing archive has unexpected dimensions or categories.")
evaluator = sketch_evaluation.SketchEvaluator(paths["evaluator"])
if evaluator.fingerprint != manifest["evaluator_fingerprint"]:
    raise ValueError("Restore the matching scoring model.")
reference_images = np.stack([validation_images[validation_labels == index] for index in range(5)])
public_references = evaluator.prepare_references(reference_images)
print(f"Loaded {len(train_images):,} training drawings and {len(validation_images):,} public validation drawings.")
print("Model weights will be initialized randomly and learned from training drawings.")

# %% cellView="form" jupyter={"source_hidden": true}
# @title Setup 3: prepare the experiment table and local export { display-mode: "form" }
if globals().get("configuration_selector") is not None:
    configuration_selector.disabled = True
    configuration_selector.close()
configuration_selector = None
experiments = []
last_result = None
training_batch_size = None
HTML_FONT = "font-family:system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;"


def show_notice(title, message):
    display(HTML('<div style="' + HTML_FONT + 'padding:16px 20px;margin:12px 0;border:1px solid #bfdbfe;'
                 'border-left:5px solid #2563eb;border-radius:10px;background:#eff6ff;color:#172c4d;line-height:1.6;">'
                 '<strong>' + html.escape(str(title)) + '</strong><br>' + html.escape(str(message)) + '</div>'))


def configuration_key(config):
    return json.dumps(diffusion_model.settings(**config), sort_keys=True, separators=(",", ":"))


def training_key(config):
    return json.dumps({key: value for key, value in diffusion_model.settings(**config).items()
                       if key != "inference_steps"}, sort_keys=True, separators=(",", ":"))


def describe_settings(config):
    return (f"width={config['model_channels']} · epochs={config['epochs']} · "
            f"generation steps={config['inference_steps']} · lr={config['learning_rate']:g}")


def recorded_run(key):
    return next((record for record in experiments if configuration_key(record["config"]) == key), None)


def draw_grid(images, *, title, columns=8, categories=CLASSES):
    fig, axes = plt.subplots(len(categories), columns, figsize=(10, 1.35 * len(categories)),
                             squeeze=False, layout="constrained")
    for row, category in enumerate(categories):
        for column in range(columns):
            axis = axes[row, column]
            axis.imshow(images[row, column], cmap="gray_r", vmin=0, vmax=255, interpolation="nearest")
            axis.set(xticks=[], yticks=[])
            for spine in axis.spines.values():
                spine.set_visible(False)
        axes[row, 0].set_ylabel(category, rotation=0, ha="right", va="center", fontsize=10)
    fig.suptitle(title, fontsize=12)
    plt.show()
    plt.close(fig)


def draw_training(record):
    history = record["checkpoint"]["history"]
    fig, axis = plt.subplots(figsize=(8, 3), layout="constrained")
    for key, label in (("noise_mse", "Training"), ("validation_noise_mse", "Validation")):
        axis.plot([row["epoch"] for row in history], [row[key] for row in history], label=label)
    axis.set(xlabel="Training epoch", ylabel="Mean squared noise error", title="Predicting the added noise")
    axis.legend()
    axis.grid(alpha=0.15)
    plt.show()
    plt.close(fig)
    print("Training averages fitting minibatches. Validation uses the completed epoch's model with fixed noise and noise levels.")


def experiment_table():
    headers = ("Run", "Width", "Epochs", "Batch", "Weight updates", "Generation steps", "Learning rate", "Validation score ↓", "Val. noise MSE", "Sampling seconds")
    rows = []
    for index, record in enumerate(experiments, start=1):
        config = record["config"]
        rows.append((index, config["model_channels"], config["epochs"], record["checkpoint"]["batch_size"],
                     record["checkpoint"]["optimizer_steps"], config["inference_steps"],
                     f"{config['learning_rate']:g}", f"{record['validation_score']:.4f}",
                     f"{record['noise_mse']:.4f}", f"{record['sampling_seconds']:.1f}"))
    heading = ''.join('<th style="padding:10px;text-align:left;background:#eff4fa;">' + html.escape(str(value)) + '</th>' for value in headers)
    body = ''.join('<tr>' + ''.join('<td style="padding:10px;border-bottom:1px solid #e6edf5;">' + html.escape(str(value)) + '</td>' for value in row) + '</tr>' for row in rows)
    display(HTML('<div style="' + HTML_FONT + 'overflow-x:auto;margin:14px 0;color:#172c4d;">'
                 '<table style="border-collapse:collapse;min-width:780px;width:100%;font-size:13px;">'
                 '<thead><tr>' + heading + '</tr></thead><tbody>' + body + '</tbody></table></div>'))


def refresh_configuration_selector(preferred=None):
    if configuration_selector is None:
        return
    choice = configuration_selector.value if preferred is None else preferred
    options = [(f"Run {index} · score {record['validation_score']:.4f} · " + describe_settings(record["config"]),
                configuration_key(record["config"])) for index, record in enumerate(experiments, start=1)]
    configuration_selector.options = options
    configuration_selector.disabled = not options
    if options:
        best = min(experiments, key=lambda record: record["validation_score"])
        configuration_selector.value = choice if choice in [key for _, key in options] else configuration_key(best["config"])


def show_configuration_selector():
    global configuration_selector
    previous = configuration_selector.value if configuration_selector is not None else None
    if configuration_selector is not None:
        configuration_selector.disabled = True
        configuration_selector.close()
    if not experiments:
        configuration_selector = None
        show_notice("Train the starting model", "The selector lists completed experiments retained in this runtime.")
        return
    import ipywidgets as widgets
    try:
        from google.colab import output
    except ImportError:
        pass
    else:
        output.enable_custom_widget_manager()
    configuration_selector = widgets.Dropdown(description="Configuration:", layout=widgets.Layout(width="100%"),
                                              style={"description_width": "initial"})
    refresh_configuration_selector(preferred=previous)
    display(configuration_selector)


def selected_record():
    if configuration_selector is None or configuration_selector.disabled:
        return None
    return recorded_run(configuration_selector.value)


def preview_record():
    return selected_record() if selected_record() is not None else last_result


def run_experiment(model_channels=16, epochs=20, inference_steps=25, learning_rate=0.0003):
    global last_result, training_batch_size
    try:
        config = diffusion_model.settings(model_channels, epochs, inference_steps, learning_rate)
        existing = recorded_run(configuration_key(config))
        if existing is not None:
            last_result = existing
            show_notice("Recorded result reused", "This exact configuration is already complete. Its model, score and image batch stay fixed.")
            draw_grid(existing["samples"], title="Retained drawings for this configuration")
            return
        if len(experiments) >= 100:
            raise ValueError("The table contains 100 configurations. Rerun Setup 3 to start a new investigation.")
        matching = next((record for record in experiments if training_key(record["config"]) == training_key(config)), None)
        if matching is not None:
            checkpoint = {**matching["checkpoint"], "config": config}
            print("Reusing trained weights and changing only the generation step count.")
        else:
            if training_batch_size is None:
                print("Finding a training batch that fits both network widths…")
                training_batch_size = diffusion_model.select_training_batch_size(
                    len(train_images), device=DEVICE,
                    progress=lambda size, fits: print(f"Batch {size:,}: " + ("fits" if fits else "trying a smaller batch")))
                print(f"Training batch: {training_batch_size:,} drawings. This batch stays fixed across experiments.")
            print("Training", describe_settings(config))
            progress = display(HTML("Preparing the first epoch…"), display_id=True)

            def update(row):
                progress.update(HTML('<p>Epoch ' + str(row["epoch"]) + ' / ' + str(config["epochs"]) +
                                     ' · noise MSE ' + f"{row['noise_mse']:.4f}" + '</p>'))

            checkpoint = diffusion_model.train_diffusion(train_images, train_labels, config, device=DEVICE,
                                                          validation_images=validation_images, validation_labels=validation_labels,
                                                          progress=update, batch_size=training_batch_size)
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        print("Generating the fixed batch of 1,000 drawings…")
        samples = diffusion_model.generate_samples(checkpoint, device=DEVICE, seed=2026, samples_per_class=200)
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        sampling_seconds = time.perf_counter() - start
        result = evaluator.score(samples, public_references)
        record = {"config": config, "checkpoint": checkpoint, "samples": samples.copy(),
                  "validation_score": float(result["score"]), "validation_result": result,
                  "noise_mse": float(checkpoint["history"][-1]["validation_noise_mse"]),
                  "sampling_seconds": sampling_seconds}
        if not np.isfinite([record["validation_score"], record["noise_mse"], sampling_seconds]).all():
            raise ValueError("The experiment produced non-finite results. Reduce the learning rate and try again.")
    except (ValueError, TypeError, RuntimeError) as error:
        show_notice("Experiment needs attention", str(error))
        return
    except KeyboardInterrupt:
        show_notice("Experiment interrupted", "Earlier completed experiments stay available. Run this configuration again to finish it.")
        return
    experiments.append(record)
    last_result = record
    refresh_configuration_selector()
    show_notice("Result retained", f"Validation Sketch distance: {record['validation_score']:.4f}. Training batch: {checkpoint['batch_size']:,}. Weight updates: {checkpoint['optimizer_steps']:,}. Sampling: {sampling_seconds:.1f} seconds. The final-epoch weights and scored drawings are saved in memory.")
    draw_training(record)
    draw_grid(samples, title="New drawings generated from random pixel noise, grouped by the requested category")
    experiment_table()


def inspect_denoising(category="apple", preview_seed=7):
    record = preview_record()
    if record is None:
        show_notice("Train a model first", "The preview uses a retained model.")
        return
    if category not in CLASSES or isinstance(preview_seed, bool) or not isinstance(preview_seed, int) or not 0 <= preview_seed <= 9999:
        show_notice("Check the preview controls", "Choose a category and an integer seed from 0 to 9999.")
        return
    samples = diffusion_model.generate_samples(record["checkpoint"], device=DEVICE, seed=preview_seed, samples_per_class=8)
    index = CLASSES.index(category)
    draw_grid(samples[index:index+1], title="Fresh previews using separate randomness", categories=(category,))
    path = diffusion_model.denoising_trajectory(record["checkpoint"], index, device=DEVICE, seed=preview_seed, frames=6)
    draw_grid(path["images"][None], title="One generation trajectory: noise to drawing. Completed steps: " +
              ", ".join(str(value) for value in path["completed_steps"]), columns=6, categories=(category,))
    print("The recorded scoring batch stays fixed while you explore previews.")


def completion_issues():
    configs = [record["config"] for record in experiments]
    issues = []
    if recorded_run(configuration_key(diffusion_model.BASELINE)) is None:
        issues.append("Train the supplied five-epoch starting model.")
    if len(configs) < 5:
        issues.append("Record at least five distinct configurations.")
    for field, label in (("epochs", "training duration"), ("inference_steps", "generation steps"), ("model_channels", "model size")):
        if not any(a[field] != b[field] and {k: v for k, v in a.items() if k != field} ==
                   {k: v for k, v in b.items() if k != field} for a in configs for b in configs):
            issues.append("Compare " + label + " while holding the other settings fixed.")
    return issues


def export_selected():
    record = selected_record()
    if record is None:
        show_notice("Choose a recorded configuration", "Run the configuration selector and choose a completed experiment.")
        return
    issues = completion_issues()
    if issues:
        show_notice("Remaining investigations", " ".join(issues))
        return
    samples = np.ascontiguousarray(record["samples"], dtype=np.uint8)
    if samples.shape != (5, 200, 28, 28):
        raise ValueError("The recorded batch must contain 200 images for each category.")
    metadata = {"experiment_version": EXPERIMENT_VERSION, "model_type": "class-conditional-pixel-diffusion",
                "diffusers_version": DIFFUSERS_VERSION, "model_helper_sha256": DIFFUSION_MODEL_SHA256,
                "model_helper_revision": HELPER_REVISION,
                "dataset_version": DATA_VERSION, "dataset_fingerprint": manifest["dataset_fingerprint"],
                "evaluator_fingerprint": evaluator.fingerprint, "config": record["config"],
                "sampling": {"seed": 2026, "samples_per_class": 200, "sampler": "DDIM", "eta": 0.0},
                "pixel_convention": "0=background;255=ink", "validation_result": record["validation_result"],
                "training_history": record["checkpoint"]["history"], "sampling_seconds": record["sampling_seconds"],
                "training_batch_size": record["checkpoint"]["batch_size"],
                "optimizer_steps": record["checkpoint"]["optimizer_steps"],
                "images_sha256": hashlib.sha256(samples.tobytes(order="C")).hexdigest()}
    destination = ROOT / "sketch-diffusion_result.npz"
    np.savez_compressed(destination, images=samples, classes=np.array(CLASSES),
                        metadata_json=np.array(json.dumps(metadata, sort_keys=True, allow_nan=False)))
    show_notice("Result saved", "The archive contains the scored drawings, settings and public results. Selection and export reuse the retained model without further training. Keep this experimental archive for local comparison.")
    try:
        from google.colab import files
    except ImportError:
        display(FileLink(str(destination)))
    else:
        files.download(str(destination))


show_notice("Ready to train", "Inspect the drawings and added noise, then train the starting model.")

# %% [markdown]
# ## 1. Inspect drawings and added noise
# There are **5,000 training drawings** and **512 public validation drawings** per category. The training split supplies weight updates. The fixed validation split supports model comparisons.
#
# A diffusion training example starts with a real drawing, adds a known amount of Gaussian noise, and asks the network to predict that noise. Look at the lower rows: which details remain visible as the noise level increases?

# %% cellView="form" jupyter={"source_hidden": true}
# @title Inspect real drawings and the forward noise process { display-mode: "form" }
examples = np.stack([train_images[train_labels == index][:8] for index in range(5)])
draw_grid(examples, title="Human drawings: variation within each category")
noise_times = (0, 100, 300, 600, 999)
noisy = diffusion_model.noise_images(examples[2], timesteps=noise_times, seed=7)
draw_grid(noisy, title="The same fish drawings with increasing noise", categories=tuple("t=" + str(t) for t in noise_times))

# %% [markdown]
# ## 2. Train a denoising network
# The **U-Net** receives a noisy image, its **noise level** (a training timestep from 0 to 999), and the requested **category**. It predicts the noise added to that image. Its weights start randomly. Hugging Face **Diffusers** supplies the network and noise schedulers.
#
# Training minimizes the **mean squared error (MSE)** between the predicted and known noise, averaged over pixels and drawings. Each minibatch uses randomly selected noise levels. An **epoch** is one pass through the training drawings. Public validation noise and noise levels stay fixed for diagnostic comparisons.
#
# For **generation**, begin with Gaussian pixel noise and a category. A **DDIM sampler** repeatedly uses the network's prediction to update the image toward a drawing. Here it uses deterministic updates once the initial noise is fixed. Training uses 1,000 possible noise levels; generation visits a chosen subset of those levels.
#
# The **starting model** uses U-Net channel widths **16, 32 and 64**, five training epochs, learning rate **0.0003**, and **25 generation steps**. The width control sets the first of these three widths; later widths are twice and four times that value.
#
# Before the first run, the notebook finds a large training batch that fits the GPU and keeps it fixed for both network widths. The table records the batch size and number of weight updates. Larger batches produce fewer updates per epoch, so compare results using the same runtime and batch setting.

# %% cellView="form" jupyter={"source_hidden": true}
# @title Train the five-epoch diffusion starting model { display-mode: "form" }
run_experiment(model_channels=16, epochs=5, inference_steps=25, learning_rate=0.0003)

# %% [markdown]
# ### Read the two kinds of evidence
# **Noise MSE** measures the training task: predicting added noise. **Sketch distance** compares the distribution of generated images with public validation drawings using a fixed feature extractor. Lower is better for both. Inspect image grids alongside the scores to assess recognizable shapes and variety.
#
# Each recorded configuration generates **1,000 drawings**, 200 per category, from the fixed initial-noise seed **2026**. The public score evaluates those exact quantized images. The VAE activity uses the same evaluator and public references, supporting comparison on this public metric.

# %% [markdown]
# ## 3. Compare training and generation choices
# Predict the effect before each run. Hold the learning rate at **0.0003** and change one setting at a time.
#
# | Run | First width | Epochs | Generation steps | Question |
# |---|---:|---:|---:|---|
# | Starting model | 16 | 5 | 25 | Which shapes emerge after a short training run? |
# | More training | 16 | 20 | 25 | Which details change with more weight updates? |
# | Fewer generation steps | 16 | 20 | 10 | How much sampling time can we save? |
# | More generation steps | 16 | 20 | 50 | How do sampling time and drawing quality change? |
# | Wider network | 32 | 20 | 25 | Does extra capacity improve the generated distribution? |
#
# Changing **generation steps** reuses the trained weights and fixed initial noise. It only reruns sampling and scoring. Changing width, epochs or learning rate trains a fresh model from the fixed initialization seed. Width changes also change the parameter shapes. Each training run retains its final epoch; validation supplies diagnostics throughout training.

# %% cellView="form" jupyter={"source_hidden": true}
# @title Choose one change and record an experiment { display-mode: "form" }
model_channels = 16 # @param [16, 32] {type:"raw"}
epochs = 20 # @param [1, 5, 10, 20] {type:"raw"}
inference_steps = 25 # @param [10, 25, 50, 100] {type:"raw"}
learning_rate = 0.0003 # @param [0.0001, 0.0003, 0.001] {type:"raw"}
run_experiment(model_channels, epochs, inference_steps, learning_rate)

# %% [markdown]
# ## 4. Watch noise become a drawing
# Choose a category and preview seed. The first grid uses fresh preview noise. The second follows one generation trajectory through the retained model. Identify the stage where a recognizable outline appears.
#
# These controls use the most recent experiment until you open the configuration selector below. After that, they use its selected result. Preview randomness is separate from the fixed scoring batch.

# %% cellView="form" jupyter={"source_hidden": true}
# @title Preview a category and inspect a denoising trajectory { display-mode: "form" }
preview_category = "fish" # @param ["apple", "house", "fish", "tree", "umbrella"]
preview_seed = 7 # @param {type:"integer"}
inspect_denoising(preview_category, preview_seed)

# %% [markdown]
# ## 5. Choose and save a result
# Prepare to explain the effect of additional training, the cost and quality of generation steps, and the difference between the two network widths. Which categories are recognizable? Which variations are missing? Use the same public score when comparing with VAE results.
#
# Complete the starting model and at least **five distinct configurations**, including a comparison for each of epochs, generation steps and width with other settings fixed. Exact reruns reuse their recorded result. The five runs in the table satisfy these requirements.
#
# The dropdown initially selects the lowest public validation score. It keeps your choice when new runs appear. The archive saves that configuration's retained images, category order, settings, training history, fingerprints and public scores. It contains numeric/string arrays that can be read with `np.load(path, allow_pickle=False)`.

# %% cellView="form" jupyter={"source_hidden": true}
# @title Review experiments and choose a recorded configuration { display-mode: "form" }
experiment_table()
show_configuration_selector()

# %% cellView="form" jupyter={"source_hidden": true}
# @title Download the selected experiment and its 1,000 drawings { display-mode: "form" }
export_selected()

# %% [markdown]
# Drawings: Google's [Quick, Draw! dataset](https://github.com/googlecreativelab/quickdraw-dataset), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Public data: **sketch-generation-v1**. Experiment: **sketch-diffusion-experimental-v1**.
#
# Implementation references: Hugging Face's [from-scratch diffusion tutorial](https://huggingface.co/docs/diffusers/v0.36.0/en/tutorials/basic_training), [U-Net API](https://huggingface.co/docs/diffusers/v0.36.0/en/api/models/unet2d), and [DDIM sampler](https://huggingface.co/docs/diffusers/v0.36.0/en/api/schedulers/ddim).
