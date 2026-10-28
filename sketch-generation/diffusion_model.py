"""Small class-conditioned diffusion model for the Quick, Draw! activity.

The U-Net starts from random weights and trains with Hugging Face Diffusers.
Images use uint8 ink intensity: zero is background and 255 is a dark stroke.
Students use a Colab GPU; instructors may pass ``device="cpu"`` for rehearsals.
"""

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import diffusers
from diffusers import DDIMScheduler, DDPMScheduler, UNet2DModel
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

DIFFUSERS_VERSION = "0.36.0"
MODEL_TYPE = "class-conditional-ddpm-v1"
CLASSES = ("apple", "house", "fish", "tree", "umbrella")
MODEL_CHANNELS = (16, 32)
EPOCHS = (1, 5, 10, 20)
INFERENCE_STEPS = (10, 25, 50, 100)
LEARNING_RATES = (0.0001, 0.0003, 0.001)
BATCH_SIZE = 128
MODEL_SEED = 42
SAMPLE_SEED = 2026
TRAIN_TIMESTEPS = 1000


def settings(model_channels=16, epochs=20, inference_steps=25,
             learning_rate=0.0003, seed=42):
    """Return the settings offered by the classroom controls."""
    for name, value, choices in (("model_channels", model_channels, MODEL_CHANNELS),
                                  ("epochs", epochs, EPOCHS),
                                  ("inference_steps", inference_steps, INFERENCE_STEPS),
                                  ("seed", seed, (MODEL_SEED,))):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value not in choices:
            raise ValueError(f"Choose {name} from {list(choices)}.")
    if (isinstance(learning_rate, (bool, np.bool_))
            or not isinstance(learning_rate, (int, float, np.integer, np.floating))
            or learning_rate not in LEARNING_RATES):
        raise ValueError(f"Choose learning_rate from {list(LEARNING_RATES)}.")
    return {"model_channels": int(model_channels), "epochs": int(epochs),
            "inference_steps": int(inference_steps), "learning_rate": float(learning_rate),
            "seed": int(seed)}


BASELINE = settings(epochs=5)


def _images(images):
    images = np.asarray(images)
    if images.dtype != np.uint8 or images.ndim != 3 or images.shape[1:] != (28, 28) or len(images) == 0:
        raise ValueError("Images must be a nonempty uint8 array with shape [N,28,28].")
    return np.ascontiguousarray(images)


def _labels(labels, count):
    labels = np.asarray(labels)
    if (labels.ndim != 1 or len(labels) != count or labels.dtype.kind not in "iu"
            or np.any((labels < 0) | (labels >= len(CLASSES)))):
        raise ValueError("Each image needs an integer class label from 0 to 4.")
    return np.ascontiguousarray(labels, dtype=np.int64)


def _seed(seed):
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))
            or not 0 <= seed < 2**32):
        raise ValueError("Choose an integer seed between 0 and 4,294,967,295.")
    return int(seed)


def require_gpu():
    if not torch.cuda.is_available():
        raise RuntimeError("This activity needs a GPU. In Colab, choose Runtime > Change runtime type > T4 GPU, then reconnect and rerun Setup 1.")
    return torch.device("cuda")


def _device(device):
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        require_gpu()
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Use a CUDA GPU for students, or CPU for an instructor rehearsal.")
    return device


@contextmanager
def _deterministic(device):
    mode = torch.get_deterministic_debug_mode()
    benchmark, deterministic = torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic
    matmul_tf32, cudnn_tf32 = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    try:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(MODEL_SEED)
            yield
    finally:
        torch.set_deterministic_debug_mode(mode)
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = benchmark, deterministic
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = matmul_tf32, cudnn_tf32


def _network(model_channels):
    return UNet2DModel(sample_size=28, in_channels=1, out_channels=1,
                       block_out_channels=(model_channels, 2 * model_channels, 4 * model_channels),
                       layers_per_block=1, down_block_types=("DownBlock2D",) * 3,
                       up_block_types=("UpBlock2D",) * 3, norm_num_groups=8,
                       add_attention=False, num_class_embeds=len(CLASSES))


def _scheduler():
    return DDPMScheduler(num_train_timesteps=TRAIN_TIMESTEPS,
                         beta_schedule="squaredcos_cap_v2", prediction_type="epsilon",
                         clip_sample=True)


def _tensor_images(images, device):
    return torch.from_numpy(_images(images).copy()).unsqueeze(1).to(device=device, dtype=torch.float32).div_(127.5).sub_(1.0)


def _pixels(images):
    if not torch.isfinite(images).all().item():
        raise ValueError("Generation became non-finite. Reduce the learning rate and train again.")
    return ((images.clamp(-1.0, 1.0) + 1.0) * 127.5).round().to(torch.uint8).squeeze(1).cpu().numpy()


def _validation_mse(network, scheduler, images, labels, noise, timesteps):
    total = torch.zeros((), device=images.device)
    with torch.no_grad():
        for start in range(0, len(images), BATCH_SIZE):
            batch = images[start:start+BATCH_SIZE]
            target = noise[start:start+BATCH_SIZE]
            times = timesteps[start:start+BATCH_SIZE]
            noisy = scheduler.add_noise(batch, target, times)
            predicted = network(noisy, times, class_labels=labels[start:start+BATCH_SIZE]).sample
            total += F.mse_loss(predicted, target) * len(batch)
    return float((total / len(images)).cpu())


def train_diffusion(images, labels, config=None, *, device="cuda", progress=None,
                    validation_images=None, validation_labels=None):
    """Train from random weights for exactly the selected number of epochs.

    Validation uses fixed diagnostic noise and timesteps. The final epoch
    supplies the checkpoint; the validation curve never selects its weights.
    ``inference_steps`` affects generation only.
    """
    config = settings(**(config or {}))
    images = _images(images)
    labels = _labels(labels, len(images))
    device = _device(device)
    if (validation_images is None) != (validation_labels is None):
        raise ValueError("Supply validation images and labels together.")
    with _deterministic(device):
        network = _network(config["model_channels"]).to(device)
        scheduler = _scheduler()
        x = _tensor_images(images, device)
        y = torch.from_numpy(labels).to(device)
        order_rng = torch.Generator().manual_seed(43)
        noise_rng = torch.Generator(device=device).manual_seed(44)
        time_rng = torch.Generator(device=device).manual_seed(45)
        xv = yv = validation_noise = validation_times = None
        if validation_images is not None:
            validation_images = _images(validation_images)
            validation_labels = _labels(validation_labels, len(validation_images))
            xv = _tensor_images(validation_images, device)
            yv = torch.from_numpy(validation_labels).to(device)
            validation_rng = torch.Generator().manual_seed(2027)
            validation_noise = torch.randn(xv.shape, generator=validation_rng).to(device)
            validation_times = torch.randint(TRAIN_TIMESTEPS, (len(xv),), generator=validation_rng).to(device)
        optimizer = torch.optim.AdamW(network.parameters(), lr=config["learning_rate"], weight_decay=0.0001)
        history = []
        start_time = time.perf_counter()
        for epoch in range(1, config["epochs"] + 1):
            epoch_start = time.perf_counter()
            network.train()
            total = torch.zeros((), device=device)
            order = torch.randperm(len(x), generator=order_rng).to(device)
            for indices in order.split(BATCH_SIZE):
                batch, condition = x[indices], y[indices]
                noise = torch.randn(batch.shape, generator=noise_rng, device=device)
                times = torch.randint(TRAIN_TIMESTEPS, (len(batch),), generator=time_rng, device=device)
                noisy = scheduler.add_noise(batch, noise, times)
                optimizer.zero_grad(set_to_none=True)
                predicted = network(noisy, times, class_labels=condition).sample
                loss = F.mse_loss(predicted, noise)
                if not torch.isfinite(loss).item():
                    raise ValueError("Training became non-finite. Reduce the learning rate and rerun the experiment.")
                loss.backward()
                nn.utils.clip_grad_norm_(network.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                total += loss.detach() * len(batch)
            record = {"epoch": epoch, "noise_mse": float((total / len(x)).cpu())}
            if xv is not None:
                network.eval()
                value = _validation_mse(network, scheduler, xv, yv, validation_noise, validation_times)
                if not np.isfinite(value):
                    raise ValueError("Validation became non-finite. Reduce the learning rate and rerun the experiment.")
                record["validation_noise_mse"] = value
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            record["seconds"] = time.perf_counter() - epoch_start
            history.append(record)
            if progress is not None:
                progress(record.copy())
        if not all(torch.isfinite(value).all().item() for value in network.state_dict().values()):
            raise ValueError("The final weights are non-finite. Reduce the learning rate and rerun the experiment.")
        state = {name: value.detach().cpu().clone() for name, value in network.state_dict().items()}
        training_seconds = time.perf_counter() - start_time
    return {"model_type": MODEL_TYPE, "diffusers_version": diffusers.__version__,
            "config": config, "state_dict": state, "history": history,
            "training_count": len(images), "training_seconds": training_seconds}


def checkpoint_network(checkpoint, *, device="cpu"):
    config = settings(**checkpoint["config"])
    if checkpoint.get("model_type") != MODEL_TYPE:
        raise ValueError("Choose a checkpoint from this diffusion activity.")
    device = _device(device)
    with _deterministic(device):
        network = _network(config["model_channels"]).to(device)
        network.load_state_dict(checkpoint["state_dict"], strict=True)
    if not all(torch.isfinite(value).all().item() for value in network.state_dict().values()):
        raise ValueError("The checkpoint contains non-finite weights.")
    return network.eval()


def _initial_noise(seed, samples_per_class):
    noise = np.random.default_rng(_seed(seed)).standard_normal(
        (len(CLASSES), samples_per_class, 1, 28, 28)).astype(np.float32)
    return torch.from_numpy(noise.reshape(-1, 1, 28, 28))


def _sample(network, initial_noise, labels, inference_steps, *, capture=()):
    scheduler = DDIMScheduler.from_config(_scheduler().config, timestep_spacing="trailing")
    scheduler.set_timesteps(inference_steps, device=initial_noise.device)
    images = initial_noise
    snapshots = [_pixels(images)] if 0 in capture else []
    with torch.no_grad():
        for completed, timestep in enumerate(scheduler.timesteps, start=1):
            predicted = network(images, timestep, class_labels=labels).sample
            images = scheduler.step(predicted, timestep, images, eta=0.0).prev_sample
            if completed in capture:
                snapshots.append(_pixels(images))
    return images, snapshots


def generate_samples(checkpoint, *, device="cuda", seed=SAMPLE_SEED, samples_per_class=200):
    """Return class-major uint8 drawings [5,samples_per_class,28,28].

    DDIM uses eta=0; the initial Gaussian noise supplies the randomness.
    The same seed and batch shape reuse that noise across sampler step counts.
    """
    if isinstance(samples_per_class, bool) or not isinstance(samples_per_class, int) or not 1 <= samples_per_class <= 2000:
        raise ValueError("Choose between 1 and 2,000 samples per class.")
    device = _device(device)
    network = checkpoint_network(checkpoint, device=device)
    noise = _initial_noise(seed, samples_per_class)
    labels = torch.arange(len(CLASSES)).repeat_interleave(samples_per_class)
    pixels = []
    with _deterministic(device), torch.no_grad():
        for start in range(0, len(noise), BATCH_SIZE):
            final, _ = _sample(network, noise[start:start+BATCH_SIZE].to(device),
                               labels[start:start+BATCH_SIZE].to(device),
                               checkpoint["config"]["inference_steps"])
            pixels.append(_pixels(final))
    return np.concatenate(pixels).reshape(len(CLASSES), samples_per_class, 28, 28)


def denoising_trajectory(checkpoint, class_index=0, *, device="cuda", seed=7, frames=6):
    """Return clipped display snapshots and the completed DDIM step counts."""
    if isinstance(class_index, bool) or not isinstance(class_index, int) or class_index not in range(len(CLASSES)):
        raise ValueError("Choose a class from 0 to 4.")
    steps = settings(**checkpoint["config"])["inference_steps"]
    if isinstance(frames, bool) or not isinstance(frames, int) or not 2 <= frames <= steps + 1:
        raise ValueError("Choose between 2 and inference_steps + 1 display frames.")
    device = _device(device)
    network = checkpoint_network(checkpoint, device=device)
    noise = _initial_noise(seed, 1)[class_index:class_index+1].to(device)
    labels = torch.tensor([class_index], device=device)
    capture = np.linspace(0, steps, frames).round().astype(np.int64)
    with _deterministic(device), torch.no_grad():
        _, images = _sample(network, noise, labels, steps, capture=capture)
    return {"images": np.concatenate(images), "completed_steps": capture}


def noise_images(images, timesteps=(0, 100, 300, 600, 999), *, seed=7):
    """Add a fixed Gaussian noise draw at each specified training timestep.

    Return clipped uint8 display images with shape [len(timesteps),N,28,28].
    Training uses the full floating-point noisy values before display clipping.
    """
    images = _images(images)
    times = np.asarray(timesteps)
    if times.ndim != 1 or len(times) == 0 or times.dtype.kind not in "iu" or np.any((times < 0) | (times >= TRAIN_TIMESTEPS)):
        raise ValueError("Choose integer noise timesteps from 0 to 999.")
    x = _tensor_images(images, torch.device("cpu"))
    noise = torch.randn(x.shape, generator=torch.Generator().manual_seed(_seed(seed)))
    scheduler = _scheduler()
    return np.stack([_pixels(scheduler.add_noise(x, noise, torch.full((len(x),), int(t), dtype=torch.long)))
                     for t in times])


def save_checkpoint(checkpoint, path):
    """Save weights and training history separately from image exports."""
    torch.save(checkpoint, Path(path))


def load_checkpoint(path):
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
    checkpoint_network(checkpoint)
    return checkpoint


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train")
    train.add_argument("--data", required=True)
    train.add_argument("--validation")
    train.add_argument("--output", required=True)
    train.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    train.add_argument("--model-channels", type=int, default=16)
    train.add_argument("--epochs", type=int, default=20)
    train.add_argument("--inference-steps", type=int, default=25)
    train.add_argument("--learning-rate", type=float, default=0.0003)
    sample = commands.add_parser("sample")
    sample.add_argument("--checkpoint", required=True)
    sample.add_argument("--output", required=True)
    sample.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    sample.add_argument("--inference-steps", type=int, choices=INFERENCE_STEPS)
    args = parser.parse_args()
    if args.device == "cpu":
        torch.set_num_threads(1)
    if args.command == "train":
        with np.load(args.data, allow_pickle=False) as data:
            images, labels = data["images"], data["labels"]
        validation = {}
        if args.validation:
            with np.load(args.validation, allow_pickle=False) as data:
                validation = {"validation_images": data["images"], "validation_labels": data["labels"]}
        config = settings(args.model_channels, args.epochs, args.inference_steps, args.learning_rate)
        checkpoint = train_diffusion(images, labels, config, device=args.device,
                                     progress=lambda row: print(json.dumps(row), flush=True), **validation)
        save_checkpoint(checkpoint, args.output)
        print(json.dumps({"config": config, "training_seconds": checkpoint["training_seconds"]}), flush=True)
    else:
        checkpoint = load_checkpoint(args.checkpoint)
        if args.inference_steps is not None:
            checkpoint = {**checkpoint, "config": settings(**{**checkpoint["config"], "inference_steps": args.inference_steps})}
        samples = generate_samples(checkpoint, device=args.device)
        np.save(args.output, samples, allow_pickle=False)
        print(json.dumps({"shape": list(samples.shape), "dtype": str(samples.dtype)}))


if __name__ == "__main__":
    _main()
