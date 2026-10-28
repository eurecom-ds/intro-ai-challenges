"""Conditional VAE used by the Quick, Draw! classroom challenge.

Students train on a Colab GPU. Instructor rehearsals may pass ``device="cpu"``.
Images use uint8 ink intensity: zero is background and 255 is a dark stroke.
"""

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

CLASSES = ("apple", "house", "fish", "tree", "umbrella")
LATENT_DIMS = (2, 4, 8, 16, 32)
BETAS = (0.0, 0.1, 0.5, 1.0, 2.0, 4.0, 8.0)
EPOCHS = (5, 10, 20, 40)
LEARNING_RATES = (0.0003, 0.001, 0.003)
LIKELIHOODS = ("binomial", "gaussian")
BATCH_SIZE = 256
MODEL_SEED = 42
SAMPLE_SEED = 2026


def settings(latent_dim=16, beta=1.0, epochs=20, learning_rate=0.001, seed=42,
             likelihood="binomial"):
    """Return the active settings accepted by the classroom controls."""
    for name, value, choices in (("latent_dim", latent_dim, LATENT_DIMS),
                                  ("epochs", epochs, EPOCHS), ("seed", seed, (MODEL_SEED,))):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value not in choices:
            raise ValueError(f"Choose {name} from {list(choices)}.")
    for name, value, choices in (("beta", beta, BETAS), ("learning_rate", learning_rate, LEARNING_RATES)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)) or value not in choices:
            raise ValueError(f"Choose {name} from {list(choices)}.")
    if not isinstance(likelihood, str) or likelihood not in LIKELIHOODS:
        raise ValueError(f"Choose likelihood from {list(LIKELIHOODS)}.")
    return {"latent_dim": int(latent_dim), "beta": float(beta), "epochs": int(epochs),
            "learning_rate": float(learning_rate), "seed": int(seed), "likelihood": likelihood}


BASELINE = settings(epochs=5)


class ConditionalVAE(nn.Module):
    def __init__(self, latent_dim=16):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder_convs = nn.Sequential(
            nn.Conv2d(1, 16, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(16, 32, 4, 2, 1), nn.ReLU(), nn.Flatten())
        self.encoder_hidden = nn.Sequential(nn.Linear(1568 + 5, 128), nn.ReLU())
        self.mu = nn.Linear(128, latent_dim)
        self.logvar = nn.Linear(128, latent_dim)
        self.decoder_hidden = nn.Sequential(
            nn.Linear(latent_dim + 5, 128), nn.ReLU(),
            nn.Linear(128, 1568), nn.ReLU(), nn.Unflatten(1, (32, 7, 7)))
        self.decoder_convs = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 4, 2, 1))

    def encode(self, images, labels):
        condition = F.one_hot(labels, 5).to(dtype=images.dtype)
        hidden = self.encoder_hidden(torch.cat((self.encoder_convs(images), condition), dim=1))
        return self.mu(hidden), self.logvar(hidden).clamp(-10.0, 10.0)

    def decode(self, latent, labels):
        condition = F.one_hot(labels, 5).to(dtype=latent.dtype)
        return self.decoder_convs(self.decoder_hidden(torch.cat((latent, condition), dim=1)))


def _images(images):
    images = np.asarray(images)
    if images.dtype != np.uint8 or images.ndim != 3 or images.shape[1:] != (28, 28) or len(images) == 0:
        raise ValueError("Images must be a nonempty uint8 array with shape [N,28,28].")
    return np.ascontiguousarray(images)


def _labels(labels, count):
    labels = np.asarray(labels)
    if labels.ndim != 1 or len(labels) != count or labels.dtype.kind not in "iu" or np.any((labels < 0) | (labels >= 5)):
        raise ValueError("Each image needs an integer class label from 0 to 4.")
    return np.ascontiguousarray(labels, dtype=np.int64)


def require_gpu():
    if not torch.cuda.is_available():
        raise RuntimeError("This activity needs a GPU. In Colab, choose Runtime > Change runtime type > T4 GPU, then reconnect and rerun Setup 1.")
    return torch.device("cuda")


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


def _terms(logits, images, mu, logvar, likelihood="binomial"):
    if likelihood == "binomial":
        # Binomial count out of 255: NLL / 255, omitting the fixed count term.
        pixel_loss = F.binary_cross_entropy_with_logits(logits, images, reduction="none")
    elif likelihood == "gaussian":
        # Gaussian mean in [0,1], variance 1; omit the fixed normalizing term.
        pixel_loss = 0.5 * (torch.sigmoid(logits) - images).square()
    else:
        raise ValueError(f"Choose likelihood from {list(LIKELIHOODS)}.")
    reconstruction = pixel_loss.flatten(1).sum(1)
    kl = 0.5 * (mu.square() + logvar.exp() - 1.0 - logvar).sum(1)
    return reconstruction.mean(), kl.mean()


def _tensor_images(images, device):
    return torch.from_numpy(_images(images).copy()).unsqueeze(1).to(device=device, dtype=torch.float32).div_(255.0)


def _validation_terms(network, images, labels, noise, likelihood="binomial"):
    total = torch.zeros(2, device=images.device)
    with torch.no_grad():
        for start in range(0, len(images), BATCH_SIZE):
            batch, condition = images[start:start+BATCH_SIZE], labels[start:start+BATCH_SIZE]
            mu, logvar = network.encode(batch, condition)
            z = mu + (0.5 * logvar).exp() * noise[start:start+BATCH_SIZE, :network.latent_dim]
            r, kl = _terms(network.decode(z, condition), batch, mu, logvar, likelihood)
            total += torch.stack((r, kl)) * len(batch)
    return (total / len(images)).cpu().tolist()


def train_cvae(images, labels, config=None, *, device="cuda", progress=None,
               validation_images=None, validation_labels=None):
    """Train exactly the chosen epochs and return the final CPU checkpoint.

    ``progress`` receives a numeric dictionary after each completed epoch.
    Validation is diagnostic; it never selects a checkpoint or changes training.
    """
    config = settings(**(config or {}))
    images = _images(images)
    labels = _labels(labels, len(images))
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        require_gpu()
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Use a CUDA GPU for students, or CPU for an instructor rehearsal.")
    if (validation_images is None) != (validation_labels is None):
        raise ValueError("Supply validation images and labels together.")
    with _deterministic(device):
        network = ConditionalVAE(config["latent_dim"]).to(device)
        x = _tensor_images(images, device)
        y = torch.from_numpy(labels).to(device)
        order_rng = torch.Generator(device="cpu").manual_seed(43)
        noise_rng = torch.Generator(device=device).manual_seed(44)
        xv = yv = validation_noise = None
        if validation_images is not None:
            validation_images = _images(validation_images)
            validation_labels = _labels(validation_labels, len(validation_images))
            xv = _tensor_images(validation_images, device)
            yv = torch.from_numpy(validation_labels).to(device)
            validation_noise = torch.randn((len(xv), 32), generator=torch.Generator().manual_seed(2027)).to(device)
        optimizer = torch.optim.Adam(network.parameters(), lr=config["learning_rate"], weight_decay=0)
        history = []
        start_time = time.perf_counter()
        for epoch in range(1, config["epochs"] + 1):
            epoch_start = time.perf_counter()
            network.train()
            totals = torch.zeros(2, device=device)
            order = torch.randperm(len(x), generator=order_rng).to(device)
            for indices in order.split(BATCH_SIZE):
                batch, condition = x[indices], y[indices]
                optimizer.zero_grad(set_to_none=True)
                mu, logvar = network.encode(batch, condition)
                noise = torch.randn((len(batch), 32), generator=noise_rng, device=device)[:, :config["latent_dim"]]
                z = mu + (0.5 * logvar).exp() * noise
                reconstruction, kl = _terms(network.decode(z, condition), batch, mu, logvar, config["likelihood"])
                objective = reconstruction + config["beta"] * kl
                if not torch.isfinite(objective).item():
                    raise ValueError("Training became non-finite. Reduce the learning rate and rerun the experiment.")
                objective.backward()
                nn.utils.clip_grad_norm_(network.parameters(), 5.0, error_if_nonfinite=True)
                optimizer.step()
                totals += torch.stack((reconstruction.detach(), kl.detach())) * len(batch)
            r, kl = (totals / len(x)).cpu().tolist()
            record = {"epoch": epoch, "reconstruction": r, "kl": kl, "objective": r + config["beta"] * kl}
            if xv is not None:
                network.eval()
                vr, vk = _validation_terms(network, xv, yv, validation_noise, config["likelihood"])
                if not np.isfinite([vr, vk]).all():
                    raise ValueError("Validation became non-finite. Reduce the learning rate and rerun the experiment.")
                record.update(validation_reconstruction=vr, validation_kl=vk,
                              validation_objective=vr + config["beta"] * vk)
            record["seconds"] = time.perf_counter() - epoch_start
            history.append(record)
            if progress is not None:
                progress(record.copy())
        if not all(torch.isfinite(value).all().item() for value in network.state_dict().values()):
            raise ValueError("The final weights are non-finite. Reduce the learning rate and rerun the experiment.")
        state = {name: value.detach().cpu().clone() for name, value in network.state_dict().items()}
    return {"config": config, "state_dict": state, "history": history,
            "training_count": len(images), "training_seconds": time.perf_counter() - start_time}


def checkpoint_network(checkpoint, *, device="cpu"):
    config = settings(**checkpoint["config"])
    device = torch.device(device)
    with _deterministic(device):
        network = ConditionalVAE(config["latent_dim"]).to(device)
        network.load_state_dict(checkpoint["state_dict"], strict=True)
    return network.eval()


def _pixels(logits):
    return torch.round(torch.sigmoid(logits) * 255.0).to(torch.uint8).squeeze(1).cpu().numpy()


def generate_samples(checkpoint, *, device="cuda", seed=SAMPLE_SEED, samples_per_class=200):
    """Return class-major grayscale samples [5,samples_per_class,28,28]."""
    if isinstance(samples_per_class, bool) or not isinstance(samples_per_class, int) or not 1 <= samples_per_class <= 2000:
        raise ValueError("Choose between 1 and 2,000 samples per class.")
    network = checkpoint_network(checkpoint, device=device)
    device = torch.device(device)
    latent = np.random.default_rng(int(seed)).standard_normal((5, samples_per_class, 32)).astype(np.float32)
    latent = torch.from_numpy(latent[:, :, :network.latent_dim].reshape(5 * samples_per_class, network.latent_dim).copy()).to(device)
    labels = torch.arange(5).repeat_interleave(samples_per_class).to(device)
    with _deterministic(device), torch.no_grad():
        pixels = [_pixels(network.decode(latent[start:start+BATCH_SIZE], labels[start:start+BATCH_SIZE]))
                  for start in range(0, len(latent), BATCH_SIZE)]
    return np.concatenate(pixels).reshape(5, samples_per_class, 28, 28)


def reconstruct_images(checkpoint, images, labels, *, device="cuda"):
    network = checkpoint_network(checkpoint, device=device)
    images = _images(images)
    labels = _labels(labels, len(images))
    x = _tensor_images(images, torch.device(device))
    y = torch.from_numpy(labels).to(device)
    with _deterministic(torch.device(device)), torch.no_grad():
        mu, _ = network.encode(x, y)
        return _pixels(network.decode(mu, y))


def interpolate_images(checkpoint, class_index=0, *, device="cuda", seed=7, steps=9):
    if isinstance(class_index, bool) or not isinstance(class_index, int) or class_index not in range(5):
        raise ValueError("Choose a class from 0 to 4.")
    network = checkpoint_network(checkpoint, device=device)
    ends = torch.from_numpy(np.random.default_rng(seed).standard_normal((2, 32)).astype(np.float32)[:, :network.latent_dim].copy())
    alpha = torch.linspace(0, 1, steps).unsqueeze(1)
    latent = ((1 - alpha) * ends[0] + alpha * ends[1]).to(device)
    labels = torch.full((steps,), class_index, device=device, dtype=torch.long)
    with _deterministic(torch.device(device)), torch.no_grad():
        return _pixels(network.decode(latent, labels))


def save_checkpoint(checkpoint, path):
    """Save instructor or student model state separately from submissions."""
    torch.save(checkpoint, Path(path))


def load_checkpoint(path):
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
    settings(**checkpoint["config"])
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
    train.add_argument("--latent-dim", type=int, default=16)
    train.add_argument("--beta", type=float, default=1.0)
    train.add_argument("--epochs", type=int, default=20)
    train.add_argument("--learning-rate", type=float, default=0.001)
    train.add_argument("--likelihood", choices=LIKELIHOODS, default="binomial")
    sample = commands.add_parser("sample")
    sample.add_argument("--checkpoint", required=True)
    sample.add_argument("--output", required=True)
    sample.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.command == "train":
        with np.load(args.data, allow_pickle=False) as data:
            images, labels = data["images"], data["labels"]
        validation = {}
        if args.validation:
            with np.load(args.validation, allow_pickle=False) as data:
                validation = {"validation_images": data["images"], "validation_labels": data["labels"]}
        config = settings(args.latent_dim, args.beta, args.epochs, args.learning_rate,
                          likelihood=args.likelihood)
        if args.device == "cpu":
            torch.set_num_threads(1)
        checkpoint = train_cvae(images, labels, config, device=args.device,
                                progress=lambda row: print(json.dumps(row), flush=True), **validation)
        save_checkpoint(checkpoint, args.output)
        print(json.dumps({"config": config, "training_seconds": checkpoint["training_seconds"]}), flush=True)
    else:
        samples = generate_samples(load_checkpoint(args.checkpoint), device=args.device)
        np.save(args.output, samples, allow_pickle=False)
        print(json.dumps({"shape": list(samples.shape), "dtype": str(samples.dtype)}))


if __name__ == "__main__":
    _main()
