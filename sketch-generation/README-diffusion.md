# Teach a diffusion model to draw

Train a small class-conditional pixel diffusion model from scratch using
Hugging Face Diffusers. Generate apples, houses, fish, trees and umbrellas, then
compare training duration, generation steps and network width. Students use
Colab form controls without editing model code.

The [diffusion notebook](notebook-diffusion.ipynb) is an experimental alternative
to the [VAE activity](README.md). Both use the released 28 × 28 public drawings,
fixed train/validation split and public Sketch distance evaluator. The original
VAE notebook and helper remain available.

## Run the activity

1. Open `notebook-diffusion.ipynb` in Colab and select a T4 GPU runtime.
2. Run the three setup cells. Setup installs `diffusers==0.36.0`, checks public
   artifact hashes, and loads the diffusion helper.
3. Inspect real drawings and the forward noise process. Train the five-epoch
   starting model.
4. Complete the five controlled comparisons below, checking generated grids,
   public scores, noise-prediction errors and sampling time.
5. Preview category-specific drawings and an intermediate denoising trajectory.
6. Choose a recorded configuration and download its result archive.

Publish the new notebook and `diffusion_model.py` together before using remote
downloads in Colab. Until then, upload the local notebook and matching helper
through Colab's Files panel. The helper is fetched from the repository's `main`
branch and checked against a SHA-256 hash pinned in the notebook. Existing data
and evaluator downloads retain the immutable `sketch-generation-v1` release.
From a local checkout, setup finds the files in `sketch-generation/`.

GPU timings and image quality require a fresh Colab instructor rehearsal before
setting a classroom time budget. CPU is supported explicitly for instructor
checks; the student notebook requires a CUDA GPU.

## Model and comparisons

The randomly initialized Diffusers `UNet2DModel` predicts the Gaussian noise
added to a drawing. Inputs include the noisy image, timestep and class label.
The three U-Net widths are `(c, 2c, 4c)`, with three resolution stages and no
attention. Training uses AdamW and pixel-averaged noise MSE.
The DDPM training scheduler has 1,000 cosine-scheduled noise levels. The DDIM
sampler uses deterministic updates (`eta=0`) and starts at timestep 999.

Before the first experiment, CUDA calibration probes two forward/backward and
AdamW steps on disposable weights for the wider network, measuring memory with
optimizer state already allocated. It grows batches from
1,024, handles memory exhaustion by trying smaller batches, then refines the
memory limit in increments of 128 drawings. Calibration reserves 10% of free
GPU memory for working space. Both widths use the largest tested fitting batch
throughout that experiment table. Setup 3 resets this selection. CPU instructor
rehearsals use at most 128 drawings per batch. Validation and sampling retain
separate batches of 128.

The experiment table, checkpoints and export record the actual training batch
and total optimizer steps. Larger batches give fewer weight updates per epoch;
evaluate drawing quality again when changing the training batch.

| Run | First width c | Epochs | Generation steps | Learning rate |
| --- | ---: | ---: | ---: | ---: |
| Starting model | 16 | 5 | 25 | 0.0003 |
| More training | 16 | 20 | 25 | 0.0003 |
| Fewer generation steps | 16 | 20 | 10 | 0.0003 |
| More generation steps | 16 | 20 | 50 | 0.0003 |
| Wider network | 32 | 20 | 25 | 0.0003 |

Changing the generation step count reuses matching trained weights and the
same initial image noise. It resamples and scores images without retraining.
Changing training settings initializes and trains a fresh model. Training
retains the final epoch. Diagnostic validation uses fixed noise and timesteps,
with randomness separate from training. Reproducibility applies within the same
hardware and software environment.

Every recorded result contains the exact scored uint8 image batch with shape
`[5,200,28,28]`, using seed 2026 and the VAE category order. Lower public Sketch
distance is better. Compare the images as well as the score when assessing
recognizable shapes and variety.

## Results and leaderboard

The unchanged leaderboard validates VAE-specific latent dimensions, β and loss
fields. The diffusion notebook therefore saves local results and public scores.
Use the VAE notebook for current leaderboard submissions.

After all five comparisons, the selector initially chooses the lowest public
score and preserves subsequent choices. Download saves
`sketch-diffusion_result.npz`, containing `images`, `classes` and
`metadata_json`. Metadata records model settings, helper and image hashes,
dataset/evaluator fingerprints, training history, sampler configuration and
public results, plus training batch size and update count. Read the archive
with `np.load(path, allow_pickle=False)` and
parse `metadata_json.item()` with `json.loads`. Export uses retained images
without further fitting or generation.

## Maintenance and instructor checks

- `notebook-diffusion.py` is the editable Jupytext source;
  `notebook-diffusion.ipynb` is its unexecuted Colab version.
- `diffusion_model.py` supplies the model, training, sampling, trajectory and
  weights-only checkpoint helpers. Update `DIFFUSION_MODEL_SHA256` in the
  notebook after changing this helper, then regenerate the notebook using the
  repository README's pinned Jupytext command.
- Preserve the VAE notebook, helper, manifest, public data and evaluator when
  working on the diffusion variant.

Keep rehearsal outputs in a disposable directory outside the public repository:

```bash
mkdir -p /private/tmp/sketch-diffusion-rehearsal

python sketch-generation/diffusion_model.py train \
  --data sketch-generation/data/train.npz \
  --validation sketch-generation/data/validation.npz \
  --output /private/tmp/sketch-diffusion-rehearsal/baseline.pt \
  --device cpu --epochs 1

python sketch-generation/diffusion_model.py sample \
  --checkpoint /private/tmp/sketch-diffusion-rehearsal/baseline.pt \
  --output /private/tmp/sketch-diffusion-rehearsal/baseline.npy \
  --device cpu --inference-steps 10
```

The full dataset can make this CPU rehearsal slow. Use a balanced subset for
smoke checks and the complete public training split for the Colab quality and
timing rehearsal. Inspect both widths, checkpoint loading, fixed-seed sampling,
sampling-step reuse, selection, trajectory previews and exported image hashes.

References: Hugging Face's [from-scratch training tutorial](https://huggingface.co/docs/diffusers/v0.36.0/en/tutorials/basic_training),
[U-Net API](https://huggingface.co/docs/diffusers/v0.36.0/en/api/models/unet2d), and
[DDIM scheduler](https://huggingface.co/docs/diffusers/v0.36.0/en/api/schedulers/ddim).
Drawings: Google's [Quick, Draw! dataset](https://github.com/googlecreativelab/quickdraw-dataset),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
