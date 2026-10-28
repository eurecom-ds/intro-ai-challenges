# Teach a model to draw

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eurecom-ds/intro-ai-challenges/blob/main/sketch-generation/notebook.ipynb)

An alternative [diffusion activity](README-diffusion.md) trains a small model
from scratch using Hugging Face Diffusers. Both versions use the same public
drawings and evaluator. This VAE version supports leaderboard uploads; the
diffusion version saves experiments locally.

Train a conditional variational autoencoder (VAE) to generate new drawings of
an apple, house, fish, tree or umbrella. This 60-minute activity investigates
training time, latent regularization and latent-space size. Students use the
supplied controls without writing code.

## Start the activity

1. Open the notebook in Colab. Choose **Runtime → Change runtime type → T4 GPU**
   or another available GPU, then connect.
2. Run the three setup cells. Public drawings, the model helper and the scoring
   helper download automatically and are checked against their release hashes.
3. Inspect examples from the five categories. Train the supplied five-epoch
   baseline and compare reconstructed images with fresh drawings.
4. Enter your name, surname and the instructor's access code, then run Upload 1.
5. Run the controlled comparisons. Predict each effect before training, then
   inspect the generated drawings and public validation score.
6. Choose a recorded run in the configuration dropdown and run Upload 2.

The notebook requires a GPU for student training. An unavailable GPU produces
instructions for changing the runtime. Setup 3 clears the local experiment
table, retained checkpoints and current selection.

## Five controlled experiments

The encoder maps a drawing and its category to a distribution over latent
variables. The decoder maps sampled latent variables and a chosen category to
pixel intensities. Fresh generation starts from random latent variables rather
than an input drawing.

| Run | Latent dimensions | Regularization β | Epochs | Learning rate |
| --- | ---: | ---: | ---: | ---: |
| Starting model | 16 | 1 | 5 | 0.001 |
| More training | 16 | 1 | 20 | 0.001 |
| Weaker regularization | 16 | 0.1 | 20 | 0.001 |
| Stronger regularization | 16 | 4 | 20 | 0.001 |
| Smaller latent space | 2 | 1 | 20 | 0.001 |

Use the same training and validation drawings throughout the activity. The
default experiment has 16 latent dimensions, β = 1 and 20 epochs. The controls
also offer other learning rates and model settings for further investigation.

Completion requires the baseline, at least five distinct configurations, an
epoch comparison, weaker and stronger regularization comparisons, and a latent
dimension comparison. Controlled comparisons change one setting at a time.
Equivalent configurations can share runs. Identical reruns count once.

Inspect the reconstruction and KL components separately. Their weighted sum
uses the selected β, so use the public validation distribution score to compare
generation across configurations. Prepare to explain which drawings improve,
which variations the model captures and where it repeats a similar pattern.

## Scoring and uploads

Each recorded run generates 1,000 images, 200 for each category. The latent
draws use the fixed seed 2026. The preview controls use separate randomness.
The notebook evaluates the exact quantized images retained for upload.

A fixed small image classifier maps each image to a feature vector. The score
compares generated and reference feature distributions within each category,
then averages across categories. A lower score is better. Public reference
drawings support model selection; the server uses separate private references
for the ranked score. Generated grids help interpret the score's view of
similarity and variation.

The configuration dropdown initially selects the run with the lowest public
validation score and preserves your choice when new runs appear. Upload 2 uses
that run's retained final-epoch checkpoint and frozen image batch. It performs
no further training. The uploaded images contain no student code or model
weights. The server computes the ranked score from those images.

Both uploads save a `sketch-generation_<name-slug>_<hash>.json` file containing
the generated images, model settings and fingerprints. Your name and surname
appear on the course dashboard. The final upload replaces the local file; the
server retains accepted score history. If the service is unavailable, download
the saved JSON for manual upload later.

Rerunning an upload with the same identity, model and images reuses its saved
timestamp and fingerprint. The server recognizes that retry as a duplicate.
Choosing another model or changing your identity prepares a new export.

## Source files and release

- `notebook.py` is the editable Jupytext source; `notebook.ipynb` is its
  unexecuted Colab version.
- `model.py` contains the conditional VAE, deterministic training and image
  generation helpers.
- `evaluator.py` contains the fixed feature extractor and distribution score.
- `data/manifest.json` records file hashes, data versions and evaluator identity.
- `data/train.npz` and `data/validation.npz` contain the approved public rasters.
- `data/evaluator.npz` contains the fixed classifier's numeric parameters.
- `assets/` contains static SVG notebook panels.

Version: `sketch-generation-v1`. Downloads use that immutable release tag.
The release tag must exist before distributing its Colab link. See the
repository README for the pinned Jupytext conversion command; generate
`sketch-generation/notebook.ipynb` from `sketch-generation/notebook.py` and keep
all cell outputs and execution counts empty.

The drawings come from Google's [Quick, Draw! dataset](https://github.com/googlecreativelab/quickdraw-dataset),
available under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The approved release contains a fixed subset and preserves the source
attribution. The manifest and instructor preparation guide describe selection
and splitting.

## Instructor rehearsal

Use the instructor preparation commands to produce public files, private
references and evaluator parameters. Refresh the manifest after changing
`model.py`, `evaluator.py` or an artifact, then update the notebook's pinned
manifest hash. Publish compatible files together under a new immutable version
when approved data or scoring changes.

The reusable model helper also accepts CPU explicitly for an instructor
rehearsal. Keep checkpoints and generated rehearsal outputs in a disposable
private directory:

```bash
python sketch-generation/model.py train \
  --data sketch-generation/data/train.npz \
  --validation sketch-generation/data/validation.npz \
  --output /private/tmp/sketch-generation-rehearsal/baseline.pt \
  --device cpu --epochs 5

python sketch-generation/model.py sample \
  --checkpoint /private/tmp/sketch-generation-rehearsal/baseline.pt \
  --output /private/tmp/sketch-generation-rehearsal/baseline.npy \
  --device cpu
```

Create the disposable output directory first. Instructor checkpoints contain
configuration, CPU weights and numeric training history. Load them through
`model.load_checkpoint`, which uses PyTorch's weights-only loader. Submission
files contain generated images and metadata; the server reads them as data.

Rehearse all five configurations, generated-image quality, the local validation
score and its agreement with server scoring against the same references.
Check reconstruction, sampling, interpolation, nearest-neighbor displays,
configuration selection, duplicate uploads, access-code errors, manual upload
and unavailable-service recovery using disposable student identities and score
storage. Inspect the generated notebook and SVG panels. Finish with a separate
fresh Colab GPU rehearsal before classroom distribution.
