# Prepare the sketch-generation challenge

The preparation CLI creates the public student archives and trains the frozen
sketch evaluator. Run it from the challenge repository. Data preparation and
manifest commands require NumPy; evaluator training also requires PyTorch.
The course compute image supplies these libraries independently of Quarto.

Keep the source cache and private preparation directory outside the public
repository. The example below uses disposable local preparation storage.

```bash
python sketch-generation/instructor/prepare.py prepare-data \
  --cache-dir /private/tmp/intro-ai-sketch-generation/sources \
  --private-dir /private/tmp/intro-ai-sketch-generation/private \
  --public-dir sketch-generation/data \
  --seed 20261028

python sketch-generation/instructor/prepare.py train-evaluator \
  --private-dir /private/tmp/intro-ai-sketch-generation/private \
  --public-dir sketch-generation/data \
  --device cpu --epochs 20 --seed 20261028

python sketch-generation/instructor/prepare.py manifest \
  --public-dir sketch-generation/data
```

## Source and partitions

The source is Google's official 28 × 28 QuickDraw bitmap export. Categories, in
label order, are `apple`, `house`, `fish`, `tree`, and `umbrella`. Bitmap values
describe ink intensity: 0 is background and 255 is full ink. Empty images are
discarded. The first copy of an exact same-category bitmap is retained; all
copies of a bitmap that appears in conflicting categories are discarded.
The bitmap export supplies image arrays. Preparation uses no `recognized`
metadata filter.

The remaining source indices are shuffled reproducibly, independently per
category, before assignment. Every partition below is disjoint by bitmap hash.

| Partition | Images per category | Location and purpose |
| --- | ---: | --- |
| Student training | 5,000 | Public `data/train.npz` |
| Public validation | 512 | Public `data/validation.npz` |
| Evaluator training | 10,000 | Private classifier fitting and feature normalization |
| Evaluator validation | 2,000 | Private classifier epoch selection and recognition checks |
| Calibration | 512 | Private fixed kernel bandwidth preparation |
| Private reference | 512 | Private server reference features |
| Audit | 1,000 | Private instructor score and memorization investigations |

Each public data archive contains `images` (`uint8[N,28,28]`) and `labels`
(`int64[N]`). The public manifest binds their SHA-256 hashes to the dataset
fingerprint. It also supplies hashes for the evaluator weights and the notebook
helper modules. Run `manifest` again after changing a helper module.

Private `split_indices.npz` stores each partition's original source indices and
bitmap SHA-256 hashes. Private `preparation.json` records source-file hashes,
the preparation seed, filter counts and partition sizes. Preserve these files
with an instructor backup when preparing the approved release.

## Frozen evaluator and score

The classifier is an MLP with widths 784, 128, 64 and 5, using ReLU hidden
activations. It trains for at most 20 epochs with batch size 256, Adam learning
rate 0.001 and weight decay 0.0001. The lowest evaluator-validation
cross-entropy selects its frozen epoch.

The scoring representation combines its 64-dimensional hidden activation with
49 pooled image features. Separate means and standard deviations are fitted on
evaluator-training images only, with a standard-deviation floor of 0.05. Each
branch is normalized to unit length, then the two branches are concatenated and
divided by the square root of two. This yields 113 features.

Each category's bandwidth is the median nonzero pairwise feature distance in
its calibration partition. The kernel averages three Gaussian RBF kernels at
half, one and twice that bandwidth. The score is 1,000 times the equally
weighted mean of the five biased empirical squared MMD values. Lower scores
indicate closer feature distributions. The server compares exactly 200 uploaded
images per category with 512 reference images per category.

The canonical NumPy implementation is `evaluator.py`. Identical module bytes
must be used in the notebook and leaderboard repositories. Its fingerprint
binds the module bytes, frozen weight bytes and metric version. The student
model helper has a separate manifest checksum.

The preparation gates require at least 90% balanced recognition accuracy,
at least 85% recall in every category, NumPy/PyTorch logit agreement within
0.0001, and a lower score for held-out human sketches than blank images, noise,
wrong-category images, repeated drawings and repeated category means in each of
20 independently seeded comparisons. Each human batch samples 200 distinct
images per category from the disjoint audit pool. Images may recur between
batches. Wrong-category rotations, noise and the repeated drawing vary between
comparisons. Blank images and category means are fixed.

Private `evaluator_report.json` records all batch scores, their means, sample
standard deviations, minima and maxima, human win counts, per-category
comparisons and image diagnostics. The preparation gate uses every aggregate
comparison; per-category comparisons support inspection of uneven behavior.
The same report records classifier checks and training history.

The public status `evaluator_ready` covers data and evaluator preparation. The
separate release status remains `pending_instructor_colab_rehearsal` until the
instructor rehearses the complete student activity in fresh GPU Colab.

The supplied evaluator reached 94.64% balanced accuracy, with category recall
between 92.85% and 96.50%. Held-out human batches beat each of the five
degenerate comparators in all 20 aggregate comparisons. Every individual
category comparison also passed 20 out of 20. The human-batch score averaged
2.96, with a sample standard deviation of 0.23. Resampled training images
averaged 3.11, consistent with the metric's role as a distribution comparison.

A local Linux ARM Docker rehearsal restricted to one CPU and one BLAS thread
scored the 1,000-image trained baseline in a median of 38 milliseconds over
seven runs. This timing covers feature extraction, MMD and image diagnostics;
HTTP, authentication, JSON decoding and persistence have separate costs.
The private `rehearsal/metric-robustness.json` records full comparison results
and timings. Hosted GPU Colab remains an instructor rehearsal step.

Training-image resampling is also scored as an instructor baseline. The metric
assesses the submitted image distribution. Inspect nearest training images and
the notebook investigations when discussing memorization and generation.

## Server artifacts

The private preparation directory retains raw image partitions. Server-ready
files are written separately under its `server/` subdirectory:

- `evaluator.npz`: identical bytes to the public frozen weights.
- `private_reference.npz`: numeric reference arrays, with keys `features`
  (`float64[5,512,113]`), `squared_norms` (`float64[5,512]`), `kernel_mean`
  (`float64[5]`) and `evaluator_digest` (`uint8[32]`).
- `servermanifest.json`: dataset, evaluator and reference fingerprints, file
  hashes and the preparation gate status.

The digest array contains the raw bytes of the evaluator fingerprint.
Reference features and their kernel means are prepared once. Submission
scoring performs only NumPy inference and array calculations.

For maintenance changes that preserve the feature definition, refresh the
fingerprints and reference archive using the already trained weights:

```bash
python sketch-generation/instructor/prepare.py freeze-evaluator \
  --private-dir /private/tmp/intro-ai-sketch-generation/private \
  --public-dir sketch-generation/data
```

This command verifies the weight hash, repeats classifier and metric checks,
and rewrites the private server files. Changes to the feature definition or
classifier training require fresh evaluator preparation and a new approved
challenge version. Source caches, private partitions, audit samples and server
reference artifacts belong in instructor storage.

## References

The [QuickDraw dataset](https://github.com/googlecreativelab/quickdraw-dataset)
is made available by Google, Inc. under CC BY 4.0. This challenge selects and
deduplicates subsets of its raster images.
[Demystifying MMD GANs](https://arxiv.org/abs/1801.01401) describes related
distribution-based generative-model evaluation. The course score uses a
frozen sketch representation and an RBF kernel mixture.
