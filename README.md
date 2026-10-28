# Introduction to AI challenges

Hands-on activities for the Introduction to Artificial Intelligence course at EURECOM.
Open a notebook in Google Colab, choose the runtime listed below, and run its setup cells.
The notebook downloads its public data automatically. Use the supplied controls
to run experiments and upload submissions to the course leaderboard.
For the final network-latency upload, choose a recorded configuration from the
dropdown and run Upload 2.

| Challenge | Activity | Runtime | Open notebook |
| --- | --- | --- | --- |
| [Network latency](network-latency/README.md) | Compare polynomial regression and a neural network using validation RMSE | CPU | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eurecom-ds/intro-ai-challenges/blob/main/network-latency/notebook.ipynb) |
| [Teach a model to draw](sketch-generation/README.md) | Train a conditional VAE and compare generated sketch distributions | GPU | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eurecom-ds/intro-ai-challenges/blob/main/sketch-generation/notebook.ipynb) |
| [Teach a diffusion model to draw](sketch-generation/README-diffusion.md) | Train a small diffusion model from scratch and compare public scores locally | GPU | [Diffusion notebook](sketch-generation/notebook-diffusion.ipynb) |

[Course website](https://eurecom-ds.github.io/intro-ai/) ·
[Leaderboard](https://eurecom-probai-intro-ai-leaderboard.hf.space/)

## Maintaining a challenge

Edit each activity's Jupytext percent-format `notebook.py` source.
Generate its student notebook from the repository root:

```bash
uvx --from jupytext==1.19.6 jupytext --to ipynb network-latency/notebook.py \
  --output network-latency/notebook.ipynb

uvx --from jupytext==1.19.6 jupytext --to ipynb sketch-generation/notebook.py \
  --output sketch-generation/notebook.ipynb

uvx --from jupytext==1.19.6 jupytext --to ipynb sketch-generation/notebook-diffusion.py \
  --output sketch-generation/notebook-diffusion.ipynb
```

Commit the source and generated notebook together. Keep notebook execution
counts empty and outputs cleared. Maintain the approved public CSVs separately;
the notebook checks their SHA-256 hashes before use. Data downloads reference
the immutable `network-latency-v1` release tag. Publish a new version when the
approved data changes, updating the tag reference and checksums together.

Sketch generation uses the immutable `sketch-generation-v1` release tag for
its public drawings and helper modules. Its manifest pins every artifact hash.
The [instructor preparation guide](sketch-generation/instructor/README.md)
describes dataset selection, evaluator training and private scoring artifacts.
Complete the local rehearsal and the separate instructor Colab GPU rehearsal
before publishing this release tag or distributing the activity.

The [experimental diffusion version](sketch-generation/README-diffusion.md)
preserves the VAE activity and reuses its public data and evaluator. It saves
results locally. The current leaderboard accepts VAE metadata. Publish the new
notebook and helper together before using their remote Colab downloads.

Static banners, step labels and activity cards are SVG images in each
activity's `assets/` directory. Embed these panels with Markdown image links so
their styling appears in Colab before any cells run. Keep descriptive image
text alongside the SVG title and description when editing a panel.

Rehearse the notebook and upload flow with disposable submissions before a
release. The instructor performs a separate rehearsal in a fresh Colab session.
