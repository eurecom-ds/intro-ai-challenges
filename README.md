# Introduction to AI challenges

Hands-on activities for the Introduction to Artificial Intelligence course at EURECOM.
Open a notebook in Google Colab, choose a CPU runtime, and run its setup cells.
The notebook downloads its public data automatically. Use the supplied controls
to run experiments and submit predictions to the course leaderboard.

| Challenge | Activity | Open notebook |
| --- | --- | --- |
| [Network latency](network-latency/README.md) | Compare polynomial regression and a neural network using validation RMSE | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eurecom-ds/intro-ai-challenges/blob/main/network-latency/notebook.ipynb) |

[Course website](https://eurecom-ds.github.io/intro-ai/) ·
[Leaderboard](https://eurecom-probai-intro-ai-leaderboard.hf.space/)

## Maintaining a challenge

Edit the Jupytext percent-format source, `network-latency/notebook.py`.
Generate the student notebook from the repository root:

```bash
uvx --from jupytext==1.19.6 jupytext --to ipynb network-latency/notebook.py \
  --output network-latency/notebook.ipynb
```

Commit the source and generated notebook together. Keep notebook execution
counts empty and outputs cleared. Maintain the approved public CSVs separately;
the notebook checks their SHA-256 hashes before use. Data downloads reference
the immutable `network-latency-v1` release tag. Publish a new version when the
approved data changes, updating the tag reference and checksums together.

Rehearse the notebook and upload flow with disposable submissions before a
release. The instructor performs a separate rehearsal in a fresh Colab session.
