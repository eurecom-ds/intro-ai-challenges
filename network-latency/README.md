# Will this video call lag?

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eurecom-ds/intro-ai-challenges/blob/main/network-latency/notebook.ipynb)

Predict video-frame delivery latency from ten network measurements. This
45–60 minute activity compares polynomial regression and a small neural
network. The data are a fictional teaching example, and latency is measured
in milliseconds.

## Start the activity

1. Open the notebook in Colab and choose a CPU runtime.
2. Run the three setup cells. The two public CSVs download automatically.
3. Inspect the data and run the degree-0 baseline.
4. Enter your name, surname and the instructor's code in the `access_code`
   field in Upload 1, then upload the baseline.
5. Complete the investigations and compare models using validation RMSE.
6. Run the selection cell before Upload 2 and choose a recorded configuration
   from the dropdown.
7. Run Upload 2 to upload its predictions.

Use the notebook controls for each experiment. Predict the effect of a change
before running it, then compare the result with your prediction. Keep the
training/validation split fixed during controlled comparisons. The default
split uses 60% of the 1,000 labelled frames for training and 40% for validation.

Completion requires at least five distinct configurations: the baseline,
a comparison that changes only polynomial degree, a comparison that changes
only regularization at a positive degree, and a neural-network configuration.
Prepare to explain your comparisons and residual plots to the instructor.

The dropdown lists each recorded run, its settings and validation RMSE. It
initially selects the lowest validation RMSE. New experiments refresh the list
while keeping your choice. Changing the data split or rerunning Setup 3 clears
the selection.

Upload 2 reruns the chosen configuration on the training split before refitting
it on all public labels. To upload another recorded configuration, change the
dropdown and run Upload 2 again.

Both uploads refit their chosen settings on all public labels before predicting
the 2,400 test frames. Validation RMSE describes the model-selection fit;
public fitting RMSE describes the final refit. The server computes private
test RMSE in milliseconds. Keep using validation evidence to select settings.

The upload saves a `network-latency_<name-slug>_<hash>.json` file. If the service
is unavailable, download this file and use the dashboard's manual upload form
later. The final upload replaces the local JSON; the server keeps accepted
score history. Your name and surname appear on the course dashboard.

[Open the leaderboard](https://eurecom-probai-intro-ai-leaderboard.hf.space/challenges/network-latency)

## Files and version

- [Student notebook](notebook.ipynb)
- [Labelled public frames](data/public.csv)
- [Feature-only test frames](data/test.csv)

Version: `network-latency-v1`. Keep the original data files unchanged.
For a local Jupyter run, open the notebook from this directory so it finds
`data/` beside it. NumPy, Matplotlib, IPython, ipywidgets and CPU PyTorch are
required; Colab supplies these libraries.

The editable source is `notebook.py`. See the repository README for the pinned
Jupytext generation command.
