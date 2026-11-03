# Teach an AI to play Snake

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eurecom-ds/intro-ai-challenges/blob/main/snake/notebook.ipynb)

Start from a small imitation-trained Snake policy and improve it using policy
gradients. In this 60-minute activity, choose rewards for food, collisions, movement and progress toward food, then inspect
what your agent learns. Use a **CPU runtime** in Colab. All controls are supplied.

## Start the activity

1. Open the notebook and run its three setup cells.
2. Inspect the 15 × 15 game and watch an untrained agent.
3. Train the 256-episode baseline, enter your name and surname, and prepare Upload 1.
4. Enter the instructor's code in the `access_code` field in Upload 1. Keep the supplied `submission_url` unless your instructor gives you another one. The code stays out of the saved JSON.
5. Record at least three distinct configurations, including the baseline.
6. Compare one setting at a time. Predict the behaviour before training, then inspect the replay and public evaluation results.
7. Select a retained agent from the dropdown and run Upload 2.

After the 256-episode baseline, try 512, 1,024 and 2,048 episodes to inspect
early learning. Use 4,096 episodes (the default), or 8,192, for broader
investigations.
Keep the episode budget fixed when comparing settings. Rerun promising settings
with 16,384 episodes, then consider 32,768 or 65,536 for further training.
Episode counts are total training budgets. Every run reloads the same imitation policy
and initialization seed, creates a fresh value network and optimizers, and
retains its final actor. Repeating a configuration
counts once. Resetting Setup 3 clears the local experiment table and selection.

## The game and the policy

The snake begins with three body cells. It chooses between moving straight,
turning left and turning right. Eating food grows its body. A wall collision,
body collision, full board or 1,500 attempted moves ends a game. Moving into the
departing tail's cell is legal on a move without growth.

A small multilayer perceptron (MLP), with two 64-unit hidden layers, maps the
board and game information to probabilities for the three actions. Training
samples actions from those probabilities. The supplied starting policy learned
to predict actions from successful demonstration games through imitation
learning. Student experiments continue with rewards from their own games.
REINFORCE uses discounted future rewards to update the policy. A separate
value network supplies a baseline: the expected future reward from a state.
Actions with better-than-expected outcomes receive a positive advantage.

The 685 numeric inputs include the full ordered body, head, food, heading,
remaining moves, relative food position and immediate collision indicators.
The indicators provide information; all three actions remain available.

## Choose the training rewards

| Setting | Default | Range |
| --- | ---: | ---: |
| Food reward | 10 | −100 to 100 |
| Collision reward | −10 | −100 to 100 |
| Each-move reward | −0.01 | −10 to 10 |
| Distance-progress reward | 1 | −10 to 10 |

Choose **constant** or **linear** food rewards. Constant mode gives the same
food bonus for every apple. Linear mode multiplies the food reward by the
apple number within the current game: with weight 1, the second apple gives 2
and the tenth gives 10. The count resets each game and is independent of the
move number. The default is constant; evaluation always counts each apple once.

Every attempted move receives the movement reward. An eating move also
receives the food reward. A collision also receives the collision reward.
Distance progress uses Manhattan distance: one cell toward the current food
contributes +1, and one cell away contributes −1. On an eating move, progress
uses the food's position before it was eaten. A collision has zero distance
progress. These four controls define the base reward.

You can also choose learning rate, discount factor and training episodes.
The default learning rate is 0.003 and the discount factor is 0.999. An episode
is one complete game. Record the number of moves as well as episodes when
comparing training experience.

Training adds a fixed potential-based shaping term. For a live state, the
potential Φ is 1 when the head can reach the current tail through free cells,
and 0 otherwise. The path calculation treats the head and tail as available
and blocks the other body cells. Every terminal state has potential 0.

The added reward is

$$
F(s_t,s_{t+1})=\gamma\Phi(s_{t+1})-\Phi(s_t).
$$

For complete episodes, the trainer computes the shaped return as
$G'_t=G_t-\Phi(s_t)$, where $G_t$ is the discounted base return. The value
baseline fits the shaped return. Charts show the average base and shaped
reward sums per game separately. Apples and collection time provide the common
evaluation metrics.

The notebook suggests comparisons for longer training and each reward term.
Equivalent controlled comparisons can share runs. Completion requires the
baseline and at least three distinct configurations. Prepare to explain the
behaviour you observe and the settings you would investigate next.

## Evaluation and ranking

Every recorded agent plays the same 30 public evaluation games using sampled
actions and fixed independent game/action seeds. Compare agents using:

1. **Food collected**, higher first.
2. **Food-collection time**, earlier first when food totals tie.

Collection time is the move number when an apple is eaten. The table displays
apples per game and the average collection move across all eaten apples.
For a zero-food result, collection time is shown as “—”. Four apples always
rank above three, regardless of collection time.

The server applies the same rules on 100 separate private games. It reads the
uploaded policy weights as numeric data and simulates the agent. Your chosen
training rewards guide learning; ranked performance uses actual food and
collection time. Training returns use different scales when rewards change,
so use the common evaluation metrics to compare configurations.

The dropdown initially selects the best public result, preserves your choice
as new runs appear, and uploads that retained final actor without additional
training. Changing a replay seed leaves the recorded score and policy intact.

Upload 2 reuses your name, surname, submission URL and access code from Upload 1.
To correct those values, edit the fields in Upload 1 and run it again. Your
retained agents stay available for selection.

Both uploads save a `snake_<name-slug>_<hash>.json` file. If the service is
unavailable, download it for the dashboard's manual upload form. An unchanged
retry keeps its timestamp and fingerprint. The server retains accepted score
history, with exact ties favouring the earlier accepted submission. Your name
and surname appear on the public course dashboard.

[Open the Snake leaderboard](https://eurecom-probai-intro-ai-leaderboard.hf.space/challenges/snake)

## Files and release

The editable notebook source is `notebook.py`; its unexecuted Colab version is
`notebook.ipynb`. `environment.py` implements the game, `evaluator.py` supplies
trusted NumPy policy inference and scoring, `protocol.py` defines versioned
rules and controls, and `model.py` supplies the CPU trainer. Public helper
hashes and evaluation seeds are recorded in `data/manifest.json`. The compact
`data/starter-policy.npz` stores the six float32 parameter arrays for the shared
imitation initialization. Setup verifies its hash and loads numeric arrays
with pickle disabled.

The notebook uses the Numba simulator and tail-connectivity kernel in
`numba_environment.py`, with CPU PyTorch for neural-network updates. The first
training run compiles the kernels; later runs reuse the compiled functions.
`torch_environment.py` provides alternative simulator backends for instructor
CPU benchmarks.

Version: `snake-v2`. Downloads use the immutable revision recorded in the
notebook. Setup verifies the manifest and helper hashes before loading code.
For local Jupyter, open the notebook from this directory. NumPy, Numba, CPU
PyTorch, Matplotlib, IPython and ipywidgets are required. Colab supplies the
core libraries; notebook setup installs Numba when needed.

See the [instructor guide](instructor/README.md) for private evaluation assets,
CPU rehearsal, helper parity, snapshot initialization and release checks.
