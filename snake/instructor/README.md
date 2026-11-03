# Snake instructor guide

The activity trains a stochastic policy for a 15 × 15 Snake board on CPU using
episodic REINFORCE with a separate Monte Carlo value baseline. Each game lasts
at most 1,500 attempted moves. Students choose signed food,
collision, movement and distance rewards. Public and private evaluation rank
food collected first and collection speed second.

The reference configuration trains for 16,384 episodes with learning rate
0.003, discount 0.999, food reward 10, collision reward −10, movement reward
−0.01 and distance-progress reward 1. The starting agent uses the same settings
for 256 episodes. Both runs initialize the actor from the checked
`data/starter-policy.npz` artifact and create fresh critics and optimizers.
Teacher training, demonstration selection and imitation preparation use
[`warm_start.py`](warm_start.py) in this directory. See the
[imitation guide](imitation.md) and [experiment report](results-20261103.md).
Generated teachers, games and evaluation artifacts stay in the repository's
ignored `_private/` directory. The notebook artifact contains only the fixed
actor parameters.

## Public release

The simulator, protocol and NumPy evaluator have matching copies in the
leaderboard repository. After editing a shared helper, update both copies and
check them before creating new fingerprints:

```bash
python snake/instructor/prepare.py check --leaderboard-root ../intro-ai-leaderboard
python snake/instructor/prepare.py public
```

The public manifest records the challenge version, protocol fingerprint,
public evaluation seeds and SHA-256 hashes of the required and optional CPU
helpers and the starting-policy artifact. Run the commands with NumPy installed.
The protocol fingerprint binds the rules, observation order and the three trusted source modules.

Commit the prepared helpers and manifest. Pin the notebook's
`RELEASE_REVISION` to that immutable commit and `MANIFEST_SHA256` to its
manifest hash. Generate the notebook with the pinned Jupytext command in the
repository README. Keep the source and unexecuted notebook together.

Changes to game rules, observations, policy interpretation or scoring require
a new protocol release. Keep previously distributed revisions available.

## Private evaluation assets

Create private seeds in a separate directory. For a disposable rehearsal:

```bash
python snake/instructor/prepare.py private \
  --private-dir /private/tmp/snake-rehearsal/bucket/challenges/snake
```

The command creates `private_seeds.json` and `servermanifest.json` with
restricted file permissions. It refuses to replace existing assets. The
manifest binds 100 private environment/action seed pairs to the protocol and
the seed file's checksum. Public, demonstration and private seeds are
disjoint. Keep the private files outside public materials and Git.

Place these files in the private bucket's `challenges/snake/` prefix when
publishing the activity. Before enabling Snake in a production application
that requires existing state, create and verify its initial empty score
snapshot. Follow the leaderboard repository's Snake preparation instructions.

## CPU rehearsal

Run the reusable trainer and the instructor rehearsal command on CPU. Store
policies, reports and submitted files in a disposable private directory.
Measure the baseline, longer training and each reward comparison. Repeat the
principal comparisons with three initialization seeds and evaluate them on
separate instructor games.

```bash
uv run --no-project --with numpy --with torch --with numba python snake/instructor/rehearse.py \
  --output /private/tmp/snake-rehearsal-learning --episodes 16384 --seeds 42,43,44
```

Record episodes, environment steps, updates, elapsed time, policy entropy,
public food scores, independent evaluation scores and representative replays.
Use these measurements to choose release episode presets. Different rewards
can change episode length and the number of moves in a fixed episode budget.

The trainer adds potential-based shaping with fixed strength 1. A live state's
potential Φ is 1 when a static four-neighbor path joins head and current tail
through free cells; the head and tail are traversable and other body cells are
blocked. Collision, board-completion and move-limit terminal states have Φ = 0.
The shaping reward is $\gamma\Phi(s_{t+1})-\Phi(s_t)$.

Complete-episode Monte Carlo returns satisfy $G'_t=G_t-\Phi(s_t)$. The actor
maximizes the episode-average sum of
$\gamma^t(G'_t-V(s_t))\log\pi(a_t\mid s_t)$, with the separate value network
fitting $G'_t$. The training curves show the undiscounted base and shaped reward
sums per game separately. The network architecture and sampled actions remain
fixed. Public validation measurements leave the final training budget and
retained actor unchanged.

## Notebook and upload rehearsal

1. Run the three setup cells, verify the imitation policy loads, inspect the untrained demonstration agent and train the baseline.
2. Prepare Upload 1 with a disposable identity and verify its saved JSON.
3. Record at least three configurations, inspect rewards, learning curves and replays.
4. Check the configuration dropdown and upload its retained final actor.
5. Retry the same file and confirm that its timestamp, fingerprint and history entry stay fixed.
6. Check invalid access codes, unavailable-service recovery and manual upload.
7. Check that food ties use collection time in selection, upload feedback and the dashboard.
8. Save and restore a disposable score snapshot, then verify the reset hook.

Finish with a fresh Colab CPU session. For unpublished helpers, upload the
local notebook and a verified helper bundle into the runtime:

```bash
python snake/instructor/prepare.py bundle --output /private/tmp/snake-helpers.zip
```

After uploading the ZIP through Colab's Files panel, run this instructor cell
before the three setup cells:

```python
import zipfile
with zipfile.ZipFile('/content/snake-helpers.zip') as archive:
    archive.extractall('/content')
INSTRUCTOR_HELPER_DIR = '/content/snake-helpers'
```

Setup verifies the same manifest and file hashes. After publication, rehearse
the normal pinned download path separately.

The server evaluates every policy on the same private games. Repeated uploads
therefore provide feedback against that fixed evaluation set. Select models
using the public evidence and discuss observations with students.

## CPU simulator backends

The trainer supports `backend="reference"`, `"torch_eager"`, `"torch_compile"`
and `"numba"`. The notebook selects `backend="numba"` for training. Direct calls
to `model.train_policy` default to `"reference"`; pass `backend="numba"` to
reproduce notebook training and provide the starting policy as shown below.
Direct calls use random actor initialization when `starting_policy` is omitted.
All backends preserve the game, policy network,
32-game batches, action sampling and policy-gradient updates. Neural-network
updates use CPU PyTorch.

The tensor simulators store fixed-size body-age grids and mask completed games.
They precompute each game's food ranks from its canonical PCG64 stream.
Parity checks compare observations, transitions, rewards, trajectories and
retained actors. The trusted NumPy leaderboard evaluator stays independent of
the training backend.

Run a comparison with a retained instructor actor. The command below measures
three warm collection repeats, cold compilation and two or three complete
training runs:

```bash
uv run --no-project --with numpy --with torch --with numba \
  python snake/instructor/benchmark_cpu.py \
  --output /private/tmp/snake-cpu-benchmark \
  --policy /private/tmp/instructor-policy.json \
  --batches 16 --repeats 3 --train-episodes 8192 --fresh-cache
```

Omit `--policy` to use an untrained actor. For representative long games,
provide a trained numeric checkpoint. Timing excludes trajectory hashing;
complete training timing includes collection, updates and public evaluation.
The JSON report records versions, source hashes, first-call costs, warmed
throughput and policy parity. Compilation costs should be considered alongside
the number of experiments in a runtime.

The PyTorch compiler backend requires a working C++ compiler and OpenMP runtime.
Numba supplies the default simulator and the tail-connectivity kernel used by
every training backend. Notebook setup installs it when needed. The first
training run compiles the required kernels, and later runs reuse their compiled
specializations within the process. Its disk cache is disabled so local,
package and release-scoped notebook imports remain independent.
For an individual instructor experiment:

```python
checkpoint = model.train_policy(config, backend="numba",
                                starting_policy=model.read_policy("snake/data/starter-policy.npz"))
print(checkpoint["training_seconds"], checkpoint["compile_seconds"])
```

Backends and compilation timing are checkpoint metadata. The submitted model
settings and grading protocol retain the same format. Rehearse optional
backends in the intended Colab CPU runtime before using them in class.
