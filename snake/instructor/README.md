# Snake instructor guide

The activity trains a stochastic Snake policy on CPU using episodic REINFORCE
with a separate Monte Carlo value baseline. Students choose signed food,
collision, movement and distance rewards. Public and private evaluation rank
food collected first and collection speed second.

## Public release

The simulator, protocol and NumPy evaluator have matching copies in the
leaderboard repository. After editing a shared helper, update both copies and
check them before creating new fingerprints:

```bash
python snake/instructor/prepare.py check --leaderboard-root ../intro-ai-leaderboard
python snake/instructor/prepare.py public
```

The public manifest records the challenge version, protocol fingerprint,
public evaluation seeds and SHA-256 hashes of all four helpers. Run the
commands with NumPy installed. The protocol fingerprint binds the rules,
observation order and the three trusted source modules.

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
uv run --no-project --with numpy --with torch python snake/instructor/rehearse.py \
  --output /private/tmp/snake-rehearsal-learning --episodes 8192 --seeds 42,43,44
```

Record episodes, environment steps, updates, elapsed time, policy entropy,
public food scores, independent evaluation scores and representative replays.
Use these measurements to choose release episode presets. Different rewards
can change episode length and the number of moves in a fixed episode budget.

The policy update averages episode sums of discounted reward-to-go minus the
old value baseline. The value network then fits the Monte Carlo returns.
Training uses raw rewards, fixed architecture and sampled actions. Public
validation measurements leave the final training budget and retained actor
unchanged.

## Notebook and upload rehearsal

1. Run the three setup cells, inspect the untrained agent and train the baseline.
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
