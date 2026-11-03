# Snake imitation initialization

This instructor pipeline trains a teacher, saves its best games, and trains a
small policy to imitate the recorded actions. It then compares continued
REINFORCE training from imitation and random initialization. The scripts live
in `snake/instructor/` in this challenge repository. Generated teachers, games
and evaluations belong in untracked instructor storage.

## Run the pipeline

Use an isolated environment with NumPy, PyTorch and Numba. Keep numerical
libraries on one thread for reproducible timing. Run these commands from the
challenge repository root. Set `SNAKE_RUN_DIR` to an output directory accepted
by the CLI before running the pipeline:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
uv run --no-project --with numpy --with torch --with numba python snake/instructor/warm_start.py all \
  --output "${SNAKE_RUN_DIR:?Set the experiment output directory}" --evaluation-stream 5000000
```

The scripts load the helpers from this repository's `snake/` directory.
Set `SNAKE_HELPER_DIR` to use another checkout.

The four stages can also run separately by replacing `all` with `expert`,
`demos`, `imitate`, or `compare`. Completed RL runs are reused when their
settings and starting-policy hash agree. Keep one output directory per recipe
and source revision. An interrupted RL run restarts that run; completed stages
remain available.

1. **Teacher:** 262,144 complete games, seed 42, 32 games per update, one critic
   update, learning rate 0.003, discount 0.999, food reward 10, collision reward
   −10, movement reward −0.01, distance reward 1, tail-connectivity potential
   strength 1, and trap-entry penalty 0.25. The trap penalty applies to a live
   reachable-to-unreachable transition. The saved actor is the final actor.
2. **Games:** generate 10,000 independent stochastic games and save the top
   1,000, ranked by apples and then food-collection time. Exact ties use the
   generation index. `elite-games.npz` stores seeds, every action, episode
   lengths and scores. Replaying these data reconstructs the observations.
3. **Imitation:** split the elite games into 900 training and 100 validation
   games. Sample up to 512 positions uniformly from each training game, then
   train the standard 685–64–64–3 actor for one epoch using action cross-entropy, Adam at
   0.001, and batches of 1,024 transitions. Retain the final actor. `starter-policy.npz` contains six float32 arrays, about 190 KiB
   before compression. Temporary observation caches are removed afterward.
4. **Comparison:** run 16,384 RL episodes for random and imitation starts with
   each seed 42, 43 and 44. Both arms use the standard classroom rewards,
   potential shaping, fresh critics and fresh optimizers. The extra teacher
   trap penalty is confined to teacher training. Evaluate every starting and
   final policy on the same 1,000 independent games. Monitoring, demonstration,
   teacher evaluation and comparison seed ranges are separate.

`--expert-episodes`, `--trap`, `--games`, `--il-epochs`, `--compare-episodes`,
`--examples-per-game`, `--evaluation-games`, `--evaluation-stream` and `--seeds`
expose the experiment budgets and independent evaluation seed range. Use
`--il-epochs 3 --examples-per-game 0` to reproduce the full-imitation variant.
Game generation requires a multiple of 10 with at least 10,000 games, so the retained top 10%
contains at least 1,000 complete games. The independent evaluator verifies a
30-game subset for every final policy. Demonstration replay verifies every
retained game's length, apples and food-collection time.

## Decision and release

`comparison.json` reports apples per game, collection time, training moves,
elapsed time, and paired score differences. Its `pass` field requires a
positive lower bound of the approximate 95% paired game interval for all three
comparisons in every training seed:

- imitation initialization versus random initialization;
- imitation followed by RL versus imitation initialization;
- imitation followed by RL versus random initialization followed by RL.

Intervals describe variation across the evaluation games. The three training
seeds give a separate check of sensitivity to training randomness. Teacher,
demonstration and imitation costs are incurred once by the instructor; report
student RL time separately.

After a passing comparison, copy `starter-policy.npz` into `snake/data/`,
add it to the checked release manifest, and pin
the notebook to the committed helpers and artifact. Keep the teacher and its
games private. Rehearse loading and continued training, then in a fresh
Colab CPU runtime before publication. Publication requires a separate request.

The [experiment report](results.md) records both imitation
variants, the passing starter policy, and their complete comparison scores.
Generate comparison figures with:

```bash
uv run --no-project --with numpy --with matplotlib python snake/instructor/plot_comparison.py "${SNAKE_RUN_DIR:?Set the experiment output directory}"
```

## Imitation-only budget checks

The `finetune` stage trains only from a saved imitation policy. Use a separate
output directory for each episode budget, containing that policy's
`imitation.json`. Set `SNAKE_SHORT_RUN_DIR` and `SNAKE_LONG_RUN_DIR` to those
directories. For one initialization seed and four PyTorch CPU threads:

```bash
uv run --no-project --with numpy --with torch --with numba python snake/instructor/warm_start.py finetune \
  --output "${SNAKE_SHORT_RUN_DIR:?Set the short-run output directory}" --compare-episodes 256 \
  --seeds 42 --threads 4 --evaluation-stream 5000000

uv run --no-project --with numpy --with torch --with numba python snake/instructor/warm_start.py finetune \
  --output "${SNAKE_LONG_RUN_DIR:?Set the long-run output directory}" --compare-episodes 262144 \
  --seeds 42 --threads 4 --evaluation-stream 5000000
```

`finetune.json` records the initial and final scores on the same evaluation
games, their paired differences, training moves, elapsed time and CPU thread
count. Each budget starts from the saved imitation policy with a fresh critic
and optimizers. `--threads` controls PyTorch parallelism during RL; the Numba
game simulator remains single-threaded. The default is one thread. Thread count
is part of the cache key, and changing it requires a separate output directory. Floating-point reduction
order can change between thread counts.

## Compare food reward modes

Use `--food-reward-mode constant` or `--food-reward-mode linear` with the
training and comparison stages. Constant is the default. Linear multiplies the
food weight by the apple number within the current game. Keep the food weight,
training budget, initialization seed, CPU thread count and evaluation games
fixed when comparing the modes. Use a separate output directory for each mode;
the mode is included in the cached-run settings.
