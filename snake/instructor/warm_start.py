#!/usr/bin/env python3
"""Train a Snake teacher, retain elite games, clone its actions, and compare RL starts."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch
from torch import nn

DEFAULT_CHALLENGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.environ.get('SNAKE_HELPER_DIR', str(DEFAULT_CHALLENGE)))
import environment
import evaluator
import model
import numba_environment as fast
import protocol


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pairs(count, stream):
    # Disjoint, explicit ranges for demonstration, monitoring, and final evaluation.
    return [(stream + i, stream + 100_000 + i) for i in range(count)]


def trap_penalty(batch, config, strength=0.25):
    """Penalize reachable-to-unreachable transitions within each live game."""
    connected = fast.tail_connected_batch(batch['observations']).astype(bool)
    offset = 0
    totals = []
    for length in batch['lengths']:
        stop = offset + int(length)
        penalties = np.zeros(length, dtype=np.float64)
        penalties[:-1] = -strength * (connected[offset:stop - 1] & ~connected[offset + 1:stop])
        batch['returns'][offset:stop] += model.discounted_returns(penalties, config['gamma'])
        totals.append(float(penalties.sum()))
        offset = stop
    mean = float(np.mean(totals))
    batch['mean_return'] += mean
    batch['reward_components']['trap_entry'] = mean
    return batch


def rollouts(policy, seed_pairs, *, record=False):
    """Canonical NumPy sampling with accelerated game transitions, in batches of 64."""
    policy = evaluator.validate_policy(policy)
    rows, action_rows = [], []
    for first in range(0, len(seed_pairs), 64):
        chunk = seed_pairs[first:first + 64]
        rngs = [np.random.default_rng(a) for a, _ in chunk]
        action_rngs = [np.random.default_rng(b) for _, b in chunk]
        state, food_ranks = fast.initialize(rngs)
        observation = fast.observe(state)
        actions = np.zeros(len(chunk), dtype=np.int64)
        saved = np.full((len(chunk), protocol.MAX_STEPS), -1, dtype=np.int8) if record else None
        apples = np.zeros(len(chunk), dtype=np.int64)
        food_steps = np.zeros(len(chunk), dtype=np.int64)
        for move in range(protocol.MAX_STEPS):
            active = np.flatnonzero(state.reason == 0)
            if not len(active):
                break
            probabilities = evaluator.policy_probabilities(policy, observation[active], validated=True)
            actions[active] = evaluator.sample_actions(probabilities, [action_rngs[i] for i in active])
            if record:
                saved[active, move] = actions[active]
            state, events, observation = fast.step(state, actions, food_ranks)
            apples += events[:, 0].astype(np.int64)
            food_steps += events[:, 0].astype(np.int64) * (move + 1)
        for i, pair in enumerate(chunk):
            rows.append({'seeds': list(pair), 'apples': int(apples[i]),
                         'food_steps': int(food_steps[i]), 'steps': int(state.steps[i]),
                         'reason': fast.REASONS[int(state.reason[i])]})
        if record:
            action_rows.append(saved)
    return rows, np.concatenate(action_rows) if record else None


def score(rows):
    apples = np.asarray([row['apples'] for row in rows])
    total = int(apples.sum())
    food_steps = sum(row['food_steps'] for row in rows)
    return {'games': len(rows), 'mean_apples': float(apples.mean()),
            'total_apples': total, 'sum_food_steps': food_steps,
            'mean_food_step': food_steps / total if total else None,
            'standard_error': float(apples.std(ddof=1) / np.sqrt(len(rows)))}


def verify_rollouts(policy, seed_pairs):
    rows, _ = rollouts(policy, seed_pairs)
    canonical = evaluator.evaluate(policy, seed_pairs)
    result = score(rows)
    for key in ('total_apples', 'sum_food_steps'):
        if result[key] != canonical[key]:
            raise AssertionError(f'Canonical replay mismatch: {key}')
    return result


@contextmanager
def training_threads(count):
    """Override the classroom thread count inside this single-process driver."""
    original = model._cpu_randomness

    @contextmanager
    def configured(seed):
        with original(seed):
            torch.set_num_threads(count)
            yield

    model._cpu_randomness = configured
    try:
        yield
    finally:
        model._cpu_randomness = original


def train(output, name, episodes, seed, *, starting_policy=None, trap=0.0, cpu_threads=1):
    destination = output / f'{name}.json'
    if destination.exists():
        result = json.loads(destination.read_text())
        result['run'].setdefault('cpu_threads', 1)
        expected = {'episodes': episodes, 'seed': seed, 'trap': trap, 'cpu_threads': cpu_threads,
                    'starting_policy_sha256': policy_hash(starting_policy)}
        if result['run'] != expected:
            raise ValueError(f'Existing run settings differ: {destination}')
        return result
    last_log = [0.0]
    def progress(row):
        if row['episodes'] == 0 or time.monotonic() - last_log[0] >= 30 or row['episodes'] == episodes:
            brief = {key: row[key] for key in ('episodes', 'steps', 'seconds')}
            if 'validation' in row:
                brief['monitor_apples'] = row['validation']['mean_apples']
            else:
                brief['batch_apples'] = row.get('training_apples')
            print(json.dumps({'run': name, **brief}), flush=True)
            write_json(output / f'{name}-progress.json', brief)
            last_log[0] = time.monotonic()
    with training_threads(cpu_threads):
        result = model.train_policy(protocol.settings(episodes=16384), seed=seed,
                                starting_policy=starting_policy, episode_budget=episodes,
                                batch_transform=(lambda batch, config: trap_penalty(batch, config, trap)) if trap else None,
                                evaluation_seed_pairs=pairs(30, 2_000_000), validation_interval=256,
                                backend='numba', progress=progress)
    result['run'] = {'episodes': episodes, 'seed': seed, 'trap': trap, 'cpu_threads': cpu_threads,
                     'starting_policy_sha256': policy_hash(starting_policy)}
    write_json(destination, result)
    return result


def policy_hash(policy):
    if policy is None:
        return None
    return hashlib.sha256(json.dumps(policy, sort_keys=True).encode()).hexdigest()


def expert(args):
    result = train(args.output, 'expert', args.expert_episodes, 42, trap=args.trap, cpu_threads=args.threads)
    rows, _ = rollouts(result['policy'], pairs(1000, 3_000_000))
    verify_rollouts(result['policy'], pairs(30, 3_000_000))
    write_json(args.output / 'expert-evaluation.json', {'score': score(rows), 'games': rows})
    print('Teacher:', json.dumps(score(rows)), flush=True)


def demonstrations(args):
    teacher = json.loads((args.output / 'expert.json').read_text())['policy']
    rows, actions = rollouts(teacher, pairs(args.games, 1_000_000), record=True)
    count = args.games // 10
    chosen = sorted(range(args.games), key=lambda i: (-rows[i]['apples'], rows[i]['food_steps'], i))[:count]
    np.savez_compressed(args.output / 'elite-games.npz',
                        seeds=np.asarray([rows[i]['seeds'] for i in chosen], dtype=np.uint32),
                        actions=actions[chosen], lengths=np.asarray([rows[i]['steps'] for i in chosen]),
                        apples=np.asarray([rows[i]['apples'] for i in chosen]),
                        food_steps=np.asarray([rows[i]['food_steps'] for i in chosen]))
    write_json(args.output / 'demonstrations.json', {
        'teacher_sha256': policy_hash(teacher), 'generated': args.games, 'retained': count,
        'selection': 'apples descending, food-step sum ascending, generation index ascending',
        'all_score': score(rows), 'elite_score': score([rows[i] for i in chosen]),
        'selected_indices': chosen, 'games': rows,
        'dataset_sha256': digest(args.output / 'elite-games.npz')})
    print('Demonstrations:', json.dumps(score([rows[i] for i in chosen])), flush=True)


def materialize(args):
    """Replay compact action sequences into temporary, memory-mapped examples."""
    with np.load(args.output / 'elite-games.npz', allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    total = int(data['lengths'].sum())
    observations = np.lib.format.open_memmap(args.output / 'observations.npy', mode='w+', dtype=np.float32,
                                            shape=(total, protocol.OBSERVATION_SIZE))
    labels = np.lib.format.open_memmap(args.output / 'actions.npy', mode='w+', dtype=np.int64, shape=(total,))
    offsets = np.concatenate(([0], np.cumsum(data['lengths'])))
    for first in range(0, len(data['seeds']), 64):
        stop = min(first + 64, len(data['seeds']))
        state, ranks = fast.initialize([np.random.default_rng(int(pair[0])) for pair in data['seeds'][first:stop]])
        obs = fast.observe(state)
        apples = np.zeros(stop - first, dtype=np.int64)
        food_steps = np.zeros(stop - first, dtype=np.int64)
        for move in range(protocol.MAX_STEPS):
            active = np.flatnonzero(state.reason == 0)
            if not len(active):
                break
            positions = offsets[first + active] + move
            observations[positions] = obs[active]
            selected = data['actions'][first:stop, move].astype(np.int64)
            labels[positions] = selected[active]
            if np.any(selected[active] < 0):
                raise AssertionError('Recorded game ended before replay.')
            state, events, obs = fast.step(state, np.maximum(selected, 0), ranks)
            apples += events[:, 0].astype(np.int64)
            food_steps += events[:, 0].astype(np.int64) * (move + 1)
        np.testing.assert_array_equal(state.steps, data['lengths'][first:stop])
        np.testing.assert_array_equal(apples, data['apples'][first:stop])
        np.testing.assert_array_equal(food_steps, data['food_steps'][first:stop])
    observations.flush()
    labels.flush()
    return observations, labels, offsets


def save_numeric(path, policy):
    validated = evaluator.validate_policy(policy)
    np.savez_compressed(path, **{f'{name}_{i}': layer[name].astype(np.float32)
                                for i, layer in enumerate(validated['layers']) for name in ('weight', 'bias')})


def imitate(args):
    start = time.perf_counter()
    observations, labels, offsets = materialize(args)
    rng = np.random.default_rng(514)
    games = rng.permutation(len(offsets) - 1)
    holdout_count = max(1, len(games) // 10)
    validation = np.concatenate([np.arange(offsets[i], offsets[i + 1]) for i in games[:holdout_count]])
    training_games = []
    for i in games[holdout_count:]:
        indices = np.arange(offsets[i], offsets[i + 1])
        if args.examples_per_game:
            indices = rng.choice(indices, min(args.examples_per_game, len(indices)), replace=False)
        training_games.append(indices)
    training = np.concatenate(training_games)
    history = []
    with model._cpu_randomness(123):
        actor = model.PolicyNetwork()
        optimizer = torch.optim.Adam(actor.parameters(), lr=0.001)
        for epoch in range(args.il_epochs):
            rng.shuffle(training)
            loss_sum = 0.0
            for first in range(0, len(training), 1024):
                indices = training[first:first + 1024]
                x = torch.from_numpy(np.asarray(observations[indices]))
                y = torch.from_numpy(np.asarray(labels[indices]))
                loss = nn.functional.cross_entropy(actor(x), y)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                loss_sum += float(loss.detach()) * len(indices)
            valid_loss, correct = 0.0, 0
            with torch.no_grad():
                for first in range(0, len(validation), 1024):
                    indices = validation[first:first + 1024]
                    logits = actor(torch.from_numpy(np.asarray(observations[indices])))
                    y = torch.from_numpy(np.asarray(labels[indices]))
                    valid_loss += float(nn.functional.cross_entropy(logits, y, reduction='sum'))
                    correct += int((logits.argmax(1) == y).sum())
            row = {'epoch': epoch + 1, 'train_cross_entropy': loss_sum / len(training),
                   'validation_cross_entropy': valid_loss / len(validation),
                   'validation_accuracy': correct / len(validation)}
            history.append(row)
            print('Imitation:', json.dumps(row), flush=True)
        policy = model.export_policy(actor)
    save_numeric(args.output / 'starter-policy.npz', policy)
    write_json(args.output / 'imitation.json', {
        'policy': policy, 'history': history, 'epochs': args.il_epochs, 'seed': 123,
        'learning_rate': 0.001, 'batch_size': 1024, 'training_games': len(games) - holdout_count,
        'validation_games': holdout_count, 'training_examples': len(training),
        'examples_per_game': args.examples_per_game,
        'validation_examples': len(validation), 'seconds': time.perf_counter() - start,
        'dataset_sha256': digest(args.output / 'elite-games.npz'),
        'artifact_sha256': digest(args.output / 'starter-policy.npz')})
    # The compressed games retain every example exactly; the large replay cache is disposable.
    del observations, labels
    (args.output / 'observations.npy').unlink()
    (args.output / 'actions.npy').unlink()


def difference(after, before):
    delta = np.asarray([a['apples'] - b['apples'] for a, b in zip(after, before)], dtype=np.float64)
    mean = float(delta.mean())
    half = 1.96 * float(delta.std(ddof=1) / np.sqrt(len(delta)))
    return {'mean': mean, 'paired_95_percent_interval': [mean - half, mean + half]}


def compare(args):
    policy = json.loads((args.output / 'imitation.json').read_text())['policy']
    evaluation_pairs = pairs(args.evaluation_games, args.evaluation_stream)
    initial, _ = rollouts(policy, evaluation_pairs)
    verify_rollouts(policy, evaluation_pairs[:30])
    report = {'evaluation_games': args.evaluation_games, 'evaluation_stream': args.evaluation_stream,
              'rl_episodes': args.compare_episodes,
              'imitation_initial': score(initial), 'imitation_games': initial, 'seeds': [],
              'rules': 'Each paired 95% interval must have a positive lower bound in every training seed.'}
    for seed in args.seeds:
        random_initial, _ = rollouts(model.initial_policy(seed), evaluation_pairs)
        arms = {}
        for name, starting in [('random', None), ('imitation', policy)]:
            result = train(args.output, f'{name}-{seed}', args.compare_episodes, seed,
                           starting_policy=starting, cpu_threads=args.threads)
            rows, _ = rollouts(result['policy'], evaluation_pairs)
            verify_rollouts(result['policy'], evaluation_pairs[:30])
            arms[name] = {'score': score(rows), 'games': rows,
                          'seconds': result['training_seconds'], 'steps': result['training_steps']}
        deltas = {'il_vs_random_initial': difference(initial, random_initial),
                  'rl_gain_from_il': difference(arms['imitation']['games'], initial),
                  'il_rl_vs_random_rl': difference(arms['imitation']['games'], arms['random']['games'])}
        row = {'seed': seed, 'random_initial': score(random_initial), 'arms': arms, 'differences': deltas}
        report['seeds'].append(row)
        report['pass'] = (len(report['seeds']) == len(args.seeds) and
                          all(d['paired_95_percent_interval'][0] > 0 for r in report['seeds'] for d in r['differences'].values()))
        write_json(args.output / 'comparison.json', report)
        print('Comparison:', json.dumps({'seed': seed, 'il_initial': score(initial),
                                          'arms': {k: v['score'] for k, v in arms.items()}, 'differences': deltas}), flush=True)
    print('PASS:', report['pass'], flush=True)


def finetune(args):
    """Evaluate RL budgets from the saved imitation policy only."""
    policy = json.loads((args.output / 'imitation.json').read_text())['policy']
    evaluation_pairs = pairs(args.evaluation_games, args.evaluation_stream)
    initial, _ = rollouts(policy, evaluation_pairs)
    verify_rollouts(policy, evaluation_pairs[:30])
    report = {'initial': score(initial), 'initial_games': initial, 'seeds': [],
              'rl_episodes': args.compare_episodes, 'cpu_threads': args.threads,
              'evaluation_stream': args.evaluation_stream}
    for seed in args.seeds:
        result = train(args.output, f'imitation-{seed}', args.compare_episodes, seed,
                       starting_policy=policy, cpu_threads=args.threads)
        rows, _ = rollouts(result['policy'], evaluation_pairs)
        verify_rollouts(result['policy'], evaluation_pairs[:30])
        row = {'seed': seed, 'score': score(rows), 'games': rows,
               'seconds': result['training_seconds'], 'steps': result['training_steps'],
               'gain': difference(rows, initial)}
        report['seeds'].append(row)
        write_json(args.output / 'finetune.json', report)
        print('Fine-tuning:', json.dumps({k: v for k, v in row.items() if k != 'games'}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('expert', 'demos', 'imitate', 'compare', 'finetune', 'all'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expert-episodes', type=int, default=262144)
    parser.add_argument('--trap', type=float, default=0.25)
    parser.add_argument('--games', type=int, default=10000)
    parser.add_argument('--il-epochs', type=int, default=1)
    parser.add_argument('--examples-per-game', type=int, default=512,
                        help='Uniformly sample this many transitions per training game; 0 uses all.')
    parser.add_argument('--compare-episodes', type=int, default=16384)
    parser.add_argument('--evaluation-games', type=int, default=1000)
    parser.add_argument('--evaluation-stream', type=int, default=4_000_000,
                        help='Start of a fresh paired evaluation seed range.')
    parser.add_argument('--seeds', type=lambda s: [int(x) for x in s.split(',')], default=[42, 43, 44])
    parser.add_argument('--threads', type=int, default=1, help='PyTorch CPU threads during RL training.')
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.games < 10000 or args.games % 10:
        parser.error('Generate at least 10,000 games, in multiples of 10, to retain at least 1,000.')
    if args.il_epochs <= 0 or args.evaluation_games < 30 or args.evaluation_games > 10000 or not args.seeds:
        parser.error('Use positive IL epochs, 30–10,000 evaluation games, and at least one training seed.')
    if len(args.seeds) != len(set(args.seeds)):
        parser.error('Use distinct training seeds.')
    if args.threads < 1:
        parser.error('Use at least one CPU thread.')
    if args.examples_per_game < 0 or args.evaluation_stream < 4_000_000 or args.evaluation_stream > 2**32 - 110_000:
        parser.error('Use a nonnegative example cap and an evaluation stream between 4,000,000 and 2**32 - 110,000.')
    if not np.isfinite(args.trap) or args.trap < 0:
        parser.error('Use a finite nonnegative trap penalty.')
    if '_private' not in args.output.parts and Path('/private/tmp') not in args.output.parents:
        parser.error('Store generated artifacts under _private or /private/tmp.')
    args.output.mkdir(parents=True, exist_ok=True)
    os.umask(0o077)
    torch.set_num_threads(1)
    metadata = {'python': platform.python_version(), 'numpy': np.__version__, 'torch': torch.__version__,
                'numba': __import__('numba').__version__, 'protocol_fingerprint': protocol.protocol_fingerprint(),
                'sources': {name: digest(Path(model.__file__).parent / name) for name in
                            ('model.py', 'environment.py', 'evaluator.py', 'protocol.py', 'numba_environment.py')},
                'driver_sha256': digest(__file__)}
    metadata['rl_cpu_threads'] = args.threads
    write_json(args.output / f'{args.stage}-environment.json', metadata)
    if args.stage == 'finetune':
        finetune(args)
        return
    for name, function in [('expert', expert), ('demos', demonstrations), ('imitate', imitate), ('compare', compare)]:
        if args.stage in ('all', name):
            function(args)


if __name__ == '__main__':
    main()
