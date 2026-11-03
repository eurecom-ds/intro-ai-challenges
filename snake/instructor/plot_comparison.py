#!/usr/bin/env python3
"""Plot starting and final Snake scores from a completed comparison."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    report = json.loads((args.directory / 'comparison.json').read_text())
    seeds = report['seeds']
    random_initial = [row['random_initial']['mean_apples'] for row in seeds]
    random_final = [row['arms']['random']['score']['mean_apples'] for row in seeds]
    il_final = [row['arms']['imitation']['score']['mean_apples'] for row in seeds]
    il_initial = report['imitation_initial']['mean_apples']
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), constrained_layout=True, sharey=True)
    values = [(random_initial, [il_initial]), (random_final, il_final)]
    colors = ['#64748b', '#2563eb']
    for axis, title, groups in zip(axes, ['Before RL', f"After {report['rl_episodes']:,} RL episodes"], values):
        axis.bar([0, 1], [np.mean(x) for x in groups], color=colors, width=0.6, alpha=0.8)
        for i, group in enumerate(groups):
            offsets = np.linspace(-0.10, 0.10, len(group)) if len(group) > 1 else [0]
            axis.scatter(i + np.asarray(offsets), group, color='#0f172a', s=28, zorder=3)
            axis.text(i, max(group) + 1.3, f'{np.mean(group):.2f}', ha='center', fontsize=12,
                      bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': 1})
        axis.set_xticks([0, 1], ['Random start', 'Imitation start'])
        axis.set_title(title)
        axis.spines[['top', 'right']].set_visible(False)
        axis.set_axisbelow(True)
        axis.grid(axis='y', alpha=.2)
    axes[0].set_ylabel('Apples per game')
    axes[1].axhline(il_initial, linestyle='--', color='#2563eb', linewidth=1.3, label='Imitation before RL')
    axes[1].legend(loc='upper left')
    axes[0].set_ylim(0, max(il_initial, *random_final, *il_final) * 1.2 + 2)
    fig.suptitle(f"Snake imitation initialization · {report['evaluation_games']:,} evaluation games\n"
                 f"Dots: {len(seeds)} RL training seeds; one shared imitation policy", fontsize=13)
    fig.savefig(args.directory / 'comparison.png', dpi=180)
    fig.savefig(args.directory / 'comparison.svg')


if __name__ == '__main__':
    main()
