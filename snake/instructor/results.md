# Snake imitation learning results

The limited imitation policy starts at 14.773 apples per game and reaches
39.723 on average after 16,384 RL episodes across three training seeds. Random initialization reaches
11.840 after the same episode budget. The classroom reference budget is
16,384 episodes, with the shared imitation policy loaded at the start of every
experiment.

## Teacher and demonstrations

The teacher trained for 262,144 episodes and 347,894,041 moves. It used
learning rate 0.003, discount 0.999, food reward 10,
collision reward −10, movement reward −0.01, distance reward 1, potential
strength 1, trap-entry penalty 0.25, batch size 32 and one critic update. Its
final score was 58.746 apples per game on 1,000 separate games.

Generation produced 10,000 games. Selection retained exactly the best 1,000,
ordered by food collected and then collection time. These games averaged
68.190 apples. Each demonstration contains a complete sequence of chosen
actions; game seeds allow its observations to be reconstructed.

## Imitation variants

Full imitation trained for three epochs on every transition from 900 games,
with 100 complete games held out for action validation. Its independent score
was 55.020 apples per game. Continued RL reduced its scores to 48.472, 49.805
and 48.597 (mean 48.958).
The benchmark used 1,000 games beginning at environment seed 4,000,000.

A shorter fit supports the classroom starting-policy objective. A 128-position
cap and one epoch scored 0.367 apples per game on the 30 monitoring games.
The retained recipe uses one epoch and 512 positions sampled uniformly without
replacement from each of the 900 training games. It scored 13.400 on those
monitoring games. The independent comparison uses a fresh set of 1,000 games
beginning at environment seed 5,000,000. The 100 held-out demonstration games
remain separate from training.

The retained fit used 460,800 training examples, Adam at 0.001,
batches of 1,024, seed 123, and the fixed 685–64–64–3 actor. Validation action
accuracy was 93.37%. The policy contains six float32 parameter arrays.

## Independent comparison

Scores are apples per game on the same 1,000 environment/action seed pairs.
Each RL run used 16,384 complete games, classroom rewards without the extra
teacher trap penalty, and fresh critics and optimizers. The teacher and
imitation policy are shared across the three RL training seeds.

| Training seed | Random before RL | Imitation before RL | Random + RL | Imitation + RL |
| --- | ---: | ---: | ---: | ---: |
| 42 | 0.093 | 14.773 | 11.296 | 35.813 |
| 43 | 0.093 | 14.773 | 12.414 | 42.767 |
| 44 | 0.094 | 14.773 | 11.810 | 40.589 |

For every seed, all three paired score differences have positive lower bounds
in their approximate 95% game intervals: imitation versus random before RL,
RL improvement from imitation, and imitation plus RL versus random plus RL.
The intervals describe evaluation-game variation. Three training seeds
separately check sensitivity to training randomness.

The retained policy starts at 16.2 apples per game on the notebook's 30 public
games; the 256-episode classroom baseline scores 9.733. Short RL runs can reduce
food collection temporarily. The longer 16,384-episode comparison above
establishes the improvement for the selected recipe and tested seeds.

## Training budgets from imitation, seed 42

Both runs begin independently from the imitation policy, which
scores 14.773 apples per game before RL. They use seed 42, four PyTorch CPU
threads, fresh critics and optimizers, the classroom reward configuration,
and the same 1,000 evaluation games beginning at environment seed 5,000,000.
The Numba simulator remains single-threaded. The training budgets are 256
and 262,144 episodes.

| RL episodes | Apples/game | Change from imitation | Paired 95% interval for change | Training moves | Training seconds |
| --- | ---: | ---: | --- | ---: | ---: |
| 256 | 9.628 | -5.145 | [-6.023, -4.267] | 211,981 | 2.23 |
| 262,144 | 59.584 | +44.811 | [+43.866, +45.756] | 368,233,955 | 1421.09 |

At 256 episodes, RL reduces food collection by 5.145 apples per game. At
262,144 episodes, it improves food collection by 44.811 apples per game and
reaches 59.584. These results describe one training seed. The intervals quantify
variation across paired evaluation games.

## Constant and linear food rewards

Both policies train for 16,384 episodes from the same starting policy, using
seed 42 and four PyTorch CPU threads. Evaluation uses the same 1,000 games.
Food weight is 10, collision reward −10, movement reward −0.01, distance reward
1, learning rate 0.003 and discount 0.999. The food reward mode is the only
configuration difference. Constant mode awards 10 per apple; linear mode
awards 10 times the apple number within the game. The starting policy scores
14.773 apples per game.

| Food reward mode | Apples/game | Moves/game | Collisions | Reached 1,500 moves |
| --- | ---: | ---: | ---: | ---: |
| Constant | 40.723 | 1450.419 | 134/1,000 (13.4%) | 866/1,000 (86.6%) |
| Linear | 27.796 | 1314.703 | 321/1,000 (32.1%) | 679/1,000 (67.9%) |

Linear mode collects 12.927 fewer apples per game. The approximate paired
95% interval for linear minus constant is [-13.617, -12.237] apples. Linear
mode also produces shorter games and more collisions in this seed. Both modes
improve food collection over the starting policy.

These results describe one training seed. The interval quantifies variation
across evaluation games. For n apples, the undiscounted food component totals
10n in constant mode and 5n(n+1) in linear mode, so this comparison changes both
the progression and the total magnitude of food rewards.
