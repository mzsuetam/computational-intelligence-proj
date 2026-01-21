|                             | loss          |   valid_loss |   accuracy |   jaccard_multi |
|:----------------------------|:--------------|-------------:|-----------:|----------------:|
| cse_phase_4_1               | Cross Entropy |        0.434 |      0.864 |           0.681 |
| baseline_34_pretrained      | Cross Entropy |        0.485 |    nan     |           0.678 |
| cse_phase_2                 | Cross Entropy |        0.508 |      0.868 |           0.671 |
| cse_phase_4_1_combined_loss | CE + Dice     |        0.53  |      0.86  |           0.669 |
| cse_phase_3                 | Cross Entropy |        0.529 |      0.855 |           0.652 |
| cse_phase_4_2_combined_loss | CE + Dice     |        0.454 |      0.834 |           0.646 |
| cse                         | Cross Entropy |        0.548 |      0.846 |           0.634 |
| baseline_34                 | Cross Entropy |        0.481 |      0.825 |           0.633 |
| cse_phase_4_3_focal_loss    | Focal         |        0.295 |      0.818 |           0.613 |
| cse_combined_loss           | CE + Dice     |        0.677 |      0.774 |           0.535 |