|                             | loss          |   best_valid_loss |   best_accuracy |   best_dice_multi |   best_jaccard_multi |
|:----------------------------|:--------------|------------------:|----------------:|------------------:|---------------------:|
| cse_phase_4_1               | Cross Entropy |             0.417 |           0.864 |             0.793 |                0.681 |
| baseline_34_pretrained      | Cross Entropy |             0.473 |         nan     |             0.796 |                0.678 |
| cse_phase_2                 | Cross Entropy |             0.46  |           0.868 |             0.782 |                0.671 |
| cse_phase_4_1_combined_loss | CE + Dice     |             0.521 |           0.86  |             0.782 |                0.669 |
| cse_phase_3                 | Cross Entropy |             0.524 |           0.855 |             0.763 |                0.652 |
| cse_phase_4_2_combined_loss | CE + Dice     |             0.454 |           0.838 |             0.766 |                0.646 |
| cse                         | Cross Entropy |             0.548 |           0.846 |             0.752 |                0.634 |
| baseline_34                 | Cross Entropy |             0.481 |           0.825 |             0.755 |                0.633 |
| cse_phase_4_3_focal_loss    | Focal         |             0.256 |           0.818 |             0.743 |                0.613 |
| cse_combined_loss           | CE + Dice     |             0.616 |           0.774 |             0.667 |                0.535 |