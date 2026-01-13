https://www.kaggle.com/datasets/humansintheloop/semantic-segmentation-of-aerial-imagery
https://www.kaggle.com/datasets/bulentsiyah/semantic-drone-dataset?select=class_dict_seg.csv

https://forums.fast.ai/t/semtorch-a-semantic-segmentation-library-build-above-fastai/79194

Transfer Learning
SegFormer
Fine-tune a pretrained transformer-based model
https://huggingface.co/blog/fine-tune-segformer


Supervised
U-Net, DeepLabv3+
Train both from scratch on labeled aerial dataset

---

251105 notes:

- Baseline proposition: https://www.kaggle.com/code/abhinavsp0730/semantic-segmentation-by-implementing-fcn
- implement CSE-Unet <- the main goal, most interesting solution
- unet implementation: https://towardsdatascience.com/cook-your-first-u-net-in-pytorch-b3297a844cf3/
- what next?
	- maybe take classic cnn, and unet 
	- maybe implement the multiresnet 
	- MAYBE connect cse with multi
	- maybe just take some other solutions (easy way – SemTorch: https://forums.fast.ai/t/semtorch-a-semantic-segmentation-library-build-above-fastai/79194) and compare them to the CSE-Unet appraoch
	

--- 

17.12.2025 notes

**changes for cse-updated**:

- dropout in double conv
- custom loss
- SaveModelCallback
- Continue Training with Lower LR: After the initial 100 epochs, the model is likely "hot." Run a second phase of training with a much lower learning rate to let it settle into the minimum.
- history plotter update


**to be discussed in the report**:

- show the differences on the subset for the first performance comparison (resnet50 net vs cse-unet)
- then show the final results on the full dataset for the baseline (resnet34 unet) vs cse-unet (simmilar param count)
- dropout made the learning process visibly reduce overfitting for cse-unet (sce-unet w/ and w/o dropout comparison)
- technicalities: new tools used, new methods, full experimental process (shame about the limited time)  

---

? micro batching 

---

for report:

- old-ds subset 500
- old-ds
- new-ds full
    - new (full) dataset, cleaning
    - target/pred on same pictures
    - conf matrices
    - learning curves
- training process
    - data preparation
        - splitting
        - cleaning
        - augmentations
    - model architectures
        - cse unet vs unet
    - training tricks
        - overfitting reduction
            - dropout
            - weight decay
            - data augmentation (ablumenations)
        - early stopping
        - fine-tuning with low lr
        - gradient accumulation
        - learning rate finder
- results analysis
    - summary old-ds
        - unet
        - pretrained unet
        - cse unet
        - cse unet + dropout
    - summary new-ds
        - unet
        - cse unet + dropout
- conclusions
    - cse unet works better than unet
    - dropout helps
    - pretrained weights help
    - data cleaning helps
    - more data helps

    A. Stability is the Hidden Feature of CSE-UNet While your slides highlight that CSE-UNet has fewer parameters (36M vs 41M for ResNet34), the new full-dataset training reveals a more critical advantage: Training Stability.

        Evidence: In baseline.png, the ResNet34 model exhibits high volatility (huge loss spikes at epoch ~13 and a total crash at epoch ~50).

        Contrast: In cse.png, the training is smooth. The validation Jaccard score climbs steadily without the violent fluctuations seen in the baseline.

        Conclusion: The Multi-level Receptive Field Blocks (RFB) and dual-path encoder in CSE-UNet likely provide better feature regularization, preventing the model from overfitting or diverging as easily as the standard ResNet backbone.

    B. The "Convergence Illusion"

        Observation: You mentioned "CSE will be trained more as it does not converge."

        Analysis: Looking at cse.png, the model has not stopped learning; the slope is still positive. The Baseline, however, appeared to converge faster (flattens around epoch 30) before it crashed.

        Conclusion: CSE-UNet is a "slow burner." It learns more complex semantic relationships (context) which takes longer to optimize but results in a more robust model. The Baseline learned simple features quickly but failed to sustain them.

    C. Intra-class Heterogeneity Handling

        Context: Your slides mention "Intra-class heterogeneity" (objects of the same class looking different) as a key challenge.

    Evidence: If your CSE-UNet scores are higher (or more stable) on the "Buildings" or "Impervious Surfaces" classes specifically, this proves the architecture is successfully using context to unify these varied appearances, validating the theoretical claims from the literature.

