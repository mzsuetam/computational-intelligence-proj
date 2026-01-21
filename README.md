![](docs/pred-target-best.png)

# Analysis of Deep Learning Architectures for Aerial Image Segmentation – Case Study of CSE-UNet vs. Classic UNet architectures

**Course:** Computational Intelligence

**Project slides:** [Analysis of Deep Learning Architectures for Aerial Image Segmentation](docs/slides/out/slides-latest.pdf)

## Project Overview

Deep learning architectures for **semantic segmentation of high-resolution aerial imagery** were investigated in this project. This task is critical for urban planning, environmental monitoring, and automated mapping.

The primary objective was to compare standard deep learning baselines (U-Net with ResNet backbones) against specialized architectures designed for remote sensing, specifically the **Context and Semantic Enhanced U-Net (CSE-Unet)**. These models were evaluated on stability, efficiency, and segmentation accuracy across complex urban classes.

## Motivation

Accurate segmentation is vital for up-to-date geospatial analysis, but manual annotation is costly and slow. While deep learning offers scalability, standard models often struggle with specific aerial challenges:

* **Intra-class heterogeneity:** Objects of the same class (e.g., roofs) may appear vastly different.
* **Inter-class homogeneity:** Different classes (e.g., concrete vs. buildings) may appear similar.
* **Small objects:** Detecting cars or narrow vegetation remains challenging.

## Dataset

The project initially utilized the "Humans in the Loop" (Dubai) dataset but was later migrated to a higher-quality benchmark due to annotation inconsistencies.

* **Final Dataset:** **ISPRS Potsdam 2D Semantic Labeling Benchmark**.
* **Content:** High-resolution aerial tiles were cleaned, split, and resized.
* **Classes:** Impervious surfaces, Buildings, Low vegetation, Trees, Cars, Clutter.

Sample image and class distribution:

![](docs/slides/src/img/04-postdam.png)

![](docs/slides/src/img/class-balance.png)

## Architectures

### 1. Baseline: U-Net (ResNet34)

A standard U-Net architecture utilizing a **ResNet34 backbone**. Both a version trained from scratch and a version **pre-trained on ImageNet** were tested to serve as strong benchmarks.

* **Parameters:** ~41M.

### 2. CSE-Unet (Context and Semantic Enhanced U-Net)

A specialized architecture designed to address semantic gaps in aerial imagery:

* **Multi-kernel Dual-path Encoder:** Extracts rich semantic information to distinguish similar-looking classes.
* **RFB-based Skip Pathways:** Uses Receptive Field Blocks to capture multi-scale context.
* **Parameters:** ~36M (approx. 12% fewer than baseline).

## Experiments & Methodology

Experiments were conducted in multiple phases to isolate the effects of architecture, data volume, and regularization techniques.

### Phase 1: Preliminary Comparison (Subset)

ResNet50 Baseline and CSE-Unet were tested on a small data subset (500 images).

* **Result:** CSE-Unet achieved competitive validation scores with significantly faster training and fewer parameters (36M vs 339M for ResNet50).

### Phase 2: Full Dataset & Optimization

The full ISPRS Potsdam dataset (~3300 tiles) was utilized.

* **Baseline (From Scratch):** Showed high volatility and "catastrophic collapse" around Epoch 50.
* **CSE-Unet:** Exhibited smooth, linear convergence. **Dropout (0.2)** and **Weight Decay** were introduced to address early overfitting.

### Phase 3: Advanced Loss Functions

Efforts were made to maximize performance using a combined **CrossEntropy + Dice Loss**.

* **Outcome:** No significant improvements over CrossEntropy alone were observed. Auxiliary losses introduced gradient instability rather than refinement for this specific dataset.

### Phase 4: Fine-Tuning & Warm Restarts (Final)

The CSE-Unet was recognized as a "slow burner," and a **warm restart with 10x smaller learning rates** was applied to the best Phase 2 model.

* **Outcome:** Validation loss was significantly reduced, and fine-grained detection of small classes like Cars and Clutter was improved.

## Key Results

The table below summarizes the best performance metrics on the validation set.

**Notably, the custom CSE-Unet (Phase 4) successfully outperformed the heavy Pretrained Baseline.**

| Model | Status | Accuracy | Jaccard (IoU) |
| --- | --- | --- | --- |
| **CSE-Unet (Phase 4)** | **Best & Stable** | **0.864** | **0.681** |
| **Baseline (ResNet34 Pretrained)** | Strong Benchmark | N/A | 0.678 |
| **CSE-Unet (Phase 2)** | Stable / Under-converged | 0.868 | 0.671 |
| **CSE-Unet (Combined Loss)** | Unstable | 0.860 | 0.669 |
| **Baseline (ResNet34 From Scratch)** | Volatile / Collapsed | 0.825 | 0.633 |

The following visualizations provide further insight into the best model's performance:

* **Confusion Matrix Analysis:** The matrix confirms the model's robustness, highlighting over **90% accuracy** for dominant classes like Impervious Surfaces and Buildings. Remarkably, despite severe class imbalance, the model achieves **81% accuracy for the minority "Car" class**, validating the fine-grained feature recovery from the Phase 4 warm restart. The primary remaining errors stem from semantic ambiguity, specifically where "Clutter" is misclassified as the background "Impervious Surfaces" it sits upon.
* **Prediction vs. Target Comparison:** Visual inspection demonstrates the CSE-Unet's ability to generate sharp, coherent segmentation maps that closely match ground truth. The model excels at isolating small, detached objects, successfully defining individual cars and narrow vegetation strips that are often lost by standard architectures. This qualitative success confirms that the improved metrics translate into practical, high-fidelity mapping capabilities for complex urban scenes.

![](experiments/new-ds/cse_phase_4_1/confusion_matrix.png)

![](experiments/new-ds/cse_phase_4_1/pred-targets.png)




## Conclusions

1. **Surpassing the Baseline:** Through iterative optimization, the custom **CSE-Unet** (36M params) outperformed the standard **ResNet34 Pretrained** baseline (41M params), demonstrating that specialized architectural features (RFB modules) can rival heavy transfer learning with the right training strategy.
2. **The "Slow Burner" Effect:** The CSE-Unet learns complex semantic relationships gradually. The success of **Phase 4** (warm restart) confirmed that the model required a multi-stage learning rate schedule to resolve fine-grained features (Cars/Clutter) and escape local minima.
3. **Stability > Raw Power:** While the ResNet baseline trained from scratch was prone to catastrophic collapse, the CSE-Unet demonstrated robust, linear stability throughout all phases.
4. **Efficiency:** The CSE architecture achieved these superior results with **~12% fewer parameters** than the baseline.

---

### References

* *Wang, L. et al. (2020).* "A context and semantic enhanced UNet for semantic segmentation of high-resolution aerial imagery"
* *Kaiser, P. et al. (2017).* "Learning Aerial Image Segmentation From Online Maps."
* *Abdollahi, A. et al. (2021).* "Integrating semantic edges and segmentation information for building extraction from aerial images using UNet"