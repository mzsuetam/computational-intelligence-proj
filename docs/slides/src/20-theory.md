# Theory Overview

## Learning Aerial Image Segmentation From Online Maps [@kaiser2017learning]

### Using Addapted FCNs to Leverage Noisy Crowdsourced Data

<!-- **Challenge:** Deep Convolutional Neural Networks (CNNs) are the top-performing method for high-resolution semantic labeling, but they are **extremely data hungry**. Manual annotation is costly and scales poorly. -->

<!-- **Approach: Exploiting Crowdsourced/Legacy Maps**
The research investigates whether large-scale, low-accuracy training data—specifically **OpenStreetMap (OSM) maps**—can replace substantial manual labeling efforts. -->

**Technology: Adapted Fully Convolutional Network (FCN)**

*   The study utilized a variant of the FCN architecture, which performs pixel-to-pixel classification, returning a structured spatially explicit label image.
*   **Adaptation:** The FCN variant introduced a **third skip connection** (in addition to the original two) to preserve even finer image details and deliver sufficiently sharp edges in the segmentation results.
*   **Label Generation:** OSM coordinates for buildings (polygons) and roads (centerlines with estimated widths based on highway tags) were automatically transformed into pixel-wise label maps.
  
<!-- 
**Sources of Weakness in OSM Data:**
Training data derived from OSM is inherently noisy due to: coregistration errors, limitations (e.g., only road centerlines, not boundaries), temporal changes, and generally sloppy/low-accuracy annotations by volunteers. -->

---

### Key Insights

- **Improved Generalization:** Training on a large variety of data spanning multiple different cities (e.g., Chicago, Paris, Zurich, Berlin) improves the classifier’s ability to generalize to new, unseen locations (e.g., Tokyo).
- **Complete Substitution:** Semantic segmentation can be learned without any manual labeling by relying solely on large-scale noisy OSM labels, achieving acceptable results. The sheer volume of training data can largely compensate for lower accuracy.
- **Augmentation:** Even when a comfortable amount of accurate training data is available (the "gold standard"), pretraining with massive OSM data from other sites further improves results (e.g., boosting F1-score for the road class).

<!-- 4.  **Partial Substitution (The Trade-Off):** Pretraining with large-scale OSM labels allows for the replacement of the vast majority (85%) of costly, high-quality manual labels. Fine-tuning a model pretrained on OSM data with only a small dedicated training set compensates for a large portion of the potential performance loss. -->

<!-- **Conclusion:** Large-scale pretraining on inexpensive, weakly labeled geospatial data, followed by domain adaptation using a small set of project-specific, high-accuracy data, is highly effective and recommended as standard practice. -->

## A context and semantic enhanced UNet for semantic segmentation of high-resolution aerial imagery [@wang2020context]


### Core Challenges in High-Resolution Aerial Imagery:

1.  **Intra-class heterogeneity:** Objects of the same category (e.g., cars) have wide-ranging visual appearances (colors, characteristics), leading to difficulty in categorization. This stems mainly from insufficient contextual information.
2.  **Inter-class homogeneity:** Objects of different categories (e.g., buildings and impervious surfaces) have similar appearances, leading to semantic ambiguity. This stems from poor semantic information.

**Proposed solution**: CSE-UNet: Context and Semantic Enhanced UNet

---

### Key Insights – Addressing Heterogeneity via Context


**Technology 1: Multi-level RFB-based Skip Pathways (Context Enhancement)**

*   **Purpose:** To strengthen the representational capacity for **multi-scale contextual features** and mitigate **intra-class heterogeneity**.
*   **Mechanism:** Inspired by the concept of receptive fields in human visual systems, Receptive Field Blocks (RFB) are utilized in the skip pathways.
*   **Implementation:** These pathways exploit varying convolution kernels and dilated convolutions to control the sizes and eccentricities of receptive fields, effectively highlighting informative regions.

---

### Key Insights – Addressing Homogeneity via Semantic Enhancement and Performance

**Technology 2: Multi-kernel Dual-path Encoder (Semantic Enhancement)**

*   **Purpose:** To extract and fuse multi-level features with **rich semantic information** and tackle **inter-class homogeneity** by enlarging the inter-class differences.
*   **Mechanism:** The dual-path encoder contains an auxiliary multi-kernel based feature encoding path that provides additional semantics during the downsampling process.
*   **Feature Fusion:** Feature outputs from the original UNet encoding path and the auxiliary path are fused via element-wise addition at each level to generate rich semantic representations.
  
<!-- 
**Performance and Insights:**

*   CSE-UNet, integrating both the dual-path encoder and RFB-based skip pathways, consistently achieved the best accuracy (highest mean Intersection Over Union (mIoU) and mean F1 (mF1)) across the ISPRS Potsdam and Vaihingen datasets.
*   **Qualitative Improvement:** CSE-UNet produced **more accurate and coherent segmentation results** compared to standard UNet. For example, it generated less blurring edges for buildings and was competent in segmenting cars with large intra-class variances. -->


## Integrating semantic edges and segmentation information for building extraction from aerial images using UNet [@abdollahi2021integrating]

### MultiRes-UNet Architecture and Multi-Scale Feature Learning

- **Goal:** To achieve **accurate mapping of building objects** from aerial imagery, overcoming challenges posed by vegetation and shadows which exhibit similar spectral values to buildings.

- **Technology: MultiRes-UNet**
The MultiRes-UNet is an improved version of the original UNet network, designed to enhance feature assimilation and address inconsistencies between encoder/decoder features.

---

1.  **MultiRes Block:** This block replaces the traditional series of two convolutions in the original UNet structure.
    *   **Function:** Assimilates features learned from the data at various scales to comprise more spatial details.
    *   **Mechanism:** It mimics inception-like blocks by approximating larger convolutions (like 5x5 and 7x7) using a sequence of lightweight and smaller 3x3 convolutions to extract spatial features from various scales while attempting to manage memory requirements.
    <!-- *   *Image reference: Design of the proposed MultiRes block operation (Fig. 1(g)).* -->

2.  **Res Path:** New shortcut path replaces the common skip connections used in UNet.
    *   **Function:** Mitigates the **semantic gap** between the low-level features computed in the encoder and the notable higher-level features computed in the decoder.
    *   **Mechanism:** Uses a **chain of convolutional operations** and residual connections instead of straightforwardly merging feature maps. Extra non-linear operations are expected to decrease semantic gaps.
    <!-- *   *Image reference: Suggested Res path structure (Fig. 2).* -->

<!-- **Approach:** The network integrates **semantic edge information with semantic polygons** for highly accurate building detection. -->

---


### Key Insights

*   **Enhanced Boundaries:** Semantic edges are specifically used to enhance the boundary of semantic polygons.
*   **Irregular Polygon Correction:** Edges help solve the issue of irregular semantic polygons and make them more appropriate for actual building forms.
*   **Distinction:** Edges realize the distinction between adjacent buildings.
*   **Performance Gain:** Integrating semantic edges enhanced the average quantitative results for Intersection Over Union (IOU) by **0.78%** (from 93.35% to 94.13%).
*   **Overall Competency:** MultiRes-UNet achieved 93.14% IOU accuracy (with data augmentation), proving its success in building object extraction compared to state-of-the-art models like UNet (92.40%), DeeplabV3 (89.48%), and ResNet (88.84%).

## Architecture Selection

After the overview, the **CSE-Unet** [@wang2020context] architecture was selected for further exploration and analysis in this project due to its promising results in addressing key challenges in aerial image segmentation.