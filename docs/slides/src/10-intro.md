# Introduction

## Motivation

Accurate aerial image segmentation is vital for up-to-date mapping and urban analysis, yet manual methods are slow and costly. Deep learning offers scalable solutions, but architectures differ in handling challenges like shadows, small objects, and class variability. By comparing key models, the project aims to identify the most effective approach for reliable segmentation, supporting geospatial research where high-quality segmented imagery is essential for deeper analysis and improved results.

## Project overview

The project focuses on **aerial image segmentation**, a key task in remote sensing with applications in mapping, urban planning, and environmental monitoring. The objective is to **compare the performance of different deep learning architectures**, ranging from classic convolutional neural networks (CNNs) to advanced models such as **U-Net** and U-Net–derived architectures (e.g., MultiRes-UNet, and CSE-UNet). Using publicly available datasets like **iSAID** or the **Dubai Aerial Imagery dataset**, model performance is evaluated to identify the most reliable and efficient architecture.

## Literature overview

- **[Learning Aerial Image Segmentation From Online Maps](https://ieeexplore-1ieee-1org-1000047wi0086.wbg2.bg.agh.edu.pl/stamp/stamp.jsp?tp=&arnumber=7987710)** [@kaiser2017learning]  
\footnotesize
Shows that **CNNs** can learn to segment aerial images using noisy labels from online maps like OpenStreetMap. The study demonstrates that large, imperfect datasets can reduce manual annotation needs while maintaining strong performance.
\normalsize

- **[A Context and Semantic Enhanced UNet for Semantic Segmentation of High-Resolution Aerial Imagery](https://iopscience.iop.org/article/10.1088/1742-6596/1607/1/012083/pdf)** [@wang2020context]  
\footnotesize
Introduces **CSE-UNet**, which enhances segmentation in high-resolution aerial imagery using multi-level receptive field blocks and a dual-path encoder. The model effectively addresses intra-class heterogeneity and inter-class homogeneity, outperforming UNet and other baselines.
\normalsize

- **[Integrating Semantic Edges and Segmentation Information for Building Extraction from Aerial Images Using UNet](https://www.sciencedirect.com/science/article/pii/S2666827021000979)** [@abdollahi2021integrating]  
\footnotesize
Proposes **MultiRes-UNet**, an improved model for building extraction that integrates multi-scale feature learning and semantic edge information. Results show superior boundary accuracy compared to UNet, DeeplabV3, and ResNet.
\normalsize

## Related datasets

To evaluate and compare different segmentation architectures, two benchmark datasets are considered:


1. [Semantic segmentation dataset – The Humans in the Loop](https://humansintheloop.org/resources/datasets/semantic-segmentation-dataset-2/) ([Kaggle](https://www.kaggle.com/datasets/humansintheloop/semantic-segmentation-of-aerial-imagery)) \
\footnotesize
The Humans in the Loop dataset contains aerial images of Dubai annotated with pixel-wise semantic segmentation across six classes. It is publicly available under a CC0 1.0 license, making it free for use in research and analysis.
\normalsize


![Sample mask for the dataset](https://storage.googleapis.com/kaggle-datasets-images/681625/1196732/a1d8dfe666ca94f660058a2f56a5639d/dataset-cover.png?t=2020-05-29-09-01-54)

---

1. [iSAID: A Large-scale Dataset for Instance Segmentation in Aerial Images](https://openaccess.thecvf.com/content_CVPRW_2019/papers/DOAI/Zamir_iSAID_A_Large-scale_Dataset_for_Instance_Segmentation_in_Aerial_Images_CVPRW_2019_paper.pdf) [@waqas2019isaid]
\footnotesize
This paper introduces iSAID, the first large-scale benchmark dataset for instance segmentation in aerial imagery, featuring 655,451 objects across 15 categories. It combines object detection and pixel-level segmentation, addressing challenges like dense scenes and tiny objects. Results show that standard models like Mask R-CNN and PANet perform suboptimally, highlighting the need for specialized methods for aerial images.
\normalsize

---

![Sample images from the iSAID dataset](img/image.png){height=90%}
