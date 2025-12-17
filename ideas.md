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

to be discussed in the report:

- show the differences on the subset for the first performance comparison (resnet50 net vs cse-unet)
- then show the final results on the full dataset for the baseline (resnet34 unet) vs cse-unet (simmilar param count)
- dropout made the learning process visibly reduce overfitting for cse-unet (sce-unet w/ and w/o dropout comparison)
- technicalities: new tools used, new methods, full experimental process (shame about the limited time)  
