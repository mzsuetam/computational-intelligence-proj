# %%
import torch
from fastai.basics import *
from fastai.vision.all import *
# from fastai.callback.tensorboard import TensorBoardCallback
from fastai.callback.tracker import SaveModelCallback

from pathlib import Path
import pandas as pd
import json

# %%
# !pwd
# !hostname

# %%
ROOT_DIR = Path('.').resolve().parent
if ROOT_DIR.name != 'proj': # jupyter 
    ROOT_DIR = ROOT_DIR / 'proj'

DATASETS_DIR = ROOT_DIR / "datasets"
POTSDAM_DIR = DATASETS_DIR / "Potsdam-tiles-512"
assert POTSDAM_DIR.exists(), f"Potsdam dataset not found in {POTSDAM_DIR}"

EXPERIMENTS_DIR = ROOT_DIR / 'experiments'
LOGS_DIR = EXPERIMENTS_DIR / 'logs'
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# %%
postdam_static = DATASETS_DIR/'Potsdam-static'
df_classes = pd.read_csv(postdam_static/'classes.csv')
df_classes

# %%
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap

def hex2rgb(hex_color):
    if isinstance(hex_color, tuple) or isinstance(hex_color, list):
        return tuple(hex_color)
    if hex_color.startswith('#'):
        hex_color = hex_color.lstrip('#')
        t = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    else:
        t = tuple(int(x.strip()) for x in hex_color.strip('()').split(','))
    return t

ssd_cmap = ListedColormap(
    df_classes['color'].apply(lambda x: np.array(hex2rgb(x))/255.0).to_list(),
    name='skyscapes',
    N = len(df_classes)
) 

potsdam_plot_kwargs = {
    'vmin': 0,
    'vmax': len(df_classes) - 1,
    'cmap': ssd_cmap
}

potsdam_legend_kwargs = {
    'handles': [
        Patch(color=potsdam_plot_kwargs['cmap'](i), label=name) 
        for i, name in enumerate(df_classes.name.values)
    ],
    'bbox_to_anchor': (1.05, 1),
    'loc': 'upper left',
    'borderaxespad': 0.,
}

potsdam_plot_kwargs['cmap']

# %%
image_paths = L(sorted([f for f in get_image_files(POTSDAM_DIR) if f.parent != 'labels']))
image_paths

# %%
import albumentations as A

class AlbumentationsTransform(Transform):
    split_idx = 0
    
    def __init__(self, aug): self.aug = aug
    def encodes(self, img: PILImage):
        aug_img = self.aug(image=np.array(img))['image']
        return PILImage.create(aug_img)

# heavy lighting augmentation
a_tfms = A.Compose([
    A.RandomGamma(p=0.5),
    A.CLAHE(p=0.5), # Great for seeing details in shadows
    A.RandomBrightnessContrast(p=0.5),
    A.HueSaturationValue(p=0.3),
    # A.GridDistortion(p=0.3), # Helps with road undulations # cannot be used as the segmentation masks get distorted
])

# %%
# IMG_SIZE = 300
IMG_SIZE = 512

DB_MEAN = [0.3362390100955963, 0.35974231362342834, 0.3330190181732178]
DB_STD = [0.11687745898962021, 0.11572890728712082, 0.12002932280302048]
    
def get_y(img_path):
    img_path = Path(img_path)
    label_dir = img_path.parent.parent / "labels"
    label_name = img_path.name.replace("_RGB", "_label")
    return label_dir / label_name

db = DataBlock(
    blocks=(
        ImageBlock, 
        MaskBlock(codes=df_classes.name)
    ),
    get_y = get_y,
    splitter=GrandparentSplitter(valid_name='val'),
    item_tfms = [
        Resize((IMG_SIZE), method='nearest'),
        AlbumentationsTransform(a_tfms)
    ],
    batch_tfms=[
        Dihedral(), 
        *aug_transforms(
            max_rotate=10.0,    # Still useful for small "tilts" (non-90 degree)
            min_zoom=0.9,
            max_zoom=1.1,       # Keep conservative to preserve context
            
            max_lighting=0.4,    
            p_lighting=0.8,
            
            # Warp helps simulate slightly off-nadir camera angles
            max_warp=0.2,       
            p_affine=0.75,
        ),
        Normalize.from_stats(
            mean=tensor(DB_MEAN),
            std=tensor(DB_STD)
        )
    ],
)

# %%
desired_bs = 16
dls_bs = 4

dls = db.dataloaders(
    source=image_paths, 
    bs=dls_bs,
    num_workers=8 # Use 4 or 8 to keep the GPU fed
)

gradient_accum = GradientAccumulation(n_acc=desired_bs//dls_bs)
print(f"Using gradient accumulation with {desired_bs//dls_bs} steps")

# %%
# dls.show_batch(max_n=4, **potsdam_plot_kwargs)

# %%
loss_func = CrossEntropyLossFlat(axis=1)

# %%
pixel_accuracy = partial(accuracy, axis=1)

metrics = [
    pixel_accuracy,
    
    DiceMulti(), # F1 Score

    JaccardCoeffMulti(), # IoU
]

# # %%
# # resnet50       (best cost/perf)
# # efficientnet_b3
# # swin_t         (amazing on aerial)
# # convnext_tiny  (excellent)

# # arch = resnet50
# model = resnet34

# # %%
# learn = unet_learner(
#     dls, 
#     model,
#     loss_func=loss_func,
#     metrics=metrics,
#     pretrained=False,
# ).to_fp16()

# learn.summary()

# # %% [markdown]
# # resnet50: Total params: 339,071,460
# # 
# # resnet34: Total params: 41,221,668

# # %%
# suggested_lr = None
# suggested_lr = learn.lr_find()

# # %%
# lr = suggested_lr.valley if suggested_lr is not None else 2.5e-3
# print(f"Using learning rate: {lr} {'(hardcoded)' if suggested_lr is None else '(suggested)'}")

# %%
from fastai.callback.tracker import SaveModelCallback
from fastai.callback.tracker import EarlyStoppingCallback

SaveCB = lambda fname: SaveModelCallback(
    monitor='jaccard_coeff_multi', 
    fname=fname, 
    comp=np.greater,
    with_opt=False
)

LogCB = lambda fname: CSVLogger(
    fname=LOGS_DIR/fname
)

ValidLossEarlyStoppingCB = lambda patience, min_delta=0.001: EarlyStoppingCallback(
    monitor='valid_loss',
    min_delta=min_delta,
    patience=patience
)

# %%
# learn.fit_one_cycle(
#     100, 
#     lr,
#     cbs=[
#         SaveCB('best_segmentation'),
#         LogCB('segmentation_log.csv'),
#         ValidLossEarlyStoppingCB(10),
#         gradient_accum,
#     ]
# )

# # %%
# learn.show_results(max_n=4, **potsdam_plot_kwargs)

# # %%
# learn.load('../notebooks/models/baseline_34_pretrained')

# # %%
# learn.export('experiments/new-ds/baseline_34_pretrained/baseline_34_pretrained.pkl')

# %% [markdown]
# Reference approaches:
# - [InceptionResNetV2-UNet (81% Dice Coeff. & 86% Acc)](https://www.kaggle.com/code/ayushdabra/inceptionresnetv2-unet-81-dice-coeff-86-acc)
# - [Aerial Image for Semantic Segmentation (~85% Dice Coeff.)](https://www.kaggle.com/code/aletbm/aerial-image-for-semantic-segmentation#Evaluation-metrics)
# - [Unet segmentation implementation with ASPP](https://www.kaggle.com/discussions/general/205141)

# %% [markdown]
# ? Edge-Refinement Network (ERN) are extremely effective for your aerial imagery task.

# %% [markdown]
# # CSE-UNet

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

# %%
class ConvBlock(nn.Module):
    """
    Standard Conv -> BN -> ReLU block used throughout the network.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class DoubleConv(nn.Module):
    """
    The standard U-Net encoder block: Two Conv3x3 blocks.
    """
    def __init__(self, in_channels, out_channels, dropout_rate=0.2):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            ConvBlock(in_channels, out_channels),
            ConvBlock(out_channels, out_channels),
            nn.Dropout2d(dropout_rate)
        )

    def forward(self, x):
        return self.double_conv(x)

class RFB_Skip(nn.Module):
    """
    Multi-level RFB-based skip pathways (Section 2.2).
    It consists of two parallel paths:
    1. A stack of standard 3x3 convolutions (approximating a larger kernel).
    2. A single dilated convolution.
    The outputs are concatenated.
    """
    def __init__(self, in_channels, num_stack, dilation_rate):
        super(RFB_Skip, self).__init__()
        
        # Path 1: Stack of standard convolutions (Green blocks in diagram)
        # To emulate large kernels like 7x7, we stack 3x3 convs.
        stack_layers = []
        for _ in range(num_stack):
            # 1x1 convs are used in the bottom level as per text, 3x3 elsewhere
            k = 3 if num_stack > 1 else 1 
            p = 1 if num_stack > 1 else 0
            stack_layers.append(ConvBlock(in_channels, in_channels, kernel_size=k, padding=p))
        self.stack_path = nn.Sequential(*stack_layers)

        # Path 2: Dilated Convolution (Yellow blocks in diagram)
        # Note: Section 2.2 mentions matching receptive fields.
        # For the bottom level (dilation 1), it acts as a standard conv.
        self.dilated_path = ConvBlock(in_channels, in_channels, 
                                      kernel_size=3, 
                                      padding=dilation_rate, # Padding must equal dilation to keep size
                                      dilation=dilation_rate)

    def forward(self, x):
        out_stack = self.stack_path(x)
        out_dilated = self.dilated_path(x)
        # Concatenate features from both paths
        return torch.cat([out_stack, out_dilated], dim=1)

class CSE_UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1):
        super(CSE_UNet, self).__init__()
        
        filters = [64, 128, 256, 512, 1024]

        # --- ENCODER (Dual-Path) ---
        
        # Level 1
        self.enc1_main = DoubleConv(in_channels, filters[0])
        self.pool1 = nn.MaxPool2d(2)
        # Aux Path 1: Conv7x7 stride 2 (Section 2.3)
        self.enc1_aux = nn.Sequential(
            nn.Conv2d(in_channels, filters[0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(filters[0])
        )

        # Level 2
        self.enc2_main = DoubleConv(filters[0], filters[1])
        self.pool2 = nn.MaxPool2d(2)
        # Aux Path 2: Conv5x5 stride 2
        self.enc2_aux = nn.Sequential(
            nn.Conv2d(filters[0], filters[1], kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(filters[1])
        )

        # Level 3
        self.enc3_main = DoubleConv(filters[1], filters[2])
        self.pool3 = nn.MaxPool2d(2)
        # Aux Path 3: Conv3x3 stride 2
        self.enc3_aux = nn.Sequential(
            nn.Conv2d(filters[1], filters[2], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(filters[2])
        )

        # Level 4
        self.enc4_main = DoubleConv(filters[2], filters[3])
        self.pool4 = nn.MaxPool2d(2)
        # Aux Path 4: Conv2x2 stride 2
        # Note: 2x2 kernel with stride 2 and padding 0 halves the dimension perfectly
        self.enc4_aux = nn.Sequential(
            nn.Conv2d(filters[2], filters[3], kernel_size=2, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(filters[3])
        )

        # Bridge
        self.bridge = DoubleConv(filters[3], filters[4])

        # --- RFB SKIP PATHWAYS ---
        
        # According to Section 2.2 and Diagram
        # Top (Level 1): Stack of 3, Dilation 7
        self.rfb1 = RFB_Skip(filters[0], num_stack=3, dilation_rate=7)
        # Level 2: Stack of 2, Dilation 5
        self.rfb2 = RFB_Skip(filters[1], num_stack=2, dilation_rate=5)
        # Level 3: Stack of 1 (3x3), Dilation 3
        self.rfb3 = RFB_Skip(filters[2], num_stack=1, dilation_rate=3)
        # Level 4: Stack of 1 (1x1), Dilation 1
        # Text says: "one convolution layer with 1x1... and one dilated... dilation rate of 1"
        # We pass num_stack=1, but internal logic handles the 1x1 kernel switch for the bottom layer
        self.rfb4 = RFB_Skip(filters[3], num_stack=1, dilation_rate=1) 
        
        # Note on RFB4: The stack logic in `RFB_Skip` uses 3x3 by default. 
        # We need to manually override or create a specific block if strict adherence to "1x1" 
        # for the stack path is required. I added logic in RFB_Skip to handle this.

        # --- DECODER ---
        
        # Since RFB concatenates output, skip channels are doubled
        
        # Decoder 4
        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_conv4 = nn.Conv2d(filters[4], filters[3], kernel_size=1) # Conv1x1 after upsample
        self.dec4 = DoubleConv(filters[3] + (filters[3] * 2), filters[3]) # Input = Prev + Skip(RFB x2)

        # Decoder 3
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_conv3 = nn.Conv2d(filters[3], filters[2], kernel_size=1)
        self.dec3 = DoubleConv(filters[2] + (filters[2] * 2), filters[2])

        # Decoder 2
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_conv2 = nn.Conv2d(filters[2], filters[1], kernel_size=1)
        self.dec2 = DoubleConv(filters[1] + (filters[1] * 2), filters[1])

        # Decoder 1
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_conv1 = nn.Conv2d(filters[1], filters[0], kernel_size=1)
        self.dec1 = DoubleConv(filters[0] + (filters[0] * 2), filters[0])

        # Final Output
        self.final_conv = nn.Conv2d(filters[0], num_classes, kernel_size=1)

    def forward(self, x):
        # --- ENCODER + DUAL PATH FUSION ---
        
        # Stage 1
        x1_main = self.enc1_main(x)
        x1_pool = self.pool1(x1_main)
        x1_aux = self.enc1_aux(x)
        # Fusion: Main (Pooled) + Aux
        x1_fused = x1_pool + x1_aux 
        # Apply ReLU after addition (Standard ResNet practice, though not explicitly drawn, implied by block logic)
        x1_fused = F.relu(x1_fused)

        # Stage 2
        x2_main = self.enc2_main(x1_fused)
        x2_pool = self.pool2(x2_main)
        x2_aux = self.enc2_aux(x1_fused)
        x2_fused = x2_pool + x2_aux
        x2_fused = F.relu(x2_fused)

        # Stage 3
        x3_main = self.enc3_main(x2_fused)
        x3_pool = self.pool3(x3_main)
        x3_aux = self.enc3_aux(x2_fused)
        x3_fused = x3_pool + x3_aux
        x3_fused = F.relu(x3_fused)

        # Stage 4
        x4_main = self.enc4_main(x3_fused)
        x4_pool = self.pool4(x4_main)
        x4_aux = self.enc4_aux(x3_fused)
        x4_fused = x4_pool + x4_aux
        x4_fused = F.relu(x4_fused)

        # Bridge
        x_bridge = self.bridge(x4_fused)

        # --- RFB SKIP GENERATION ---
        # The skips come from the fused encoder features before they went to the next level
        # Note: The diagram arrows for skips originate from the output of the "Main" double conv 
        # BEFORE pooling/fusion? Or from the fused result?
        # Looking at Fig 2: The arrows go Inputs -> Conv -> [Conv] -> right arrow to RFB.
        # This implies the skip comes from the STANDARD encoder path before pooling.
        
        s1 = self.rfb1(x1_main)
        s2 = self.rfb2(x2_main)
        s3 = self.rfb3(x3_main)
        s4 = self.rfb4(x4_main)

        # --- DECODER ---
        
        # Block 4
        d4 = self.up4(x_bridge)
        d4 = self.up_conv4(d4)
        # Concatenate with RFB Skip 4
        d4 = torch.cat([d4, s4], dim=1)
        d4 = self.dec4(d4)

        # Block 3
        d3 = self.up3(d4)
        d3 = self.up_conv3(d3)
        d3 = torch.cat([d3, s3], dim=1)
        d3 = self.dec3(d3)

        # Block 2
        d2 = self.up2(d3)
        d2 = self.up_conv2(d2)
        d2 = torch.cat([d2, s2], dim=1)
        d2 = self.dec2(d2)

        # Block 1
        d1 = self.up1(d2)
        d1 = self.up_conv1(d1)
        d1 = torch.cat([d1, s1], dim=1)
        d1 = self.dec1(d1)

        return self.final_conv(d1)

# %%
arch_cse = CSE_UNet(in_channels=3, num_classes=len(df_classes)).to('cuda')
# arch_cse.smooth_loss = torch.tensor(0.0)

# %%
learn_cse: Learner = Learner(
    dls, 
    arch_cse,
    loss_func = CrossEntropyLossFlat(axis=1),
    # loss_func = CombinedLoss(dice_weight=0.4),
    metrics = metrics,
    wd=0.0005
).to_fp16()

# learn_cse.summary()

learn_cse.load('../../notebooks/models/cse_phase_2')

# %% [markdown]
# Total params: 36,988,807

# %%
# suggested_lr = None
# suggested_lr = learn_cse.lr_find()

# # %%
# suggested_lr = None # bacause it was 0.001 in the paper

# # %%
# lr = suggested_lr.valley if suggested_lr is not None else 2.5e-4
# print(f"Using learning rate: {lr} {'(hardcoded)' if suggested_lr is None else '(suggested)'}")


# lr_max=slice(1e-6, 1e-4) # previous phase 2 
lr = slice(1e-7, 1e-5)

# %%
learn_cse.fit_one_cycle(
    100, 
    lr,
    cbs=[
        SaveCB('cse_phase_4_1'),
        LogCB('cse_phase_4_1_log.csv'),
        ValidLossEarlyStoppingCB(20),
        gradient_accum,
    ]
)

learn_cse.load('cse_phase_4_1', weights_only=True)


import torch.nn.functional as F
class CombinedLoss:
    def __init__(self, dice_weight=0.5):
        self.dice_weight = dice_weight
        self.ce_weight = 1.0 - dice_weight
        
        # Your DiceLoss applies softmax internally (line 33) -> Expects Logits
        self.dice = DiceLoss(axis=1, reduction='mean') 
        
        # CE expects Logits
        self.ce = CrossEntropyLossFlat(axis=1, reduction='mean')

    def __call__(self, pred, targ):
        # Pass raw logits (pred) and integer targets (targ) to both.
        # DiceLoss._one_hot handles the target conversion automatically.
        return (
            self.dice(pred, targ) * self.dice_weight +
            self.ce(pred, targ) * self.ce_weight
        )
    
    def decodes(self, x):    
        return x.argmax(dim=1)

loss_func = CombinedLoss(dice_weight=0.5)

# %%
learn_cse.loss_func = loss_func

# %%
learn_cse

# %%
learn_cse.fit_one_cycle(
    100, 
    lr_max=slice(1e-8, 1e-6),
    cbs=[
        SaveCB('cse_phase_4_1_combined_loss'),
        LogCB('cse_phase_4_1_combined_loss_log.csv'),
        ValidLossEarlyStoppingCB(20),
        gradient_accum,
    ]
)
