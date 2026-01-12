import matplotlib.pyplot as plt
import torch
from fastai.interpret import SegmentationInterpretation
# from fastcore.basics import is_listy, tuplify
import seaborn as sns
import numpy as np
import pandas as pd


class MySegmentationInterpretation(SegmentationInterpretation):
    def __init__(self, learn, dl, preds, targs, losses, act=None, df_classes=None, test_mode=False):
        # Initialize the parent Interpretation class
        super().__init__(learn, dl, losses, act)
        self.preds = preds
        self.targs = targs
        self.df_classes = df_classes

        if df_classes is not None:
            assert('class' in df_classes.columns), "'class' column not found in df_classes (class id)"
            assert('category' in df_classes.columns), "'category' column not found in df_classes (class name)"

    @classmethod
    def from_learner(cls, learn, ds_idx=1, dl=None, act=None):
        "Construct interpretation object from a learner"
        if dl is None: 
            dl = learn.dls[ds_idx].new(shuffle=False, drop_last=False)
        
        # Get predictions, targets, and losses
        # with_decoded=False gives us probabilities (needed for your uncertainty maps)
        preds, targs, losses = learn.get_preds(dl=dl, with_input=False, with_loss=True, 
                                               with_decoded=False, with_preds=True, 
                                               with_targs=True, act=act)
        
        return cls(learn, dl, preds, targs, losses, act)

    def _stringify_classes(self, idx=None, skip_indices=None):
        if skip_indices is None:
            skip_indices = []

        if self.df_classes is None:
            l = list(range(self.preds.shape[1]))
        else:
            l = self.df_classes[['class', 'category']].astype(str).apply('. '.join, axis=1).str.strip().tolist() 

        if idx is not None:
            return l[idx]

        l = [name for i, name in enumerate(l) if i not in skip_indices]

        return l

    def _flatten(self, p, t, cls):
        p = (p == cls).float()
        t = (t == cls).float()
        return p.view(-1), t.view(-1)


    # def show_results(self,
    #     idxs:list, # Indices of predictions and targets
    #     **kwargs
    # ):
    #     if self.test_mode:
    #         if isinstance(idxs, torch.Tensor): idxs = idxs.tolist()
    #         if not is_listy(idxs): idxs = [idxs]
    #         inps, _, _, decoded, _ = self[idxs]
    #         b = tuplify(inps)
    #         self.dl.show_results(b, tuplify(decoded), max_n=len(idxs), **kwargs)
    #     else:
    #         super().show_results(idxs, **kwargs)


    # --- GLOBAL METRICS (Micro-Average) ---
    # These flatten the entire dataset first.
    # Good for "Overall Dataset Performance" and handling rare classes.
    
    def accuracy(self):
        "Global Pixel Accuracy (Total Correct / Total Pixels)"
        pred = self.preds.argmax(1)
        correct = (pred == self.targs).float()
        return correct.sum() / correct.numel()

    def precision(self, cls):
        "Global Precision for class `cls`"
        pred = self.preds.argmax(1)
        p, t = self._flatten(pred, self.targs, cls)
        
        tp = (p * t).sum()
        fp = (p * (1 - t)).sum()
        
        return tp / (tp + fp + 1e-6)
    
    def recall(self, cls):
        "Global Recall for class `cls`"
        pred = self.preds.argmax(1)
        p, t = self._flatten(pred, self.targs, cls)
        
        tp = (p * t).sum()
        fn = ((1 - p) * t).sum()
        
        return tp / (tp + fn + 1e-6)

    # --- PER-IMAGE METRICS (Averaged) ---
    # These calculate the metric for each image individually, then mean.
    # Good for seeing how the model performs "on average per image".

    def _flatten_binary(self, p, t, cls):
        return (p == cls).float().view(-1), (t == cls).float().view(-1)

    def dice(self, cls=None, eps=1e-6):
        pred = self.preds.argmax(1)
        
        # If no class specified, loop over all and average
        if cls is None:
            return torch.stack([self.dice(c) for c in range(self.preds.shape[1])]).mean()

        dices = []
        for p, t in zip(pred, self.targs):
            p, t = self._flatten_binary(p, t, cls)
            
            inter = (p * t).sum()
            union = p.sum() + t.sum()
            
            if union == 0:
                # Both target and pred are empty for this class -> Perfect match
                dices.append(torch.tensor(1.0))
            else:
                dices.append((2 * inter) / (union + eps))
                
        return torch.stack(dices).mean()

    def iou(self, cls, eps=1e-6):
        pred = self.preds.argmax(1)
        ious = []
        for p, t in zip(pred, self.targs):
            p, t = self._flatten_binary(p, t, cls)
            
            inter = (p * t).sum()
            union = p.sum() + t.sum() - inter
            
            if union == 0:
                # No ground truth and no prediction -> Perfect match
                ious.append(torch.tensor(1.0))
            else:
                ious.append(inter / (union + eps))
                
        return torch.stack(ious).mean()

    # --- GLOBAL DATASET STATS (Micro-Average) ---
    # These flatten the entire dataset first.
    # Good for "Overall Dataset Performance" and handling rare classes.

    def stats_per_class(self, agg_func=np.mean):
        print("Calculating global per-class statistics...")
        
        # 1. Flatten entire dataset (Predictions and Targets)
        #    This is faster than looping and ensures 'Micro' averaging
        pred_flat = self.preds.argmax(1).flatten()
        targs_flat = self.targs.flatten()
        
        results = []
        
        for _, row in self.df_classes.iterrows():
            cls = row['class']
            
            # Binary masks for the whole dataset
            mask_pred = (pred_flat == cls)
            mask_targ = (targs_flat == cls)

            # Confusion Matrix Elements
            tp = (mask_pred & mask_targ).sum().item()
            fp = (mask_pred & ~mask_targ).sum().item()
            fn = (~mask_pred & mask_targ).sum().item()
            
            # --- Calculations ---
            
            # IoU = TP / (TP + FP + FN)
            union = tp + fp + fn
            iou = tp / (union + 1e-6) if union > 0 else float('nan')
            
            # Dice = 2TP / (2TP + FP + FN)
            dice_denom = 2 * tp + fp + fn
            dice = (2 * tp) / (dice_denom + 1e-6) if dice_denom > 0 else float('nan')
            
            # Precision = TP / (TP + FP)
            precision = tp / (tp + fp + 1e-6) if (tp + fp) > 0 else float('nan')
            
            # Recall = TP / (TP + FN)
            # (Note: This was labeled 'cls_acc' in your original code)
            recall = tp / (tp + fn + 1e-6) if (tp + fn) > 0 else float('nan')

            results.append({
                'category': row['category'],
                'class': cls,
                'iou': iou,
                'dice': dice,
                'precision': precision,
                'recall': recall
            })

        # add total/average row
        if agg_func is not None:
            agg_row = {'category': 'average', 'class': -1}
            for key in results[0].keys():
                if key in ['category', 'class']:
                    continue
                values = [r[key] for r in results if not np.isnan(r[key])]
                if len(values) > 0:
                    agg_row[key] = agg_func(values)
                else:
                    agg_row[key] = float('nan')
            results.append(agg_row)

        return pd.DataFrame(results)

    def plot_df_result_classes(self, df_results=None, skip_indices=None):
        """Plot per-class metrics from a dataframe returned by `stats_per_class`."""

        if df_results is None:
            df_results = self.stats_per_class(agg_func=None)
        
        if skip_indices is None:
            skip_indices = [-1]
        else:
            skip_indices.append(-1)

        if skip_indices is not None:
            df_results = df_results[~df_results['class'].isin(skip_indices)]
            
        # plot actual bars
        ax = df_results.drop('class', axis=1).plot(kind='bar')
        ax.set_xticklabels(self._stringify_classes(skip_indices=skip_indices), rotation=-45, ha='left')

        # Plot averages for each metric as dashed lines
        metrics = df_results.columns.tolist()[2:]
        for metric in metrics:
            if metric in df_results.columns:
                avg = df_results[metric].mean()
            color = plt.get_cmap('tab10')(metrics.index(metric) % 10)
            plt.axhline(avg, linestyle='--', color=color, alpha=0.5, label=f'Avg {metric}', zorder=-1)

        plt.legend(
            bbox_to_anchor=(1.05, 1), borderaxespad=0.,fontsize='small',
            framealpha=0.7
        )

        plt.ylim(0, 1)

        plt.title('Per-Class Segmentation Metrics')

        plt.tight_layout()
        plt.show()
        

    # -- CONFIDENCE / UNCERTAINTY MAPS ---
    
    def confidence_map(self, i):
        # self.preds[i] is likely already probabilities (sum to 1)
        # Just take the max
        return self.preds[i].max(0).values

    def uncertainty_map(self, i):
        probs = self.preds[i]
        # Add epsilon to avoid log(0) returning -inf
        return -torch.sum(probs * (probs + 1e-8).log(), dim=0)

    def show_confidence(self, i, ctx=None):
        # If no axis is provided, get the current one
        if ctx is None: ctx = plt.gca()
        
        # 1. Capture the 'mappable' object returned by imshow
        im = ctx.imshow(self.confidence_map(i), cmap='viridis')
        
        # 2. Add colorbar targeting this specific axis
        # fraction and pad keep the colorbar size proportional to the image
        plt.colorbar(im, ax=ctx, fraction=0.046, pad=0.04)
        
        # 3. Use set_title (standard for Axes objects)
        ctx.set_title("Confidence")

    def show_uncertainty(self, i, ctx=None):
        if ctx is None: ctx = plt.gca()
        
        im = ctx.imshow(self.uncertainty_map(i), cmap='magma')
        plt.colorbar(im, ax=ctx, fraction=0.046, pad=0.04)
        ctx.set_title("Uncertainty")

    def show_confidence_and_uncertainty(self, i):
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        
        # --- Handle the Input Image ---
        # Note: self.dl.dataset[i][0] is likely a Tensor (C, H, W) on GPU.
        # We need to move it to CPU and rearrange to (H, W, C) for matplotlib.
        img_tensor, _ = self.dl.dataset[i] # Deconstruct tuple (x,y)
        
        # If using fastai, it's safer to use the decode method to handle normalization
        # But if doing it manually:
        if hasattr(img_tensor, 'cpu'): img_tensor = img_tensor.cpu()
        if img_tensor.shape[0] == 3:   img_tensor = img_tensor.permute(1, 2, 0)
            
        axs[0].imshow(img_tensor)
        axs[0].set_title("Input Image")
        axs[0].axis('off')  
        
        # --- Plot Maps ---
        self.show_confidence(i, ctx=axs[1])
        self.show_uncertainty(i, ctx=axs[2])

        for ax in axs[1:]:
            ax.axis('off')  

        
        plt.tight_layout()
        plt.show()


    def confusion_matrix(self):
        pred = self.preds.argmax(1)
        n_cls = self.preds.shape[1]

        cm = torch.zeros(n_cls, n_cls, dtype=torch.int64)

        for p, t in zip(pred, self.targs):
            for gt in range(n_cls):
                for pr in range(n_cls):
                    cm[gt, pr] += ((t == gt) & (p == pr)).sum()

        return cm


    def plot_confusion_matrix(self, normalize=True):
        cm = self.confusion_matrix().float()

        if normalize:
            cm = cm / cm.sum(1, keepdim=True)

        cm = cm.cpu().numpy()

        class_names = self._stringify_classes()

        plt.figure(figsize=(8,6))
        sns.heatmap(
            cm,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names
        )
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.title("Pixel-wise Confusion Matrix")


    def show_results_with_class(self, cls, n_samples=1, **kwargs):
        pred = self.preds.argmax(1)

        # Find indices where the target contains the class
        indices = [i for i, t in enumerate(self.targs) if (t == cls).any()]
        np.random.shuffle(indices)

        # Limit to n_samples
        indices = indices[:n_samples]
        
        self.show_results(indices, **kwargs)

    def _make_rgba_overlay(self, mask, color, alpha=0.5):
        """Creates an RGBA array where the mask is the given color/alpha and the rest is transparent."""
        h, w = mask.shape
        rgba = np.zeros((h, w, 4), dtype=np.float32)
        
        # Set RGB channels (broadcast color to mask pixels)
        rgba[mask, :3] = color 
        # Set Alpha channel
        rgba[mask, 3] = alpha
        return rgba
    
    def plot_overlay(self, img, mask, class_color, title=None, alpha=0.5, ax=None):
        """
        Plot an image with an RGBA overlay for a given mask and class color.
        """
        if ax is None:
            ax = plt.gca()
        # Grayscale background
        if img.ndim == 3 and img.shape[2] == 3:
            img_gray = np.mean(img, axis=2)
        else:
            img_gray = img
        overlay = self._make_rgba_overlay(mask, class_color, alpha=alpha)
        ax.imshow(img_gray, cmap='gray')
        ax.imshow(overlay)
        if title is not None:
            ax.set_title(title)
        ax.axis('off')

    def show_results_per_class(self, cls, n_samples=3, crop=True, pad=50, figsize=(10, 4)):
        # 1. Determine the color for this class
        # Default to Green if no info provided
        class_color = np.array([0.0, 1.0, 0.0]) 
        
        if self.df_classes is not None:
            # Look up the row where 'class' matches the current cls
            try:
                # Assuming 'class' col holds the int index and 'rgb' holds (R,G,B)
                row = self.df_classes[self.df_classes['class'] == cls]
                if not row.empty:
                    rgb = row.iloc[0]['rgb']
                    class_color = np.array(rgb) / 255.0
            except Exception as e:
                print(f"Warning: Could not extract color for class {cls}. using default. ({e})")

            # 2. Find samples
            idxs = [i for i, t in enumerate(self.targs) if (t == cls).any()]
            if len(idxs) == 0:
                print(f"No samples found with class {cls}")
                return
            print(f"Found {len(idxs)} samples with class {cls}")
            np.random.shuffle(idxs)
            idxs = idxs[:n_samples]

            # 3. Get data
            inps, _, targs, decoded, _ = self[idxs]
            decoded_batch = self.dl.decode_batch((inps, targs))
            decoded_batch = decoded_batch.argmax(1)  # Convert to class indices

            # 4. Plot
            fig, ax = plt.subplots(len(idxs), 2, figsize=(figsize[0], figsize[1] * len(idxs)))
            if len(idxs) == 1: ax = ax[None, :]

            for i, idx in enumerate(idxs):
                # --- PROCESS IMAGE ---
                img_disp = decoded_batch[i][0]
                if isinstance(img_disp, torch.Tensor):
                    img_disp = img_disp.permute(1, 2, 0).cpu().numpy()
                elif hasattr(img_disp, 'permute'):
                     img_disp = img_disp.permute(1, 2, 0).cpu().numpy()
                else:
                    img_disp = np.array(img_disp)

                targ_mask = targs[i]
                pred_mask = decoded[i]

                # --- CROP ---
                if crop:
                    mask_focus = (targ_mask == cls) | (pred_mask == cls)
                    if mask_focus.any():
                        rows, cols = torch.where(mask_focus)
                        rmin, rmax = rows.min(), rows.max()
                        cmin, cmax = cols.min(), cols.max()

                        # Pad & Clamp
                        h, w = img_disp.shape[:2]
                        rmin = max(0, int(rmin) - pad)
                        rmax = min(h, int(rmax) + pad)
                        cmin = max(0, int(cmin) - pad)
                        cmax = min(w, int(cmax) + pad)

                        # Apply Crop
                        img_disp = img_disp[rmin:rmax, cmin:cmax]
                        targ_mask = targ_mask[rmin:rmax, cmin:cmax]
                        pred_mask = pred_mask[rmin:rmax, cmin:cmax]

                # --- DISPLAY ---
                # Column 1: Ground Truth
                self.plot_overlay(
                    img_disp,
                    targ_mask == cls,
                    class_color,
                    title=f"GT (Class {self._stringify_classes(idx=cls)})",
                    ax=ax[i, 0]
                )

                # Column 2: Prediction
                self.plot_overlay(
                    img_disp,
                    pred_mask == cls,
                    class_color,
                    title=f"Pred (Class {self._stringify_classes(idx=cls)})",
                    ax=ax[i, 1]
                )

        plt.tight_layout()
        plt.show()
