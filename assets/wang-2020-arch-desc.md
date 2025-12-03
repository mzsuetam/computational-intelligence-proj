![alt text](../docs/slides/src/img/4-cse-unet-arch.png)

Based on the image you provided (Figure 2), here is exactly where you can find each component I summarized.

I have broken down the diagram by color and location to help you spot them.

### 1. The Dual-Path Encoder (Far Left)
Look at the **left-hand side** of the diagram. You will see two vertical columns of boxes that merge together.

* **The Auxiliary Path (Pink/Peach Boxes):** These are the boxes on the far left edge labeled `Conv7x7`, `Conv5x5`, etc., with `Stride=2`. These correspond to the "Multi-kernel" part of the encoder.
* **The Main Path (White Boxes):** Just to the right of the pink boxes, you see the standard U-Net blocks labeled `Conv3x3` and `MaxPool 2X`.
* **The Fusion (The `+` Circle):** Notice the arrows from the Pink box and the White box meeting at a circle with a plus sign **(+)**. This is where the two paths are added together (element-wise addition).

### 2. RFB-Based Skip Pathways (Center)
Look at the **middle section** of the diagram, bridging the gap between the left (encoder) and right (decoder).

* **The Stacks (Green Boxes):** These are the upper blocks in the middle section. You see stacks of `Conv3x3`. This represents the "stack of convolutional layers" mentioned in the text.
* **The Dilated Convs (Yellow Boxes):** These are the single blocks below the green ones. They are explicitly labeled with `D=7`, `D=5`, etc. This stands for **Dilation Rate**.
* **The Split:** You can see the arrow coming from the encoder (left) splitting into two arrows: one going into the Green boxes and one into the Yellow box.

### 3. The Decoder (Far Right)
Look at the **right-hand side** of the diagram moving upwards.

* **Upsampling (White Boxes):** These boxes are labeled `BilinearUP 2X`. This is the upsampling layer.
* **Concatenation (The `C` Circle):** Above the upsampling blocks, there is a circle with the letter **(C)**.
    * One arrow comes from the bottom (the upsampled features).
    * One arrow comes from the left (the output of the **Green/Yellow RFB blocks**).
    * This confirms that the Decoder fuses features from the RFB module, not the raw encoder.

### Visual Summary Map
* **Left (Pink + White + Plus Sign):** Dual-Path Encoder.
* **Middle (Green + Yellow):** RFB Context Module.
* **Right (Up Arrow + 'C'):** Decoder.
