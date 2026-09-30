# MSCP-Mamba (Balanced Sampling) — Detailed Block Diagram

> A CNN + VSS (Vision State-Space) Hybrid Architecture for Background Subtraction on CDnet 2014 Thermal Scenes

---

## 1. High-Level Pipeline

```mermaid
flowchart TB
    subgraph INPUT["📥 Input Pipeline"]
        A["CDnet 2014 Thermal Dataset<br/>(5 scenes: corridor, diningRoom,<br/>lakeSide, library, park)"]
        B["CDDnetThermal Dataset Class<br/>(Balanced sampling: 40 frames/scene)"]
        C["DataLoader<br/>(batch_size=1, shuffle=True)"]
        A --> B --> C
    end

    subgraph MODEL["🧠 MSCPNet (8.73M params)"]
        D["CNN Encoder<br/>(4 blocks × 5 modules)"]
        E["MSCP Block<br/>(Multi-Scale Contrast Preservation)"]
        F["VSS Stack<br/>(2 × VSSBlock with SS2D)"]
        G["CNN Decoder<br/>(3 blocks + 2 Fusion Layers)"]
        D -->|"F [B,192,60,80]"| E -->|"Y [B,240,60,80]"| F -->|"Y' [B,240,60,80]"| G
        D -->|"low_full [B,64,240,320]"| G
    end

    subgraph TRAIN["🔄 Training Loop"]
        H["Weighted BCE Loss<br/>(ROI-masked, fp32)"]
        I["AdamW Optimizer<br/>(lr=5e-4, wd=0.05)"]
        J["ReduceLROnPlateau<br/>(factor=0.1, patience=5)"]
        K["AMP GradScaler<br/>(Mixed Precision)"]
        L["Early Stopping<br/>(patience=10)"]
        H --> I --> J
        K --> I
        L -->|"monitors val F1"| J
    end

    subgraph CKPT_HI["💾 Checkpointing"]
        M["Drive Persistence<br/>(atomic writes, every epoch)"]
        N["Auto-Resume<br/>(model + optimizer + RNG states)"]
        M --> N
    end

    subgraph EVAL_HI["📊 Evaluation"]
        O["Per-Scene Metrics<br/>(all annotated frames, batch=8)"]
        P["Inference Speed<br/>(fps benchmark)"]
        O --> P
    end

    subgraph XAI_HI["🔍 Explainability"]
        Q["SHAP Expected Gradients<br/>(per-pixel attribution, N=200)"]
        R["LIME SLIC Superpixels<br/>(num_samples=400, 12 features)"]
    end

    subgraph OUT_HI["📦 Output Artifacts"]
        S["Model .pth + Results .csv<br/>+ Explainability .png<br/>saved locally + Google Drive"]
    end

    C -->|"[B,3,240,320]"| D
    G -->|"logits [B,1,240,320]"| H
    TRAIN --> CKPT_HI
    CKPT_HI --> EVAL_HI
    EVAL_HI --> XAI_HI
    XAI_HI --> OUT_HI

    style INPUT fill:#1a1a2e,stroke:#16213e,color:#e0e0ff
    style MODEL fill:#0f3460,stroke:#533483,color:#e0e0ff
    style TRAIN fill:#16213e,stroke:#e94560,color:#e0e0ff
    style CKPT_HI fill:#1a1a2e,stroke:#533483,color:#e0e0ff
    style EVAL_HI fill:#16213e,stroke:#e94560,color:#e0e0ff
    style XAI_HI fill:#533483,stroke:#e94560,color:#e0e0ff
    style OUT_HI fill:#0f3460,stroke:#533483,color:#e0e0ff
```


---

## 2. CNN Encoder — Detailed Architecture

> 4 residual blocks, each with 5 parallel modules (1 sparse + 4 atrous). Outputs bottleneck `F [B,192,H/4,W/4]` and low-level features `low_full [B,64,H,W]`.

```mermaid
flowchart TB
    subgraph ENCODER["CNN Encoder"]

        IN["Input<br/>[B, 3, 240, 320]"]

        subgraph B1["EncoderBlock 1 (in=3, mid=64, out=64, n_sparse=2)"]
            direction LR
            SM1["SparseModule<br/>Conv3x3 - CN - ReLU x2<br/>Conv1x1 - CN - ReLU"]
            A2_1["AtrousModule<br/>dilation=2"]
            A4_1["AtrousModule<br/>dilation=4"]
            A8_1["AtrousModule<br/>dilation=8"]
            A16_1["AtrousModule<br/>dilation=16"]
        end
        SUM1(("Sum"))
        SM1 --> SUM1
        A2_1 --> SUM1
        A4_1 --> SUM1
        A8_1 --> SUM1
        A16_1 --> SUM1
        IN --> B1

        O1["o1 = Block1 output [B,64,240,320]<br/>also used as low_full"]
        SUM1 --> O1

        SD1["SpatialDropout p=0.25"]
        MP1["MaxPool2d 2x2"]
        O1 --> SD1 --> MP1

        subgraph B2["EncoderBlock 2 (in=64, mid=128, out=64, n_sparse=2)"]
            direction LR
            SM2["SparseModule"]
            A2_2["Atrous d=2"]
            A4_2["Atrous d=4"]
            A8_2["Atrous d=8"]
            A16_2["Atrous d=16"]
        end
        SUM2(("Sum + Residual"))
        MP1 --> B2
        SM2 --> SUM2
        A2_2 --> SUM2
        A4_2 --> SUM2
        A8_2 --> SUM2
        A16_2 --> SUM2
        O2["o2 [B,64,120,160]"]
        SUM2 --> O2

        SD2["SpatialDropout"]
        MP2["MaxPool2d 2x2"]
        O2 --> SD2 --> MP2

        subgraph B3["EncoderBlock 3 (in=64, mid=256, out=64, n_sparse=3)"]
            direction LR
            SM3["SparseModule"]
            A2_3["Atrous d=2"]
            A4_3["Atrous d=4"]
            A8_3["Atrous d=8"]
            A16_3["Atrous d=16"]
        end
        SUM3(("Sum + Residual"))
        MP2 --> B3
        SM3 --> SUM3
        A2_3 --> SUM3
        A4_3 --> SUM3
        A8_3 --> SUM3
        A16_3 --> SUM3
        O3["o3 [B,64,60,80]"]
        SUM3 --> O3

        SD3["SpatialDropout"]
        O3 --> SD3

        subgraph B4["EncoderBlock 4 (in=64, mid=512, out=64, n_sparse=3)<br/>No pooling between Block3 and Block4"]
            direction LR
            SM4["SparseModule"]
            A2_4["Atrous d=2"]
            A4_4["Atrous d=4"]
            A8_4["Atrous d=8"]
            A16_4["Atrous d=16"]
        end
        SUM4(("Sum + Residual"))
        SD3 --> B4
        SM4 --> SUM4
        A2_4 --> SUM4
        A4_4 --> SUM4
        A8_4 --> SUM4
        A16_4 --> SUM4
        O4["o4 [B,64,60,80]"]
        SUM4 --> O4

        PROJ["proj1: Conv1x1 64 to 64"]
        O1 --> PROJ

        INTERP1["Bilinear down to 60x80"]
        INTERP2["Bilinear down to 60x80"]
        PROJ --> INTERP1
        O2 --> INTERP2

        CAT["Concat o1down o2down o3 o4<br/>[B, 256, 60, 80]"]
        INTERP1 --> CAT
        INTERP2 --> CAT
        O3 --> CAT
        O4 --> CAT

        FUSE_ENC["fuse: Conv1x1 256 to 192<br/>F [B, 192, 60, 80]"]
        CAT --> FUSE_ENC
    end

    FUSE_ENC -->|"F [B,192,60,80]"| MSCP_OUT["To MSCP Block"]
    O1 -->|"low_full [B,64,240,320]"| DEC_OUT["To Decoder"]

    style ENCODER fill:#0f3460,stroke:#533483,color:#e0e0ff
    style B1 fill:#1a1a2e,stroke:#0f3460,color:#e0e0ff
    style B2 fill:#1a1a2e,stroke:#0f3460,color:#e0e0ff
    style B3 fill:#1a1a2e,stroke:#0f3460,color:#e0e0ff
    style B4 fill:#1a1a2e,stroke:#0f3460,color:#e0e0ff
```

> [!NOTE]
> **SparseModule** applies `n_sparse` stacked 3x3 convolutions (each followed by ContrastNorm + ReLU), then a 1x1 reduction conv.
>
> **AtrousModule** applies a single dilated 3x3 conv (with dilation `d`) followed by ContrastNorm then ReLU.
>
> **ContrastNorm** is InstanceNorm2d (per-sample, per-channel normalization — preferred over BatchNorm for batch_size=1).

---

## 3. MSCP Block — Multi-Scale Contrast Preservation (Sec. III-B)

> The core novelty of the paper. Extracts contrast features at multiple scales and concatenates with a max-pooled global cue.

```mermaid
flowchart TB
    subgraph MSCP["MSCP Block in=192 out=240"]
        F_IN["F [B,192,60,80]"]

        subgraph BRANCH_M["Branch M: Global MaxPool Cue"]
            MP_M["MaxPool2d 2x2"]
            CONV_M["Conv1x1 192 to 64"]
            RELU_M["ReLU"]
            UP_M["Bilinear up restore resolution"]
            MP_M --> CONV_M --> RELU_M --> UP_M
        end

        subgraph BRANCH_X1["Contrast Branch X1 Eq. 6"]
            CV1["Conv 3x3 pad=1 192 to 64"]
            AVG1["AvgPool 3x3 stride=1 pad=1"]
            SUB1["CV1 minus AvgPool of CV1"]
            RELU1["ReLU"]
            CV1 --> SUB1
            CV1 --> AVG1 --> SUB1
            SUB1 --> RELU1
        end

        subgraph BRANCH_X2["Contrast Branch X2 dilation=4"]
            CV2["Conv 3x3 dil=4 pad=4 192 to 64"]
            AVG2["AvgPool 3x3"]
            SUB2["CV2 minus AvgPool of CV2"]
            RELU2["ReLU"]
            CV2 --> SUB2
            CV2 --> AVG2 --> SUB2
            SUB2 --> RELU2
        end

        subgraph BRANCH_X3["Contrast Branch X3 dilation=8"]
            CV3["Conv 3x3 dil=8 pad=8 192 to 64"]
            AVG3["AvgPool 3x3"]
            SUB3["CV3 minus AvgPool of CV3"]
            RELU3["ReLU"]
            CV3 --> SUB3
            CV3 --> AVG3 --> SUB3
            SUB3 --> RELU3
        end

        subgraph BRANCH_X4["Contrast Branch X4 dilation=16"]
            CV4["Conv 3x3 dil=16 pad=16 192 to 64"]
            AVG4["AvgPool 3x3"]
            SUB4["CV4 minus AvgPool of CV4"]
            RELU4["ReLU"]
            CV4 --> SUB4
            CV4 --> AVG4 --> SUB4
            SUB4 --> RELU4
        end

        F_IN --> BRANCH_M
        F_IN --> BRANCH_X1
        F_IN --> BRANCH_X2
        F_IN --> BRANCH_X3
        F_IN --> BRANCH_X4

        CATM["Concat M X1 X2 X3 X4<br/>[B, 320, 60, 80]"]
        UP_M --> CATM
        RELU1 --> CATM
        RELU2 --> CATM
        RELU3 --> CATM
        RELU4 --> CATM

        PROJ_MSCP["proj: Conv1x1 320 to 240"]
        CN_MSCP["ContrastNorm 240"]
        RELU_MSCP["ReLU"]
        SD_MSCP["SpatialDropout p=0.25"]

        CATM --> PROJ_MSCP --> CN_MSCP --> RELU_MSCP --> SD_MSCP

        Y_OUT["Y [B, 240, 60, 80]"]
        SD_MSCP --> Y_OUT
    end

    style MSCP fill:#533483,stroke:#e94560,color:#e0e0ff
    style BRANCH_M fill:#1a1a2e,stroke:#533483,color:#e0e0ff
    style BRANCH_X1 fill:#1a1a2e,stroke:#533483,color:#e0e0ff
    style BRANCH_X2 fill:#1a1a2e,stroke:#533483,color:#e0e0ff
    style BRANCH_X3 fill:#1a1a2e,stroke:#533483,color:#e0e0ff
    style BRANCH_X4 fill:#1a1a2e,stroke:#533483,color:#e0e0ff
```

> [!IMPORTANT]
> **Eq. 6 — Contrast Computation**: `X_j = ReLU(CV_j(F) - AvgPool(CV_j(F)))`
>
> This preserves contrast information at different spatial scales by subtracting the local average from the convolved features.

---

## 4. VSS (Mamba) Stack — 2D Selective State-Space Model

> Two VSSBlocks applied sequentially after the MSCP block. Each block uses SS2D (4-direction cross-scan) for global context.

```mermaid
flowchart TB
    subgraph VSS_STACK["VSS Stack 2 x VSSBlock"]
        Y_IN["Y [B, 240, 60, 80]"]

        subgraph VSS1["VSSBlock 1"]
            LN1["LayerNorm 240<br/>[B,H,W,C] channel-last"]
            SS2D_1["SS2D Module"]
            DROP1["Dropout"]
            RES1(("+ Residual"))
            LN1 --> SS2D_1 --> DROP1 --> RES1
        end

        subgraph VSS2["VSSBlock 2"]
            LN2["LayerNorm 240"]
            SS2D_2["SS2D Module"]
            DROP2["Dropout"]
            RES2(("+ Residual"))
            LN2 --> SS2D_2 --> DROP2 --> RES2
        end

        Y_IN --> VSS1
        Y_IN --> RES1
        RES1 --> VSS2
        RES1 --> RES2

        Y_PRIME["Y_prime [B, 240, 60, 80]"]
        RES2 --> Y_PRIME
    end

    style VSS_STACK fill:#e94560,stroke:#533483,color:#ffffff
    style VSS1 fill:#1a1a2e,stroke:#e94560,color:#e0e0ff
    style VSS2 fill:#1a1a2e,stroke:#e94560,color:#e0e0ff
```

### SS2D — 2D Selective Scan (VMamba-style, 4-direction)

```mermaid
flowchart TB
    subgraph SS2D["SS2D Module d_model=240 expand=2 d_state=16"]
        X_IN["x [B, H, W, d_model]"]

        IN_PROJ["in_proj: Linear 240 to 960<br/>x + z gate, 2 x d_inner"]
        X_IN --> IN_PROJ

        SPLIT["Split into x_val and z<br/>each [B,H,W,480]"]
        IN_PROJ --> SPLIT

        CONV2D["Depthwise Conv2d 3x3<br/>480 groups=480"]
        SILU1["SiLU activation"]
        SPLIT -->|"x_val"| CONV2D --> SILU1

        subgraph SCAN_ORDERS["4-Direction Cross-Scan"]
            direction LR
            S0["rows forward"]
            S1["rows backward"]
            S2["cols downward"]
            S3["cols upward"]
        end
        SILU1 --> SCAN_ORDERS

        subgraph SCAN1D["1-D Selective Scan per direction"]
            direction TB
            X_PROJ["x_proj: Linear d_inner to dt_rank+2N"]
            DT_PROJ["dt_proj: Linear dt_rank to d_inner"]
            SSM["SSM Recurrence:<br/>hi = exp delta_i A hi_minus_1 + delta_i ui Bi<br/>yi = hi dot Ci"]
            X_PROJ --> DT_PROJ --> SSM
        end
        SCAN_ORDERS -->|"4 sequences [B,d_inner,L]"| SCAN1D

        UNSCAN["Unscan: reshape back to [B,d_inner,H,W]"]
        AVG_DIR["Average 4 directions"]
        SCAN1D --> UNSCAN --> AVG_DIR

        Z_GATE["z-gate: y times SiLU of z"]
        AVG_DIR --> Z_GATE
        SPLIT -->|"z"| Z_GATE

        OUT_PROJ["out_proj: Linear 480 to 240"]
        Z_GATE --> OUT_PROJ

        X_OUT["output [B, H, W, d_model]"]
        OUT_PROJ --> X_OUT
    end

    style SS2D fill:#0f3460,stroke:#e94560,color:#e0e0ff
    style SCAN_ORDERS fill:#1a1a2e,stroke:#0f3460,color:#e0e0ff
    style SCAN1D fill:#1a1a2e,stroke:#0f3460,color:#e0e0ff
```

> [!TIP]
> **Selective Scan Dispatch**:
> - **sm80+ GPUs** (A100, L4): Uses official CUDA kernels from `mamba-ssm`
> - **T4/P100** (sm75 or lower): Falls back to a **pure-PyTorch chunked selective scan** (2-pass algorithm, O(L/chunk) sequential steps)
>
> All scans run in **fp32** (AMP-excluded) for numerical stability of delta discretization.

---

## 5. CNN Decoder

> 3 decoder blocks with 2 Fusion Layers that inject low-level encoder features via Global MaxPool attention.

```mermaid
flowchart TB
    subgraph DECODER["CNN Decoder in=240"]
        Y_PRIME_IN["Y_prime [B, 240, 60, 80]"]
        LOW["low_full [B, 64, 240, 320]<br/>from Encoder Block 1"]

        subgraph DB1["Decoder Block 1"]
            DCONV1["Conv 3x3 240 to 64 pad=1"]
            DCN1["ContrastNorm 64"]
            DRELU1["ReLU"]
            DCONV1 --> DCN1 --> DRELU1
        end
        Y_PRIME_IN --> DB1

        subgraph FL1["FusionLayer 1 dec=64 low=64"]
            LOW_CONV1["Conv1x1 64 to 64 on low_full"]
            GMP1["AdaptiveMaxPool2d 1<br/>gives [B,64,1,1]"]
            INTERP_F1["Bilinear up to dec_out size"]
            MUL1(("Multiply"))
            ADD1(("Add"))
            LOW_CONV1 --> GMP1 --> INTERP_F1 --> MUL1
            MUL1 --> ADD1
        end
        DRELU1 --> MUL1
        DRELU1 --> ADD1
        LOW --> LOW_CONV1

        UP1["Bilinear up x2<br/>[B,64,60,80] to [B,64,120,160]"]
        ADD1 --> UP1

        subgraph DB2["Decoder Block 2"]
            DCONV2["Conv 3x3 64 to 64"]
            DCN2["ContrastNorm 64"]
            DRELU2["ReLU"]
            DCONV2 --> DCN2 --> DRELU2
        end
        UP1 --> DB2

        subgraph FL2["FusionLayer 2"]
            LOW_CONV2["Conv1x1 64 to 64 on low_full"]
            GMP2["AdaptiveMaxPool2d 1"]
            INTERP_F2["Bilinear up"]
            MUL2(("Multiply"))
            ADD2(("Add"))
            LOW_CONV2 --> GMP2 --> INTERP_F2 --> MUL2
            MUL2 --> ADD2
        end
        DRELU2 --> MUL2
        DRELU2 --> ADD2
        LOW --> LOW_CONV2

        UP2["Bilinear up x2<br/>[B,64,120,160] to [B,64,240,320]"]
        ADD2 --> UP2

        subgraph DB3["Decoder Block 3"]
            DCONV3["Conv 3x3 64 to 64"]
            DCN3["ContrastNorm 64"]
            DRELU3["ReLU"]
            DCONV3 --> DCN3 --> DRELU3
        end
        UP2 --> DB3

        HEAD["head: Conv1x1 64 to 1<br/>raw logits no sigmoid"]
        DRELU3 --> HEAD

        LOGITS["logits [B, 1, 240, 320]"]
        HEAD --> LOGITS
    end

    style DECODER fill:#16213e,stroke:#e94560,color:#e0e0ff
    style DB1 fill:#1a1a2e,stroke:#16213e,color:#e0e0ff
    style DB2 fill:#1a1a2e,stroke:#16213e,color:#e0e0ff
    style DB3 fill:#1a1a2e,stroke:#16213e,color:#e0e0ff
    style FL1 fill:#0f3460,stroke:#16213e,color:#e0e0ff
    style FL2 fill:#0f3460,stroke:#16213e,color:#e0e0ff
```

> [!NOTE]
> **FusionLayer formula**: `fused = (GlobalMaxPool(Conv1x1(low)) * dec_out) + dec_out`
>
> This applies channel attention from low-level features onto the decoded features, then adds a residual.

---

## 6. Loss, Optimizer and Training Details

```mermaid
flowchart LR
    subgraph LOSS["Weighted BCE Loss Eq. 8"]
        LOGITS_IN["logits [B,1,H,W]"]
        TARGET["target fg=1 bg=0"]
        VALID["valid mask<br/>ROI and not unknown"]
        WEIGHT["per-pixel class weight<br/>w_fg = N_fg+N_bg over 2 N_fg<br/>w_bg = N_fg+N_bg over 2 N_bg"]
        BCE["binary_cross_entropy_with_logits<br/>weight = weight times valid<br/>reduction = sum / valid.sum"]
        LOGITS_IN --> BCE
        TARGET --> BCE
        VALID --> BCE
        WEIGHT --> BCE
    end

    subgraph OPTIM["Optimizer"]
        ADAMW["AdamW<br/>lr = 5e-4"]
        DECAY["decay group: wd=0.05<br/>all 2D+ params"]
        NODECAY["no_decay group: wd=0<br/>1D params biases A_log"]
        ADAMW --- DECAY
        ADAMW --- NODECAY
    end

    subgraph SCHED["Scheduler"]
        RLROP["ReduceLROnPlateau<br/>metric = 1 minus val_F1<br/>factor = 0.1<br/>patience = 5"]
    end

    subgraph AMP_BLK["Mixed Precision"]
        AMP_CAST["torch.amp.autocast cuda"]
        SCALER["GradScaler"]
        CLIP["clip_grad_norm_ 1.0"]
        AMP_CAST --> SCALER --> CLIP
    end

    BCE -->|"loss"| OPTIM
    OPTIM --> SCHED
    OPTIM --> AMP_BLK

    style LOSS fill:#e94560,stroke:#533483,color:#ffffff
    style OPTIM fill:#0f3460,stroke:#533483,color:#e0e0ff
    style SCHED fill:#16213e,stroke:#e94560,color:#e0e0ff
    style AMP_BLK fill:#1a1a2e,stroke:#e94560,color:#e0e0ff
```

---

## 7. Data Pipeline — Balanced Sampling

```mermaid
flowchart TB
    subgraph DATA["Dataset: CDDnetThermal"]
        SCENES["5 Thermal Scenes<br/>corridor diningRoom lakeSide<br/>library park"]

        subgraph SAMPLING["Balanced Train Sampling"]
            PER["n_train=200 n_scenes=5<br/>40 frames per scene"]
            SHUFFLE["Per-scene seeded shuffle<br/>seed=42"]
            RESERVE["Reserve val tail<br/>10 percent of each scene"]
            TAKE["Take first 40 from available"]
            PER --> SHUFFLE --> RESERVE --> TAKE
        end

        subgraph VAL_SPLIT["Validation Split"]
            VAL_PROP["Proportional tail per scene<br/>val_frac=0.10"]
        end

        subgraph GT_DECODE["Ground Truth Decoding"]
            GT_FG["gt ge 200 foreground 1"]
            GT_BG["gt lt 200 and gt neq 170 background 0"]
            GT_IGN["gt = 170 IGNORE unknown"]
            GT_ROI["ROI mask ge 127 valid"]
        end

        subgraph AUGMENT["Preprocessing"]
            RESIZE["Resize to 240x320"]
            NORMALIZE["Normalize to 0 1"]
            TO_TENSOR["Convert to C H W tensor"]
            RESIZE --> NORMALIZE --> TO_TENSOR
        end

        SCENES --> SAMPLING
        SCENES --> VAL_SPLIT
    end

    subgraph OUTPUTS["Dataset getitem returns"]
        X_OUT_D["x: C H W frame tensor"]
        LABEL_D["label: 1 H W fg bg"]
        VALID_D["valid: 1 H W ROI mask"]
        WEIGHT_D["weight: 1 H W<br/>inverse-frequency class weight"]
    end

    DATA --> OUTPUTS

    style DATA fill:#1a1a2e,stroke:#16213e,color:#e0e0ff
    style SAMPLING fill:#0f3460,stroke:#1a1a2e,color:#e0e0ff
    style VAL_SPLIT fill:#0f3460,stroke:#1a1a2e,color:#e0e0ff
    style GT_DECODE fill:#0f3460,stroke:#1a1a2e,color:#e0e0ff
    style AUGMENT fill:#0f3460,stroke:#1a1a2e,color:#e0e0ff
    style OUTPUTS fill:#16213e,stroke:#e94560,color:#e0e0ff
```

---

## 8. Complete Forward Pass — Tensor Shapes

| Stage | Component | Input Shape | Output Shape |
|-------|-----------|-------------|--------------|
| 1 | **Input** | — | `[B, 3, 240, 320]` |
| 2 | **Encoder Block 1** | `[B, 3, 240, 320]` | `o1 [B, 64, 240, 320]` |
| 3 | **SpatialDropout + MaxPool** | `[B, 64, 240, 320]` | `[B, 64, 120, 160]` |
| 4 | **Encoder Block 2 + Residual** | `[B, 64, 120, 160]` | `o2 [B, 64, 120, 160]` |
| 5 | **SpatialDropout + MaxPool** | `[B, 64, 120, 160]` | `[B, 64, 60, 80]` |
| 6 | **Encoder Block 3 + Residual** | `[B, 64, 60, 80]` | `o3 [B, 64, 60, 80]` |
| 7 | **SpatialDropout** (no pool) | `[B, 64, 60, 80]` | `[B, 64, 60, 80]` |
| 8 | **Encoder Block 4 + Residual** | `[B, 64, 60, 80]` | `o4 [B, 64, 60, 80]` |
| 9 | **Multi-scale Fusion (Concat+1x1)** | `[B, 256, 60, 80]` | `F [B, 192, 60, 80]` |
| 10 | **MSCP Block** | `[B, 192, 60, 80]` | `Y [B, 240, 60, 80]` |
| 11 | **VSSBlock 1 (SS2D)** | `[B, 240, 60, 80]` | `[B, 240, 60, 80]` |
| 12 | **VSSBlock 2 (SS2D)** | `[B, 240, 60, 80]` | `Y' [B, 240, 60, 80]` |
| 13 | **Decoder Block 1 + Fusion** | `[B, 240, 60, 80]` | `[B, 64, 60, 80]` |
| 14 | **Bilinear up x2** | `[B, 64, 60, 80]` | `[B, 64, 120, 160]` |
| 15 | **Decoder Block 2 + Fusion** | `[B, 64, 120, 160]` | `[B, 64, 120, 160]` |
| 16 | **Bilinear up x2** | `[B, 64, 120, 160]` | `[B, 64, 240, 320]` |
| 17 | **Decoder Block 3** | `[B, 64, 240, 320]` | `[B, 64, 240, 320]` |
| 18 | **Head Conv1x1** | `[B, 64, 240, 320]` | `logits [B, 1, 240, 320]` |
| 19 | **Sigmoid + Threshold (ge 0.9)** | `logits [B, 1, 240, 320]` | `prediction [B, 1, 240, 320]` |

---

## 9. Key Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `arch` | `cnn_mamba` | Hybrid CNN + VSS |
| `sampling` | `balanced` | 40 frames/scene (evenly split) |
| `n_vss` | `2` | VSS blocks after MSCP |
| `vss_expand` | `2` | SS2D inner dim = 2 x 240 = 480 |
| `vss_d_state` | `16` | SSM state size N |
| `scan_chunk` | `64` | Chunk length for PyTorch fallback scan |
| `img_size` | `(240, 320)` | H x W |
| `n_train` | `200` | Total training frames |
| `batch_size` | `1` | Paper specification |
| `adamw_lr` | `5e-4` | For CNN+Mamba hybrid |
| `adam_wd` | `0.05` | Excludes norms/SSM state |
| `threshold` | `0.9` | Pixel decision threshold |
| `sd_rate` | `0.25` | SpatialDropout rate |
| `max_epochs` | `100` | With early stopping |
| `early_stop_patience` | `10` | Epochs without improvement |
| **Total params** | **8.73M** | CNN-only baseline: 7.92M |

---

## 10. Evaluation Metrics

```mermaid
flowchart LR
    subgraph METRICS["SegMetrics over valid/ROI pixels only"]
        direction TB
        TP["TP = pred AND gt AND valid sum"]
        FP["FP = pred AND NOT gt AND valid sum"]
        FN["FN = NOT pred AND gt AND valid sum"]
        TN["TN = NOT pred AND NOT gt AND valid sum"]

        PREC["Precision = TP / TP + FP"]
        REC["Recall = TP / TP + FN"]
        F1["F-measure = 2 P R / P + R"]
        PWC["PWC = FP + FN / TP+FP+FN+TN times 100"]

        TP --> PREC
        FP --> PREC
        TP --> REC
        FN --> REC
        PREC --> F1
        REC --> F1
        FP --> PWC
        FN --> PWC
        TP --> PWC
        TN --> PWC
    end

    style METRICS fill:#0f3460,stroke:#533483,color:#e0e0ff
```

---

## 11. Checkpointing & Auto-Resume Pipeline

> Full training state is persisted to Google Drive every epoch (via atomic tmp-copy to avoid partial writes). On Colab runtime reset, the notebook auto-resumes from the last saved checkpoint — including optimizer, scheduler, scaler, RNG states, and history.

```mermaid
flowchart TB
    subgraph CKPT["Checkpointing Pipeline"]
        direction TB
        RUN_TAG["RUN_TAG = arch_sampling<br/>e.g. cnn_mamba_balanced"]

        subgraph SAVE["save_ckpt(path, epoch)"]
            BUNDLE["Bundle:<br/>model state_dict<br/>optimizer state_dict<br/>scheduler state_dict<br/>scaler state_dict<br/>best_val_f1, no_improve<br/>history dict<br/>RNG states (torch, cuda, numpy, python)"]
            TMP["Write to path.tmp"]
            COPY["shutil.copy(tmp, path)"]
            RM["os.remove(tmp)"]
            BUNDLE --> TMP --> COPY --> RM
        end

        subgraph PATHS["Checkpoint Paths on Drive"]
            LAST["CKPT_LAST: last__cnn_mamba_balanced.pt<br/>saved EVERY epoch"]
            BEST["CKPT_BEST: best__cnn_mamba_balanced.pt<br/>saved when val F1 improves"]
            SNAP["Epoch snapshots: epoch_NNN__cnn_mamba_balanced.pt<br/>saved every 10 epochs"]
        end

        subgraph RESUME["Auto-Resume on Restart"]
            CHECK["os.path.exists(CKPT_LAST)?"]
            LOAD["Load all state dicts<br/>+ restore RNG states<br/>+ set start_epoch = ckpt_epoch + 1"]
            FRESH["Start from epoch 1"]
            CHECK -->|"Yes"| LOAD
            CHECK -->|"No"| FRESH
        end

        RUN_TAG --> PATHS
        SAVE --> PATHS
        PATHS --> RESUME
    end

    style CKPT fill:#1a1a2e,stroke:#533483,color:#e0e0ff
    style SAVE fill:#0f3460,stroke:#1a1a2e,color:#e0e0ff
    style PATHS fill:#0f3460,stroke:#1a1a2e,color:#e0e0ff
    style RESUME fill:#0f3460,stroke:#1a1a2e,color:#e0e0ff
```

> [!NOTE]
> **Atomic writes** prevent corrupted checkpoints on Colab resets: the bundle is first written to a `.tmp` file, then `shutil.copy`'d to the final path, and the tmp file is removed.
>
> **RNG state restoration** ensures the exact same training trajectory after a resume — `torch`, `cuda`, `numpy`, and Python `random` states are all saved and restored.

---

## 12. Per-Scene Evaluation Pipeline

> After training, every annotated frame from each scene is evaluated (not just the validation subsample). Inference speed is also benchmarked.

```mermaid
flowchart TB
    subgraph EVAL["Full Quantitative Evaluation"]
        direction TB
        BEST_MODEL["Load best model weights"]

        subgraph PER_SCENE["Per-Scene Loop (5 scenes)"]
            DS_ALL["CDDnetThermal<br/>val_frac=1.0, split=val<br/>ALL annotated frames"]
            DL_ALL["DataLoader batch_size=8"]
            SM["SegMetrics per scene"]
            OVERALL["SegMetrics overall<br/>accumulates across all scenes"]
            DS_ALL --> DL_ALL --> SM
            SM --> OVERALL
        end

        subgraph INFER_LOOP["Per-Batch Inference"]
            AMP_INF["torch.autocast cuda"]
            FWD["model(x)"]
            SIGMOID_TH["sigmoid ge 0.9<br/>binary prediction"]
            UPDATE["sm.update(pred, label, valid)<br/>overall.update(pred, label, valid)"]
            AMP_INF --> FWD --> SIGMOID_TH --> UPDATE
        end

        subgraph REPORT["Results Table"]
            DF["pandas DataFrame<br/>Scene | Precision | Recall | F-measure | PWC<br/>+ OVERALL row"]
        end

        BEST_MODEL --> PER_SCENE
        PER_SCENE --> INFER_LOOP
        INFER_LOOP --> REPORT
    end

    subgraph SPEED["Inference Speed Benchmark"]
        DUMMY["Random input [1,3,240,320]"]
        WARMUP["3 warmup forward passes"]
        TIMED["20 timed forward passes<br/>with cuda.synchronize"]
        FPS["Report: fps and ms per frame"]
        DUMMY --> WARMUP --> TIMED --> FPS
    end

    EVAL --> SPEED

    style EVAL fill:#16213e,stroke:#e94560,color:#e0e0ff
    style PER_SCENE fill:#0f3460,stroke:#16213e,color:#e0e0ff
    style INFER_LOOP fill:#0f3460,stroke:#16213e,color:#e0e0ff
    style REPORT fill:#0f3460,stroke:#16213e,color:#e0e0ff
    style SPEED fill:#1a1a2e,stroke:#e94560,color:#e0e0ff
```

---

## 13. Explainability — SHAP (Expected Gradients) + LIME (SLIC)

> Both methods explain ONE scalar target: the **ROI-mean sigmoid foreground probability** of a frame — the same quantity the segmentation metrics are computed on.

### 13a. SHAP — Expected Gradients (Erion et al. 2021)

> Per-pixel attribution computed directly (not via the `shap` library) to control batch size and memory. Uses gradient checkpointing on SS2D scans to avoid OOM on T4 GPUs.

```mermaid
flowchart TB
    subgraph SHAP["SHAP Expected Gradients"]
        direction TB

        subgraph WRAP["ROIProbNet Wrapper"]
            NET["MSCPNet model"]
            SIG["sigmoid"]
            ROI_MASK["ROI mask [1,1,H,W]"]
            MEAN["ROI-masked spatial mean"]
            SCALAR["output: one scalar [B,1]<br/>ROI-mean fg probability"]
            NET --> SIG --> ROI_MASK --> MEAN --> SCALAR
        end

        subgraph BG["Background References"]
            BG_SEL["1 training frame per scene<br/>(5 diverse references)"]
            BG_TENSOR["background [5,3,H,W]"]
            BG_SEL --> BG_TENSOR
        end

        subgraph EG_LOOP["Expected Gradients (N=200 samples)"]
            RAND_REF["Random ref from background"]
            RAND_T["t ~ Uniform(0,1)"]
            INTERP["x_i = t * x + (1-t) * bg"]
            GRAD["grad = autograd.grad(wrap(x_i), x_i)"]
            PHI["phi += (x - bg) * grad"]
            CKPT_ON["SS2D checkpointing ON<br/>memory-safe backward"]
            RAND_REF --> RAND_T --> INTERP --> GRAD --> PHI
            CKPT_ON --> GRAD
        end

        subgraph OUTPUT["SHAP Output"]
            PHI_MAP["phi_map = mean(phi)[H,W]<br/>sum over 3 channels"]
            VERIFY["Additivity check:<br/>sum(phi) approx f(x) - E[f(bg)]"]
            PHI_MAP --> VERIFY
        end

        WRAP --> EG_LOOP
        BG --> EG_LOOP
        EG_LOOP --> OUTPUT
    end

    style SHAP fill:#533483,stroke:#e94560,color:#e0e0ff
    style WRAP fill:#1a1a2e,stroke:#533483,color:#e0e0ff
    style BG fill:#1a1a2e,stroke:#533483,color:#e0e0ff
    style EG_LOOP fill:#1a1a2e,stroke:#533483,color:#e0e0ff
    style OUTPUT fill:#1a1a2e,stroke:#533483,color:#e0e0ff
```

### 13b. LIME — SLIC Superpixels

> Superpixel-based perturbation explainer. SLIC segmentation is chosen over LIME's default quickshift because thermal frames have low contrast.

```mermaid
flowchart TB
    subgraph LIME["LIME Explainer"]
        direction TB

        subgraph SEG["SLIC Segmentation"]
            SLIC_PARAMS["SegmentationAlgorithm: slic<br/>n_segments=80<br/>compactness=10<br/>sigma=3"]
        end

        subgraph CLF["Classifier Function"]
            INPUT_IMGS["Perturbed images (N,H,W,3)"]
            ROI_PROB["roi_mean_prob via model"]
            PROBS["[N,2] columns: (1-p, p)<br/>bg prob and fg prob"]
            INPUT_IMGS --> ROI_PROB --> PROBS
        end

        subgraph EXPLAIN["explain_instance"]
            FRAME["Input frame [0,1] float64"]
            PARAMS["labels=(1,)<br/>num_features=12<br/>num_samples=400<br/>batch_size=8"]
            EXP_OBJ["Explanation object"]
            FRAME --> PARAMS --> EXP_OBJ
        end

        subgraph VIZ["LIME Output"]
            IMG_MASK["get_image_and_mask(1)<br/>positive_only=False<br/>num_features=12"]
            LIME_IMG["LIME overlay image<br/>green=positive, red=negative"]
            IMG_MASK --> LIME_IMG
        end

        SEG --> EXPLAIN
        CLF --> EXPLAIN
        EXPLAIN --> VIZ
    end

    style LIME fill:#e94560,stroke:#533483,color:#ffffff
    style SEG fill:#1a1a2e,stroke:#e94560,color:#e0e0ff
    style CLF fill:#1a1a2e,stroke:#e94560,color:#e0e0ff
    style EXPLAIN fill:#1a1a2e,stroke:#e94560,color:#e0e0ff
    style VIZ fill:#1a1a2e,stroke:#e94560,color:#e0e0ff
```

> [!IMPORTANT]
> **XAI Target**: Both SHAP and LIME explain `ROI_mean_prob = mean(sigmoid(model(x)) * ROI_mask)` — a single scalar per image. SHAP gives per-pixel attribution; LIME gives per-superpixel attribution.
>
> **Output grid**: Each scene shows 5 panels — **Input | Ground Truth | Prediction | SHAP heatmap | LIME overlay**.

---

## 14. Output Artifacts

> Final model weights and per-scene evaluation results are saved locally and copied to Google Drive for persistence.

```mermaid
flowchart LR
    subgraph OUTPUTS["Output Artifacts"]
        direction TB
        MODEL_FILE["outputs/mscp_mamba_balanced_cdnet2014.pth<br/>state_dict + config + arch"]
        CSV_FILE["outputs/cdnet2014_thermal_results_mamba_balanced.csv<br/>Scene | Precision | Recall | F-measure | PWC"]
        EXPL_IMG["outputs/explainability_shap_lime.png<br/>5-scene x 5-panel grid (150 dpi)"]
    end

    subgraph DRIVE["Google Drive Persistence"]
        DRIVE_PTH["mscp_mamba_balanced/mscp_mamba_balanced_cdnet2014.pth"]
        DRIVE_CSV["mscp_mamba_balanced/cdnet2014_thermal_results_mamba_balanced.csv"]
        DRIVE_PNG["mscp_mamba_balanced/explainability_shap_lime.png"]
    end

    MODEL_FILE -->|"shutil.copy"| DRIVE_PTH
    CSV_FILE -->|"shutil.copy"| DRIVE_CSV
    EXPL_IMG -->|"shutil.copy"| DRIVE_PNG

    style OUTPUTS fill:#0f3460,stroke:#533483,color:#e0e0ff
    style DRIVE fill:#16213e,stroke:#e94560,color:#e0e0ff
```
