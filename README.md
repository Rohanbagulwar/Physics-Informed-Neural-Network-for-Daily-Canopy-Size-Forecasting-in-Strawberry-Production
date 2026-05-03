# 🍓 PINN — Strawberry Canopy Forecasting

**Physics-Informed Neural Network for Daily Canopy Size Forecasting in Strawberry Production Using Fused Weather and Image Embeddings**

> Rohan Bagulwar¹ · Won Suk Daniel Lee² · Shinsuke Agehara³ · Hongyoung Jeon⁴ · Heping Zhu⁴
>
> ¹ Herbert Wertheim College of Engineering, University of Florida
> ² Department of Agricultural and Biological Engineering, IFAS, University of Florida
> ³ Gulf Coast Research and Education Center, University of Florida
> ⁴ USDA ARS Application Technology Research Unit, Wooster, OH

---

## Overview

This repository contains the research pipeline, trained models, and results for a Physics-Informed Neural Network (PINN) that forecasts daily strawberry canopy size from low-cost field camera images and weather station data. The system was deployed at Citra, FL during the 2025 growing season using a solar-powered, fully off-grid Raspberry Pi 4 imaging station.

The core idea is simple: a logistic growth curve captures the smooth biological trend (driven by temperature), while an LSTM residual encoder learns the day-to-day deviations caused by weather events like cold snaps, rainfall, and high vapour pressure deficit. Neither component works as well alone — together they outperform both a physics-only and a naive persistence baseline.

---

## Field Setup

| Component | Specification |
|---|---|
| Processing unit | Raspberry Pi 4 Model B |
| Camera | Raspberry Pi Camera Module 3 (Sony IMX708, 11.9 MP) |
| Resolution | 4608 × 2592 pixels |
| Capture interval | Every 15 minutes, 24 hours/day |
| Power | Monocrystalline solar panels + 100 Ah AGM battery (fully off-grid) |
| Deployment site | Citra, FL — open strawberry field |
| Observation period | 38 consecutive days, 2025 growing season |
| Varieties monitored | Brilliance FL · Medallion |

The camera was mounted on a centre-row stake above the strawberry beds. Each image captured multiple plants on raised beds over black plastic mulch. Individual plant regions were manually cropped using fixed bounding boxes to isolate each plant across all 38 days.

---

## Methods

### Target Variable

Raw green pixel counts (HSV-thresholded) were used as the primary canopy measurement for the 38-day single-plant experiment. For the extended 9-plant validation, the target was switched to **Green Chromatic Coordinate (GCC = G / (R + G + B))** computed from pre-processed lighting-corrected images — a ratio-based metric that is invariant to illumination changes.

### Weather Features

Nine daily weather features were used after removing redundant columns from the original 13-variable station dataset:

| Feature | Rationale |
|---|---|
| Air temperature @ 2 m (°C) | Primary thermal driver |
| Soil temperature (°C) | Root-zone conditions |
| Relative humidity (%) | Moisture availability |
| Dew point temperature (°C) | Atmospheric moisture |
| Rainfall (in) | Water input events |
| Wind speed (mph) | Transpiration driver |
| Solar radiation (W/m²) | Light availability |
| GDD (Tbase = 7°C) | Cumulative thermal accumulation |
| VPD (kPa) | Computed from T and RH — stomatal conductance driver |

Removed from original set: `Temp @ 60cm`, `Temp @ 10m` (redundant with `Temp @ 2m`), `Tmax`, `Tmin` (already encoded in GDD), `Wind Direction` (not predictive of canopy).

### Image Embeddings

Each daily plant image was passed through a pretrained DINOv2 vision transformer. The **CLS token** (global scene summary vector) was extracted — replacing the mean-pool of all patch tokens used in earlier versions of the pipeline. A Sparse Random Projection (Johnson–Lindenstrauss) compressed the embedding to 16 dimensions without requiring any fitting on training data, eliminating the overfitting risk of PCA on small datasets.

### PINN Architecture

The model processes a 1-day lookback window through two parallel branches:

```
Input: (1, 29)  =  25-d feature vector  +  4-d plant identity embedding

┌─────────────────────────────┐    ┌─────────────────────────────────────┐
│      Physics Branch         │    │          Data Branch (LSTM)          │
│                             │    │                                      │
│  ΔGCC_phys = α·ReLU(T−T₀)  │    │  LSTM (input=29, hidden=8, 1 layer)  │
│                             │    │  → concat T_scaled → 9-d             │
│  α, T₀ learned end-to-end  │    │  → Linear(9,16) → ReLU              │
│                             │    │  → Dropout(0.3) → Linear(16,1)      │
│  Biologically correct for   │    │  Captures cold snaps, rainfall,      │
│  daily temperature window   │    │  high-VPD deviation events           │
└────────────┬────────────────┘    └──────────────┬──────────────────────┘
             │                                    │
             └──────────────┬─────────────────────┘
                            ▼
              ΔGCC_pred = ΔGCC_phys + ΔGCC_residual

At inference:  GCC[t] = GCC[t−1] + ΔGCC_pred
```

**Why delta prediction?** Predicting the daily change (ΔGCC) rather than the absolute value removes the need to learn each plant's baseline level, substantially reducing error on short windows.

**Why temperature-response physics?** The logistic S-curve (used in season-scale models) is not identifiable from a 9–38 day window — the curve appears flat. A temperature-response law `ΔGCC = α · ReLU(T − T_base)` is biologically correct at the daily timescale and is identifiable from even a handful of training observations.

### Training Strategy

| Setting | Value |
|---|---|
| Optimiser | Adam |
| Learning rate | 3 × 10⁻³ |
| Weight decay | 10⁻² |
| Gradient clip | 1.0 |
| Max epochs | 80 (single-plant: 400) |
| Early stopping | Patience = 15 epochs on validation loss |
| Batch | Pooled across all 9 plants |
| Physics penalty λ | 0.05 |
| Loss | MSE(ΔGCC_pred, ΔGCC_actual) + λ · P |

The physics penalty P activates when the model predicts canopy decline on a thermally accumulating day (T_scaled > 0), enforcing the biological monotonicity constraint during the vegetative phase.

---

## Results

### 38-Day Single-Plant Experiment (5-Day Test Window)

| Variety | MAE (px) | MAPE (%) | R² | Physics-only MAE | Naive MAE |
|---|---|---|---|---|---|
| Brilliance FL | 6,966 | 4.53 | 0.27 | 10,260 | 9,416 |
| Medallion | 7,300 | 4.11 | 0.41 | 10,880 | 9,580 |

- PINN reduced error vs physics-only baseline by **32%** (Brilliance FL) and **33%** (Medallion)
- PINN reduced error vs naive persistence by **26%** (Brilliance FL) and **28%** (Medallion)
- Best training loss: **0.0014** after 400 epochs

### 9-Plant Front Row Experiment (2-Day Test Window, GCC Target)

| Metric | Original Pipeline | Revised Pipeline |
|---|---|---|
| Target variable | Raw green pixels | GCC (illumination-normalised) |
| Mean MAPE | 15.4% | **0.30%** |
| Beats naive baseline | 1 / 9 plants | **5 / 9 plants** |
| Beats persistence | 1 / 9 plants | **4 / 9 plants** |
| Model parameters | ~21,300 | ~1,100 |
| Training windows | 5 per plant | 63 pooled |

Per-plant GCC results (revised pipeline):

| Plant | MAE (GCC) | MAPE (%) | Beats Naive | Beats Persistence |
|---|---|---|---|---|
| 11 | 0.00181 | 0.50 | ✅ | ❌ |
| 13 | 0.00148 | 0.42 | ❌ | ❌ |
| 14 | 0.00100 | 0.29 | ❌ | ❌ |
| 15 | 0.00039 | 0.11 | ✅ | ❌ |
| 21 | 0.00017 | 0.05 | ✅ | ✅ |
| 22 | 0.00082 | 0.22 | ✅ | ✅ |
| 23 | 0.00083 | 0.24 | ❌ | ❌ |
| 24 | 0.00094 | 0.27 | ❌ | ✅ |
| 25 | 0.00297 | 0.81 | ✅ | ✅ |

> **Note on Plants 13 and 14:** GCC was nearly constant across the 9-day winter window (Naive MAE ≈ 0.00024), indicating near-dormant conditions. Any model predicting a non-zero delta will appear worse than the naive mean — this is a data limitation, not a model failure.

### Learned Physics Parameters (Revised Pipeline)

| Parameter | Value | Interpretation |
|---|---|---|
| α (growth amplitude) | 0.02095 | Positive — model learned correct direction of canopy growth |
| T_base (standardised) | −0.09 std units | ≈ 14.2°C actual — within physiologically plausible range for strawberry |

---

## Key Findings

**1. The measurement artifact was the dominant error source, not the model.**
Switching from raw green pixels to GCC reduced mean MAPE from 15.4% to 0.30% — a 50× improvement. Solar radiation was negatively correlated with green pixel counts (r = −0.41 to −0.64 across 8 of 9 plants), confirming that bright days caused specular reflection artifacts in the HSV mask. GCC eliminates this because it uses channel ratios rather than absolute brightness.

**2. Physics prior direction matters more than scale.**
The learned α was always positive (canopy grows with warmth), and T_base converged to a physiologically plausible value without any supervision. The physics branch did not need to be pre-calibrated — it recovered interpretable parameters end-to-end.

**3. Pooling across plants is more important than model architecture.**
Going from 5 training windows (per-plant) to 63 (pooled) had a larger impact than any architectural change. With 5 samples, no deep learning model can generalise — the physics prior was doing all the regularisation work.

**4. LSTM residual earns its keep on weather-perturbation days.**
During a 4-day cold/low-radiation period in the Brilliance FL test window, the physics-only curve predicted continued smooth growth (GDD was still accumulating), while the LSTM residual correctly predicted a canopy plateau. Per-day error dropped from 1,842 px (physics-only) to 698 px (PINN) across those four days.

---

## Pipeline Summary

```
Field images (15-min interval)
        │
        ▼
Plant cropping (OpenCV, fixed ROI)
        │
        ├──► GCC computation from lighting-corrected images
        │         GCC = G / (R + G + B)   per pixel, averaged over crop
        │
        └──► DINOv2 CLS token extraction
                  Sparse Random Projection → 16-d embedding
        │
        ▼
Weather station (15-min) → daily aggregation → 8 vars + VPD computed
        │
        ▼
Feature concatenation: [16-d embedding | 9 weather features] = 25-d
        │
        ▼
Plant identity embedding (4-d, learnable) → LSTM input: (1, 29)
        │
        ├──► Physics branch:  ΔGCC_phys = α · ReLU(T_scaled − T_base)
        │
        └──► LSTM branch:     ΔGCC_residual
        │
        ▼
ΔGCC_pred = ΔGCC_phys + ΔGCC_residual
        │
        ▼
Loss: MSE(ΔGCC_pred, ΔGCC_actual) + 0.05 × physics_penalty
        │
        ▼
Inference: GCC[t] = GCC[t−1] + ΔGCC_pred
```

---

## Limitations

- **9–38 day window** covers only the vegetative phase. The physics prior may require re-parameterisation for flowering and fruiting phases.
- **2–5 test days** make R² unreliable as a metric. MAE and MAPE are the appropriate measures for this experiment.
- **Manual bounding boxes** introduce potential spatial drift if the camera shifts between days.
- **T_base constrained by winter data** — the model only saw 7–22°C temperatures; true biological base temperature may be lower and would be better constrained with full-season data.

---

## Outputs

| File | Description |
|---|---|
| `PINN_Paper_Complete.docx` | Full paper with Results, Discussion, Conclusion, Declarations, References |
| `PINN_Paper_With_Authors.docx` | Paper with author and affiliation block |
| `PINN_Architecture_Fig4.png` | Full PINN architecture diagram (1800×2520 px, publication-ready) |
| `Fig5_Brilliance_FL_Predictions.png` | 5-day test predictions vs baselines — Brilliance FL |
| `Fig6_Medallion_Predictions.png` | 5-day test predictions vs baselines — Medallion |
| `PINN_Complete_References.docx` | 84 curated references across 12 categories |
| `PINN_Workflow_Report.pdf` | Plain-language 7-section explanation of the full pipeline |
| `pinn_revised_pipeline.py` | Complete revised training pipeline (GCC target, pooled, early stopping) |
| `pinn_front_row_TUNED.py` | Tuned per-plant pipeline (reduced capacity, delta target, val split) |

---

## Citation

If you use this work, please cite:

```bibtex
@article{bagulwar2025pinn,
  title   = {Physics-Informed Neural Network for Daily Canopy Size Forecasting
             in Strawberry Production Using Fused Weather and Image Embeddings},
  author  = {Bagulwar, Rohan and Lee, Won Suk Daniel and Agehara, Shinsuke
             and Jeon, Hongyoung and Zhu, Heping},
  journal = {[Target Journal]},
  year    = {2025},
  note    = {Under review}
}
```

---

## Acknowledgements

This research was conducted at the University of Florida. The authors thank the Florida Strawberry Research and Education Foundation, Inc. (FSREF) and the Florida Foundation Seed Producers, Inc. (FFSP) for supporting this research. Field access and agronomic guidance were provided by the Gulf Coast Research and Education Center.

---

*Last updated: May 2026*
