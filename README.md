# Freight Rate Prediction & Multi-Seed Evaluation Pipeline

This repository provides an end-to-end, reproducible machine learning pipeline to predict freight rates (`posted_rate`) for commercial trucking loads. It implements rigorous data checks, feature engineering with zero validation leakage, and evaluates **Ridge Regression**, **LightGBM**, and a custom **PyTorch Multi-Layer Perceptron (MLP)** across multiple random seeds and split protocols.

---

## 1. Objective & Challenge Overview

* **Objective**: Predict `posted_rate` for commercial freight loads.
* **Development Data**: `assesiment/data/train-test.csv` (48,000 loads from January 1, 2025 to October 31, 2025).
* **Final Validation Set**: `assesiment/data/validation.csv` (12,000 loads requiring final predictions).
* **Deliverables**:
  1. `validation_predictions.csv`: Filled predictions for all 12,000 loads (`load_id,predicted_rate`).
  2. `december_predictions.csv` & `results/candidate_december.png`: Predicted rate curves for 31 December benchmark shipments.
  3. Multi-seed evaluation results, visualizations, and pipeline source code.

---

## 2. Data Exploration & Cleaning

A comprehensive data quality inspection was conducted without silently discarding rows:

* **Volume & Timeline**: 48,000 loads spanning 10 months (`2025-01-01` to `2025-10-31`).
* **Missing Value Imputation**:
  * `weight`: 300 missing values (0.63%).
  * `market_index`: 374 missing values (0.78%).
  * *Handling*: Imputed strictly using the **median of the training split**. No validation or test set statistics were used.
* **Geographic & Distance Audit**:
  * Haversine distance is calculated between pickup coordinates $(\phi_1, \lambda_1)$ and delivery coordinates $(\phi_2, \lambda_2)$:
    $$d_{\text{hav}} = 2 R \arcsin\left(\sqrt{\sin^2\left(\frac{\Delta \phi}{2}\right) + \cos(\phi_1)\cos(\phi_2)\sin^2\left(\frac{\Delta \lambda}{2}\right)}\right)$$
    where $R = 3,958.8\text{ miles}$, $\Delta \phi = \phi_2 - \phi_1$, and $\Delta \lambda = \lambda_2 - \lambda_1$.
  * Correlation with reported distance: $r = 0.9995$ with a median absolute difference of $144.2\text{ miles}$.
  * **Coordinate Anomaly**: In 7.77% of rows (3,731 loads), the relative discrepancy exceeded 20%:
    $$\Delta_{\text{rel}} = \frac{|d_{\text{hav}} - d_{\text{reported}}|}{\max(d_{\text{reported}}, 10^{-6})} > 0.20$$
    For example, Richmond to Baltimore coordinates indicate $\approx 220\text{ miles}$ whereas reported road distance is $274\text{ miles}$.
  * *Handling*: Coordinates were identified as perturbed/synthetic. Rather than dropping valid loads, rows were **flagged**, and both reported $d_{\text{reported}}$ and calculated $d_{\text{hav}}$ were preserved as complementary features.
* **City Consistency**: All 64 pickup and 64 delivery cities map to exactly one persistent latitude/longitude pair with zero mismatches.
* **Target (`posted_rate`) Distribution**:
  * Positively skewed (skewness = 1.90), ranging from \$57.22 to \$25,533.00 (Mean: \$2,373.98, Median: \$2,030.76).
  * Outlier detection via the interquartile range:
    $$\text{Boundaries} = [Q_1 - 1.5 \cdot \text{IQR}, \quad Q_3 + 1.5 \cdot \text{IQR}]$$
    Identified 260 loads (0.54%) as outliers. All rows were retained as legitimate heavy or long-haul loads.
* **Leakage Verification**:
  * Standalone `quote_signal` correlation with posted rate is low ($r = -0.0399$), but the interaction term correlates strongly ($r = 0.8987$):
    $$\text{corr}(\text{quote\_signal} \times \text{distance}, \; \text{posted\_rate}) = 0.8987$$
  * A configuration flag `USE_QUOTE_SIGNAL` (and CLI `--no-quote-signal`) allows toggling this signal if unavailable at prediction time.

---

## 3. Feature Engineering (Fit on Train Only)

All transformations and encoders are fit strictly on training splits and applied to validation/test sets:

1. **Numeric Features**:
   * Distance metrics: reported distance and Haversine distance $d_{\text{hav}}$.
   * Heavy right-tail normalization:
     $$x' = \log(1 + \max(x, 0)) \quad \text{for } x \in \{\text{weight}, \text{distance}\}$$
   * Interactions and ratios:
     $$\text{weight\_per\_distance} = \frac{\text{weight}}{\max(\text{distance}, 1.0)}, \quad \text{dist\_x\_market} = \text{distance} \times \text{market\_index}$$
     $$\text{quote\_x\_dist} = \text{quote\_signal} \times \text{distance}$$
   * Scaled with `StandardScaler` (fit strictly on the training set):
     $$z = \frac{x - \mu_{\text{train}}}{\sigma_{\text{train}}}$$
2. **Direction & Geography**:
   * Vectorized forward compass bearing (initial azimuth $\theta \in [0^\circ, 360^\circ)$):
     $$\theta = \left(\operatorname{atan2}\left(\sin(\Delta \lambda)\cos(\phi_2), \; \cos(\phi_1)\sin(\phi_2) - \sin(\phi_1)\cos(\phi_2)\cos(\Delta \lambda)\right) \cdot \frac{180^\circ}{\pi} + 360^\circ\right) \bmod 360^\circ$$
3. **Temporal Features**:
   * Discrete Calendar Indicators: `dayofweek`, `month`, `weekofyear`, `dayofmonth`.
4. **Categorical Encodings & Entity Embeddings**:
   * Categorical inputs: `equipment`, `pickup`, `delivery`, and `lane` ($\text{pickup} \to \text{delivery}$).
   * **For LightGBM**: Out-of-fold smoothed lane target encoding inside train ($K=5$ folds, smoothing parameter $m=10.0$):
     $$\hat{y}_{\text{lane}}(c) = \frac{\sum_{i \in \text{lane } c} y_i + m \cdot \bar{y}_{\text{train}}}{n_c + m}$$
     Validation rows map strictly using the global training statistics; unseen lanes fall back to $\bar{y}_{\text{train}}$.
   * **For PyTorch MLP**: Learned entity embeddings for all categoricals with embedding dimension:
     $$d_{\text{emb}} = \min\left(16, \; \left\lceil \frac{n_{\text{categories}}}{2} \right\rceil\right)$$
     An extra index $n_{\text{categories}}$ is reserved for unseen categories in validation.

---

## 4. Validation Strategy: Dual-Protocol Evaluation

To thoroughly test both split variance and real-world temporal generalization, every model is evaluated across **5 random seeds (0, 1, 2, 3, 4)** under two distinct protocols:

* **Protocol A (Random 80/20 Split)**: Shuffled 80% train / 20% validation per seed. Evaluates sensitivity to data partitioning.
* **Protocol B (Time-Based 80/20 Split)**: Earliest 80% of unique dates (Jan 1 – Aug 31) for train, latest 20% of dates (Sep 1 – Oct 31) for validation. Deterministic date split where $t_{\text{val}} > t_{\text{train}}$ simulates forward-looking production deployment.

---

## 5. Model Benchmarking & Results

Three model architectures were tested:
1. **Ridge Baseline**: Linear regression on scaled numerics and equipment one-hot dummies.
2. **LightGBM Regressor**: Gradient boosted trees with fixed sensible defaults (2,000 estimators, early stopping 50 rounds, lr=0.05, num_leaves=31, subsample=0.8).
3. **PyTorch MLP**: Deep neural network with categorical embeddings, standardized numerics, SmoothL1 (Huber) loss, AdamW ($\text{lr}=10^{-3}$, $\text{weight\_decay}=10^{-4}$), ReduceLROnPlateau, and early stopping.

### Loss Function & Target Standardization (PyTorch MLP)
The MLP models standardized target variables ($z_y$) using Huber loss:
$$\mathcal{L}_{\beta}(z, \hat{z}) = \begin{cases} \frac{1}{2}(z - \hat{z})^2, & \text{if } |z - \hat{z}| < 1 \\ |z - \hat{z}| - \frac{1}{2}, & \text{otherwise} \end{cases}$$

When modeling rate-per-mile ($y_{\text{rpm}} = \frac{\text{posted\_rate}}{\text{distance}}$), predictions are inverted back to original dollars:
$$\widehat{\text{posted\_rate}} = (\hat{z} \cdot \sigma_{\text{train}} + \mu_{\text{train}}) \times \text{distance}$$

### Evaluation Metrics
All metrics are evaluated in original dollar units:
$$\text{MAE} = \frac{1}{N}\sum_{i=1}^N |y_i - \hat{y}_i|$$
$$\text{RMSE} = \sqrt{\frac{1}{N}\sum_{i=1}^N (y_i - \hat{y}_i)^2}$$
$$\text{MAPE} = \frac{100\%}{N}\sum_{i=1}^N \frac{|y_i - \hat{y}_i|}{\max(y_i, 10^{-4})}$$
$$\text{sMAPE} = \frac{100\%}{N}\sum_{i=1}^N \frac{2|y_i - \hat{y}_i|}{|y_i| + |\hat{y}_i| + 10^{-4}}$$
$$R^2 = 1 - \frac{\sum_{i=1}^N (y_i - \hat{y}_i)^2}{\sum_{i=1}^N (y_i - \bar{y})^2}$$

### Multi-Seed Summary Table (5 Seeds: 0, 1, 2, 3, 4)

| Protocol | Model | MAE ($) (Mean ± Std) | MAE [Min, Max] | RMSE ($) (Mean ± Std) | MAPE (%) | sMAPE (%) | $R^2$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Protocol A** (Random) | **MLP** | **$104.89 ± $5.52** | [$96.94, $111.44] | **$584.76 ± $32.24** | **5.30%** | **4.51%** | **0.845** |
| Protocol A (Random) | LightGBM | $129.08 ± $4.92 | [$121.61, $133.68] | $593.95 ± $30.91 | 6.91% | 5.79% | 0.840 |
| Protocol A (Random) | Ridge | $148.03 ± $5.05 | [$140.82, $154.68] | $596.72 ± $31.21 | 8.45% | 7.11% | 0.839 |
| **Protocol B** (Time-based) | **MLP** | **$125.47 ± $8.38** | [$119.10, $139.50] | **$633.49 ± $4.10** | **6.26%** | **5.34%** | **0.828** |
| Protocol B (Time-based) | LightGBM | $151.56 ± $0.79 | [$150.82, $152.85] | $645.89 ± $0.42 | 7.10% | 6.13% | 0.821 |
| Protocol B (Time-based) | Ridge | $175.69 ± $0.00 | [$175.69, $175.69] | $649.72 ± $0.00 | 9.29% | 8.63% | 0.819 |

---

## 6. Result Visualizations

### MAE Across Seeds & Protocols
Comparison of Ridge, LightGBM, and MLP across 5 seeds:
![MAE Per Seed](results/mae_per_seed.png)

### MLP Training & Validation Loss
Huber loss convergence over epochs with ReduceLROnPlateau and early stopping restoration:
![MLP Training Curve](results/training_curve_mlp.png)

### LightGBM Feature Importance
Top gain features driving rate predictions:
![Feature Importance](results/feature_importance.png)

### Predicted vs. Actual Rates
Validation actuals vs. model predictions:
![Predicted vs Actual](results/predicted_vs_actual.png)

### December Fixed Predictions Chart
Output of `assesiment/score.py` showing daily predicted rates for December shipments:
![Candidate December Chart](results/candidate_december.png)

---

## 7. Model Selected for Final Validation

**Decision: Ensemble of PyTorch MLP + LightGBM (across 5 seeds)**

* **Why MLP Wins on Accuracy**:
  * Achieved the lowest error on the forward-looking Protocol B test set: **$125.47 MAE** vs. LightGBM's **$151.56 MAE** (a \$26.09 improvement) and highest $R^2$ (**0.828**).
  * Continuous entity embeddings smoothly map the 64 pickup and delivery cities and lane network interactions without discrete axis-aligned tree constraints.
* **Why LightGBM Complements the MLP**:
  * LightGBM exhibits near-zero seed variance (std = \$0.79 vs. MLP's \$8.38), providing structural regularization.
* **Ensemble Synergies**:
  * Averaging predictions from the 5-seed MLP and 5-seed LightGBM produces variance reduction and lower generalization error on unseen test dates.
* **Final Submissions Generated**:
  * `validation_predictions.csv`: 12,000 loads predicted.
  * `december_predictions.csv`: 31 December loads predicted.
  * Verified and passed with `assesiment/score.py`:
    ```
    Validated 12,000 final predictions.
    Validated 31 fixed December predictions.
    Created chart: results/candidate_december.png
    ```

---

## 8. How to Run

### Install Dependencies
```bash
pip install torch lightgbm scikit-learn pandas numpy matplotlib
```

### Run the 5-Seed Experiment
```bash
# Full 5-seed run across all models (Ridge, LightGBM, MLP)
python run_experiment.py

# Quick smoke run (e.g. 2,000 rows, 5 epochs)
python run_experiment.py --sample-rows 2000 --mlp-max-epochs 5

# Without quote_signal
python run_experiment.py --no-quote-signal
```

### Generate Final Submission Predictions & Scorer Chart
```bash
python predict_submission.py --model ensemble
python assesiment/score.py --predictions validation_predictions.csv --december-predictions december_predictions.csv --output-dir results
```
