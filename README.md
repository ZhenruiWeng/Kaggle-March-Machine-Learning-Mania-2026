# March Machine Learning Mania 2026 — Gold Medal Solution

**Kaggle Gold Medal (individual entry) — Rank 11 / 3,482 teams (top 0.3%)**

Solution for [March Machine Learning Mania 2026](https://www.kaggle.com/competitions/march-machine-learning-mania-2026), the annual NCAA March Madness prediction competition. The task is to predict the probability that Team1 beats Team2 for every possible tournament matchup, evaluated by **log loss**.

## Approach

```
Load Data → Elo (MOV-adjusted) → Advanced Stats → Massey Ordinals → Form + SOS → 19 Features → Ensemble → Seed Blend → Submission
```

### Feature Engineering (19 matchup features)

All features are computed as differences (Team1 − Team2) so the model learns symmetric matchups:

| # | Feature | Source |
|---|---------|--------|
| 1 | `elo_diff` | Margin-of-victory adjusted Elo (FiveThirtyEight-style K-factor, 75% inter-season carry-over) |
| 2 | `seed_diff` | Tournament seed |
| 3 | `ordinal_diff` | Massey Ordinals averaged across 100+ ranking systems (last day per system per season) |
| 4 | `form_diff` | Win rate over the last 10 regular-season games |
| 5 | `sos_diff` | Strength of schedule — average Elo of opponents faced |
| 6–19 | Box-score stats | Dean Oliver offensive/defensive efficiency (possession-based), shooting splits, rebound rates, AST/TO ratio, win %, average margin of victory |

### Ensemble

| Model | Base Weight | Notes |
|-------|-------------|-------|
| XGBoost | 40% | 500 estimators, depth 5 |
| LightGBM | 40% | 500 estimators, depth 5 |
| Logistic Regression | 20% | Scaled features |

The final submission improves on the fixed weights with:

1. **Temporal cross-validation** — train on seasons < N, validate on season N. Strictly no future leakage.
2. **Optimized ensemble weights** — weights tuned with `scipy.optimize` instead of fixed 40/40/20.
3. **Seed-baseline blend** — model predictions blended with the historical seed-matchup probability baseline (`ALPHA = 0.85`), injecting domain knowledge that regularizes overconfident upset predictions.
4. **Prediction clipping** — extreme probabilities are clipped to avoid catastrophic log-loss penalties.

## Results

- **Final rank: 11 / 3,482 (top 0.3%) — Gold Medal**
- Trained on tournament games from 2003+, validated with per-season temporal CV (Brier score + log loss)

## Repository Structure

```
├── notebook/
│   └── march-mania-2026-xgb-lgb-ensemble-19-features.ipynb   # Full solution
├── requirements.txt
└── README.md
```

## Reproduction

1. Join the competition and download the data from the [competition page](https://www.kaggle.com/competitions/march-machine-learning-mania-2026/data)
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the notebook end-to-end on the competition data — the notebook auto-detects the Kaggle input directory and falls back to a local `data/` folder

## Author

**Zhenrui Weng (翁振瑞)** — [Kaggle](https://www.kaggle.com/zhenruiweng) · [GitHub](https://github.com/ZhenruiWeng)
