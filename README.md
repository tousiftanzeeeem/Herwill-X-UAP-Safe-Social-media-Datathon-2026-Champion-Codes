# 🏆 Team FatWiz — HerWILL × UAP Safe Social Media Datathon 2026

### **Champion — Onsite Final Round**
**HerWILL × University of Asia Pacific**
*Explicit & Subtle Toxicity Detection in Bangla, English and Banglish*

---

<div align="center">

![Award](https://img.shields.io/badge/Award-Champion-gold?style=for-the-badge)
![Event](https://img.shields.io/badge/Event-HerWILL%20%C3%97%20UAP%20Datathon%202026-blue?style=for-the-badge)
![Score](https://img.shields.io/badge/Macro%20F1-0.77514-brightgreen?style=for-the-badge)
![Rank](https://img.shields.io/badge/Leaderboard-1st%20of%2055-purple?style=for-the-badge)
![Language](https://img.shields.io/badge/Languages-Bangla%20%7C%20English%20%7C%20Banglish-orange?style=for-the-badge)

</div>

<div align="center">

### 🥇 [**View the Competition Leaderboard on Kaggle**](https://www.kaggle.com/competitions/HerWILL-Safe-Social-Media-Datathon-2026/leaderboard)

[![Kaggle](https://img.shields.io/badge/Kaggle-Leaderboard-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/competitions/HerWILL-Safe-Social-Media-Datathon-2026/leaderboard)
[![Competition](https://img.shields.io/badge/Competition-Overview-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/competitions/HerWILL-Safe-Social-Media-Datathon-2026)

</div>

---

## 🧠 Overview

**Team FatWiz** won the HerWILL × UAP Safe Social Media Datathon 2026 with a
**six-encoder ensemble** that separates *explicitly toxic*, *subtly toxic* and
*non-toxic* social-media comments across three language registers.

Every design decision in this system traces back to a measurement on the corpus —
not to intuition about what should work.

| | |
|---|---|
| **Private leaderboard** | **0.77514** macro F1 — 1st of 55 teams |
| **Margin over 2nd** | +0.0197 (larger than the entire 2nd→4th spread) |
| **Majority-class baseline** | 0.22129 |
| **Corpus** | 47,817 train / 11,955 test, raw text, no metadata |
| **Ensemble size** | 6 encoders, ~1.6B parameters total |

---

## 🔍 What the Data Actually Said

The brief described a Bangla/Banglish corpus. Measuring it changed the whole approach.

| Finding | Measurement | Consequence |
|---|---|---|
| **Majority English** | 54.3% English, 42.1% Bangla script, 3.6% code-mixed | A Bangla-only model would address 40% of the problem |
| **No distribution shift** | Train and test language mix agree to **0.4pp** | Validation strategy is trustworthy |
| **One boundary dominates** | **71%** of all errors are explicit ↔ subtle | The real problem is not toxic-vs-clean |
| **Annotation ceiling** | 63 identical texts carry conflicting labels | Part of that boundary is unlearnable |
| **Rare class is hardest** | Non-toxic is 12.3% of rows but ⅓ of macro F1 | Needs a decision-rule fix, not a bigger model |

---

## ✨ Approach

| Stage | What we did | Why |
|---|---|---|
| **Preprocessing** | NFC-normalise (16.2% of rows), keep emoji / repeated punctuation / censored spellings | Tone *is* the label for subtle toxicity |
| **Language-split training** | Bangla family trains on Bangla rows, English family on English rows | Each route learns its own label prior (58/34/8 vs 42/42/16) |
| **Six backbones** | 3 Bangla-specialised + 3 multilingual / domain-adapted | Models whose errors differ, not models that are individually best |
| **Equal-weight soft vote** | Mean of six softmax outputs, all six score every row | Fitted blend weights did not beat equal weights on held-out folds |
| **Per-class decision priors** | `[1.0, 0.02239, 3.54813]` applied before `argmax` | Targets macro F1 directly; corrects the under-predicted rare class |

**Ensemble members**

| Route | Backbones |
|---|---|
| 🔴 **Bangla** | BanglaBERT · BanglishBERT · BanglaBERT-large |
| 🔵 **English** | HateBERT · Twitter-XLM-R · XLM-R-large |

All six share the same classification head: `concat[CLS, mean-pool] → MLP 4096–2048–1024 → 3 logits`.

---

## 📂 Repository Contents

```
notebooks/
  01_train_bangla_route.ipynb       BanglaBERT · BanglishBERT · BanglaBERT-large
  02_train_english_route.ipynb      HateBERT · Twitter-XLM-R · XLM-R-large
  03_ensemble_weight_tuning.ipynb   temperature, blend weights, class priors, kNN — all CV-gated
  04_inference_ensemble.ipynb       production inference → submission.csv
config/
  ensemble_config.json              the exact bundle used for the winning submission
FatWiz_Onsite_Presentation.pdf      slides from the onsite final round
```

> **Note:** the competition dataset is **not** included. Per the competition rules, the
> dataset may not be shared or published outside the competition.

---

## 🚀 How to Run

1. **Train** — run `01_train_bangla_route.ipynb` and `02_train_english_route.ipynb` on Kaggle.
   Each writes `{tag}_final/` checkpoint folders.
2. **Tune** — run `03_ensemble_weight_tuning.ipynb` with those checkpoints attached.
   It fits every post-processing stage on validation, cross-validates each stage's gain, and
   **switches off any stage with a negative held-out gain**. Writes `ensemble_bundle/`.
3. **Predict** — run `04_inference_ensemble.ipynb` with the bundle and checkpoints attached.
   Runs **fully offline** (`HF_HUB_OFFLINE=1`) and writes `submission.csv`.

Point the path variables in each notebook's first config cell at your own inputs.

---

## 🛠️ Tech Stack

- **Encoders:** BanglaBERT · BanglishBERT · BanglaBERT-large · HateBERT · Twitter-XLM-R · XLM-R-large
- **Frameworks:** PyTorch · Hugging Face Transformers · scikit-learn
- **Training:** 5-fold `StratifiedKFold` (seed 42) · layer-wise LR decay · AMP (bf16/fp16)
- **Bangla normalisation:** `csebuetnlp/normalizer`
- **Environment:** Kaggle T4 ×2, fully offline inference

---

## 📐 Methodology Notes

- **Frozen folds.** One `StratifiedKFold(5, seed=42)` split is the alignment contract across
  every backbone, so out-of-fold matrices stack without leakage.
- **Nested cross-validation.** Blend weights and class priors are fitted on four folds and
  scored on the fifth. A stage is kept only if its *held-out* gain is positive — global
  calibration's apparent +0.0044 shrank to +0.0024 under this test.
- **Stages we switched off.** Fitted blend weights, kNN soft mixing, temperature scaling and
  the exact-match override all failed to transfer. Only the class priors survived.
- **ZWJ/ZWNJ preserved.** `U+200C` / `U+200D` are never stripped — removing them corrupts
  Bengali orthography and emoji sequences.

---

## 📜 Documents

- 📊 **[Onsite Presentation](FatWiz_Onsite_Presentation.pdf)** — the deck presented in the
  final round: data measurements, the three findings, the final system, and the approaches
  that did not work

---

## 👥 Team FatWiz

| Member | GitHub |
|---|---|
| **Tanzim Tousif** | [tousiftanzeeeem](https://github.com/tousiftanzeeeem) |
| **Mohammed Afham Adian** | [AfhamAdian](https://github.com/AfhamAdian) |
| **Arafat Rahman** | [arafatrahman216](https://github.com/arafatrahman216) |

<!-- Add remaining members here, e.g.:
| **Name** | [handle](https://github.com/handle) |
-->

---

## 🏅 Acknowledgement

Our thanks to **HerWILL** and the **University of Asia Pacific** for hosting a datathon on a
problem that matters — making social platforms safer for Bangla and English speakers alike,
including the subtle toxicity that keyword filters never catch.

---

<div align="center">

**Team FatWiz** · Champion, HerWILL × UAP Safe Social Media Datathon 2026

</div>
