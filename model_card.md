# Model Card: Toxic Comment Classification

## Model Details

* **Name**: DistilBERT Toxic Comment Classifier
* **Version**: 1.0
* **Authors**: Wajdi Hammami
* **Date**: Jan 2025
* **Framework**: PyTorch + HuggingFace Transformers
* **Baselines**: TF-IDF + Logistic Regression

---

## Intended Use

This model predicts whether a comment is toxic across 6 categories:

* `toxic`
* `severe_toxic`
* `obscene`
* `threat`
* `insult`
* `identity_hate`

⚠️ **Not intended for**:

* Legal, medical, or safety-critical decision-making.
* Automated moderation without human oversight.

✅ **Intended for**:

* Research and educational purposes.
* Demonstrating end-to-end ML engineering, deployment, and Responsible AI practices.

---

## Training Data

* **Dataset**: Jigsaw Toxic Comment Classification Challenge (Wikipedia comments).
* **Languages**: English only.
* **Size**: 159k comments.
* **Preprocessing**: lowercasing, punctuation normalization, whitespace cleanup.

---

## Evaluation Data

* Train/validation (80/10).
* Evaluation metrics: Macro-F1, per-class F1, PR-AUC.

---

## Performance

### Baseline (TF-IDF + Logistic Regression)

| Class         | Val F1   | Test F1  | Test PR-AUC |
| ------------- | -------- | -------- | ----------- |
| Toxic         | 0.79     | 0.68     | 0.62        |
| Severe toxic  | 0.35     | 0.39     | 0.39        |
| Obscene       | 0.79     | 0.70     | 0.74        |
| Threat        | 0.48     | 0.37     | 0.55        |
| Insult        | 0.69     | 0.64     | 0.74        |
| Identity hate | 0.42     | 0.46     | 0.67        |
| **Macro-F1**  | **0.59** | **0.54** | —           |

---

### DistilBERT (Fine-Tuned)

| Label         | F1 Score  | PR-AUC    |
| ------------- | --------- | --------- |
| Toxic         | 0.818     | 0.894     |
| Severe toxic  | 0.492     | 0.430     |
| Obscene       | 0.804     | 0.862     |
| Threat        | 0.509     | 0.434     |
| Insult        | 0.732     | 0.779     |
| Identity hate | 0.565     | 0.524     |
| **Macro-F1**  | **0.653** | **0.654** |

➡️ DistilBERT improves overall Macro-F1 from **0.54 → 0.65**, with notable gains in rare classes (`severe_toxic`, `threat`).

---

## Ethical Considerations & Limitations

* May be biased against identity-related terms.
* Performance is weaker on rare classes (`severe_toxic`, `threat`).
* False positives may flag benign speech as toxic.
* Limited to English text; not validated on other languages.
* Intended for research/educational use, not production moderation pipelines.

---

## How to Use

Example API call:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"texts":["You are stupid!", "Have a good day"]}'
```

Example response:

```json
[
  {
    "labels": {"toxic": 1, "severe_toxic": 0, "obscene": 1, "threat": 0, "insult": 1, "identity_hate": 0},
    "probs": {"toxic": 0.98, "severe_toxic": 0.03, "obscene": 0.91, "threat": 0.05, "insult": 0.89, "identity_hate": 0.01}
  }
]
```
