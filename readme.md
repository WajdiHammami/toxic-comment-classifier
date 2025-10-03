# 🛡️ Toxic Comment Classification – End-to-End ML System

## 📌 Project Objective

This project builds a **production-ready multi-label toxic comment classification system** using the [Jigsaw Toxic Comment Classification Challenge dataset](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge).

The system detects the following categories:

* `toxic`
* `severe_toxic`
* `obscene`
* `threat`
* `insult`
* `identity_hate`

The goal was to not only train accurate models but also deliver a **deployable and maintainable pipeline** with modern MLOps practices.

---

## 🛠️ Tech Stack

* **Machine Learning**: PyTorch, HuggingFace Transformers
* **Baseline**: TF-IDF + Logistic Regression
* **Transformer Model**: DistilBERT (fine-tuned)
* **API Deployment**: FastAPI + Uvicorn
* **Containerization**: Docker
* **Utilities**: NumPy, Scikit-learn, argparse
* **Responsible AI**: Model Card, bias & interpretability awareness

---

## 📊 Results

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

> Baseline model performs well on common classes (`toxic`, `obscene`) but struggles with rare ones (`severe toxic`, `threat`). Transformers are expected to improve rare-class F1.

---

### Transformer (DistilBERT Fine-Tuned)

**Final Evaluation Metrics:**

* **Macro-F1**: 0.653
* **Mean PR-AUC**: 0.654

| Label             | F1 Score | PR-AUC |
| ----------------- | -------- | ------ |
| **toxic**         | 0.818    | 0.894  |
| **severe_toxic**  | 0.492    | 0.430  |
| **obscene**       | 0.804    | 0.862  |
| **threat**        | 0.509    | 0.434  |
| **insult**        | 0.732    | 0.779  |
| **identity_hate** | 0.565    | 0.524  |

➡️ DistilBERT improves overall Macro-F1 from **0.54 → 0.65**, particularly boosting performance on rare classes like `severe_toxic` and `threat`, though those remain challenging.

---

## 🚀 Quickstart


### 1. Build Docker image

```bash
docker build -t toxic-api .
```

### 2. Run container

```bash
docker run -p 8000:8000 toxic-api
```

### 3. Test API

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

---

## ⚙️ Project Structure

```
.
├── api/                  # FastAPI service
│   └── main.py
├── src/                  # Core source code
│   ├── data_pipeline.py
│   ├── train_baseline.py
│   ├── train_bert.py
│   ├── inference.py
│   └── utils.py
├── outputs/              # Model artifacts
│   ├── distilbert_model_2.pt
│   ├── thresholds.npy
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── README.md
└── model_card.md
```

---

## 📄 Model Card

See [model_card.md](model_card.md) for details on:

* Dataset
* Training setup
* Metrics
* Limitations and risks
* Responsible AI considerations

---

## 👨‍💻 Development (local, without Docker)

Create venv + install deps:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run API locally:

```bash
uvicorn api.main:app --reload
```

Visit Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 🧰 Future Improvements

* Try larger models (RoBERTa, BERT-base)
* Address rare label imbalance with focal loss or oversampling
* Add model interpretability (SHAP/LIME)
* Deploy to cloud (AWS/GCP/Azure) with CI/CD

---

## 📜 License

MIT License – free to use and modify.
