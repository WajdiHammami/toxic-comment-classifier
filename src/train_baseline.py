import numpy as np
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score, classification_report
import argparse
from scipy.sparse import hstack
from src.utils import load_data, train_val_split

logger = logging.getLogger(__name__)





def vectorize_texts(X_train: np.ndarray, X_val: np.ndarray, max_features: int = 200000
) -> tuple[np.ndarray, np.ndarray, object, object]:
    """
    Convert raw texts into TF-IDF features.

    Args:
        X_train (np.ndarray): Training texts.
        X_val (np.ndarray): Validation texts.
        max_features (int): Max features for TF-IDF.

    Returns:
        tuple: (X_train_tfidf, X_val_tfidf, vectorizer)
    """
    word_vectorizer = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), max_features=max_features // 2)
    char_vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), max_features=max_features // 2)

    X_train_word = word_vectorizer.fit_transform(X_train)
    X_val_word = word_vectorizer.transform(X_val)
    X_train_char = char_vectorizer.fit_transform(X_train)
    X_val_char = char_vectorizer.transform(X_val)

    X_train_tfidf = hstack([X_train_word, X_train_char])
    X_val_tfidf = hstack([X_val_word, X_val_char])

    return X_train_tfidf, X_val_tfidf, word_vectorizer, char_vectorizer

def train_baseline_model(X_train: np.ndarray, y_train: np.ndarray
) -> object:
    """
    Train one-vs-rest Logistic Regression baseline on TF-IDF features.

    Args:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training labels.

    Returns:
        object: Trained baseline model.
    """
    logistic = LogisticRegression(solver="sag", C=4, max_iter=1000)
    model = OneVsRestClassifier(logistic, n_jobs=-1)
    model.fit(X_train, y_train)
    return model



def evaluate_model(model: object, X_val: np.ndarray, y_val: np.ndarray, label_names: list[str]) -> dict:
    """
    Evaluate model on validation set and compute metrics.

    Args:
        model (object): Trained model.
        X_val (np.ndarray): Validation features.
        y_val (np.ndarray): Validation labels.
        label_names (list[str]): List of label names.

    Returns:
        dict: Metrics including macro-F1 and per-class F1.
    """

    predictions = model.predict(X_val)
    macro_f1 = f1_score(y_val, predictions, average="macro", zero_division=1)
    micro_f1 = f1_score(y_val, predictions, average="micro", zero_division=1)
    report = classification_report(y_val, predictions, target_names=label_names, output_dict=True)

    metrics = {"macro_f1": macro_f1, "micro_f1": micro_f1}
    for label in label_names:
        metrics[f"f1_{label}"] = report[label]["f1-score"]

    pr_auc = {}
    for label in label_names:
        pr_auc[label] = report[label]["precision"]
    metrics.update({f"pr_auc_{k}": v for k, v in pr_auc.items()})
    return metrics



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate baseline model.")
    parser.add_argument("--train_file", type=str, required=True, help="Path to the training data file.")
    parser.add_argument("--train_labels_file", type=str, required=True, help="Path to the training labels file.")
    parser.add_argument("--test_file", type=str, required=True, help="Path to the test data file.")
    parser.add_argument("--test_labels_file", type=str, required=True, help="Path to the test labels file.")
    parser.add_argument("--save_results", type=str, default="results/baseline_results.txt", help="File to save evaluation results.")
    args = parser.parse_args()

    LABEL_NAMES = [
        "toxic",
        "severe_toxic",
        "obscene",
        "threat",
        "insult",
        "identity_hate",
    ]

    # Load data
    X_train_raw, y_train_raw = load_data(args.train_file, args.train_labels_file)
    X_test, y_test = load_data(args.test_file, args.test_labels_file)

    X_train, X_val, y_train, y_val = train_val_split(X_train_raw, y_train_raw)
    logger.info(f"Train size: {len(X_train)}, Val size: {len(X_val)}, Test size: {len(X_test)}")
    # Vectorize texts
    X_train_tfidf, X_val_tfidf, word_vectorizer, char_vectorizer = vectorize_texts(X_train, X_val)
    X_test_word = word_vectorizer.transform(X_test)
    X_test_char = char_vectorizer.transform(X_test)
    X_test_tfidf = hstack([X_test_word, X_test_char])
    logger.info(f"TF-IDF feature shape: {X_train_tfidf.shape}")
    # Train baseline model
    baseline_model = train_baseline_model(X_train_tfidf, y_train)
    # Evaluate on validation set
    val_metrics = evaluate_model(baseline_model, X_val_tfidf, y_val, LABEL_NAMES)
    logger.info(f"Validation Metrics: {val_metrics}")
    # Evaluate on test set
    test_metrics = evaluate_model(baseline_model, X_test_tfidf, y_test, LABEL_NAMES)
    logger.info(f"Test Metrics: {test_metrics}")    

    # report to txt file
    if args.save_results:
        with open(args.save_results, "w") as f:
            f.write("Validation Metrics:\n")
            for k, v in val_metrics.items():
                f.write(f"{k}: {v}\n")
            f.write("\nTest Metrics:\n")
            for k, v in test_metrics.items():
                f.write(f"{k}: {v}\n")