import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel
from sklearn.metrics import f1_score, precision_recall_curve, auc
from tqdm import tqdm
import argparse
import yaml
from torch.optim import AdamW
from src.utils import load_data, train_val_split
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader


def encode_texts(tokenizer, texts: np.ndarray, max_length: int = 256) -> dict:
    """
    Tokenize and encode texts for BERT input.

    Args:
        tokenizer: Hugging Face tokenizer.
        texts (np.ndarray): Array of raw text strings.
        max_length (int): Maximum sequence length.

    Returns:
        dict: Encoded tensors (input_ids, attention_mask).
    """
    return tokenizer(
        texts.tolist(),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )



class ToxicCommentsDataset(torch.utils.data.Dataset):
    """
    Custom dataset for multi-label toxic comment classification.
    """
    def __init__(self, encodings: dict, labels: np.ndarray):
        self.encodings = encodings
        self.labels = labels
    def __len__(self) -> int:
        return len(self.labels)
    def __getitem__(self, idx: int) -> dict:
        return {key: val[idx] for key, val in self.encodings.items()}, torch.tensor(self.labels[idx], dtype=torch.float)



class ToxicCommentsModel(torch.nn.Module):
    """
    BERT-based model for multi-label toxic comment classification.
    """
    def __init__(self, model_name: str, num_labels: int):
        super(ToxicCommentsModel, self).__init__()
        self.bert = AutoModel.from_pretrained(
            model_name,
        )
        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(hidden_size, num_labels)
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(outputs.last_hidden_state[:, 0, :])
        return self.classifier(pooled)


def train_one_epoch(model, dataloader, optimizer, scheduler, device, scaler):
    """
    Train model for one epoch with mixed precision.

    Args:
        model: Transformer model.
        dataloader: Training dataloader.
        optimizer: Optimizer (AdamW).
        scheduler: LR scheduler.
        device: Device (cuda or cpu).
        scaler: GradScaler for mixed precision.
    """
    for batch in tqdm(dataloader):
        inputs, labels = batch
        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels = labels.to(device)

        optimizer.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(**inputs)
            loss_fn = nn.BCEWithLogitsLoss()
            loss = loss_fn(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

    return loss.item()

def evaluate_model(model, dataloader, device, thresholds: np.ndarray) -> dict:
    """
    Evaluate model on validation set with tuned thresholds.

    Args:
        model: Trained model.
        dataloader: Validation dataloader.
        device: Device (cuda or cpu).
        thresholds (np.ndarray): Per-class thresholds.
        label_names (list[str]): Names of the labels.

    Returns:
        dict: Evaluation metrics (macro-F1, per-class F1, PR-AUC).
    """
    model.eval()
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            inputs, labels = batch
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.cpu().numpy()
            outputs = model(**inputs)
            probs = outputs.cpu().numpy()

            all_labels.append(labels)
            all_probs.append(probs)

    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)
    pr_aucs = []
    for i in range(all_labels.shape[1]):
        precision, recall, _ = precision_recall_curve(all_labels[:, i], all_probs[:, i])
        pr_aucs.append(auc(recall, precision))
    return {
        "macro_f1": f1_score(all_labels, all_probs > thresholds, average="macro", zero_division=1),
        "per_class_f1": f1_score(all_labels, all_probs > thresholds, average=None, zero_division=1),
        "mean_pr_auc": np.mean(pr_aucs),
        'per_class_pr_auc': pr_aucs,
    }


def tune_thresholds(y_true: np.ndarray, y_probs: np.ndarray) -> np.ndarray:
    """
    Tune thresholds per class to maximize F1 score.

    Args:
        y_true (np.ndarray): Ground truth labels.
        y_probs (np.ndarray): Predicted probabilities.

    Returns:
        np.ndarray: Best threshold per class.
    """
    best_thresholds = []
    for i in range(y_true.shape[1]):
        # for multi-label problem
        precision, recall, thresholds = precision_recall_curve(y_true[:, i], y_probs[:, i])
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        best_idx = np.argmax(f1_scores)
        best_thresholds.append(thresholds[best_idx] if best_idx < len(thresholds) else 0.5)
        
    return np.array(best_thresholds)



def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    scaler,
    device,
    num_epochs: int = 3,
) -> tuple[torch.nn.Module, np.ndarray]:
    """
    Train the model and tune thresholds.

    Args:
        model: Transformer model.
        train_loader: Training dataloader.
        val_loader: Validation dataloader.
        optimizer: Optimizer (AdamW).
        scheduler: LR scheduler.
        device: Device (cuda or cpu).
        num_epochs (int): Number of training epochs.
    """


    history = [] # to store training history
    for epoch in range(num_epochs):
        loss = train_one_epoch(model, train_loader, optimizer, scheduler, device, scaler)
        eval_metrics = evaluate_model(model, val_loader, device, thresholds=np.array([0.5]*model.classifier.out_features))
        history.append({"epoch": epoch + 1, "loss": loss, **eval_metrics})
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.4f}")

    return model, np.array(history)


def save_model(model: torch.nn.Module, path: str):
    """
    Save model state dictionary.

    Args:
        model: Trained model.
        path (str): File path to save the model.
    """
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def model_predict(model, val_dataloader, device):
    val_predictions = []
    model.eval()
    for batch in tqdm(val_dataloader):
        inputs, labels = batch
        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels = labels.cpu().numpy()
        

        with torch.no_grad():
            outputs = model(**inputs)
            val_predictions.append(outputs.cpu().numpy())
    return np.concatenate(val_predictions)

if __name__ == "__main__":
    # argparse with --texts, --labels, --epochs, --batch_size, --lr, etc.
    print("Starting training script...")
    parser = argparse.ArgumentParser(description="Train BERT model for toxic comment classification")
    parser.add_argument('--File_Parameters', type=str, default='config.yaml', help='Path to the config file')
    args = parser.parse_args()
    with open(args.File_Parameters, 'r') as file:
        config = yaml.safe_load(file)
    print("Loading data...")
    comments, labels = load_data(config['texts'], config['labels'])
    print(f"Data loaded: {len(comments)} samples.")



    print(f"Loading tokenizer and encoding texts...")
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    print("Tokenizer max length:", tokenizer.model_max_length)

    X_train, X_val, y_train, y_val = train_val_split(comments, labels, test_size=0.1, random_state=42)

    print(f"Encoding training data...")
    encodings = encode_texts(tokenizer, X_train, max_length=tokenizer.model_max_length)
    encodings_val = encode_texts(tokenizer, X_val, max_length=tokenizer.model_max_length)


    train_dataset = ToxicCommentsDataset(encodings, y_train)
    val_dataset = ToxicCommentsDataset(encodings_val, y_val)


    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading model and preparing for training on {device}...")
    model = ToxicCommentsModel(config['model_name'], y_val.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])

    print("Setting up dataloaders and scheduler...")

    warmup_ratio = 0.1
    num_warmup_steps = int(config["epochs"] * len(train_dataset) * warmup_ratio)

    scaler = torch.amp.GradScaler()
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=len(train_dataset)*config["epochs"])

    train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)

    if config["is_trained"]:
        print(f"Loading pre-trained model from {config['model_path']}...")
        model.load_state_dict(torch.load(config['model_path'], map_location=device))
        print("Model loaded.")
    else:
        print("Starting training...")
        model, history = train_model(
            model,
            train_dataloader,
            val_dataloader,
            optimizer,  
            scheduler,
            scaler,
            device,
            num_epochs=config["epochs"],
        )
        print("Training complete.")
        print(f"Saving model in {config['model_save_path']}...")
        save_model(model, config['model_save_path'])

    print("Generating validation predictions for threshold tuning...")
    val_predictions = model_predict(model, val_dataloader, device)

    print("Tuning thresholds on validation set...")
    best_thresholds = tune_thresholds(y_val, val_predictions)
    print("Best thresholds per class:", best_thresholds)

    print("Evaluating final model performance...")
    results = evaluate_model(model, val_dataloader, device, best_thresholds)

    print("Saving results...")
    with open(config['save_file'], "w") as f:
        f.write(f"Training History: {history}\n")
        f.write(f"Final Evaluation Metrics: {results}\n")
        f.write(f"Best Thresholds: {best_thresholds}\n")

    print("Training script finished.")