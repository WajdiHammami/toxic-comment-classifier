from src.train_bert import ToxicCommentsModel
from transformers import AutoTokenizer
import torch
import numpy as np
from src.utils import clean_text
import argparse
import json
import math

LABELS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

def format_output(probs: list[float]) -> dict[str, float]:
    
    return [
        {LABELS[i]: prob for i, prob in enumerate(sample)}
        for sample in probs
    ]


def format_json_with_probs(results: list) -> list:
    formatted = []
    preds, logits = results
    for pred_dict, logit_dict in zip(preds, logits):
        sig_probs = {k: float(1 / (1 + math.exp(-v))) for k, v in logit_dict.items()}
        labels = {label: int(v) for label, v in pred_dict.items()}
        formatted.append({
            "labels": labels,
            "probs": sig_probs
        })
    return formatted


def format_json(input: list, results: list) -> list:
    json_output = {}
    for i, text in enumerate(input):
        json_output[text] = {k: int(v) if isinstance(v, (np.integer, int)) else float(v) for k, v in results[i].items()}
    return json_output


class ToxicCommentsInference:
    def __init__(self, model_path: str, thresholds_path:str, device: str = 'cpu'):
        """
        Initialize the inference class with a pre-trained model.

        Args:
            model_path (str): Path to the saved model file.
            device (str): Device to run the model on ('cpu' or 'cuda').
        """
        self.device = torch.device(device)
        self.model = ToxicCommentsModel("distilbert-base-uncased", num_labels=6)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.thresholds = np.load(thresholds_path)

    def predict(self, texts: list[str], return_probs: bool = False) -> list[list[int]]:
        """
        Predict toxicity labels for a list of texts.

        Args:
            texts (list[str]): List of input texts to classify.
            threshold (float): Threshold for classifying a label as positive.
        Returns:
            list[list[int]]: List of predicted labels for each text.
        """
        inputs = [clean_text(text) for text in texts]

        inputs = self.tokenizer(inputs, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = outputs.cpu().numpy()
        
        preds = (probs >= self.thresholds).astype(int)
        if return_probs:
            return format_output(preds), format_output(probs)
        return format_output(preds)




if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Toxic Comments Inference")
    argparser.add_argument("--model_path", type=str, required=True, help="Path to the saved model file.")
    argparser.add_argument("--thresholds_path", type=str, required=True, help="Path to the thresholds file.")
    argparser.add_argument("--device", type=str, default="cpu", help="Device to run the model on ('cpu' or 'cuda').")
    argparser.add_argument("--texts", type=str, nargs='+', required=False, help="List of texts to classify.")
    argparser.add_argument("--file_test", type=str, help="Path to a text file containing texts to classify json.")
    argparser.add_argument("--return_probs", action='store_true',default=False, help="Whether to return probabilities along with predictions.")
    argparser.add_argument("--output_path", type=str, default="output.json", help="Path to save the output JSON file.")
    args = argparser.parse_args()

    inference = ToxicCommentsInference(args.model_path, args.thresholds_path, args.device)

    if args.file_test:
        with open(args.file_test, 'r') as f:
            input = json.load(f)
        results = inference.predict(input.get("texts", []), return_probs=args.return_probs)
        json_output = {}
        if args.return_probs:
            json_output = format_json_with_probs(results)
        else:
            json_output = format_json(input.get("texts", []), results)
    else:
        results = inference.predict(args.texts, return_probs=args.return_probs)
        if args.return_probs:
            json_output = format_json_with_probs(results)
        else:
            json_output = format_json(args.texts, results)
                
    
    if args.output_path:
        with open(args.output_path, "w") as f:
            json.dump(json_output, f, indent=2)