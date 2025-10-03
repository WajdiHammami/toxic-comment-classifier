import re
import numpy as np
import logging
import torch
import os

logger = logging.getLogger(__name__)

def replace_url(text:str) -> str:
    """
    Replace all URLs in text with <URL> placeholder.

    Args:
        text (str): Raw text string.
    Returns:
        str: Text with URLs replaced.
    """

    url_pattern = re.compile(r"http[s]?://\S+|www\.\S+")
    return url_pattern.sub("<URL>", text)



def normalize_case(text: str, lowercase:bool = True) -> str:
    """
    Normalize text to lowercase.

    Args:
        text (str): Raw text string.
        lowercase (bool): Whether to convert text to lowercase. Default is True.
    Returns:
        str: Lowercased text.
    """
    if lowercase:
        return text.lower()
    return text


def strip_whitespace(text: str) -> str:
    """
    Remove leading and trailing whitespace.

    Args:
        text (str): Input text.
    Returns:
        str: Trimmed text.
    """
    return text.strip()


def clean_text(text: str, lowercase: bool = True) -> str:
    """
    Clean a single comment string by applying all preprocessing steps.

    Steps:
        - Replace URLs with <URL>
        - Optionally lowercase the text
        - Strip leading/trailing whitespace
        - Handle None or NaN values safely

    Args:
        text (str): Raw comment text.
        lowercase (bool): Whether to lowercase text. Default True.

    Returns:
        str: Cleaned text string.
    """
    if not isinstance(text, str):
        return ""
    
    text = replace_url(text)
    text = normalize_case(text, lowercase=lowercase)
    text = strip_whitespace(text)
    return text


def load_data(text_path: str, label_path: str) -> ' np.ndarray | str':
    """
    Load text comments and their corresponding labels from specified files.

    Args:
        text_path (str): Path to the text data file.
        label_path (str): Path to the label data file.

    Returns:
        list[str]: List of comment strings.
        np.ndarray: Array of labels.
    """
    try:
        comments = np.load(text_path, allow_pickle=True)
        labels = np.load(label_path, allow_pickle=True)
    except Exception as e:
        logger.error(f"Error extracting labels or comments: {e}")
        print(f"Error extracting labels or comments: {e}")
        print(f"Current Directory: {os.getcwd()}")
    return comments, labels



def train_val_split(texts: np.ndarray, labels: np.ndarray, test_size: float = 0.2, random_state: int = 42
) -> np.ndarray:
    """
    Split texts and labels into train and validation sets.

    Args:
        texts (np.ndarray): Array of raw text strings.
        labels (np.ndarray): Multi-label array of shape (n_samples, n_classes).
        test_size (float): Proportion for validation split.
        random_state (int): Seed for reproducibility.

    Returns:
        tuple: (X_train, X_val, y_train, y_val)
    """
    np.random.seed(random_state)
    indices = np.arange(len(texts))
    np.random.shuffle(indices)
    split_idx = int(len(texts) * (1 - test_size))
    train_indices, val_indices = indices[:split_idx], indices[split_idx:]
    X_train, X_val = texts[train_indices], texts[val_indices]
    y_train, y_val = labels[train_indices], labels[val_indices]
    return (X_train, X_val, y_train, y_val)


def print_gpu_stats():
    allocated = torch.cuda.memory_allocated(0) / 1024**2
    reserved = torch.cuda.memory_reserved(0) / 1024**2
    max_allocated = torch.cuda.max_memory_allocated(0) / 1024**2
    print(f"GPU memory: allocated={allocated:.0f}MB, reserved={reserved:.0f}MB, max={max_allocated:.0f}MB")