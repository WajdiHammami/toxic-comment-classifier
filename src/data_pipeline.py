import kaggle
import argparse
import pandas as pd
from src.utils import clean_text
import numpy as np
import logging

logger = logging.getLogger(__name__)

def download_kaggle_dataset(out_dir: str):
    try:
        kaggle.api.competition_download_files('jigsaw-toxic-comment-classification-challenge', path=out_dir)
    except Exception as e:
        logger.error(f"Error downloading dataset: {e}")
    return f"Dataset downloaded successfully to {out_dir}"



def load_csv(file_path: str, compression=None)-> 'pd.DataFrame | str':
    try:
        df = pd.read_csv(file_path, compression=compression)
    except Exception as e:
        logger.error(f"Error loading CSV file: {e}")
    return df

def extract_labels(df: pd.DataFrame, label_columns: list[str]) -> 'np.ndarray | str':
    try:
        labels = df[label_columns].values
    except KeyError as e:
        logger.error(f"Label columns not found: {e}")
    return labels

def preprocess_text(df: pd.DataFrame, text_column: str) -> 'list[str] | str':
    try:
        df[text_column] = df[text_column].apply(clean_text)
    except Exception as e:
        logger.error(f"Error preprocessing text: {e}")
    return df[text_column].tolist()


def preprocess_and_save(df: pd.DataFrame, out_dir: str, lowercase: bool = True) -> str:
    try:
        df['comment_text'] = df['comment_text'].apply(lambda x: clean_text(x, lowercase=lowercase))
        features = df.drop(columns=['id', 'comment_text']).columns
        labels = extract_labels(df, features)
        np.save(out_dir, df['comment_text'].to_numpy())
        np.save(out_dir.replace('.npy', '_labels.npy'), labels)
    except Exception as e:
        logger.error(f"Error during preprocessing or saving: {e}")
    return f"Preprocessed data saved to {out_dir}"



def extract_test(comments_path:str, label_path, label_columns: list[str], out_dir:str, lowercase: bool) -> 'np.ndarray | str':
    comments_df = pd.read_csv(comments_path, compression="zip")
    labels_df = pd.read_csv(label_path, compression="zip")
    labels_df = labels_df[labels_df["toxic"] != -1]  # filter out rows with -1

    comments_df = comments_df[comments_df["id"].isin(labels_df["id"])]
    comments_df = comments_df.sort_values(by="id")
    labels_df = labels_df.sort_values(by="id")
    comments_df['comment_text'] = comments_df['comment_text'].apply(lambda x: clean_text(x, lowercase=lowercase))
    labels = extract_labels(labels_df, label_columns)
    np.save(out_dir, comments_df['comment_text'].to_numpy())
    np.save(out_dir.replace('.npy', '_labels.npy'), labels)
    return f"Preprocessed test data and saved to {out_dir}"



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Kaggle dataset")
    parser.add_argument('--download', action="store_true", help='Download the dataset')
    parser.add_argument('--out_dir', type=str, default='data/raw', help='Output directory to save the dataset')
    parser.add_argument('--preprocess', action="store_true", help='Preprocess the dataset')
    parser.add_argument('--data_file', type=str, default='data/train.csv', help='Path to the raw data file')
    parser.add_argument('--lowercase', action="store_true", help='Convert text to lowercase during preprocessing')
    parser.add_argument('--test_comments', type=str, default='data/test.csv.zip', help='Path to the test comments file')
    parser.add_argument('--test_labels', type=str, default='data/test_labels.csv.zip', help='Path to the test labels file')
    parser.add_argument('--test_preprocess', action="store_true", help='Preprocess the test dataset')
    args = parser.parse_args()
    if args.download:
        print(download_kaggle_dataset(out_dir=args.out_dir))

    if args.preprocess:
        compression = 'zip' if args.data_file.endswith('.zip') else None
        df = load_csv(args.data_file, compression=compression)
        if isinstance(df, pd.DataFrame):
            print(preprocess_and_save(df, out_dir=args.out_dir, lowercase=args.lowercase))
        else:
            print(df)

    if args.test_preprocess:
        label_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        print(extract_test(args.test_comments, args.test_labels, label_columns, args.out_dir, args.lowercase))