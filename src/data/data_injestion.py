import re
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)

import os
from sklearn.model_selection import train_test_split
import yaml
import logging
from src.logger import logging

# Load the original CSV
# df = pd.read_csv("/home/sehar/INTELLIGENT-EXPENSE-TRACKER/data/raw/fouth_batch.csv")

# df = df.drop("merchant_clean", axis=1)

# df.to_csv("/home/sehar/INTELLIGENT-EXPENSE-TRACKER/data/raw/final_batch.csv",index=False)

def load_data(data_url: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(data_url)
        logging.info('Data loaded from %s', data_url)
        return df
    except pd.errors.ParserError as e:
        logging.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the data: %s', e)
        raise

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the data."""
    try:
        # df.drop(columns=['tweet_id'], inplace=True)
        logging.info("pre-processing...")
        # Example columns: ['Date', 'Merchant', 'Description', 'Amount']

        # Load Splitwise CSV
        # df_splitwise = pd.read_csv("splitwise_expenses.csv")
        # # Example columns: ['Date', 'Group', 'Note', 'Amount', 'PaidBy']
        # Rename columns explicitly
        df.columns = ["date", "merchant", "debit", "credit", "card_no"]

        # Convert debit/credit to numeric
        df["debit"] = pd.to_numeric(df["debit"], errors="coerce")
        df["credit"] = pd.to_numeric(df["credit"], errors="coerce")

        # Create signed amount column
        df["amount"] = df["credit"].fillna(0) - df["debit"].fillna(0)

        df["date"] = pd.to_datetime(df["date"])
        final_df = df[["date", "merchant", "amount"]]
        logging.info('Data preprocessing completed')
        return final_df
    
    except KeyError as e:
        logging.error('Missing column in the dataframe: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error during preprocessing: %s', e)
        raise

def save_data(train_data: pd.DataFrame, data_path: str) -> None:
    """Save the train and test datasets."""
    try:
        raw_data_path = os.path.join(data_path, 'raw')
        os.makedirs(raw_data_path, exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_path, "first_batch.csv"), index=False)
        # test_data.to_csv(os.path.join(raw_data_path, "first_batch_test.csv"), index=False)
        logging.debug('data saved to %s', raw_data_path)
    except Exception as e:
        logging.error('Unexpected error occurred while saving the data: %s', e)
        raise

def main():
    try:
        df = load_data(data_url='data/external/cibc_data.csv')
        # s3 = s3_connection.s3_operations("bucket-name", "accesskey", "secretkey")
        # df = s3.fetch_file_from_s3("data.csv")

        final_df = preprocess_data(df)
        # train_data, test_data = train_test_split(final_df, test_size=0.2, random_state=42)
        save_data(final_df, data_path='./data')
    except Exception as e:
        logging.error('Failed to complete the data ingestion process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()

