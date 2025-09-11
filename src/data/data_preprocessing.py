# data preprocessing

import pandas as pd
import os
import re
import string
import logging
from src.logger import logging

def preprocess_dataframe(df):
    # df = pd.read_csv("/home/sehar/INTELLIGENT-EXPENSE-TRACKER/data/raw/second_batch.csv")

    # Make a fresh copy of relevant columns
    df_new = df[["date", "merchant", "amount"]].copy()

    def clean_merchant(text):
        text = text.upper()
        text = re.sub(r"\d{4,}", "", text)      # remove long numbers
        text = re.sub(r"\s+[A-Z]{2}$", "", text) # remove province codes like "ON"
        text = re.sub(r"[^A-Z ]", "", text)      # keep only letters/spaces
        text = re.sub(r"\s+", " ", text).strip()
        return text

    # If df_clean was a slice like df[df["amount"] > 0]
    # df_clean = df_clean.copy()  # now it's a true independent DataFrame

    # Now apply your cleaning safely
    df_new["merchant_clean"] = df_new["merchant"].apply(clean_merchant)

    # Define a mapping of keywords → categories
    category_rules = {
        "WALMART": "Grocery",
        "OUTLET": "Shopping",
        "POPEYES": "Food",
        "UBER": "Food",
        "DOORDASH": "Food",
        "FIDO": "Utility",
        "MOBILE": "Utility",
        "DOLLAR": "Shopping",
        "CAW MARKET": "Grocery",
        "PAYMENT": "Bills",
        "FRESHCO": "Grocery",
        "TIM HORTONS": "Food",
        "AMAZON": "Shopping",
        "HM": "Shopping",
        "MCDONALDS": "Food",
        "SHOPPERS": "Shopping",
        "SK": "Grocery",
        "BANFF": "trip",
        "WINNIPEG": "trip",
        "CALGARY": "trip",
        "VICTORIA": "trip",
        "CASHBACK": "cashback",
        "ONROUTE": "trip",
        "RESTAURANT": "Food",
        "INDIA": "Food",
        "CHICKF": "Food",
        "INSTACART": "Grocery",
        "PIZZA": "Food",
        "PAAN": "Food",
        "AW": "Food",
        "LCBO": "Food",
        "WINNERS": "Shopping",
        "MARKET": "Food",
        "ARDEN": "Shopping",
        "CEI": "Food",
        "BASICS": "Grocery",
        "FALLS": "trip",
        "STARBUCKS": "Food",
        "PAAN": "Food",
        "SEPHORA": "Shopping"

        }

    # Function to assign category based on merchant name
    def categorize_merchant(merchant):
        for keyword, category in category_rules.items():
            if keyword in merchant:
                return category
        return "Other"  # default category

    # Apply mapping
    df_new["category"] = df_new["merchant_clean"].apply(categorize_merchant)

    # Check results
    # print(df_new.head())

    new_df = df_new.drop("merchant", axis=1)

    # Specify the new order of columns
    new_order = ["date", "merchant_clean", "amount", "category"]

    # Reorder the DataFrame
    new_df = new_df[new_order]

    # df.to_csv("/home/sehar/INTELLIGENT-EXPENSE-TRACKER/data/raw/final_batch.csv",index=False)
    # df_new.to_csv("/home/sehar/INTELLIGENT-EXPENSE-TRACKER/data/raw/fouth_batch.csv",index=False)
    logging.info("Data pre-processing completed")
    return new_df


def main():
    try:
        # Fetch the data from data/raw
        train_data = pd.read_csv('./data/raw/first_batch_train.csv')
        test_data = pd.read_csv('./data/raw/first_batch_test.csv')
        logging.info('data loaded properly')

        # Transform the data
        train_processed_data= preprocess_dataframe(train_data)
        test_processed_data= preprocess_dataframe(test_data)

        # Store the data inside data/processed
        data_path = os.path.join("./data", "interim")
        os.makedirs(data_path, exist_ok=True)
        
        train_processed_data.to_csv(os.path.join(data_path, "train_processed.csv"), index=False)
        test_processed_data.to_csv(os.path.join(data_path, "test_processed.csv"), index=False)
        
        logging.info('Processed data saved to %s', data_path)
    except Exception as e:
        logging.error('Failed to complete the data transformation process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()