from imap_tools import MailBox, AND
import re
import pandas as pd
from src.data.data_preprocessing import preprocess_dataframe
from src.logger import logging
"""
Step 1: Extract transactions from emails
Connect to Gmail using IMAP (Python imaplib or imap_tools).
Search for new bank alert emails (FROM: cibc or subject filter).
Parse the email body with regex to extract fields like:
date
merchant
amount
"""

def fetch_new_transactions():
    transactions = []

    with MailBox("imap.gmail.com").login("seharkansal1@gmail.com", "rfbu zfrl njym yvdg") as mailbox:
        # Fetch unseen CIBC alerts
        msgs = mailbox.fetch(AND(from_="Mailbox.noreply@cibc.com",seen=False))

        for msg in msgs:
            body = msg.text or msg.html
            email_date = msg.date.date()  # Use the email’s date as transaction date
            
            # Regex for "for $amount at MERCHANT"
            match = re.search(r"for \$([0-9]+\.[0-9]{2}) at ([A-Z &]+)\.", body)
            if match:
                amount, merchant = match.groups()
                transactions.append({
                    "date": email_date, 
                    "merchant_clean": merchant.strip(),
                    "amount": float(amount)
                })
                print(transactions)

    df_new = pd.DataFrame(transactions)
    logging.info(f"Fetched {len(df_new)} new transactions from emails.")
    return df_new
    
df_new = fetch_new_transactions()

if df_new.empty:
    logging.info("No new transactions found.")
else:

    df_processed= preprocess_dataframe(df_new)

    print(df_processed.head(10))
        
    """
    Step 4: Save or append to your dataset
    """
    # df_processed.to_csv("/home/sehar/INTELLIGENT-EXPENSE-TRACKER/data/interim/test_processed.csv", mode="a", header=False, index=False)
    df_processed.to_csv("/home/sehar/INTELLIGENT-EXPENSE-TRACKER/data/interim/train_processed.csv", mode="a", header=False, index=False)
    print(f"Appended {len(df_processed)} new transactions to master CSV.")
