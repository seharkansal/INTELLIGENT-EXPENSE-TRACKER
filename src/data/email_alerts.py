from imap_tools import MailBox, AND
import re
import pandas as pd
from src.data.data_preprocessing import preprocess_dataframe
from src.logger import logging
import os
import bs4  # for HTML parsing if needed

def fetch_new_transactions():
    transactions = []

    with MailBox("imap.gmail.com").login(
        "seharkansal1@gmail.com",
        "rfbu zfrl njym yvdg"
    ) as mailbox:

        # # Fetch all emails from the last 3 days (adjust as needed)
        # from datetime import datetime, timedelta
        # since_date = datetime.now() - timedelta(days=7)

        msgs = mailbox.fetch(
            AND(from_="Mailbox.noreply@cibc.com")
        )

        for msg in msgs:
            # Use text or HTML
            body = msg.text or msg.html
            if body is None:
                continue

            # Strip HTML if needed
            if body.strip().startswith("<"):
                body = bs4.BeautifulSoup(body, "html.parser").get_text()

            # Normalize whitespace
            body = " ".join(body.split())

            # Regex to capture amount and merchant, very flexible
            # match = re.search(
            #     # r"for \$([0-9]+\.[0-9]{2}) at ([A-Za-z0-9 &/#'\-]+)\.",
            #     r"(?:for|of) \$([0-9]+\.[0-9]{2}) (?:at|to) ([A-Za-z0-9 &/#'\-]+)",
            #     body,
            #     re.IGNORECASE
            # )
            match = re.search(
            r"(?:for|of) \$([0-9]+\.[0-9]{2}) (?:at|to|from)?\s*([A-Za-z0-9 &/#'\-]+?)(?=(?: on your|\.|$))",
            body,
            re.IGNORECASE
        )

            if match:
                amount, merchant = match.groups()
                transactions.append({
                    "date": msg.date.date(),
                    "merchant": merchant.strip(),
                    "amount": -float(amount),
                    "email_uid": msg.uid   # <-- added
                })


    df_new = pd.DataFrame(transactions)
    logging.info(f"Fetched {len(df_new)} new transactions from emails.")
    return df_new

df_new = fetch_new_transactions()

if df_new.empty:
    logging.info("No new transactions found.")
else:
    df_processed = preprocess_dataframe(df_new)
    df_processed["source"] = "new"  # mark new emails

    master_csv = "/home/sehar/INTELLIGENT-EXPENSE-TRACKER/data/interim/new_email_transaction_data.csv"

    if os.path.exists(master_csv):
        df_master = pd.read_csv(master_csv)

        # Mark all existing rows as historical
        df_master["source"] = "historical"

        # Combine old + new
        df_combined = pd.concat([df_master, df_processed], ignore_index=True)

        # Remove duplicates
        df_combined["tx_hash"] = (
            df_combined["date"].astype(str) +
            df_combined["merchant_clean"].str.lower().str.strip() +
            df_combined["amount"].astype(str)
        ).apply(hash)

        df_combined.drop_duplicates(subset=["tx_hash"], inplace=True)
        df_combined.drop(columns=["tx_hash"], inplace=True)

        df_combined.to_csv(master_csv, index=False)
        logging.info(f"Master CSV updated with {len(df_combined['source']=='new')} unique transactions.")
    else:
        # First time, just save new rows
        df_processed.to_csv(master_csv, index=False)
        logging.info(f"Created new master CSV with {len(df_processed)} records.")


# if df_new.empty:
#     logging.info("No new transactions found.")
# else:
#     df_processed = preprocess_dataframe(df_new)
#     master_csv = "/home/sehar/INTELLIGENT-EXPENSE-TRACKER/data/interim/new_email_transaction_data.csv"
#     write_header = not os.path.exists(master_csv)  # True if file doesn't exist

#     df_processed.to_csv(
#         master_csv,
#         mode="a",
#         header=write_header,
#         index=False
#     )


#     logging.info(f"Appended {len(df_processed)} new transactions to master CSV.")
