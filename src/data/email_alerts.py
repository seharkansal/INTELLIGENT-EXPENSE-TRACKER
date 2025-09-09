from imap_tools import MailBox, AND
import re
import pandas as pd

def fetch_new_transactions():
    with MailBox("imap.gmail.com").login("your_email", "app_password") as mailbox:
        msgs = mailbox.fetch(AND(from_="alerts@cibc.com", seen=False))
        transactions = []
        for msg in msgs:
            body = msg.text or msg.html
            # Example regex (will need tuning based on email format)
            match = re.search(r"(\d{4}-\d{2}-\d{2}).*?(\w+.*?)\s+\$?(-?\d+\.\d{2})", body)
            if match:
                date, merchant, amount = match.groups()
                transactions.append({"date": date, "merchant": merchant, "amount": float(amount)})
        return pd.DataFrame(transactions)
