# import imaplib

# try:
#     mail = imaplib.IMAP4_SSL("imap.gmail.com")
#     mail.login("seharkansal1@gmail.com", "rfbu zfrl njym yvdg")  # use App Password if 2FA is on
#     print("IMAP is accessible ✅")
#     mail.logout()
# except Exception as e:
#     print("IMAP not accessible ❌:", e)
import pandas as pd

# Load bank CSV
df_bank = pd.read_csv("/home/sehar/INTELLIGENT-EXPENSE-TRACKER/data/raw/cibc_data.csv")
# Example columns: ['Date', 'Merchant', 'Description', 'Amount']

# Load Splitwise CSV
# df_splitwise = pd.read_csv("splitwise_expenses.csv")
# # Example columns: ['Date', 'Group', 'Note', 'Amount', 'PaidBy']

print("Bank transactions sample:")
print(df_bank.head())




