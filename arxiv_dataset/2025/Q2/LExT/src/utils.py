import os
import pandas as pd

REF_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "references.csv")

def save_to_references(row_data):
    """
    Save or update a row in the references.csv file.
    This function appends the row_data as a new row.
    """

    df = pd.DataFrame([row_data])  

    if not os.path.exists(REF_FILE):
        df.to_csv(REF_FILE, index=False, mode='w')
    else:
        old_df = pd.read_csv(REF_FILE)
        df = pd.concat([old_df, df], ignore_index=True) 
        df.to_csv(REF_FILE, index=False)
        