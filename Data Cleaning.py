import pandas as pd
import numpy as np
import re

df = pd.read_csv("/Users/macos/Downloads/messy_dataset.csv")

df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)


def clean_text(text, is_email=False, is_name=False):
    if isinstance(text, str):
        if is_email:
            text = re.sub(r"[^a-zA-Z0-9@._-]", "", text)
        elif is_name:
            text = re.sub(r"\d", "", text)
            text = re.sub(r"[^a-zA-Z0-9.\s-]", "", text)
            text = " ".join(word.capitalize() for word in text.split())
        else:
            text = re.sub(r"[^a-zA-Z0-9.\s-]", "", text)  # Убирается @
        text = re.sub(r"\s+", " ", text).strip()
    return text


for column in df.columns:
    if "email" in column.lower():
        df[column] = df[column].apply(lambda x: clean_text(x, is_email=True))
    elif "name" in column.lower():
        df[column] = df[column].apply(lambda x: clean_text(x, is_name=True))
    else:
        df[column] = df[column].apply(lambda x: clean_text(x, is_email=False))


df = df.replace({0: "Unknown", np.nan: "Unknown"})
print(df)


data = {
    "ID": [1, 2, 3, 4, 5, 2],
    "Name": ["Alice", "Bob", "Charlie", "David", "Eve", "Bob"],
}
df1 = pd.DataFrame(data)
duplicates = df1.duplicated(keep=False)
df1[duplicates]

df1_cleaned = df1.drop_duplicates(keep="last")
df1_cleaned
