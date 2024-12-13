import pandas as pd
import re
from sqlalchemy import create_engine
from sklearn.preprocessing import LabelEncoder

# Connect to the database
engine = create_engine('mysql+pymysql://yash:mypass123@localhost/mydatabase')

# Load the data into a DataFrame
df = pd.read_sql_table('my_table', engine)

# Handle missing values
df = df.fillna('')

# Clean and preprocess the 'description' column
def clean_text(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
    text = text.lower()  # Convert to lowercase
    return text

df['clean_description'] = df['description'].apply(clean_text)

# Encode the 'category' column
le = LabelEncoder()
df['category_encoded'] = le.fit_transform(df['category'])

# Example: Create additional derived columns (e.g., word count of 'description')
df['description_word_count'] = df['description'].apply(lambda x: len(x.split()))

# Save the preprocessed data back to the database
df.to_sql('preprocessed_table', engine, if_exists='replace', index=False)

print("Data preprocessing completed and saved to 'preprocessed_table'")
