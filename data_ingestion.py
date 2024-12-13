import pandas as pd
import json
from sqlalchemy import create_engine

# Load sample dataset
data_file = 'dataset.json'
with open(data_file, 'r') as f:
    data = json.load(f)

# Convert JSON data to a Pandas DataFrame
df = pd.DataFrame(data)

# Connect to a MySQL database
db_engine = create_engine('mysql://yash:mypass123@localhost:3306/mydatabase')

# Create a table and load the data
df.to_sql('my_table', db_engine, if_exists='replace', index=False)