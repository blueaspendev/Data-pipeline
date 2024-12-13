import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec, Index
from sqlalchemy import create_engine

# Initialize Pinecone instance with API key
api_key = "c6ea5e3d-425c-48e2-96a1-74adfe830758"
pc = Pinecone(api_key=api_key)

# Set your index name
index_name = "my-text-embeddings"

# Check if the index exists, otherwise delete it and create a new one
if index_name in pc.list_indexes().names():
    pc.delete_index(index_name)  # Delete the existing index with the wrong dimension

# Create a new index with the correct dimension
pc.create_index(
    name=index_name,
    dimension=768,  # Correct dimension for 'all-mpnet-base-v2'
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1")  # Adjust region as needed
)

# Get the index details to retrieve the host
index_details = pc.describe_index(index_name)
host = index_details.host

# Now connect to the index with the host
index = Index(name=index_name, host=host, api_key=api_key)

# Load the pre-trained Sentence Transformer model
model = SentenceTransformer('all-mpnet-base-v2')

# Connect to the database and load the preprocessed data
engine = create_engine('mysql+pymysql://yash:mypass123@localhost/mydatabase')
df = pd.read_sql_table('preprocessed_table', engine)

# Vectorize the 'clean_description' column and store the embeddings
# Vectorize the 'clean_description' column and store the embeddings
def vectorize_and_store(df):
    embeddings = model.encode(df['clean_description'].tolist(), show_progress_bar=True)
    # Prepare vectors for upsert in the correct format
    vectors = [
        {"id": str(idx), "values": embedding.tolist(), "metadata": {}}
        for idx, embedding in zip(df.index, embeddings)
    ]
    index.upsert(vectors=vectors)

# Process the data in batches to handle large datasets
batch_size = 1000
num_batches = (len(df) + batch_size - 1) // batch_size

for i in range(num_batches):
    start = i * batch_size
    end = min((i + 1) * batch_size, len(df))
    batch_df = df.iloc[start:end]
    vectorize_and_store(batch_df)

print("Text vectorization and storage complete.")