from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from sqlalchemy import create_engine, text
from typing import List, Dict
import google.generativeai as genai
import uvicorn
import json
import math
from fastapi.responses import JSONResponse
import numpy as np
import re
from collections import Counter

app = FastAPI(title="Text Search and RAG API")

# Initialize configurations
PINECONE_API_KEY = "c6ea5e3d-425c-48e2-96a1-74adfe830758"
GOOGLE_API_KEY = "AIzaSyBkm2xLI-FdaKRmh7Xv46W0-9haos7HGEM"  # Replace with your Gemini API key
DB_CONNECTION = 'mysql+pymysql://yash:mypass123@localhost/mydatabase'
INDEX_NAME = "my-text-embeddings"

# Initialize services
model = SentenceTransformer('all-mpnet-base-v2')
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)
engine = create_engine(DB_CONNECTION)

# Configure Gemini
genai.configure(api_key=GOOGLE_API_KEY)
gemini_model = genai.GenerativeModel('gemini-pro')

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return None
        return super().default(obj)

class SearchQuery(BaseModel):
    query: str
    top_k: int = 5

class RAGQuery(BaseModel):
    query: str
    top_k: int = 3

@app.on_event("startup")
async def startup_event():
    # Verify connections to all services
    try:
        # Test database connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            conn.commit()

        # Test Pinecone connection
        index.describe_index_stats()

        print("All services connected successfully")
    except Exception as e:
        print(f"Error during startup: {str(e)}")
        raise e

def compute_text_similarity(query: str, text: str) -> float:
    query_words = set(re.findall(r'\w+', query.lower()))
    text_words = Counter(re.findall(r'\w+', text.lower()))
    
    if not query_words or not text_words:
        return 0.0
    
    matches = sum(text_words[word] for word in query_words)
    total_words = sum(text_words.values())
    
    return matches / total_words if total_words > 0 else 0.0


@app.post("/search")
async def search_similar_texts(query: SearchQuery):
    try:
        # Generate embedding for the query
        query_embedding = model.encode(query.query).tolist()

        # Search in Pinecone
        search_results = index.query(
            vector=query_embedding,
            top_k=query.top_k * 2,  # Increase to get more potential matches
            include_metadata=True
        )

        print("Pinecone search results:", search_results)

        # Get the IDs of similar documents and their scores
        similar_items = {str(match['id']): match['score'] for match in search_results['matches']}

        if not similar_items:
            print("No similar items found in Pinecone")
            return JSONResponse(content={
                "query": query.query,
                "results": []
            })

        # Fetch the corresponding records from the database
        placeholders = ', '.join([':id' + str(i) for i in range(len(similar_items))])
        query_text = f"SELECT * FROM preprocessed_table WHERE id IN ({placeholders})"
        params = {f'id{i}': int(id) for i, id in enumerate(similar_items.keys())}
        similar_records = pd.read_sql_query(text(query_text), engine, params=params)

        print("Records fetched from database:", similar_records.to_dict(orient='records'))

        if similar_records.empty:
            print("No records found in database")
            return JSONResponse(content={
                "query": query.query,
                "results": []
            })

        # Add scores to the records
        similar_records['score'] = similar_records['id'].astype(str).map(similar_items)

        # Apply text similarity ranking
        query_text = query.query.lower()
        similar_records['text_similarity'] = similar_records.apply(
            lambda row: compute_text_similarity(
                query_text,
                f"{row['name']} {row['description']}".lower()
            ),
            axis=1
        )

        # Check for exact matches in name or category
        similar_records['exact_match'] = similar_records.apply(
            lambda row: 1 if query_text in row['name'].lower() or query_text in row['category'].lower() else 0,
            axis=1
        )

        # Combine vector similarity, text similarity, and exact match scores
        similar_records['combined_score'] = (
            0.4 * similar_records['score'] + 
            0.4 * similar_records['text_similarity'] +
            0.2 * similar_records['exact_match']
        )

        # Sort by combined score and get top_k results
        similar_records = similar_records.sort_values(
            'combined_score', 
            ascending=False
        ).head(query.top_k)

        # Check if any results contain the query word
        if not any(query_text in row['name'].lower() or query_text in row['description'].lower() for _, row in similar_records.iterrows()):
            # If not, perform a broader search in the database
            broader_query = f"SELECT * FROM preprocessed_table WHERE LOWER(name) LIKE :query OR LOWER(description) LIKE :query"
            broader_params = {'query': f'%{query_text}%'}
            broader_results = pd.read_sql_query(text(broader_query), engine, params=broader_params)
            
            if not broader_results.empty:
                broader_results['text_similarity'] = broader_results.apply(
                    lambda row: compute_text_similarity(
                        query_text,
                        f"{row['name']} {row['description']}".lower()
                    ),
                    axis=1
                )
                broader_results['exact_match'] = broader_results.apply(
                    lambda row: 1 if query_text in row['name'].lower() or query_text in row['category'].lower() else 0,
                    axis=1
                )
                broader_results['combined_score'] = 0.4 * broader_results['text_similarity'] + 0.2 * broader_results['exact_match']
                broader_results = broader_results.sort_values('combined_score', ascending=False).head(query.top_k)
                similar_records = pd.concat([similar_records, broader_results]).drop_duplicates(subset='id').sort_values('combined_score', ascending=False).head(query.top_k)

        # Replace NaN, inf, and -inf with None
        similar_records = similar_records.replace([np.inf, -np.inf], np.nan).where(pd.notnull(similar_records), None)

        # Convert to dictionary and handle any remaining problematic float values
        results_dict = similar_records.to_dict(orient='records')
        for record in results_dict:
            for key, value in record.items():
                if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
                    record[key] = None

        print("Final results:", results_dict)

        return JSONResponse(content={
            "query": query.query,
            "results": results_dict
        })
    except Exception as e:
        print(f"Error in search_similar_texts: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rag")
async def generate_rag_response(query: RAGQuery):
    try:
        # Generate embedding for the query
        query_embedding = model.encode(query.query).tolist()

        # Search in Pinecone
        search_results = index.query(
            vector=query_embedding,
            top_k=query.top_k,
            include_metadata=True
        )

        # Get the IDs of similar documents
        similar_ids = [match['id'] for match in search_results['matches']]

        if not similar_ids:
            return JSONResponse(content={
                "query": query.query,
                "response": "No relevant documents found to answer the query.",
                "source_documents": []
            })

        # Fetch the corresponding records from the database
        placeholders = ', '.join([':id' + str(i) for i in range(len(similar_ids))])
        query_text = f"SELECT description FROM preprocessed_table WHERE id IN ({placeholders})"
        params = {f'id{i}': id for i, id in enumerate(similar_ids)}
        similar_records = pd.read_sql_query(text(query_text), engine, params=params)

        # Prepare context for Gemini
        context = "\n".join(similar_records['description'].tolist())

        # Prepare the prompt for Gemini
        prompt = f"""Context: {context}

Question: {query.query}

Please provide a detailed answer based on the context provided. If the context contains relevant information about the query, summarize it. If there's no relevant information, state that clearly."""

        # Generate response using Gemini
        response = gemini_model.generate_content(prompt)

        return JSONResponse(content={
            "query": query.query,
            "response": response.text,
            "source_documents": similar_records['description'].tolist()
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)