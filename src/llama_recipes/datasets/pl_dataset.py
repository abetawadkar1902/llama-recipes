import psycopg2
from psycopg2.extras import RealDictCursor
import torch
from transformers import PreTrainedTokenizer

def load_data_from_postgres():
    # Replace with your PostgreSQL connection details
    conn = psycopg2.connect("dbname=aitrndb user=dbadmin password=$dAOuGwz8DMa%JcT16E4 host=aitrn-db-stack-postgresdb-e28pitwsqx40.c4zcidhe0q79.us-east-2.rds.amazonaws.com")
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    cursor.execute("SELECT content, language FROM repositorytrainingdata")
    rows = cursor.fetchall()
    conn.close()
    return rows

def preprocess_custom_dataset(dataset_config, tokenizer, split: str):
    # Load data from PostgreSQL
    data = load_data_from_postgres()
    
    # Preprocess data
    processed_data = []
    for item in data:
        content = item['content']
        language = item['language']
        
        # Tokenize the content
        tokenized = tokenizer(content, truncation=True, padding='max_length', return_tensors='pt')
        
        # Add language to labels (if needed, e.g., as a separate field)
        # Here we just include it in a dictionary. Adapt according to your use case.
        processed_data.append({
            "input_ids": tokenized["input_ids"].squeeze(),
            "attention_mask": tokenized["attention_mask"].squeeze(),
            "labels": tokenized["input_ids"].squeeze(),  # Use the same content for labels
            "language": language  # Include language
        })
    
    return processed_data

def get_pl_dataset(dataset_config, tokenizer, split: str):
    return preprocess_custom_dataset(dataset_config, tokenizer, split)
