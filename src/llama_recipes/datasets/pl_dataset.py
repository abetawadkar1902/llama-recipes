import psycopg2
from psycopg2.extras import RealDictCursor
import torch
from transformers import PreTrainedTokenizer
import pandas as pd
from datasets import Dataset

def load_data_from_postgres():
    # Replace with your PostgreSQL connection details
    conn = psycopg2.connect("dbname=aitrndb user=dbadmin password=$dAOuGwz8DMa%JcT16E4 host=aitrn-db-stack-postgresdb-e28pitwsqx40.c4zcidhe0q79.us-east-2.rds.amazonaws.com")
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    cursor.execute("SELECT content, language FROM repositorytrainingdata")
    rows = cursor.fetchall()
    conn.close()
    return rows

def get_pl_dataset(dataset_config, tokenizer, split: str):
    # Load data from PostgreSQL
    data = load_data_from_postgres()
    
    # Convert to a list of dictionaries with 'content' and 'language'
    df = pd.DataFrame(data)
    
    # Convert DataFrame to a Dataset
    dataset = Dataset.from_pandas(df)

    # Preprocess data
    def preprocess_function(examples):
        # Tokenization and processing
        input_texts = examples['content']  # Adjust based on your columns
        tokenized_inputs = tokenizer(input_texts, padding='max_length', truncation=True)
        return {
            'input_ids': tokenized_inputs['input_ids'],
            'attention_mask': tokenized_inputs['attention_mask'],
            'labels': tokenized_inputs['input_ids'],  # Or adjust if needed
            'language': examples['language']  # Include language if needed
        }
    
    dataset = dataset.map(preprocess_function, batched=True)
    
    return dataset
