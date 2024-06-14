from pymilvus import (
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection, connections,
)
import datasets
import os
import numpy as np
import json
from tqdm import tqdm
import argparse


def insert_embeddings(lang, overwrite=False):
    connections.connect("default", host="localhost", port="19530")
    col_name = f'miracl_{lang}'
    if overwrite == True:
        utility.drop_collection(col_name)
 
    dense_dim = 1024

    fields = [
        # Use auto generated id as primary key
        FieldSchema(name="pk", dtype=DataType.VARCHAR,
                    is_primary=True, auto_id=True, max_length=100),
        # Store the original text to retrieve based on semantically distance
        FieldSchema(name="docid", dtype=DataType.VARCHAR, max_length=512),
        # Milvus now supports both sparse and dense vectors, we can store each in
        # a separate field to conduct hybrid search on both vectors.
        FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
        FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR,
                    dim=dense_dim),
    ]

    schema = CollectionSchema(fields, "")
    col = Collection(col_name, schema, consistency_level="Strong")
    
    sparse_index = {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"}
    col.create_index("sparse_vector", sparse_index)
    dense_index = {"index_type": "FLAT", "metric_type": "IP"}
    col.create_index("dense_vector", dense_index)

    embnames = os.listdir('embeddings')

    for embname in tqdm(embnames):
        docid = embname.split('.')[0]
        emb = np.load('embeddings/' + embname, allow_pickle=True)
        dense_emb = emb.item()['dense']
        sparse_emb = emb.item()['sparse']
        col.insert([[docid], sparse_emb, [dense_emb]])
    col.flush()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Insert embeddings into Milvus database.")
    parser.add_argument("--lang", type=str, default="fi", help="Language code for the embeddings to insert.")
    parser.add_argument("--overwrite", action='store_true', help="Flag to overwrite existing collection if exists.")

    args = parser.parse_args()

    insert_embeddings(lang=args.lang, overwrite=args.overwrite)
    

