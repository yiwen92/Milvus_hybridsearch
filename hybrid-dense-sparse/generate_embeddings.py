import argparse
import json
import os
import torch
import datasets
from transformers import AutoTokenizer, AutoModel
import numpy as np
from pymilvus import model
from tqdm import tqdm  

def generate_corpus_embeddings(lang, device, batch_size, num):
    ef = model.hybrid.BGEM3EmbeddingFunction(return_sparse=True, device=device)
    miracl_corpus = datasets.load_dataset('miracl/miracl-corpus', lang)['train']
    
    batch_doc_ids = []
    batch_titles = []
    batch_texts = []

    if num is None:
        num = len(miracl_corpus)
        
    os.makedirs('embeddings', exist_ok=True) 
    processed_count = 0
    for i in tqdm(range(min(num, len(miracl_corpus)))):
        doc = miracl_corpus[i]
        docid = doc['docid']
        title = doc['title']
        text = doc['text']

        batch_doc_ids.append(docid)
        batch_titles.append(title)
        batch_texts.append(f"{title}\n{text}")
        processed_count += 1  

        if len(batch_texts) == batch_size or processed_count == num:
            embs = ef(batch_texts)
            for i in range(len(batch_texts)):
                np.save(f'embeddings/{batch_doc_ids[i]}.npy', {'dense': embs['dense'][i], 'sparse': embs['sparse'].getrow(i)})
            batch_doc_ids = []
            batch_titles = []
            batch_texts = []
        if processed_count == num:
            break
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process and embed documents from the MIRACL corpus.")
    parser.add_argument("--lang", type=str, default="fi", help="Language code for the documents. Default is 'fi'.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device for the embedding model. Default is 'cuda:1'.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for processing documents. Default is 8.")
    parser.add_argument("--num", type=int, default=None, help="Number of documents to process. Processes all documents if not specified.")
    
    args = parser.parse_args()
    
    generate_corpus_embeddings(lang=args.lang, device=args.device, batch_size=args.batch_size, num=args.num)

