from pymilvus import (
    Collection,
    utility,
    AnnSearchRequest, connections, WeightedRanker
)
import datasets
import os
import numpy as np
import json
from pymilvus import model
from tqdm import tqdm
import argparse

def search(lang, k, dense_weight, sparse_weight, output):
    connections.connect("default", host="localhost", port="19530")
    col_name = f'miracl_{lang}'
    col = Collection(col_name)

    ef = model.hybrid.BGEM3EmbeddingFunction(return_sparse=True, device='cuda:1')

    miracl_queries = datasets.load_dataset('miracl/miracl', lang)['dev']
    results = {}

    for i in tqdm(range(len(miracl_queries))):
        q = miracl_queries[i]
        emb = ef([q['query']])

        dense_emb = emb['dense']
        sparse_emb = emb['sparse']

        sparse_search_params = {"metric_type": "IP"}
        sparse_req = AnnSearchRequest(sparse_emb, "sparse_vector", sparse_search_params, limit=k)
        dense_search_params = {"metric_type": "IP", "params": {}}
        dense_req = AnnSearchRequest(dense_emb, "dense_vector", dense_search_params, limit=k)

        res = col.hybrid_search([sparse_req, dense_req], rerank=WeightedRanker(sparse_weight, dense_weight), limit=k, output_fields=['docid'])

        results[q['query_id']] = []
        for doc in res[0]:
            results[q['query_id']].append({'name':doc.docid , 'score': doc.distance})
        
    with open(output, 'w') as fw:
        fw.write(json.dumps(results))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Search embeddings in Milvus database.")
    parser.add_argument("--lang", type=str, default="fi", help="Language code for the embeddings to search. Default is 'fi'.")
    parser.add_argument("--k", type=int, default=1000, help="Number of results to return. Default is 1000.")
    parser.add_argument("--dense", type=float, default=1.0, help="Weight for dense vector in search. Default is 1.0.")
    parser.add_argument("--sparse", type=float, default=0.4, help="Weight for sparse vector in search. Default is 0.4.")
    parser.add_argument("--output", type=str, default='results_hybrid.json', help="Output file name.")

    args = parser.parse_args()

    search(lang=args.lang, k=args.k, dense_weight=args.dense, sparse_weight=args.sparse, output=args.output)

