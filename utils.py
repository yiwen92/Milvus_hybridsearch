import datasets
import pytrec_eval
import json

def eval_miracl(lang, filename):
    miracl_queries = datasets.load_dataset('miracl/miracl', lang)['dev']
    with open(filename) as fw:
        results = json.loads(fw.read())
    qrel = {}
    for q in miracl_queries:
        qrel[q['query_id']] = {}
        for pos_passages in q['positive_passages']:
            qrel[q['query_id']][pos_passages['docid']] = 1
        for neg_passages in q['negative_passages']:
            qrel[q['query_id']][neg_passages['docid']] = 0
    
    run = {}
    for qid in results:
        run[qid] = {}
        for doc in results[qid]:
            root_name = doc['name']
            run[qid][root_name] = doc['score']
 
    evaluator = pytrec_eval.RelevanceEvaluator( qrel, {'map', f'ndcg_cut_10'})
    ndcgs = []
    res = evaluator.evaluate(run)
    for qid in res:
        ndcgs.append(res[qid]['ndcg_cut_10'])
    
    print(filename, 'NDCG@10:', sum(ndcgs)/len(ndcgs))


