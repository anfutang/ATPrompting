import numpy as np

def dcg(relevances, p):
    relevances = np.asfarray(relevances)[:p]
    if relevances.size:
        return np.sum((2**relevances - 1) / np.log2(np.arange(2, relevances.size + 2)))
    return 0.0

def ndcg_per_query(relevances, p):
    dcg_val = dcg(relevances, p)
    idcg_val = dcg(sorted(relevances, reverse=True), p)
    return dcg_val / idcg_val if idcg_val > 0 else 0.0

def ndcg_score(hyps,refs,p):
    assert len(hyps) == len(refs), "prediction and gold relevance do not have the same length."
    
    # print(f"{len(hyps)} queries.")
    scores = []
    for hyp, ref in zip(hyps,refs):
        if not ref:
            continue
        rel = [ref.get(docid,0) for docid in hyp]
        scores.append(ndcg_per_query(rel,p))
    # print(f"{len(scores)} non-empty gold standard.")
    
    return np.mean(scores)

def index(l,v):
    if v in l:
        return l.index(v) + 1
    else:
        return float("inf")

def mrr_score(hyps,refs,cutoff):
    assert len(hyps) == len(refs), "prediction and gold relevance do not have the same length."

    scores = []
    for hyp, ref in zip(hyps,refs):
        hyp, ref = hyp[:cutoff], list(ref.keys())[:cutoff]
        if not ref:
            continue
        rank = min([index(hyp,r) for r in ref])
        if rank < float("inf"):
            scores.append(1.0/rank)
        else:
            scores.append(0.0)
    return np.mean(scores)