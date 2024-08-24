from bert_score import score

def cq_score(refs,hyps):
    scores = []
    for ref, hyp in zip(refs,hyps):
        # print(ref)
        # print(hyp)
        tmp_scores = []
        for r in ref:
            _, _, f1 = score(hyp,[r]*len(hyp),lang="en",model_type="microsoft/deberta-large-mnli")
            tmp_scores.append(list(f1.numpy()))
        scores.append(tmp_scores)
    return scores