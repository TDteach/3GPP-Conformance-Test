import json
import numpy as np
from sklearn import metrics
from sentence_transformers import SentenceTransformer, util


model = SentenceTransformer("GPL/quora-tsdae-msmarco-distilbert-margin-mse")
# model = SentenceTransformer('all-MiniLM-L6-v2')
# model = SentenceTransformer('all-mpnet-base-v2')

if __name__ == '__main__':
    line_list = list()
    with open('sentence_pair_test.txt','r') as f:
        for line in f:
            line_list.append(line.strip())

    nline = len(line_list)
    sentences1 = [line_list[i] for i in range(0,nline,2)]
    sentences2 = [line_list[i] for i in range(1,nline,2)]
    npairs = len(sentences2)

    embeddings1 = model.encode(sentences1, convert_to_tensor=True)
    embeddings2 = model.encode(sentences2, convert_to_tensor=True)

    cosine_scores = util.cos_sim(embeddings1, embeddings2).cpu().numpy()

    # for i in range(len(sentences1)):
    #     print("{} \t\t {} \t\t Score: {:.4f}".format(sentences1[i], sentences2[i], cosine_scores[i][i]))
    scores = list()
    labels = list()
    for i in range(npairs):
        for j in range(npairs):
            scores.append(cosine_scores[i][j])
            labels.append(int(i==j))
    scores, labels = np.asarray(scores), np.asarray(labels)

    fpr, tpr, thresholds = metrics.roc_curve(labels, scores)
    auc = metrics.auc(fpr, tpr)
    print(auc)

    k = np.argmin(fpr+1-tpr)
    thr = thresholds[k]
    conf_mat = metrics.confusion_matrix(labels, scores>thr)
    print(conf_mat)


'''
with open('generated/3GPP/corpus.jsonl', 'r') as json_file:
    json_list = list(json_file)


rst_str = list()
nw_list = list()
for json_str in json_list:
        result = json.loads(json_str)
        test = result['text']
        nwords = test.split(' ')
        nw_list.append(len(nwords))
        if len(nwords) > 16:
            rst_str.append(json_str)

hist, bin_edges = np.histogram(nw_list, bins=[0,2,4,8,16,32,64])
print(hist)
print(bin_edges)


with open('haha.jsonl','w') as f:
    for json_str in rst_str:
        f.write(json_str)
'''