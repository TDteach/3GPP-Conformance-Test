import os
import json
import numpy as np
import hashlib
from sklearn import metrics
from sentence_transformers import SentenceTransformer, util
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

# model = SentenceTransformer("GPL/quora-msmarco-distilbert-gpl")  #0.899
# model = SentenceTransformer("GPL/quora-tsdae-msmarco-distilbert-margin-mse")  #0.933
# model = SentenceTransformer("GPL/quora-tsdae-msmarco-distilbert-gpl")  #0.905
# model = SentenceTransformer("GPL/quora-distilbert-tas-b-gpl-self_miner")  #0.896
# model = SentenceTransformer("GPL/scifact-tsdae-msmarco-distilbert-margin-mse")  #0.933
# model = SentenceTransformer("GPL/scidocs-tsdae-msmarco-distilbert-margin-mse")  #0.933
# model = SentenceTransformer("GPL/scidocs-tsdae-msmarco-distilbert-gpl")  #0.908
# model = SentenceTransformer("GPL/fiqa-tsdae-msmarco-distilbert-margin-mse")  #0.933
# model = SentenceTransformer("GPL/webis-touche2020-tsdae-msmarco-distilbert-margin-mse")  #0.933
# model = SentenceTransformer("GPL/trec-news-tsdae-msmarco-distilbert-margin-mse")  #0.933
# model = SentenceTransformer("GPL/trec-news-tsdae-msmarco-distilbert-gpl")  #0.908
# model = SentenceTransformer("GPL/hotpotqa-tsdae-msmarco-distilbert-margin-mse")  #0.933
# model = SentenceTransformer("cross-encoder/quora-roberta-large")
# model = SentenceTransformer('all-MiniLM-L12-v2')   #0.901
# model = SentenceTransformer('all-distilroberta-v1')   #0.928
# model = SentenceTransformer('all-mpnet-base-v2')   #0.909
# model = SentenceTransformer('stsb-mpnet-base-v2')   #0.927
# model = SentenceTransformer('stsb-distilbert-base')   #0.934
# model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')   #0.934
# model = SentenceTransformer('bert-large-nli-stsb-mean-tokens')   #0.906
# model = SentenceTransformer('distilbert-base-nli-stsb-quora-ranking')   #0.894
# model = SentenceTransformer('stsb-roberta-large')   #0.901
# model = SentenceTransformer('stsb-bert-large')   #0.906
# model = SentenceTransformer('cross-encoder/stsb-TinyBERT-L-4')   #0.6967
# model = SentenceTransformer('ChrisZeng/electra-large-discriminator-nli-efl-tweeteval')   #0.652


init_file = 'toTD_similarity_testing_data.v1'

def calc_t5_auc(return_fps=False):
    line_list = list()
    with open(test_file,'r') as f:
        for line in f:
            line_list.append(line.strip())

    nline = len(line_list)
    sentences1 = [line_list[i] for i in range(0,nline,2)]
    sentences2 = [line_list[i] for i in range(1,nline,2)]
    npairs = len(sentences2)

    case_list = list()
    for i in range(npairs):
        for j in range(npairs):
            s1 = sentences1[i]
            s2 = sentences2[j]
            case = 'stsb sentence1: '+s1+'.  sentence2: '+s2+'.'
            case_list.append(case)

    tokenizer = T5Tokenizer.from_pretrained("t5-large")
    device='cuda:0'
    model = T5ForConditionalGeneration.from_pretrained("t5-large")
    model = model.to(device)

    scores = list()
    batch_size = 100
    for i in range(0, len(case_list), batch_size):
        encoding = tokenizer(
            case_list[i:i+batch_size],
            padding="longest",
            max_length=512,
            truncation=True,
            return_tensors="pt",
        )
        input_ids, attention_mask = encoding.input_ids, encoding.attention_mask

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, do_sample=False)
        scs = tokenizer.batch_decode(outputs)

        for  sc in scs:
            st = sc.find('>')
            ed = sc.find('<',st)
            sc=sc[st+2:ed]
            scores.append(float(sc))

    score_mat = np.zeros([100,100])
    k = 0
    for i in range(npairs):
        for j in range(npairs):
            score_mat[i][j] = scores[k]
            k += 1

    label_mat = compare_fps_data_list()
    ret = evaluate_btw_mats(score_mat, label_mat, sentences1, sentences2, return_fps)
    return ret


def read_init_data():
    line_list = list()
    with open(init_file, 'r') as f:
        for line in f:
            line_list.append(line.strip())

    hex_id_map = dict()
    nsent = 0
    for line in line_list:
        hex = hashlib.md5(line.encode()).hexdigest()
        if not hex in hex_id_map:
            hex_id_map[hex] = nsent
            nsent += 1

    lab_mat = np.eye(nsent, dtype=np.int32)
    for i in range(0, len(line_list), 2):
        s1 = line_list[i]
        s2 = line_list[i + 1]
        hex1 = hashlib.md5(s1.encode()).hexdigest()
        hex2 = hashlib.md5(s2.encode()).hexdigest()
        id1 = hex_id_map[hex1]
        id2 = hex_id_map[hex2]
        lab_mat[id1][id2] = 1
        lab_mat[id2][id1] = 1

    sentences = [None] * nsent
    for line in line_list:
        hex = hashlib.md5(line.encode()).hexdigest()
        id = hex_id_map[hex]
        sentences[id] = line

    return sentences, hex_id_map, lab_mat


def embeddings_to_scores(embeddings, lab_mat):
    n = len(lab_mat)
    cosine_scores = util.cos_sim(embeddings, embeddings).cpu().numpy()
    scores = list()
    labels = list()

    k_to_pair = dict()
    k = 0
    for i in range(n):
        for j in range(i):
            scores.append(cosine_scores[i][j])
            labels.append(lab_mat[i][j] > 0)
            k_to_pair[k] = (i, j)
            k += 1
    scores, labels = np.asarray(scores), np.asarray(labels)

    return scores, labels, k_to_pair


def calc_auc_on_scores(scores, labels, return_fps=False):
    fpr, tpr, thresholds = metrics.roc_curve(labels, scores)
    auc = metrics.auc(fpr, tpr)
    print('auc: ', auc)

    k = np.argmin(fpr + 1 - tpr)
    thr = thresholds[k]

    '''
    max_f1 = None
    max_f1_thr = None
    for thr in thresholds:
        f1 = metrics.f1_score(labels, scores>thr)
        if max_f1 is None or f1 > max_f1:
            max_f1 = f1
            max_f1_thr = thr
    thr = max_f1_thr
    #'''

    conf_mat = metrics.confusion_matrix(labels, scores > thr)
    print(conf_mat)

    f1 = metrics.f1_score(labels, scores > thr)
    print('f1-score: ', f1)

    if return_fps:
        fps_list = list()
        k = 0
        for sc, lb in zip(scores, labels):
            if sc > thr and lb == 0:
                fps_list.append((k, sc))
            k += 1
        return auc, fps_list
    return auc


def read_fps_data(file_path, n):
    line_list = list()
    with open(file_path, 'r') as f:
        for line in f:
            line_list.append(line.strip())

    label_mat = np.eye(n, dtype=np.int32)

    for i in range(0, len(line_list), 3):
        line1 = line_list[i]
        line2 = line_list[i + 1]
        i = int(line1[1:line1.find(']')])
        j = int(line2[1:line2.find(']')])
        label_mat[i][j] = 1
        label_mat[j][i] = -1

    return label_mat


def compare_fps_data(predicted_path, labeled_path, n):
    pre_mat = read_fps_data(predicted_path, n)
    lab_mat = read_fps_data(labeled_path, n)

    rst_mat = np.ones([n, n], dtype=np.int32) * -1
    for i in range(n):
        for j in range(n):
            if pre_mat[i][j] == 1 and lab_mat[i][j] == 1:
                rst_mat[i][j] = 1
            elif pre_mat[i][j] == 1 and lab_mat[i][j] == 0:
                rst_mat[i][j] = 0

    return rst_mat


def merge_lab_mat(ori, new, sents):
    n = len(ori)
    for i in range(n):
        for j in range(n):
            if ori[i][j] < 0:
                ori[i][j] = new[i][j]
            elif new[i][j] < 0:
                continue
            elif ori[i][j] != new[i][j]:
                print(i, j, 'ori: ', ori[i][j], 'new: ', new[i][j])
                print(sents[i])
                print(sents[j])
                raise NotImplemented
    return ori


def read_data():
    sents, hex_id_map, lab_mat = read_init_data()

    pre_list = list()
    lab_list = list()
    i = 1
    while True:
        pre_path = 'fps_v%d.txt' % i
        lab_path = 'toTD_fps_labeling_v%d.txt' % i
        if not os.path.exists(lab_path):
            break
        print(pre_path, lab_path)
        pre_list.append(pre_path)
        lab_list.append(lab_path)
        i += 1

    for pre, lab in zip(pre_list, lab_list):
        rst_mat = compare_fps_data(pre, lab, len(sents))
        lab_mat = merge_lab_mat(lab_mat, rst_mat, sents)

    return sents, hex_id_map, lab_mat


def evaluate_model(model, return_fps=False):
    sents, hex_id_map, lab_mat = read_data()

    embeddings = model.encode(sents, convert_to_tensor=True)
    scores, labels, k_to_pair = embeddings_to_scores(embeddings, lab_mat)

    if not return_fps:
        return calc_auc_on_scores(scores, labels, return_fps=False)
    else:
        auc, fps_list = calc_auc_on_scores(scores, labels, return_fps=True)
        ret_list = list()
        for k, sc in fps_list:
            i, j = k_to_pair[k]
            _dict = {
                'i':i,
                'j':j,
                'i_sent': sents[i],
                'j_sent': sents[j],
                'score': sc,
            }
            ret_list.append(_dict)
        return auc, ret_list


if __name__ == '__main__':
    #calc_t5_auc(return_fps=False)
    #exit(0)

    model_path_list = list()
    model_path_list.append('stsb-distilbert-base')
    #for i in range(1, 14 + 1):
    #    model_path = 'output/3GPP/{}0000'.format(i)
    #    model_path_list.append(model_path)
    #model_path_list.append("GPL/quora-tsdae-msmarco-distilbert-margin-mse")
    #model_path_list.append("GPL/scidocs-tsdae-msmarco-distilbert-margin-mse")
    #model_path_list.append('distilbert-base-nli-stsb-mean-tokens')
    #model_path_list.append('stsb-mpnet-base-v2')
    #model_path_list.append('all-distilroberta-v1')
    #model_path_list.append('all-mpnet-base-v2')

    rst_list = list()
    for i, model_path in enumerate(model_path_list):
        print(model_path)
        model = SentenceTransformer(model_path)
        auc = evaluate_model(model)
        rst_list.append((i, auc))
        print('[', i, ']-------------------------------------')

    model = SentenceTransformer(model_path_list[-1])
    auc, fps_list = evaluate_model(model, return_fps=True)
    with open('fps.txt', 'w') as f:
        for fps in fps_list:
            f.write('[' + str(fps['i']) + ']' + ' ' + fps['i_sent'] + '\n')
            f.write('[' + str(fps['j']) + ']' + ' ' + fps['j_sent'] + '\n')
            f.write(str(fps['score']) + '\n')

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
#'''
