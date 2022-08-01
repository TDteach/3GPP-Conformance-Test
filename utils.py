import os
import logging
import string

import numpy as np
# import benepar
import json
import torch
import stanza
from allennlp.predictors.predictor import Predictor
from sentence_transformers import SentenceTransformer, util

os.environ['ALLENNLP_LOG_LEVEL'] = 'ERROR'
stanza.logger.setLevel(logging.ERROR)
logging.root.setLevel(logging.ERROR)
# stdoutf = open('/dev/null', 'w')
# sys.stdout = stdoutf


PPN_list, PUNCT_set = None, None

def load_PPN_PUNCT():
    PPN_file_path = 'PPN_in_24.301.txt'
    PPN_list = list()
    with open(PPN_file_path, 'r', encoding='utf8') as f:
        for line in f:
            cont = line.strip()
            PPN_list.append(cont)
    PPN_list = np.asarray(PPN_list)
    a = np.asarray([len(z) for z in PPN_list])
    ord_a = np.flip(np.argsort(a))
    PPN_list = PPN_list[ord_a]

    PUNCT_set = set(string.punctuation)

    print('PPN_list and PUNCT_set are loaded!')
    return PPN_list, PUNCT_set


NN_VING_map = None

def load_NN_VING():
    nn_to_ving_file_path = 'noun_to_verbing.json'
    with open(nn_to_ving_file_path) as f:
        NN_VING_map = json.load(f)

    print('NN_VING_map is loaded!')
    return NN_VING_map


srl_predictor = None

def load_srl_predictor():
    predictor = Predictor.from_path(
        "https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz",
        cuda_device=torch.cuda.current_device(),
    )
    return predictor

def semantic_role_labeling(tokens):
    global srl_predictor
    if srl_predictor is None:
        srl_predictor = load_srl_predictor()
    rst = srl_predictor.predict_tokenized(
        tokenized_sentence=tokens,
        # sentence="Did Uriah honestly think he could beat the game in under three hours?."
    )
    return rst


coref_predictor = None

def load_coref_predictor():
    predictor = Predictor.from_path(
        "https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2021.03.10.tar.gz",
        cuda_device=torch.cuda.current_device(),
    )
    return predictor

def coreference_resolution_inference(tokens):
    global coref_predictor
    if coref_predictor is None:
        coref_predictor = load_coref_predictor()
    rst = coref_predictor.predict_tokenized(
        tokens,
        # tokenized_sentence=tokens,
        # sentence="Did Uriah honestly think he could beat the game in under three hours?."
    )
    return rst


# nlp = spacy.load('en_core_web_md')
# nlp.add_pipe('benepar', config={'model': 'benepar_en3_large'})
# def benepar_constituency_parsing(graph, sent):
#     doc = nlp(sent)
#     root = list(doc.sents)[0]
#     build_graph(graph, root)

# nlp0 = stanza.Pipeline('en', processors='tokenize')
nlp1 = None
nlp2 = None
def load_nlp1():
    nlp1 = stanza.Pipeline('en', processors='tokenize,mwt,pos, constituency', tokenize_pretokenized=True)
    return nlp1
def load_nlp2():
    nlp2 = stanza.Pipeline('en', processors='lemma, depparse', lemma_pretagged=True, depparse_pretagged=True,
                           tokenize_pretokenized=True)
    return nlp2


def my_tokenizer(sent):
    global PPN_list, PUNCT_set
    if PPN_list is None or PUNCT_set is None:
        PPN_list, PUNCT_set = load_PPN_PUNCT()

    a = [(0, len(sent))]
    la = [0]
    for ppn in PPN_list:
        b = list()
        lb = list()
        for itvl, lab in zip(a, la):
            if lab == 1:
                b.append(itvl)
                lb.append(1)
                continue
            s, t = itvl[0], itvl[1]
            z = sent.find(ppn, s, t)
            while z >= 0:
                b.append((s, z))
                lb.append(0)
                last_s = 0
                if z + len(ppn) < len(sent) and sent[z + len(ppn)] == 's':
                    last_s = 1
                b.append((z, z + len(ppn) + last_s))
                lb.append(1)
                s = z + len(ppn) + last_s
                z = sent.find(ppn, s, t)
            if s < t:
                b.append((s, t))
                lb.append(0)
        a, la = b, lb

    a = [sent[aa[0]:aa[1]] for aa in a]

    tokens = list()
    for ph, lab in zip(a, la):
        if lab == 1:
            tokens.append(ph)
            continue

        words = ph.split(' ')
        for wd in words:
            i = 0
            while i < len(wd) and wd[i] in PUNCT_set: i += 1
            j = len(wd) - 1
            while j >= i and wd[j] in PUNCT_set: j -= 1
            for z in range(0, i): tokens.append(wd[z])
            if i < j + 1: tokens.append(wd[i:j + 1])
            for z in range(j + 1, len(wd)): tokens.append(wd[z])

    upper_lab = list()
    for token in tokens:
        if (token.isupper() and token != 'I') or (len(token) > 1 and token != 'Is' and token.endswith('s') and token[:-1].isupper()):
            upper_lab.append(1)
        else:
            upper_lab.append(0)

    new_tokens = list()
    i = 0
    while i < len(tokens):
        lab, token = upper_lab[i], tokens[i]
        if lab == 0:
            new_tokens.append(token)
        else:
            _list = [token]
            j = i+1
            while j < len(tokens) and upper_lab[j] == 1:
                _list.append(tokens[j])
                j += 1
            i = j-1
            new_tokens.append(' '.join(_list))
        i += 1
    tokens = new_tokens

    return tokens


def exchange_nn_with_ving(sent, tokens):
    global NN_VING_map
    if NN_VING_map is None:
        NN_VING_map = load_NN_VING()
    i = 0
    a = list()
    ntokens = list()
    for k, token in enumerate(tokens):
        while sent[i] != token[0]:
            a.append(sent[i])
            i += 1
        i += len(token)
        if token.lower() in NN_VING_map:
            tokens[k] = NN_VING_map[token]
        a.append(tokens[k])
        ntokens.append(tokens[k])
    nsent = ''.join(a)
    return nsent, ntokens


def stanza_deal_sent(tokens, only_nlp1=False):
    global nlp1
    if nlp1 is None:
        nlp1 = load_nlp1()
    doc = nlp1([tokens])

    if only_nlp1:
        return doc

    words = doc.sentences[0].words
    for wd in words:
        text = wd.text
        if text.isupper() and text != 'I':
            wd.upos = 'PROPN'
            wd.xpos = 'NNP'
            wd.feats = 'Number=Sing'
        elif len(text) > 1 and text != 'Is' and text.endswith('s') and text[:-1].isupper():
            wd.upos = 'PROPN'
            wd.xpos = 'NNPS'
            wd.feats = 'Number=Plur'

    global nlp2
    if nlp2 is None:
        nlp2 = load_nlp2()
    rst = nlp2(doc)

    return rst


embedding_model = None
embedding_model_path = 'stsb-distilbert-base'
# embedding_model_path = 'stsb-mpnet-base-v2'
def load_embedding_model():
    embedding_model = SentenceTransformer(embedding_model_path)
    return embedding_model


def calc_embeddings_for_sent(sent=None, tokens=None, tags=None, model=None):
    if sent is None and tokens is None:
        raise NotImplementedError
    if tokens is None:
        tokens = my_tokenizer(sent)
    if model is None:
        global embedding_model
        if embedding_model is None:
            embedding_model = load_embedding_model()
        model = embedding_model

    # embedding = model.encode(sent, convert_to_tensor=True, normalize_embeddings=True)
    # embedding = torch.unsqueeze(embedding, 0)
    # return embedding


    if tags is None:
        srl_rst = semantic_role_labeling(tokens)

        max_verb_dict = None
        verb_dicts = srl_rst['verbs']
        for verb_dict in verb_dicts:
            tags = verb_dict['tags']
            verb_dict['n_related'] = np.sum([tag != 'O' for tag in tags])
            if max_verb_dict is None or verb_dict['n_related'] > max_verb_dict['n_related']:
                max_verb_dict = verb_dict
        if max_verb_dict is not None:
            tags = max_verb_dict['tags']

    main_parts = {
        'V': list(),
        # 'ARG0': list(),
        'ARG1': list(),
        'ARG2': list(),
        'ARG3': list(),
        # 'ARG4': list(),
    }

    if tags is not None:
        for tag, token in zip(tags, tokens):
            tag = tag.split('-')[-1]
            if tag in main_parts:
                main_parts[tag].append(token)

    sent_list = [sent]
    for tag in main_parts:
        part = main_parts[tag]
        if len(part) == 0:
            sent_list.append(sent)
        else:
            _sent = ' '.join(part)
            sent_list.append(_sent)

    embedding_list = model.encode(sent_list, convert_to_tensor=True, normalize_embeddings=True)

    for i, tag in enumerate(main_parts):
        part = main_parts[tag]
        if len(part) == 0:
            embedding_list[i, :] = 0

    embedding = embedding_list.flatten()
    # embedding = torch.mean(embedding_list, 0)
    embedding = torch.unsqueeze(embedding, 0)
    embedding = torch.nn.functional.normalize(embedding)
    return embedding
