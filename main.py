import os
import string
from tqdm import tqdm

import numpy as np
# import benepar
import json
import spacy
import copy
import torch
from allennlp.predictors.predictor import Predictor
import stanza

PPN_file_path = 'PPN_in_24.301.txt'
PPN_list = list()
with open(PPN_file_path, 'r') as f:
    for line in f:
        cont = line.strip()
        PPN_list.append(cont)
PPN_list = np.asarray(PPN_list)
a = np.asarray([len(z) for z in PPN_list])
ord_a = np.flip(np.argsort(a))
PPN_list = PPN_list[ord_a]

PUNCT_set = set(string.punctuation)

nn_to_ving_file_path = 'noun_to_verbing.json'
with open(nn_to_ving_file_path) as f:
    NN_VING_map = json.load(f)

print('PPN_list, PUNCT_set and NN_VING_map loaded!')

srl_predictor = Predictor.from_path(
    "https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz",
    cuda_device=torch.cuda.current_device(),
)


def semantic_role_labeling(tokens):
    rst = srl_predictor.predict_tokenized(
        tokenized_sentence=tokens,
        # sentence="Did Uriah honestly think he could beat the game in under three hours?."
    )
    return rst


coref_predictor = Predictor.from_path(
    "https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2021.03.10.tar.gz",
    cuda_device=torch.cuda.current_device(),
)


def coreference_resolution_inference(tokens):
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
nlp1 = stanza.Pipeline('en', processors='tokenize,mwt,pos', tokenize_pretokenized=True)
nlp2 = stanza.Pipeline('en', processors='lemma, depparse', lemma_pretagged=True, depparse_pretagged=True,
                       tokenize_pretokenized=True)


def my_tokenizer(sent):
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
                if sent[z + len(ppn)] == 's':
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

    return tokens


def exchange_nn_with_ving(sent, tokens):
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


def stanza_deal_sent(tokens):
    doc = nlp1([tokens])

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
    rst = nlp2(doc)

    print(rst)

    return rst


# data_dir = 'data'
# in_file = 'toTD_single_sentence_in_24.301_v5.txt'
# out_file = 'fromTD_single_sentence_in_24.301_v5.txt'
data_dir = '.'
in_file = 'toTD_test_sentences_v3.txt'
out_file = 'fromTD_test_sentences_v3.txt'

be_words = ['is', 'am', 'are', 'was', 'were', 'be', 'been']
have_words = ['have', 'has', 'had']
pre_words = ['if', 'upon', 'on', 'when', 'once', 'after', 'unless', 'while', 'whilst', 'only']
aft_words = ['before']
eql_words = ['via', 'through', 'by']


class Node:
    def __init__(self, type, desp):
        self.type = type
        self.desp = desp
        self.next = list()

    def __str__(self):
        return '[%s] %s' % (self.type, str(self.desp))

    def connect_to(self, nb):
        self.next.append(nb)


def read_data(in_file):
    data_path = os.path.join(data_dir, in_file)

    para_list = list()
    para = list()
    with open(data_path, 'r') as f:
        for line in f:
            cont = line.strip()
            if line.startswith('---'):
                para_list.append(para)
                para = list()
                continue

            r = cont.find('}')
            prefix = cont[:r + 1]
            param = prefix[1:r].split(',')
            num_line = int(param[0])
            pos_in_line = int(param[1])
            sent = cont[r + 2:]

            sent_dict = {
                'sent': sent,
                'num_line': num_line,
                'pos_in_line': pos_in_line,
                'prefix': prefix,
            }

            para.append(sent_dict)
    return para_list


def build_node(desp):
    return Node('ITEM', desp)


def build_and_node(nd_list):
    nn = Node('AND', 'and')
    for nd in nd_list:
        nn.connect_to(nd)
    return nn


def build_or_node(nd_list):
    nn = Node('OR', 'or')
    for nd in nd_list:
        nn.connect_to(nd)
    return nn


def edge_nodes(na, nb):
    na.connect_to(nb)


def dfs_parsing_tree(pt, node_dict):
    nd = node_dict[pt]

    if len(nd['next']) == 0:
        return None

    rst_node_dict = dict()
    for nt in nd['next']:
        rst_node_dict[nt] = dfs_parsing_tree(nt, node_dict)

    nd['conj_type'] = 'UNKNOWN'
    conj_word_list = list()
    for nt in nd['next']:
        upos = node_dict[nt]['upos']
        if upos in ['CCONJ', 'SCONJ']:
            conj_word_list.append(node_dict[nt])

    if len(conj_word_list) == 0:
        pass
    else:
        if len(conj_word_list) > 1:
            print('ATTENTION!!!', conj_word_list)
            pass
        nd['conj_type'] = conj_word_list[0]['upos']
        nd['conj_word'] = conj_word_list[0]['lemma']

    be_verb_noun = -1
    for nt in nd['next']:
        deprel = node_dict[nt]['deprel']
        text = node_dict[nt]['text']
        if text in be_words and deprel == 'cop':
            be_verb_noun = nt
            break
    if be_verb_noun >= 0:
        nd['be_verb'] = True

    ret_node = None
    if nd['upos'].startswith('VERB') or be_verb_noun >= 0:
        cond_list = list()
        if be_verb_noun >= 0:
            desp = node_dict[be_verb_noun]['desp']
        else:
            desp = nd['desp']
        coor_list = [build_node(desp)]
        coor_word = None
        for nt in nd['next']:
            if not node_dict[nt]['upos'].startswith('VERB') and not 'be_verb' in node_dict[
                nt]: continue  # exclude non-verb and non-be-verb
            deprel = node_dict[nt]['deprel']
            if deprel.split(':')[0] in ['csubj', 'nsubj', 'conj', 'cop']: continue  # exclude subjectional verb
            # print('--------------', nt, rst_nodes[k])
            if node_dict[nt]['conj_type'] == 'SCONJ':
                cond_list.append(rst_node_dict[nt])
            elif node_dict[nt]['conj_type'] == 'CCONJ':
                coor_list.append(rst_node_dict[nt])
                coor_word = node_dict[nt]['conj_word']
            else:  # UNKNOWN
                coor_list.append(rst_node_dict[nt])

        cond_node = None
        if len(cond_list) > 1:
            cond_node = build_and_node(cond_list)
        elif len(cond_list) == 1:
            cond_node = cond_list[0]
        if cond_node is not None:
            for _nd in coor_list:
                _nd.connect_to(cond_node)

        if nd['deprel'] == 'root':
            return coor_list

        if len(coor_list) == 1:
            ret_node = coor_list[0]
        else:
            if coor_word is None or coor_word == 'or':
                ret_node = build_or_node(coor_list)
            elif coor_word == 'and':
                ret_node = build_and_node(coor_list)

        print('----------------')
        print(nd)
        print(ret_node)
        print('----------------')
    return ret_node


def flatten_relation_tree(node):
    if len(node.next) == 0:
        return [[node]]

    rst_list = list()
    for nd in node.next:
        rst_list.append(flatten_relation_tree(nd))

    _dfs_rst = list()
    def _dfs_list(i, pre):
        if i >= len(rst_list):
            #to do remove replacement
            _dfs_rst.append(pre)
            return
        for z in rst_list[i]:
            zz = pre.copy()
            zz.extend(z)
            _dfs_list(i + 1, zz)

    if node.type == 'ITEM' or node.type == 'AND':
        _dfs_list(0, list())
        ret_list = _dfs_rst
        if node.type == 'ITEM':
            for ret in ret_list:
                ret.append(node)
    elif node.type == 'OR':
        z = list()
        for rst in rst_list:
            z.extend(rst)
        ret_list = z
    else:
        raise NotImplementedError

    return ret_list


def build_srl_tree(l, r, root, verb_dicts, tokens):

    a = [0] * len(tokens)
    b = list()
    for verb_dict in verb_dicts:
        verb_k = verb_dict['verb_k']
        if verb_k < l or verb_k >= r: continue
        if a[verb_k] == 1: continue
        for k in verb_dict['related_words']: a[k] = 1
        for k in verb_dict['main_part']: a[k] = 1
        b.append(verb_dict)
    b.sort(key=lambda z: z['verb_k'])

    span_dict = dict()
    for verb_dict in b:
        tags = verb_dict['tags']
        cut_span = None
        verb_dict['sub_spans'] = list()
        for k in verb_dict['related_words']:
            if k in verb_dict['main_part']: continue
            if tags[k].startswith('B'):
                if cut_span is not None:
                    verb_dict['sub_spans'].append((tuple(cut_span), tags[cut_span[0]]))
                    span_dict[tuple(cut_span)] = None
                cut_span = [k, k + 1]
            else:
                cut_span[1] = k + 1
        if cut_span is not None:
            verb_dict['sub_spans'].append((tuple(cut_span), tags[cut_span[0]]))
            span_dict[tuple(cut_span)] = None

    # print(l, r)
    # print(span_dict)
    # print(b)
    for span in span_dict:
        nd = build_node(span)
        span_dict[span] = build_srl_tree(span[0], span[1], nd, verb_dicts, tokens)

    c = list()
    for verb_dict in b:
        nd = build_node(verb_dict['desp'])
        nd.tags = list()
        for span, tag in verb_dict['sub_spans']:
            nd.next.append(span_dict[span])
            nd.tags.append(tag)
        c.append(nd)
    root.next = c

    if len(b) > 0:
        root_type = 'OR'
        for verb_dict in b[:0:-1]:
            k = verb_dict['verb_k'] - 1
            while k >= 0 and (a[k] == 1 or tokens[k].lower() not in ['and', 'or', 'nor', ',']):
                k -= 1
            if k >= 0:
                lower_token = tokens[k].lower()
                if lower_token in ['and', 'nor']:
                    root_type = 'AND'
                    break
                elif lower_token in ['or']:
                    root_type = 'OR'
                    break
                else:
                    pass
        root.type = root_type
    else:
        root.desp = list(range(root.desp[0], root.desp[1]))

    return root


def deal_one_sent(sent):
    tokens = my_tokenizer(sent)
    sent, tokens = exchange_nn_with_ving(sent, tokens)

    i = 0
    wd_st_map = list()
    wd_ed_map = list()
    for wd in tokens:
        while sent[i] != wd[0]: i += 1
        wd_st_map.append(i)
        i += len(wd)
        wd_ed_map.append(i)


    print('*********************semantic role labeling start***********************')
    srl_rst = semantic_role_labeling(tokens)
    # words = srl_rst['words']
    words = tokens

    verb_dicts = srl_rst['verbs']
    for verb_dict in verb_dicts:
        w_list = list()
        tags = verb_dict['tags']
        verb_k = None
        verb_dict['related_words'] = list()
        for k, lb in enumerate(tags):
            if lb != 'O':
                verb_dict['related_words'].append(k)
            if lb == 'B-V': verb_k = k
            zlb = lb.split('-')[1:]
            zlb = '-'.join(zlb)
            if zlb in ['ARGM-NEG']:
                if k - 1 >= 0 and (words[k - 1] in be_words or words[k - 1] in have_words):
                    w_list.append(k - 1)
                w_list.append(k)
            elif lb == 'B-V':
                if k - 1 >= 0 and words[k - 1] in have_words:
                    w_list.append(k - 1)
                elif k - 1 >= 0 and words[k - 1] in be_words:
                    if k - 2 >= 0 and words[k - 2] in have_words:
                        w_list.append(k - 2)
                    w_list.append(k - 1)
                w_list.append(k)
            elif zlb in ['V', 'ARG0', 'R-ARG0', 'ARG1', 'ARG2', 'ARGM-MOD']:
                w_list.append(k)
        verb_dict['verb_k'] = verb_k
        verb_dict['main_part'] = w_list
        verb_dict['desp'] = w_list.copy()

    print('*********************semantic role labeling end***********************')

    print('*********************build SRL tree start***********************')
    verb_dicts.sort(reverse=True, key=lambda z: (len(z['related_words']), -z['verb_k']))

    root = build_node([0, len(tokens)])
    srl_tree_root = build_srl_tree(0, len(tokens), root, verb_dicts, tokens)
    print('*********************build SRL tree end***********************')

    print('*********************coreference resolution inference start***********************')
    coref_rst = coreference_resolution_inference(tokens)
    coref_lab = list(range(len(tokens)))
    clusters = coref_rst['clusters']
    span_text_map = dict()
    for i in range(len(tokens)):
        span_text_map[i] = tokens[i]
    for cluster in clusters:
        for span in cluster:
            span_text_map[span[0]] = sent[wd_st_map[span[0]]:wd_ed_map[span[1]]]
            for i in range(span[0] + 1, span[1] + 1):
                coref_lab[i] = span[0]
        for span in cluster:
            coref_lab[span[0]] = cluster[0][0]

    for i in range(len(tokens)):
        print(i, coref_lab[i], ':', span_text_map[i])
    print('*********************coreference resolution inference end***********************')

    print('*********************replace coreferred phrase start***********************')

    def _find_root(i):
        if coref_lab[i] != i:
            coref_lab[i] = _find_root(coref_lab[i])
        return coref_lab[i]

    def _list_to_str_with_coreference(w_list):
        n_w_list = list()
        for k in w_list:
            k = _find_root(k)
            if len(n_w_list) == 0 or k != n_w_list[-1]:
                n_w_list.append(k)

        a_list = [span_text_map[n_w_list[0]]]
        for i in range(1, len(n_w_list)):
            k = n_w_list[i]
            text = span_text_map[k]
            pk = n_w_list[i - 1]
            ptext = span_text_map[pk]
            if not text in PUNCT_set and not ptext in PUNCT_set:
                a_list.append(' ')
            else:
                if wd_st_map[pk] + len(ptext) < wd_st_map[k]:
                    a_list.append(' ')
            a_list.append(text)
        desp = ''.join(a_list)
        return desp

    verb_desp_map = dict()
    for verb_dict in verb_dicts:
        verb_k = verb_dict['verb_k']
        verb_desp_map[verb_k] = _list_to_str_with_coreference(verb_dict['desp'])
        # print(tokens[verb_k], ':', verb_desp_map[verb_k])

    def _dfs_srl_tree(nd):
        if nd.type == 'ITEM' and type(nd.desp) is list:
            nd.desp = _list_to_str_with_coreference(nd.desp)
        for nt in nd.next:
            _dfs_srl_tree(nt)

    _dfs_srl_tree(srl_tree_root)

    print('*********************replace coreferred phrase end***********************')

    rst_list = list()
    for nd in srl_tree_root.next:
        cond_list = flatten_relation_tree(nd)
        rst_list.append((nd, cond_list))

        for cond in cond_list:
            for cc in cond[:-1]:
                print(cc)
            print('>>>>>>>>>>lead to<<<<<<<<<<<<<')
            print(nd)
            print('-----------')
        print('======================')

    return rst_list




    print('*********************align SRL into dependency parsing start***********************')
    stanza_sent = stanza_deal_sent(tokens).sentences[0]
    stanza_dict_list = stanza_sent.to_dict()
    stanza_dict = dict()
    for wd_dict in stanza_dict_list:
        wd_dict['next'] = list()
        wid = wd_dict['id']
        # if wd_dict['upos'] == 'VERB' or wd_dict['xpos'] == 'VBZ':
        if wd_dict['xpos'].startswith('VB'):
            print(wd_dict)
            print('desp:', verb_desp_map[wid - 1])
            wd_dict['desp'] = verb_desp_map[wid - 1]
        stanza_dict[wid] = wd_dict
    print('*********************align SRL into dependency parsing end***********************')

    root = None
    for id in stanza_dict:
        wd_dict = stanza_dict[id]
        head = wd_dict['head']
        nid = wd_dict['id']
        if head > 0:
            stanza_dict[head]['next'].append(nid)
        if wd_dict['deprel'] == 'root':
            root = nid

    print('*********************dfs parsing tree start***********************')
    node_list = dfs_parsing_tree(root, stanza_dict)
    print('*********************dfs parsing tree end***********************')
    print('*********************flattened results***********************')
    rst_list = list()
    for nd in node_list:
        cond_list = flatten_relation_tree(nd)
        rst_list.append((nd, cond_list))
        for cond in cond_list:
            for cc in cond[:-1]:
                print(cc)
            print('>>>>>>>>>>lead to<<<<<<<<<<<<<')
            print(nd)
            print('-----------')
        print('======================')

    return rst_list


def main():
    out_path = os.path.join(data_dir, out_file)
    os.system('rm -rf ' + out_path)

    para_list = read_data(in_file)

    with open(out_path, 'w') as f:
        for para in tqdm(para_list):
            for sent in para:
                try:
                    rst_list = deal_one_sent(sent['sent'])
                except:
                    continue

                f.write('[[[[[[[[[[[[' + sent['prefix'] + ']]]]]]]]]]]]]' + '\n')
                f.write(sent['sent'] + '\n')
                for nd, cond_list in rst_list:
                    for cond in cond_list:
                        for cc in cond[:-1]:
                            f.write(str(cc) + '\n')
                        f.write('>>>>>>>>>>lead to<<<<<<<<<<<<<' + '\n')
                        f.write(str(nd) + '\n')
                        f.write('-----------' + '\n')
                    f.write('======================' + '\n')


def test():
    # sent = 'When D happens or E occurs, MME should do A, remove B and create C'
    # sent = 'If the attach request is neither for emergency bearer services nor for initiating a PDN connection for emergency bearer services with attach type not set to "EPS emergency attach", upon reception of the EMM causes #95, #96, #97, #99 and #111 the UE should set the attach attempt counter to 5.'

    # sent = 'If the SECURITY MODE COMMAND message can be accepted, the UE shall send a SECURITY MODE COMPLETE message integrity protected with the selected NAS integrity algorithm and the EPS NAS integrity key based on the KASME or mapped K\'ASME if the type of security context flag is set to "mapped security context" indicated by the eKSI.'
    # sent = 'When a partial native EPS security context is taken into use through a security mode control procedure, the MME and the UE shall delete the previously current EPS security context.'
    # sent = 'When the MME and the UE create an EPS security context using null integrity and null ciphering algorithm during an attach procedure for emergency bearer services, or a tracking area updating procedure for a UE that has a PDN connection for emergency bearer services (see subclause 5.4.3.2), the MME and the UE shall delete the previous current EPS security context.'
    # sent = 'If the ATTACH REJECT message with EMM cause #25 was received without integrity protection, then the UE shall discard the message.'
    # sent = 'If the TRACKING AREA UPDATE REJECT message with EMM cause #25 was received without integrity protection, then the UE shall discard the message.'
    # sent = 'Once the secure exchange of NAS messages has been established, the receiving EMM or ESM entity in the UE shall not process any NAS signalling messages unless they have been successfully integrity checked by the NAS.'
    # to do integrity checked
    # sent = 'If the ATTACH REJECT message with EMM cause #25 was received without integrity protection, then the UE shall discard the message.'
    # sent = 'If the TRACKING AREA UPDATE REJECT message with EMM cause #25 was received without integrity protection, then the UE shall discard the message.'
    # sent = 'If the UE receives an ATTACH REJECT, TRACKING AREA UPDATE REJECT or SERVICE REJECT message without integrity protection with EMM cause value #3, #6, #7, #8,  #11, #12, #13, #14, #15, #31 or #35 before the network has established secure exchange of NAS messages for the NAS signalling connection, the UE shall start  timer T3247 with a random value uniformly drawn from the range between 30 minutes and 60 minutes, if the timer is not running.'
    # set ATTACH as a noun
    # sent = 'If the UE finds the SQN to be out of range, the UE shall send an AUTHENTICATION FAILURE message to the network, with the EMM cause #21 "synch failure" and a re-synchronization token AUTS provided by the USIM.'
    # sent = 'Except the messages listed below, no NAS signalling messages shall be processed by the receiving EMM entity in the MME or forwarded to the ESM entity, unless the secure exchange of NAS messages has been established for the NAS signalling connection.'
    # sent = 'Except for the CONTROL PLANE SERVICE REQUEST message including an ESM message container information element or a NAS message container information element, the UE shall start the ciphering and deciphering of NAS messages when the secure exchange of NAS messages has been established for a NAS signalling connection. '
    # sent = 'The MME initiates the NAS security mode control procedure by sending a SECURITY MODE COMMAND message to the UE and starting timer T3460.'
    # sent = 'Secure exchange of NAS messages via a Vehicle-to-Everything signalling connection is usually established by the MME during the attach procedure by initiating a security mode control procedure. '
    # set secure exchange of NAS message as noun
    # sent = 'In state EMM-DiiiEREGISTERED, the UE initiates the attach procedure by sending an ATTACH REQUEST message to the MME, starting timer T3410 and entering state EMM-REGISTERED-INITIATED.'
    sent = 'Upon receipt of an AUTHENTICATION REJECT message, if the message is received without integrity protection, the UE shall start timer T3247 with a random value uniformly drawn from the range between 30 minutes and 60 minutes, if the timer is not running.'
    # to do upon receipt
    # to do coreference resolution
    # sent = 'The UE shall initiate an attach or combined attach procedure on the expiry of timers T3411, T3402 or T3346; '
    # sent = 'When a partial native EPS security context is taken into use through a security mode control procedure, the MME and the UE shall delete the previously current EPS security context. '
    # sent = 'The EPS security context is taken into use by the UE and the MME, when the MME initiates a security mode control procedure or during the inter-system handover procedure from A/Gb mode to S1 mode or Iu mode to S1 mode.'
    # sent = 'The MME initiates the NAS security mode control procedure by sending a SECURITY MODE COMMAND message to the UE and starting timer T3460.'
    # to do deal with on the expiry
    sent = 'When a non-current full native EPS security context is taken into use by a security mode cnotrol procedure, then the MME and the UE shall delete the previously current mapped EPS security context.'



    # sent = 'If running, the timer should stops.'
    # sent = ' I am a boy who plays football.'
    # sent = 'I am a boy'
    deal_one_sent(sent)
    exit(0)


if __name__ == '__main__':
    test()
    main()
