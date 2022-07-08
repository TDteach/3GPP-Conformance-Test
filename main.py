import os
import numpy as np
import benepar
import spacy
import copy

nlp = spacy.load('en_core_web_md')
nlp.add_pipe('benepar', config={'model': 'benepar_en3_large'})

data_file = 'toTD_single_sentence_in_24.301_v1.txt'
out_file = 'haha.txt'

pre_words = ['if', 'upon', 'on', 'when', 'once', 'after', 'unless', 'while', 'whilst']
aft_words = ['before']
eql_words = ['via', 'through', 'by']


def clear_sent(sent):
    prefix = ['-', ']', '\t', ' ']
    while sent[0] in prefix:
        print(sent)
        sent = sent[1:]
        print(sent)
    return sent.strip()


def benepar_constituency_parsing(graph, sent):
    doc = nlp(sent)
    root = list(doc.sents)[0]
    build_graph(graph, root)


def insert_dict(d, c, l):
    if c not in d:
        d[c] = list()
    d[c].extend(l)
    return d


def merge_dict(a, b):
    for k in b:
        insert_dict(a, k, b[k])
    return a


def goover_spans(graph, root):
    if root is None: return dict()
    spans = list(root._.constituents)
    conj_dict = dict()
    ends = 0
    for sp in spans:
        if len(sp._.labels) > 0 and sp._.labels[0] in ['SBAR', 'PP'] and sp.start >= ends:
            ends = sp.end
            # print('xxxx')
            # print(root)
            # print(sp)
            # print('xxxx')
            conj = str(sp[0]).lower()
            insert_dict(conj_dict, conj, build_graph(graph, sp))
    return conj_dict


def build_graph(graph, root):
    if len(root) == 1:
        if 'NP' in root._.labels:
            return [graph.add_node(root, None)]
        elif 'VP' in root._.labels:
            return [graph.add_node(None, root)]
        return []

    child_list = list(root._.children)
    label_dict = dict()
    for k, ch in enumerate(child_list):
        labels = ch._.labels
        if len(labels) <= 0:
            label = ch.root.pos_
        else:
            label = labels[0]
        if label not in label_dict:
            label_dict[label] = list()
        label_dict[label].append(ch)
    print('--------------------------')
    print(label_dict)
    print(root._.parse_string)

    root_nodes = list()
    conj_dict = dict()
    if 'NP' in label_dict or 'VP' in label_dict:
        if 'NP' in label_dict and 'VP' in label_dict:
            NP = label_dict['NP'][0]
            VP = label_dict['VP'][0]
            _dict = goover_spans(graph, VP)
            merge_dict(conj_dict, _dict)
        elif 'NP' in label_dict:
            NP = label_dict['NP'][0]
            VP = None
        elif 'VP' in label_dict:
            NP = None
            VP = label_dict['VP'][0]
        root_nodes.append(graph.add_node(NP, VP))
        # _dict = goover_spans(graph, NP)
        # merge_dict(conj_dict, _dict)
    elif 'S' in label_dict:
        for s in label_dict['S']:
            root_nodes.extend(build_graph(graph, s))

    pp_list = list()
    if 'SBAR' in label_dict:
        pp_list.extend(label_dict['SBAR'])
    if 'PP' in label_dict:
        pp_list.extend(label_dict['PP'])
    for sb in pp_list:
        conj = str(sb[0]).lower()
        print(conj)
        insert_dict(conj_dict, conj, build_graph(graph, sb))

    for conj in conj_dict:
        if conj in pre_words:
            graph.add_edges(conj_dict[conj], root_nodes)
        elif conj in aft_words:
            graph.add_edges(root_nodes, conj_dict[conj])
        elif conj in eql_words:
            graph.add_edges(root_nodes, conj_dict[conj])
            graph.add_edges(conj_dict[conj], root_nodes)

    return root_nodes


class Node:
    def __init__(self, NP, VP, id):
        self.NP = NP
        self.VP = VP
        self.id = id

    def __str__(self):
        return '[NP]:' + str(self.NP) + ' [VP]: ' + str(self.VP)


class Graph:
    def __init__(self):
        self.node_list = list()
        self.edge_dict = dict()

    def add_node(self, NP, VP):
        new_node = Node(NP, VP, len(self.node_list) + 1)
        self.node_list.append(new_node)
        return new_node

    def add_edges(self, a_list, b_list):
        for a in a_list:
            for b in b_list:
                self.add_edge(a, b)

    def add_edge(self, a, b):
        na = a.id
        nb = b.id
        if na not in self.edge_dict:
            self.edge_dict[na] = list()
        self.edge_dict[na].append(nb)


def main():
    sent_list = list()

    data_path = os.path.join('data', data_file)
    with open(data_path, 'r') as f:
        for line in f:
            if line.startswith('---'): continue
            sent_list.append(line.strip())

    for sent in sent_list:
        GG = Graph()
        benepar_constituency_parsing(GG, sent)

        if len(GG.edge_dict) > 0:
            with open(out_file, 'a') as f:
                f.write('****************************' + '\n')
                f.write(sent + '\n')
                for nd in GG.node_list:
                    f.write(str(nd.id) + ' ' + str(nd) + '\n')
                f.write(str(GG.edge_dict) + '\n')
                f.write('****************************' + '\n')


if __name__ == '__main__':
    os.system('rm haha.txt')
    main()
    exit(0)

    # st = 'For the cases b, c, d, l, la and m, if the attach attempt counter is equal to 5, the UE shall delete any GUTI, TAI list, last visited registered TAI, list of equivalent PLMNs and KSI, shall set the update status to EU2 NOT UPDATED, and shall start timer T3402.'
    # st = 'The RAND and RES values stored in the ME shall be deleted and timer T3416, if running, shall be stopped upon receipt of a SECURITY MODE COMMAND, SERVICE REJECT, SERVICE ACCEPT, TRACKING AREA UPDATE REJECT, TRACKING AREA UPDATE ACCEPT, or AUTHENTICATION REJECT message, upon expiry of timer T3416, if the UE enters the EMM state EMM-DEREGISTERED or EMM-NULL, or if the UE enters EMM-IDLE mode.'
    # st = 'If the NAS security mode control procedure is initiated during an ongoing tracking area updating procedure in WB-S1 mode, the network supports RACS, the UE has set the RACS bit to "RACS supported" in the UE network capability IE of the TRACKING AREA UDPATE REQUEST message, and the UE has set the URCIDA bit to "UE radio capability ID available" in the UE radio capability ID availability IE of the TRACKING AREA UPDATE REQUEST message, then the MME may request the UE to include its UE radio capability ID in the SECURITY MODE COMPLETE message.'
    # st = 'The RAND and RES values stored in the ME shall be deleted and timer T3416, if running, shall be stopped, upon receiving of a SECURITY MODE COMMAND.'
    # st = 'Upon expirty of time T3247, the UE shall set the USIM to valid for EPS services, if the UE does not maintain a counter for "SIM/USIM considered invalid for GPRS services" events.'
    # st = 'The RAND and RES values stored in the ME shall be deleted and timer T3416, if running, shall be stopped.'
    # st = 'When the AS indicates a lower layer failure to NAS, the NAS signalling connection is not availabel.'
    # st = 'Once the UE is successfully attached, and the first default EPS bearer context has been activated during or after the attach procedure, the UE can request the MME to set up connections to additional PDNs.'
    # st = 'Before security can be activated, the MME and the UE need to establish an EPS security context.'
    # st = 'After successful integrity protection validation, the receiver shall update its corresponding locally stored NAS COUNT with the value of the estimated NAS COUNT for this NAS message.'
    # st = 'Unless explicitly stated otherwise, the procedures described in the following subclauses can only be executed whilst a NAS signalling exists between the UE and the MME.'
    # st = 'the procedures described in the following subclauses can only be executed whilst a NAS signalling exists between the UE and the MME and B has been sent.'
    # st = 'the procedures described in the following subclauses can only be executed.'
    # st = 'Before security can be activated, the MME and the UE need to establish an EPS security context.'
    # st = 'When a partial native EPS security context is taken into use through a security mode control procedure, the MME and the UE shall delete the previously current EPS security context.'
    GG = Graph()
    benepar_constituency_parsing(GG, st)
    print(GG.edge_dict)
