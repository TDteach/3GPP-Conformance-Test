import os
from utils import *
from graph import Graph, edge_to_string, node_to_string
from tqdm import tqdm
import copy

# data_dir = 'data'
# in_file = 'toTD_single_sentence_in_24.301_v5.txt'
# out_file = 'fromTD_single_sentence_in_24.301_v5.txt'
data_dir = '.'
in_file = 'toTD_single_sentence_in_24.301_v5.txt'
out_file = 'fromTD_single_sentence_in_24.301_v5.txt'

be_words = ['is', 'am', 'are', 'was', 'were', 'be', 'been']
have_words = ['have', 'has', 'had']
pre_words = ['if', 'upon', 'on', 'when', 'once', 'after', 'unless', 'while', 'whilst', 'only', 'via', 'through', 'by']
aft_words = ['before']


# eql_words = ['via', 'through', 'by']


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
            logging.warning('ATTENTION!!!', conj_word_list)
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

    if node.type == 'ITEM' or node.type == 'AND':
        ret_list = and_flatten_or_and_list(rst_list)
        if node.type == 'ITEM':
            for ret in ret_list:
                ret.append(node)
    elif node.type == 'OR':
        ret_list = or_flatten_or_and_list(rst_list)
    else:
        raise NotImplementedError

    return ret_list


def split_main_part(verb_dict, tokens):
    main_tags = list()
    main_tokens = list()
    tags = verb_dict['tags']
    for k in verb_dict['main_part']:
        stag = tags[k].split('-')[-1]
        main_tags.append(stag)
        main_tokens.append(tokens[k])

    ARGN = list()
    for i in range(3):
        ARGN.append('ARG' + str(i))

    n = len(main_tags)
    stag_loc_map = dict()
    for i in range(n):
        stag = main_tags[i]
        if stag in ARGN:
            if stag not in stag_loc_map:
                stag_loc_map[stag] = [i, i]
            if stag_loc_map[stag][1] + 1 < i:
                raise NotImplementedError
            stag_loc_map[stag][1] = i

    temp = list()
    i = 0
    while i < n:
        stag = main_tags[i]
        if stag in ARGN:
            i = stag_loc_map[stag][1]
            temp.append('[' + stag + ']')
        else:
            temp.append(main_tokens[i])
        i += 1
    temp = ' '.join(temp)

    print("xzsewwwwwwwwwwwwwww")
    print(temp)

    for stag in stag_loc_map:
        _span = stag_loc_map[stag]
        _tokens = main_tokens[_span[0]:_span[1] + 1]
        print(_tokens)
        doc = stanza_deal_sent(_tokens, only_nlp1=True)
        tree = doc.sentences[0].constituency
        conj = None
        non_conj_child = list()
        for ch in tree.children[0].children:
            if ch.label == 'CC':
                conj = ch.leaf_labels()[0]
            else:
                non_conj_child.append(ch)
        print(conj)
        if conj is None:
            stag_loc_map[stag] = ('and', [' '.join(_tokens)])
        else:
            _list = list()
            for ch in non_conj_child:
                words = ch.leaf_labels()
                _list.append(' '.join(words))
            if conj.lower() == 'and':
                conj = 'and'
            else:
                conj = 'or'
            stag_loc_map[stag] = (conj, _list)

    main_list = [[temp]]
    for stag in stag_loc_map:
        stag_temp = '[' + stag + ']'
        conj, _list = stag_loc_map[stag]
        if conj == 'and':
            new_main_list = list()
            for and_list in main_list:
                new_and_list = list()
                for it in and_list:
                    for z in _list:
                        new_and_list.append(it.replace(stag_temp, z))
                new_main_list.append(new_and_list)
            main_list = new_main_list
        else:
            new_main_list = list()
            for and_list in main_list:
                for z in _list:
                    new_and_list = list()
                    for it in and_list:
                        new_and_list.append(it.replace(stag_temp, z))
                    new_main_list.append(new_and_list)
            main_list = new_main_list

    return main_list


def build_srl_tree(l, r, root, verb_dicts, tokens):
    a = [0] * len(tokens)
    verb_list = list()
    for verb_dict in verb_dicts:
        verb_k = verb_dict['verb_k']
        if len(verb_dict['main_part']) == 1: continue
        if verb_k < l or verb_k >= r: continue
        if a[verb_k] == 1: continue
        for k in verb_dict['related_words']: a[k] = 1
        for k in verb_dict['main_part']: a[k] = 1
        verb_list.append(verb_dict)
    verb_list.sort(key=lambda z: z['verb_k'])

    for verb_dict in verb_list:
        main_list = split_main_part(verb_dict, tokens)
        verb_dict['main_list'] = main_list
        print("xyyyyyyyyyyyyyyyyyyy")
        print(verb_dict['description'])
        print(main_list)
        print("xyyyyyyyyyyyyyyyyyyy")

    span_dict = dict()
    for verb_dict in verb_list:
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

    print(l, r)
    print(span_dict)
    print(verb_list)
    for span in span_dict:
        if span[0] + 1 >= span[1]: continue
        nd = build_node(span)
        span_dict[span] = build_srl_tree(span[0], span[1], nd, verb_dicts, tokens)

    c = list()
    for verb_dict in verb_list:
        nd = build_node(verb_dict['desp'])
        nd.next = list()
        nd.tags = list()
        nd.span_first_word = list()
        nd.ref = verb_dict['description']
        nd.main_list = verb_dict['main_list']
        for span, tag in verb_dict['sub_spans']:
            if span_dict[span] is not None:
                nd.next.append(span_dict[span])
                nd.tags.append(tag)
                nd.span_first_word.append(tokens[span[0]].lower())
        c.append(nd)
    root.next = c

    if len(verb_list) > 0:
        root_type = 'OR'
        for verb_dict in verb_list[:0:-1]:
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

    root.desp = ' '.join(tokens[root.desp[0]:root.desp[1]])
    # root.desp = list(range(root.desp[0], root.desp[1]))

    return root


def and_flatten_or_and_list(a):
    gathered_list = list()
    na = len(a)

    def _dfs_list(i, record):
        if i >= na:
            # to do remove replacement
            gathered_list.append(copy.deepcopy(record))
            return
        for z in a[i]:
            for zz in z:
                record.append(zz)
            _dfs_list(i + 1, record)
            for zz in z:
                record.pop()

    _dfs_list(0, [])

    return gathered_list


def or_flatten_or_and_list(a):
    ret_list = list()
    for zz in a:
        ret_list.extend(zz)
    return ret_list


def build_graph_from_srl_tree(node, G, pre_conds=[[]]):
    if hasattr(node, 'main_list'):
        ret_list = node.main_list
    else:
        ret_list = [[node.desp]]
    ret_list = and_flatten_or_and_list([ret_list, pre_conds])

    if node.type == 'ITEM':
        node.has_verb = False
    else:
        node.has_verb = True

    if len(node.next) == 0:
        return ret_list

    rst_list = list()
    for nd in node.next:
        rst_list.append(build_graph_from_srl_tree(nd, G, pre_conds))
        if nd.has_verb:
            node.has_verb = True

    if node.type == 'ITEM':
        from_list = list()
        to_list = list()
        equ_list = list()
        for tag, rst, fword, ch in zip(node.tags, rst_list, node.span_first_word, node.next):
            print(tag)
            print(rst)
            if tag == 'B-ARGM-PRP' and ch.has_verb:
                to_list.append(rst)
            elif ch.has_verb:
                from_list.append(rst)
            else:
                equ_list.append(rst)

        if len(from_list) > 0:
            from_list.extend(equ_list)
            equ_list = list()

        equ_list.append(ret_list)
        ret_list = and_flatten_or_and_list(equ_list)

        print('from_list:', from_list)
        print('equ_list:', equ_list)
        print('to_list:', to_list)

        if len(from_list) > 0:
            from_list = and_flatten_or_and_list(from_list)
            for fr in from_list:
                fr_node = G.add_and_node(fr)
                for v in ret_list:
                    for vv in v:
                        vv_node = G.add_unary_node(vv)
                        print(fr_node, '->', vv_node)
                        G.add_edge_with_node(fr_node, vv_node, with_sent=node.ref)

        if len(to_list) > 0:
            to_list = and_flatten_or_and_list(to_list)
            for v in ret_list:
                v_node = G.add_and_node(v)
                for to in to_list:
                    for tt in to:
                        to_node = G.add_unary_node(tt)
                        G.add_edge_with_node(v_node, to_node, with_sent=node.ref)
    elif node.type == 'AND':
        ret_list = and_flatten_or_and_list(rst_list)
    elif node.type == 'OR':
        ret_list = or_flatten_or_and_list(rst_list)
    else:
        raise NotImplementedError

    return ret_list


def build_tree_from_sent(sent):
    tokens = my_tokenizer(sent)
    sent, tokens = exchange_nn_with_ving(sent, tokens)

    print('*********************coreference resolution inference start***********************')
    coref_rst = coreference_resolution_inference(tokens)
    coref_lab = list(range(len(tokens)))

    def _find_root(ii):
        if coref_lab[ii] != ii:
            coref_lab[ii] = _find_root(coref_lab[ii])
        return coref_lab[ii]

    clusters = coref_rst['clusters']
    print(clusters)
    span_text_map = dict()
    for i in range(len(tokens)):
        span_text_map[i] = [tokens[i]]
    for cluster in clusters:
        for span in cluster:
            span_text_map[span[0]] = tokens[span[0]:span[1] + 1]
            for i in range(span[0] + 1, span[1] + 1):
                coref_lab[_find_root(i)] = _find_root(span[0])
        for span in cluster:
            coref_lab[_find_root(span[0])] = _find_root(cluster[0][0])

    for i in range(len(tokens)):
        print(i, coref_lab[i], ':', span_text_map[i])
    print('*********************coreference resolution inference end***********************')

    print('*********************replace coreferred phrase start***********************')
    new_tokens = list()
    for k in list(range(len(tokens))):
        k = _find_root(k)
        if len(new_tokens) == 0 or k != new_tokens[-1]:
            new_tokens.append(k)
    z = list()
    for k in new_tokens:
        z.extend(span_text_map[k])
    tokens = z
    sent = ' '.join(tokens)
    print(sent)
    print('*********************replace coreferred phrase end***********************')

    print('*********************semantic role labeling start***********************')
    srl_rst = semantic_role_labeling(tokens)
    words = tokens

    main_tags = ['V', 'ARGM-MOD',
                 'ARG0', 'R-ARG0', 'C-ARG0',
                 'ARG1', 'R-ARG1', 'C-ARG1',
                 'ARG2', 'R-ARG2', 'C-ARG2',
                 'ARG3', 'R-ARG3', 'C-ARG3',
                 ]

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
            elif zlb in main_tags:
                w_list.append(k)
        verb_dict['verb_k'] = verb_k
        verb_dict['main_part'] = w_list
        # verb_dict['desp'] = w_list.copy()
        verb_dict['desp'] = ' '.join([tokens[z] for z in w_list])

    print('*********************semantic role labeling end***********************')

    print('*********************build SRL tree start***********************')
    new_verb_dicts = list()
    for verb_dict in verb_dicts:
        if verb_dict['verb_k'] is not None:
            new_verb_dicts.append(verb_dict)
    verb_dicts = new_verb_dicts
    verb_dicts.sort(reverse=True, key=lambda z: (len(z['related_words']), -z['verb_k']))

    root = build_node([0, len(tokens)])
    srl_tree_root = build_srl_tree(0, len(tokens), root, verb_dicts, tokens)
    print('*********************build SRL tree end***********************')

    return srl_tree_root

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

    ############################################################################
    ############################################################################
    ############################################################################

    '''
    print('*********************coreference resolution inference start***********************')
    coref_rst = coreference_resolution_inference(tokens)
    coref_lab = list(range(len(tokens)))

    def _find_root(i):
        if coref_lab[i] != i:
            coref_lab[i] = _find_root(coref_lab[i])
        return coref_lab[i]

    clusters = coref_rst['clusters']
    print(clusters)
    span_text_map = dict()
    for i in range(len(tokens)):
        span_text_map[i] = tokens[i]
    for cluster in clusters:
        for span in cluster:
            span_text_map[span[0]] = sent[wd_st_map[span[0]]:wd_ed_map[span[1]]]
            for i in range(span[0] + 1, span[1] + 1):
                coref_lab[_find_root(i)] = _find_root(span[0])
        for span in cluster:
            coref_lab[_find_root(span[0])] = _find_root(cluster[0][0])

    for i in range(len(tokens)):
        print(i, coref_lab[i], ':', span_text_map[i])
    print('*********************coreference resolution inference end***********************')

    print('*********************replace coreferred phrase start***********************')

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
        print(nd)
        # if nd.type == 'ITEM' and type(nd.desp) is list:
        if type(nd.desp) is list:
            nd.desp = _list_to_str_with_coreference(nd.desp)
        for nt in nd.next:
            _dfs_srl_tree(nt)

    _dfs_srl_tree(srl_tree_root)

    print('*********************replace coreferred phrase end***********************')

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

    return rst_listA
    # '''


def generate_graph_from_paras(para_list):
    sents_record = dict()
    G = Graph()
    for para in tqdm(para_list):
        u_conds = list()
        c_conds = list()
        u_line, c_line = -1, -1
        for sent in para:
            sl = sent['num_line']
            if sl > c_line:
                u_conds.extend(and_flatten_or_and_list(c_conds))
                c_conds = list()
                u_line, c_line = c_line, sl

            hashv = hash(sent['sent'])
            if hashv not in sents_record:
                # try:
                tree_root = build_tree_from_sent(sent['sent'])
                # except:
                #     continue
                sents_record[hashv] = tree_root
            else:
                tree_root = sents_record[hashv]

            rst_list = build_graph_from_srl_tree(tree_root, G, pre_conds=u_conds)
            c_conds.append(rst_list)

    return G


def main():
    para_list = read_data(in_file)
    G = generate_graph_from_paras(para_list)

    with open("graph.pkl", 'wb') as f:
        pickle.dump(G, f)
    print("write graph to graph.pkl")
    # with open("graph.pkl", 'rb') as f:
    #     G = pickle.load(f)
    # G.show()


def visit_prenodes_from_text(text, G):
    node = G.get_closet_node(text)
    node_to_string(G, node)

    if node.node_type == 'sentence':
        n_id = None
        for ee in G.DG.out_edges(node.n_id):
            n_id = ee[1]
            break
        node = G.node_list[n_id]

    def _show(record_list):
        print()
        print()
        for i in range(len(record_list) - 1, 0, -1):
            print('-------------------')
            edge_to_string(G, [record_list[i], record_list[i - 1]])
            print('-------------------')
        print()
        print()
        print('################################################')

    def _dfs(nid, record_list):
        record_list.append(nid)
        in_edge_list = list(G.DG.in_edges(nid))
        in_nid_list = [ee[0] for ee in in_edge_list]
        if len(in_nid_list) == 0:
            _show(record_list)
            return
        for nid in in_nid_list:
            if nid in record_list:
                continue
            _dfs(nid, record_list)
            record_list.pop()

    _dfs(node.n_id, [])


def test_single_sentence():
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
    # sent = 'Upon receipt of an AUTHENTICATION REJECT message, if the message is received without integrity protection, the UE shall start timer T3247 with a random value uniformly drawn from the range between 30 minutes and 60 minutes, if the timer is not running.'
    # to do upon receipt
    # to do coreference resolution
    # sent = 'The UE shall initiate an attach or combined attach procedure on the expiry of timers T3411, T3402 or T3346; '
    # sent = 'When a partial native EPS security context is taken into use through a security mode control procedure, the MME and the UE shall delete the previously current EPS security context. '
    # sent = 'The EPS security context is taken into use by the UE and the MME, when the MME initiates a security mode control procedure or during the inter-system handover procedure from A/Gb mode to S1 mode or Iu mode to S1 mode.'
    # sent = 'The MME initiates the NAS security mode control procedure by sending a SECURITY MODE COMMAND message to the UE and starting timer T3460.'
    # to do deal with on the expiry
    # sent = 'When a non-current full native EPS security context is taken into use by a security mode cnotrol procedure, then the MME and the UE shall delete the previously current mapped EPS security context.'
    # sent = 'Upon receipt of the IDENTITY REQUEST message the UE shall send an IDENTITY RESPONSE message to the network.'
    # sent = 'Furthermore, the MME may, send a SERVICE ACCEPT message to complete the service request procedure, if no NAS security mode control procedure was initiated, the MME did not send a SERVICE ACCEPT message as specified above to perform an EPS bearer context status synchronization, and the MME did not initiate any of the procedures specified in item 1 to 4 above.'
    # sent = 'When the MME and the UE create an EPS security context using null integrity and null ciphering algorithm during an attach procedure for emergency bearer services , or a tracking area updating procedure for a UE that has a PDN connection for emergency bearer services, the MME and the UE shall delete the previous current EPS security context.'

    sent = 'In state EMM-DEREGISTERED, the UE initiates the attach procedure by sending an ATTACH REQUEST message to the MME, starting timer T3410 and entering state EMM-REGISTERED-INITIATED which is provided by the UE.'
    # sent = 'Upon expiry of timer T3247, the UE shall initiate an EPS attach procedure or tracking area updating procedure, if still needed, dependent on EMM state and EPS update status, or perform PLMN selection according to 3GPP TS 23.122.'
    # sent = 'The UE shall start the attach procedure and detach procedure'
    sent = 'Upon receipt of EMM cause #99, #100, or #111, the UE shall start the attach procedure or send a message to the MME when the UE receives an ATTACH REQUEST message and the message is without integrity protection'


    # sent = 'If running, the timer should stops.'
    # sent = ' I am a boy who plays football.'
    # sent = 'I am a boy'
    p = {
        'sent': sent,
        'num_line': int(2),
        'pos_in_line': int(3),
        'prefix': '{2, 3}',
    }
    paras = [[p]]
    G = generate_graph_from_paras(paras)
    print('#' * 40, 'Graph demonstration', '#' * 40)
    for ee in G.DG.edges:
        edge_to_string(G, ee)
        print('-------------------')

    exit(0)


def test_paras():
    fpath = 'test_paras.txt'
    paras = list()
    list_paras = list()
    with open(fpath, 'r') as f:
        for k, line in enumerate(f):
            line = line.strip()
            if line.startswith('-----------'):
                list_paras.append(paras)
                paras = list()
                continue
            p = {
                'sent': line,
                'num_line': int(k),
                'pos_in_line': int(0),
                'prefix': '{%d, 0}' % k,
            }
            paras.append([p])

    for paras in list_paras:
        G = generate_graph_from_paras(paras)
        print('#'*40, 'Graph demonstration','#'*40)
        for ee in G.DG.edges:
            edge_to_string(G, ee)
            print('-------------------')
    exit(0)


if __name__ == '__main__':
    test_single_sentence()
    test_paras()
    main()

    with open("graph.pkl", 'rb') as f:
        G = pickle.load(f)

    text = 'the secure exchange of NAS messages has been established'
    visit_prenodes_from_text(text, G)

    '''
    scc_list = list(nx.strongly_connected_components(G.DG))
    scc_nid_map = dict()
    for k, scc in enumerate(scc_list):
        scc_nid_map[k] = list(scc)[0]

    CDG = nx.condensation(G.DG, scc=scc_list)
    path = nx.dag_longest_path(CDG)
    for i in range(len(path) - 1):
        scc1 = scc_list[path[i]]
        scc2 = scc_list[path[i+1]]
        found = False
        for nd1 in scc1:
            for nd2 in scc2:
                if (nd1, nd2) in G.edge_reference:
                    edge_to_string(G, [nd1, nd2])
                    break
            if found: break

        print('xxxxxxxxxxxxxxxxxxxxxxx')
    exit(0)
    # '''
