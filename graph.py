import torch
import networkx as nx
import matplotlib.pyplot as plt
import copy
from sentence_transformers import util, SentenceTransformer
from utils import calc_embeddings_for_sent

# embedding_model_path = 'stsb-distilbert-base'
embedding_model_path = 'stsb-mpnet-base-v2'
# node_merge_thr = 0.5378
# node_merge_thr = 0.6141
node_merge_thr = 0.7

class NodeContent:
    def __init__(self, n_id, text, embedding):
        self.n_id = n_id
        self.text = copy.copy(text)

        self.embedding = None
        self.gp_avg_embedding = None
        self.gp_sum_embedding = None

        self.parent = None
        self.group_size = 1
        self.group_list = [self]

        if embedding is not None:
            self.embedding = torch.clone(embedding)
            self.gp_avg_embedding = torch.clone(embedding)
            self.gp_sum_embedding = torch.clone(embedding)
            self.node_type = 'sentence'
        else:
            self.node_type = 'functional'

    def merge_with(self, node):
        if self.embedding is None:
            raise NotImplementedError
        self.group_size += 1
        self.gp_sum_embedding += node.embedding
        self.gp_avg_embedding = self.gp_sum_embedding / self.group_size
        self.group_list.append(node)

    def __str__(self):
        return str(self.n_id)+' ['+str(self.node_type)+'] '+self.text


class Graph:
    def __init__(self):
        self.DG = nx.DiGraph()
        self.node_list = list()
        self.embedding_model = SentenceTransformer(embedding_model_path)
        self.root_embeddings = None
        self.text_nid_map = dict()
        self.nid_eid_map = dict()
        self.eid_nid_map = dict()
        self.n_node = 0
        self.merge_thr = node_merge_thr
        self.edge_reference = dict()

    def _find_root(self, node):
        if node.parent is not None:
            node.parent = self._find_root(node.parent)
            return node.parent
        else:
            return node

    def _merge_to_root(self, root, node):
        root = self._find_root(root)
        if node.n_id != root.n_id:
            node.parent = root
            root.merge_with(node)
            k = self.nid_eid_map[root.n_id]
            self.root_embeddings[k] = root.gp_avg_embedding[0]

        return root

    def get_node_by_text(self, text):
        # print(text.encode())
        hashv = hash(text.encode())
        if hashv not in self.text_nid_map:
            return None, hashv
        i = self.text_nid_map[hashv]
        # print(hashv)
        node = self.node_list[i]
        # print(node)
        node = self._find_root(node)
        # print(node)
        return node, hashv


    def add_node(self, text, with_embedding=True, return_duplicated=False):
        # print(text)

        node, hashv = self.get_node_by_text(text)

        if node is not None:
            # node_to_string(self, node.n_id)
            # print('duplicated 8888888888888888888888888888888888888')
            if return_duplicated:
                return node, True
            else:
                return node

        if with_embedding:
            embedding = calc_embeddings_for_sent(sent=text, model=self.embedding_model)
            # embedding = self.embedding_model.encode(text, convert_to_tensor=True, normalize_embeddings=True)
            # embedding = torch.unsqueeze(embedding, 0)
            embedding = embedding.detach().cpu()
        else:
            embedding = None

        nid = self.n_node
        node = NodeContent(nid, text, embedding)
        self.node_list.append(node)
        self.text_nid_map[hashv] = nid
        self.n_node += 1

        if embedding is not None:
            if self.root_embeddings is None:
                self.root_embeddings = embedding
                self.nid_eid_map[node.n_id] = 0
                self.eid_nid_map[0] = node.n_id
            else:
                rst = util.semantic_search(embedding, self.root_embeddings, top_k=1, score_function=util.dot_score)
                k, sc = rst[-1][-1]['corpus_id'], rst[-1][-1]['score']
                if sc > self.merge_thr:
                    i = self.eid_nid_map[k]
                    node = self._merge_to_root(self.node_list[i], node)
                else:
                    n_root = len(self.root_embeddings)
                    self.root_embeddings = torch.cat([self.root_embeddings, embedding], 0)
                    self.nid_eid_map[node.n_id] = n_root
                    self.eid_nid_map[n_root] = node.n_id

        self.DG.add_node(node.n_id)

        # node_to_string(self, node.n_id)
        # print('8888888888888888888888888888888888888')
        if return_duplicated:
            return node, False
        else:
            return node

    def add_edge_with_nid(self, nid1, nid2, with_sent=None):
        if nid1 != nid2:
            self.DG.add_edge(nid1, nid2)
            ee = (nid1, nid2)
            if ee not in self.edge_reference:
                self.edge_reference[ee] = set()
            if with_sent:
                self.edge_reference[ee].add(with_sent)

    def add_edge_with_node(self, node1, node2, with_sent=None):
        self.add_edge_with_nid(node1.n_id, node2.n_id, with_sent)

    def add_edge_with_text(self, text1, text2, with_sent=None):
        node1 = self.add_node(text1)
        node2 = self.add_node(text2)
        self.add_edge_with_nid(node1.n_id, node2.n_id, with_sent)

    def add_unary_node(self, text):
        node = self.add_node(text)
        desp = 'unary-' + str(node.n_id)
        unary_node, duplicated = self.add_node(desp, with_embedding=False, return_duplicated=True)
        if not duplicated:
            self.add_edge_with_nid(node.n_id, unary_node.n_id)
            unary_node.content = [node.n_id]
        return unary_node

    def add_and_node(self, text_list):
        nid_list = list()
        for text in text_list:
            node = self.add_node(text)
            nid_list.append(node.n_id)
        nid_list = list(set(nid_list))
        nid_list.sort()

        if len(nid_list) == 1:
            return self.add_unary_node(text_list[0])

        desp = '-and-'.join([str(x) for x in nid_list])
        and_node, duplicated = self.add_node(desp, with_embedding=False, return_duplicated=True)
        if not duplicated:
            # for nid in nid_list:
            #     self.add_edge_with_nid(and_node.n_id, nid)
            and_node.content = nid_list

        return and_node

    def show(self):
        for i in range(len(self.node_list)):
            print(i, self.node_list[i].node_type, self.node_list[i].text)
        nx.draw(self.DG, with_labels=True)
        plt.show()

    def get_closet_node(self, text):
        node, hashv = self.get_node_by_text(text)
        if node is not None:
            return node

        embedding = calc_embeddings_for_sent(sent=text, model=self.embedding_model)
        # embedding = self.embedding_model.encode(text, convert_to_tensor=True, normalize_embeddings=True)
        # embedding = torch.unsqueeze(embedding, 0)
        embedding = embedding.detach().cpu()

        rst = util.semantic_search(embedding, self.root_embeddings, top_k=1, score_function=util.dot_score)
        k, sc = rst[-1][-1]['corpus_id'], rst[-1][-1]['score']
        i = self.eid_nid_map[k]
        node = self.node_list[i]
        return node


def node_to_string(G, node):
    if type(node) is int:
        node = G.node_list[node]
    if isinstance(node, NodeContent):
        print(node)
    else:
        raise NotImplementedError
    if node.node_type == 'functional':
        for sid in node.content:
            _node = G.node_list[sid]
            print('\t', _node)


def edge_to_string(G, ee):
    node_to_string(G, ee[0])
    print('******** point to **********')
    node_to_string(G, ee[1])

    print(':::::: explain sentences ::::::::::')
    sents = G.edge_reference[(ee[0], ee[1])]
    sents = list(set(sents))
    sents.sort()
    for k, sent in enumerate(sents):
        print('[%d]'%k, sent)
        print()



if __name__ == '__main__':
    G = Graph()
    G.add_edge('I am a boy', 'He is a girl')
    G.add_edge('I am a boy', 'I am a boy too')
    G.add_edge('gaga is not a a boy', 'I am a boy too')
    G.add_and_node(['I am a boy', 'he is a boy', 'I am a girl neither'])
    G.show()


