from allennlp.predictors.predictor import Predictor
import allennlp_models.tagging
import numpy as np
from spacy.lang.en import English

G_NLP = English()
G_NLP.add_pipe('sentencizer')


data_file = '/home/tangd/chen481/ConfTest/withTD/data/toTD_24.301.txt'


def clear_sent(sent):
    prefix = ['-',']','\t',' ']
    while sent[0] in prefix:
        print(sent)
        sent = sent[1:]
        print(sent)
    return sent.strip()


def main():
    case_list = list()
    sent_list = list()
    with open(data_file, 'r') as f:
        for line in f:
            if line.startswith('->->->'):
                if len(sent_list) > 0:
                    case_list.append(sent_list)
                sent_list = list()
            else:
                sent_list.append(line)

    sent_lens = [len(sent_list) for sent_list in case_list]
    rst = np.histogram(sent_lens, bins=np.arange(1, np.max(sent_lens)+1))
    print(rst)


    predictor = Predictor.from_path(
        "https://storage.googleapis.com/allennlp-public-models/elmo-constituency-parser-2020.02.10.tar.gz",
        cuda_device=0,
    )


    first_words = list()
    assum_words = ['when','once','if','upon','on','after','while','whenever']

    for sent_list in case_list:

        write_out = False
        cond_k = 0

        '''
        print('[raw]:')
        out_str = str()
        for k, l in enumerate(sent_list):
            print('[%d]: '%(k+1), l)
        print('-----------------------------')


        for l in sent_list[:-1]:
            cond_k += 1
            print('[cond_%d]: '%cond_k, l.strip())
        '''

        sents = sent_list[-1].strip()
        doc = G_NLP(sents)
        splited_sent = [str(s) for s in doc.sents]

        for sent in splited_sent:

            sent = clear_sent(sent)

            cond_list = list()
            opra_list = list()

            rst = predictor.predict(sentence=sent)
            rt = rst['hierplane_tree']['root']
            child_list = rt['children']
            for k, ch in enumerate(child_list):
                if ch['link'] == 'VP':
                    kk = k-1
                    while kk > 0 and child_list[kk]['link'] != 'NP':
                        kk -= 1
                    opra_list.append(child_list[kk]['word'])
                    opra_list.append(ch['word'])
                elif ch['link'] == 'SBAR' or ch['link'] == 'PP':
                    cond_list.append(ch['word'])

                    fw = ch['word'].split(' ')[0].lower()
                    first_words.append(fw)
                    if fw not in assum_words:
                        write_out = True

            cond_k += 1

            if write_out:
                print('[sent_%d]: '%cond_k, sent)
                print('[cond_%d]: '%cond_k, cond_list)
                print('[opra_%d]: '%cond_k, opra_list)

        if write_out: print('->->->')

    values, counts = np.unique(first_words, return_counts=True)
    order = np.argsort(counts)
    order = np.flip(order)
    values = values[order]
    counts = counts[order]
    for v,c in zip(values, counts):
        print(v, c)


if __name__ == '__main__':
    main()
