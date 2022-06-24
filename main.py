from allennlp.predictors.predictor import Predictor
import allennlp_models.tagging
import numpy as np


def main():
    sent_list = list()
    with open('sample_sentences.txt', 'r') as f:
        for line in f:
            line = line.strip()
            if len(line) <= 0: continue
            sent_list.append(line)

    predictor = Predictor.from_path(
        "https://storage.googleapis.com/allennlp-public-models/elmo-constituency-parser-2020.02.10.tar.gz",
    )

    for sent in sent_list:
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
        print(cond_list)
        print(opra_list)


if __name__ == '__main__':
    main()
