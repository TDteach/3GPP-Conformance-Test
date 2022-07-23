import json
import numpy as np


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



