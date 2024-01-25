import os
import pickle

gpt_res = pickle.load(open("./res_all.pkl", 'rb'))
llava_res = pickle.load(open("./llava_result_wo_msr.pkl", 'rb'))

id2index = {}
for i in range(len(gpt_res['id'])):
    id2index[gpt_res['id'][i]] = i

res_all = {"id":[], "labels":[], "step_three":[]}
for (id, pred) in zip(llava_res["id"], llava_res["cot_result"]):
    if id in id2index:
        res_all['id'].append(id)
        res_all['labels'].append(gpt_res['labels'][id2index[id]])
        res_all['step_three'].append(pred[0])

print(len(res_all["id"]))
with open("./res_all_llava_wo_msr.pkl", 'wb') as f:
    pickle.dump(res_all, f)