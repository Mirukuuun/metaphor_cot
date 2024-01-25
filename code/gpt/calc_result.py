import pickle
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

gpt_res = pickle.load(open("./res_all_llava_wo_msr.pkl", 'rb'))

def step_three_processor(id, text):
    text = text.replace(".", "")

    if len(text) == 2:
        print("label: {}, text: {}".format(0, text))
        return 0
    elif len(text) == 3:
        print("label: {}, text: {}".format(1, text))
        return 1
    else:
        print("error result:{}, id:{}".format(text, id))
        return 0

gpt_res["prediction"] = []
pos, neg = 0, 0
for (id, text, label) in zip(gpt_res["id"], gpt_res["step_three"], gpt_res["labels"]):
    pred = step_three_processor(id, text)
    gpt_res["prediction"].append(pred)
    if label == 0:
        neg += 1
    else:
        pos += 1


f1 = f1_score(gpt_res["labels"], gpt_res["prediction"])
acc = accuracy_score(gpt_res["labels"], gpt_res["prediction"])
precision = precision_score(gpt_res["labels"], gpt_res["prediction"])
recall = recall_score(gpt_res["labels"], gpt_res["prediction"])
print("f1:{}, acc:{}, precision:{}, recall:{}, pos:{}, neg:{}".format(f1, acc, precision, recall, pos, neg))

