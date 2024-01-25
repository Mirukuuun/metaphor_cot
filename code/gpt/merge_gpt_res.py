import os
import pickle

train_path = "../data/met_train.pkl"
test_path = "../data/met_test.pkl"
train = pickle.load(open(train_path, 'rb'))
test = pickle.load(open(test_path, 'rb'))

def list_allfile(path, all_files=[], all_py_files=[], ids=[]):
    if os.path.exists(path):
        files = os.listdir(path)
    else:
        print('this path not exist')
    for file in files:
        if os.path.isdir(os.path.join(path, file)):
            # list_allfile(os.path.join(path, file), all_files)
            continue
        else:
            all_files.append(os.path.join(path, file))
    for file in all_files:
        if file.endswith('.pkl'):
            all_py_files.append(file)
            ids.append(file[12:-4])
    return all_py_files, ids

files, ids = list_allfile("/aworkspace/datasets/ads/test/res35")
print(ids)
id2index_train = {}
for i in range(len(train['id'])):
    if train['id'][i] in ids:
        id2index_train[train['id'][i]] = i

id2index_test = {}
for i in range(len(test['id'])):
    if test['id'][i] in ids:
        id2index_test[test['id'][i]] = i
res_all = {"id":[], "raw_texts":[], "captions":[], "step_one":[], "step_two":[], "step_three":[], "labels":[]}
for (file, id) in zip(files, ids):
    f = pickle.load(open(file, 'rb'))
    res_all["id"].append(id)
    if id in id2index_train:
        res_all["raw_texts"].append(train["raw_texts"][id2index_train[id]])
        res_all["captions"].append(train["captions"][id2index_train[id]])
        res_all["step_one"].append(f["step_one"])
        res_all["step_two"].append(f["step_two"])
        res_all["step_three"].append(f["step_three"])
        res_all["labels"].append(train["labels"][id2index_train[id]])
    elif id in id2index_test:
        res_all["raw_texts"].append(test["raw_texts"][id2index_test[id]])
        res_all["captions"].append(test["captions"][id2index_test[id]])
        res_all["step_one"].append(f["step_one"])
        res_all["step_two"].append(f["step_two"])
        res_all["step_three"].append(f["step_three"])
        res_all["labels"].append(test["labels"][id2index_test[id]])
    else:
        print("error id: ",id)

print(len(res_all["id"]))
with open("./res_all_35.pkl", 'wb') as f:
    pickle.dump(res_all, f)