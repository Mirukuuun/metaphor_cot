import pickle

import pandas as pd

df = pd.read_csv("/aworkspace/zml/datasets/multimeme/E_text.csv", encoding="gb18030")
# print(df)

label_df = pd.read_csv("/aworkspace/zml/datasets/multimeme/label_E.csv", encoding="gb18030")
# print(label_df)
meme = {}
for i in range(len(df['file_name'])):
    # dic['id'].append(df['file_name'][i][:-4])
    # dic['raw_texts'].append(df['text'][i])
    # id2index[df['file_name'][i][:-4]] = i
    meme[df['file_name'][i][:-4]] = {}
    meme[df['file_name'][i][:-4]]['text'] = df['text'][i]

for i in range(len(label_df['file_name'])):
    # index = id2index[label_df['file_name'][i][:-4]]
    # dic['labels'].append(label_df['metaphor occurrence'][i])
    meme[label_df['file_name'][i][:-4]]['label'] = label_df['metaphor occurrence'][i]
# print(meme)
neg = []
pos = []
for k, v in meme.items():
    if v['label'] == 0:
        neg.append({k: v})
    else:
        pos.append({k: v})
pos_len, neg_len, total_len = len(pos), len(neg), len(pos) + len(neg)

# train_len = int(total_len * 0.8)
# val_len = total_len - train_len
# pos_train_len = int(pos_len * 0.8)
# pos_val_len = pos_len - pos_train_len
# neg_train_len = int(neg_len * 0.8)
# neg_val_len = neg_len - neg_train_len

# train_len = int(total_len * 0.8)
# test_len = total_len - train_len
# pos_train_len = int(pos_len * 0.8)
# pos_test_len = pos_len - pos_train_len
# neg_train_len = int(neg_len * 0.8)
# neg_test_len = neg_len - neg_train_len
def split_list(lst, ratios, num_splits):
    if len(ratios) != num_splits:
        raise ValueError("The length of ratios must equal to num_splits.")
    total_ratio = sum(ratios)
    if total_ratio != 1:
        raise ValueError("The sum of ratios must be equal to 1.")
    n = len(lst)
    result = []
    start = 0
    for i in range(num_splits):
        end = start + int(n * ratios[i])
        result.append(lst[start:end])
        start = end
    return result


# print(
#     "pos num:{}, neg num:{}, total num:{}, pos train num:{}, pos val num:{}, pos test num:{}, neg train num:{}, neg val num:{}, neg test num:{}, train num:{}, val num:{}, test num:{}".format(
#         len(pos), len(neg), len(pos) + len(neg), pos_train_len, pos_val_len, pos_test_len, neg_train_len, neg_val_len, neg_test_len, train_len,
#         val_len, test_len))
pos_split = split_list(pos, [0.6, 0.2, 0.2], 3)
neg_split = split_list(neg, [0.6, 0.2, 0.2], 3)

train, val, test = pos_split[0] + neg_split[0], pos_split[1] + neg_split[1], pos_split[2] + neg_split[2]
print(
    "pos num:{}, neg num:{}, total num:{}, pos train num:{}, pos val num:{}, pos test num:{}, neg train num:{}, neg val num:{}, neg test num:{}, train num:{}, val num:{}, test num:{}".format(
        len(pos), len(neg), len(pos) + len(neg), len(pos_split[0]), len(pos_split[1]), len(pos_split[2]), len(neg_split[0]), len(neg_split[1]), len(neg_split[2]), len(train),
        len(val), len(test)))
# train += pos[:pos_train_len]
# train += neg[:neg_train_len]
# test += pos[pos_train_len:]
# test += neg[neg_train_len:]

# val_len = int(train_len * 0.2)
# train_len = train_len - val_len
# pos_val_len = int(pos_train_len * 0.2)
# pos_train_len = pos_train_len - pos_val_len
# neg_val_len = int(pos_val_len * 0.2)
# neg_train_len = neg_train_len - neg_val_len

# val += train[]

print("Final: train len:{}, val len:{}".format(len(train), len(val)))

train_dic = {'id': [], 'raw_texts': [], 'labels': [], 'source': [], 'target': [], 'sentiment': [], 'template': [],
             'captions': [], 'text_attributes': [], 'image_attributes': [], 'image_prefix': []}
val_dic = {'id': [], 'raw_texts': [], 'labels': [], 'source': [], 'target': [], 'sentiment': [], 'template': [],
           'captions': [], 'text_attributes': [], 'image_attributes': [], 'image_prefix': []}
test_dic = {'id': [], 'raw_texts': [], 'labels': [], 'source': [], 'target': [], 'sentiment': [], 'template': [],
            'captions': [], 'text_attributes': [], 'image_attributes': [], 'image_prefix': []}

for item in train:
    for k, v in item.items():
        # print(k, ': ', v)
        train_dic['id'].append(k)
        train_dic['raw_texts'].append(v['text'])
        train_dic['labels'].append(v['label'])

for item in val:
    for k, v in item.items():
        val_dic['id'].append(k)
        val_dic['raw_texts'].append(v['text'])
        val_dic['labels'].append(v['label'])

for item in test:
    for k, v in item.items():
        test_dic['id'].append(k)
        test_dic['raw_texts'].append(v['text'])
        test_dic['labels'].append(v['label'])

print("Train dic:{}, val dic:{}, test dic:{}".format(len(train_dic['id']), len(val_dic['id']), len(test_dic['id'])))

# with open('./meme_train.pkl', 'wb') as f:
#     pickle.dump(train_dic, f)
#
# with open('./meme_dev.pkl', 'wb') as f:
#     pickle.dump(val_dic, f)
#
# with open('./meme_test.pkl', 'wb') as f:
#     pickle.dump(test_dic, f)
