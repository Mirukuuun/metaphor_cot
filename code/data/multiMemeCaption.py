import pickle

captions = pickle.load(open('./meme_captions.pkl', 'rb'))
train = pickle.load(open('./meme_train.pkl', 'rb'))
test = pickle.load(open('./meme_test.pkl', 'rb'))

print(train.keys())
print(captions.keys())

id2index = {}
for i in range(len(captions['id'])):
    id2index[captions['id'][i]] = i
# print(id2index)

for i in range(len(train['id'])):
    if train['id'][i] == 'image_ (3151)':
        train['captions'].append('')
    else:
        train['captions'].append(captions['caption'][id2index[train['id'][i]]])

for i in range(len(test['id'])):
    test['captions'].append(captions['caption'][id2index[test['id'][i]]])

with open('./meme_train.pkl', 'wb') as f:
    pickle.dump(train, f)

with open('./meme_test.pkl', 'wb') as f:
    pickle.dump(test, f)


