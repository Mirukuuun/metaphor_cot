import pickle

train = pickle.load(open('../data/met_train.pkl', 'rb'))
test = pickle.load(open('../data/met_test.pkl', 'rb'))
dev = pickle.load(open('../data/met_dev.pkl', 'rb'))

import re
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer

# nltk.download('stopwords')
# nltk.download('wordnet')
def tokenize(sentence):
    sentence = re.sub(r'\s+', ' ', sentence)
    token_words = word_tokenize(sentence)
    token_words = pos_tag(token_words)
    return token_words


wordnet_lematizer = WordNetLemmatizer()


def stem(token_words):
    words_lematizer = []
    for word, tag in token_words:
        if tag.startswith('NN'):
            word_lematizer = wordnet_lematizer.lemmatize(word, pos='n')
        elif tag.startswith('VB'):
            word_lematizer = wordnet_lematizer.lemmatize(word, pos='v')
        elif tag.startswith('JJ'):
            word_lematizer = wordnet_lematizer.lemmatize(word, pos='a')
        elif tag.startswith('R'):
            word_lematizer = wordnet_lematizer.lemmatize(word, pos='r')
        else:
            word_lematizer = wordnet_lematizer.lemmatize(word)
        words_lematizer.append(word_lematizer)
    return words_lematizer


sr = stopwords.words('english')

def delete_stopwords(token_words):
    cleaned_words = [word for word in token_words if word not in sr]
    return cleaned_words


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False


characters = [' ', ',', '.', 'DBSCAN', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%', '-', '...',
              '^', '{', '}']


def delete_characters(token_words):
    words_list = [word for word in token_words if word not in characters]
    return words_list


def to_lower(token_words):
    words_lists = [x.lower() for x in token_words]
    return words_lists


def pre_process(text):
    token_words = tokenize(text)
    token_words = delete_characters(token_words)
    token_words = stem(token_words)
    token_words = to_lower(token_words)
    token_words = delete_stopwords(token_words)
    return token_words
print(len(train['id']))
print(len(train['raw_texts']))
for i in range(len(train['id'])):
    text = train['raw_texts'][i]
    res = pre_process(text)
    # print(res)
    train['raw_texts'][i] = ' '.join(res)

for i in range(len(test['id'])):
    text = test['raw_texts'][i]
    res = pre_process(text)
    # print(res)
    test['raw_texts'][i] = ' '.join(res)

for i in range(len(dev['id'])):
    text = dev['raw_texts'][i]
    res = pre_process(text)
    # print(res)
    dev['raw_texts'][i] = ' '.join(res)

with open('../data/met_textclean_train.pkl', 'wb') as f:
    pickle.dump(train, f)

with open('../data/mettextclean_test.pkl', 'wb') as f:
    pickle.dump(test, f)

with open('../data/met_textclean_dev.pkl', 'wb') as f:
    pickle.dump(dev, f)