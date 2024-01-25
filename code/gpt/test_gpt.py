import base64
import os
import time

import requests
import pickle
import pandas as pd

# OpenAI API Key
api_key = ""
train_path = "./gpt_train.pkl"
test_path = "./gpt_test.pkl"
# train = pickle.load(open(train_path, 'rb'))
test = pickle.load(open(test_path, 'rb'))

def list_allfile(path, all_files=[], all_py_files=[]):
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
            all_py_files.append(file[15:-4])
    return all_py_files

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')



def ask_gpt(id, step, text, caption, step_one, step_two):
    # Path to your image
    base_path = r"/aworkspace/datasets/ads"
    image_path = os.path.join(base_path, id + '.jpg')

    if step == 1:
        text = f"Text: {text} Image caption: {caption} Please provide main entities in this sample.(just give the words)"
    elif step == 2:
        text = f"Text: {text} Image caption: {caption} Main entities:{step_one} What kind of connection between these entities?"
    elif step == 3:
        text = f"Text: {text} Image caption: {caption} Main entities:{step_one} Connection between these entities: {step_two} Is this sample contains metaphor? (just return yes or no)"
    else:
        raise ValueError("step error")
    # Getting the base64 string
    base64_image = encode_image(image_path)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": text
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "low"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 100
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    print(response.json())
    result = response.json()['choices'][0]['message']['content']

    return result

# print(train.keys())['id', 'raw_texts', 'labels', 'source', 'target', 'sentiment', 'template', 'captions', 'text_attributes', 'image_attributes', 'image_prefix', 'cot_result', 'image_blip']
errors = []
ided = list_allfile(r"/aworkspace/datasets/ads/test")
# print(ided)
# print(idd)

# id2index = {}
# for i in range(len(train['id'])):
#     if train['id'][i] in idd:
#         id2index[train['id'][i]] = i


for (id, text, caption) in zip(test['id'], test['raw_texts'], test['captions']):
    try:
        res = {}
        print("id:{}, text:{}, caption:{}".format(id, text, caption))
        if id in ided:
            print("already get")
            continue
        result1 = ask_gpt(id, 1, text, caption, "", "")

        print("result 1:", result1)
        # time.sleep(60)
        result2 = ask_gpt(id, 2, text, caption, result1, "")

        print("result 2:", result2)
        # time.sleep(60)
        result3 = ask_gpt(id, 3, text, caption, result1, result2)

        print("result 3:", result3)

        res["step_one"] = result1
        res["step_two"] = result2
        res["step_three"] = result3
        base = r'/aworkspace/datasets/ads/test'
        path = os.path.join(base, id+'.pkl')
        with open(path , 'wb') as f:
            pickle.dump(res, f)
        # time.sleep(60)
    except Exception as e:
        print(e)
        errors.append(id)
        break