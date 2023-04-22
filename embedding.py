# %%
import os
from tqdm import tqdm
from glob import glob
import numpy as np
import openai

CHATGPT_KEY = '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'
openai.api_key = CHATGPT_KEY
# %%
datas = glob('datas/*')

for dir_ in tqdm(datas):
    if not os.path.exists(dir_ + '/hidden.npy'):
        texts = glob(dir_ + '/text/*.txt')

        text = ''
        for i in texts:
            with open(i) as f:
                text += f.read()
        text = text.replace('\n', '').replace('\t', '')

        response = openai.Embedding.create(input=[text], model="text-embedding-ada-002")
        hidden_vector = np.array(response['data'][0]['embedding'])

        np.save(dir_ + '/hidden.npy', hidden_vector)
# %%
