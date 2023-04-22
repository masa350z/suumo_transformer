# %%
from transformers import ViTFeatureExtractor, ViTModel
from PIL import Image
from glob import glob
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


def ret_data_y(datas):
    data_y = []
    for i in datas:
        path = i + '/label.txt'
        with open(path) as f:
            label = int(f.read())

        if label == 1:
            data_y.append([1, 0])
        else:
            data_y.append([0, 1])

    return np.array(data_y)


feature_extractor = 'google/vit-base-patch16-224-in21k'
vit_model = 'google/vit-base-patch16-224-in21k'
feature_extractor = ViTFeatureExtractor.from_pretrained(feature_extractor)
vit_model = ViTModel.from_pretrained(vit_model, output_attentions=True)

datas = glob('datas/*')
data_y = ret_data_y(datas)
# %%
ar_images = []

for data in tqdm(datas):
    images = glob(data + '/img/*.jpg')

    mx_len = 20
    zeros = torch.zeros((mx_len, 3, 224, 224))

    images_ar = [Image.open(i) for i in images]
    input_ars = feature_extractor(images=images_ar, return_tensors="pt")
    input_ars = input_ars['pixel_values']

    size = input_ars.size()[0]
    zeros[:size, :, :, :] = input_ars
    input_ars = zeros

    ar_images.append(input_ars)

ar_images = torch.stack(ar_images)


# %%
class ViTNet(nn.Module):
    def __init__(self):
        super(ViTNet, self).__init__()
        vit_model = 'google/vit-base-patch16-224-in21k'
        self.vit_model = ViTModel.from_pretrained(vit_model, output_attentions=True)

        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(768*20, 12400),
            torch.nn.ReLU(),
            torch.nn.Linear(12400, 5120),
            torch.nn.ReLU(),
            torch.nn.Linear(5120, 2560),
            torch.nn.ReLU(),
            torch.nn.Linear(2560, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 2),
            torch.nn.Softmax(),
        )

    def forward(self, x):
        out_lis = []
        for i in range(20):
            out = self.vit_model(x[:, i])
            out = out['last_hidden_state'][:, 0, :]
            out_lis.append(out)

        x = self.classifier(torch.stack(out_lis, dim=1))

        return x


# %%
model = ViTNet()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model.to(device)
# %%
# まず全パラメータを勾配計算Falseにする
for param in model.parameters():
    param.requires_grad = False

# 追加したクラス分類用の全結合層を勾配計算ありに変更
for param in model.fc.parameters():
    param.requires_grad = True

optimizer = optim.Adam([
    {'params': model.fc.parameters(), 'lr': 1e-4}
])

# 損失関数
criterion = nn.CrossEntropyLoss()
# %%
