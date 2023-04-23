# %%
from transformers import ViTFeatureExtractor, ViTModel
import torch.utils.data as data_utils
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from PIL import Image
from glob import glob
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import torch


def ret_data_y(datas):
    data_y = []
    for i in datas:
        path = i + '/label.txt'
        with open(path) as f:
            label = int(f.read())

        if label == 1:
            data_y.append([1.0, 0.0])
        else:
            data_y.append([0.0, 1.0])

    return np.array(data_y, dtype='float32')


feature_extractor = 'google/vit-base-patch16-224-in21k'
vit_model = 'google/vit-base-patch16-224-in21k'
feature_extractor = ViTFeatureExtractor.from_pretrained(feature_extractor)
vit_model = ViTModel.from_pretrained(vit_model, output_attentions=True)
# %%
datas = np.array(glob('datas/*'))
data_y = ret_data_y(datas)

dy = np.zeros(len(data_y))
dy += data_y[:, 0] == 1
dy[:192] += data_y[:192][:, 1] == 1

data_y = data_y[dy > 0]
datas = datas[dy > 0]
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

hidden_inp = np.array([np.load(data + '/hidden.npy') for data in datas])
hidden_inp = hidden_inp.astype('float32')
# %%
class ViTNet(nn.Module):
    def __init__(self):
        super(ViTNet, self).__init__()
        vit_model = 'google/vit-base-patch16-224-in21k'
        self.vit_model = ViTModel.from_pretrained(vit_model, output_attentions=True)

        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(768*20, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 1024),
            torch.nn.ReLU(),
        )

        self.output = torch.nn.Sequential(
            torch.nn.Linear(2560, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 2),
            torch.nn.Softmax()
        )

    def forward(self, x, hidden):
        out_lis = []
        for i in range(20):
            out = self.vit_model(x[:, i])
            out = out['last_hidden_state'][:, 0, :]
            out_lis.append(out)

        x = self.classifier(torch.stack(out_lis, dim=1))

        return self.output(torch.cat([x, hidden], dim=-1))


# %%
model = ViTNet()
device = torch.device("cuda")
model.to(device)

# まず全パラメータを勾配計算Falseにする
for param in model.vit_model.parameters():
    param.requires_grad = False

optimizer = optim.Adam([
    {'params': model.parameters(), 'lr': 1e-4}
])

"""
optimizer = optim.Adam([
    {'params': model.parameters(), 'lr': 1e-4}
])
"""
# 損失関数
criterion = nn.BCELoss()
# %%
class CustomDataset(Dataset):
    def __init__(self, data, hidden, labels):
        self.data = data
        self.hidden = hidden
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 学習データとラベルを返す
        x = self.data[idx]
        hidden = self.hidden[idx]
        y = self.labels[idx]
        return {'img': x, 'hidden': hidden, 'label': y}

# テンソルと配列を受け取ってデータセットを作成する関数
def create_dataset(data, hidden, labels):
    # 学習データをテンソルに変換する

    # ラベルをテンソルに変換する
    tensor_labels = torch.from_numpy(labels)
    tensor_hidden = torch.from_numpy(hidden)

    # データセットを作成する
    dataset = CustomDataset(data, tensor_hidden, tensor_labels)
    return dataset
# %%
dataset = create_dataset(ar_images, hidden_inp, data_y)

batch_size = 100

train_size = int(0.6 * len(dataset))
val_size = int(0.2 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = data_utils.random_split(dataset,
                                                                   [train_size,
                                                                    val_size,
                                                                    test_size])

train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, pin_memory=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=True, pin_memory=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=True, pin_memory=True)
# %%
num_epochs = 10
calc_steps = 1
best_val_loss = float('inf')

for epoch in range(num_epochs):
    model.train()
    for i, dt in enumerate(train_dataloader):
        optimizer.zero_grad()
        inputs = dt['img']
        hidden = dt['hidden']
        labels = dt['label']
        inputs, hidden, labels = inputs.to(device), hidden.to(device), labels.to(device)
        outputs = model(inputs, hidden)

        loss = criterion(outputs, labels)
        loss.backward()  # 逆伝播を計算
        optimizer.step()  # パラメータを更新

        print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_dataloader)}], Train Loss: {loss.item():.4f}')

    # validation dataで評価
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for dt in val_dataloader:
            inputs = dt['img']
            hidden = dt['hidden']
            labels = dt['label']
            inputs, hidden, labels = inputs.to(device), hidden.to(device), labels.to(device)
            outputs = model(inputs, hidden)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    val_loss /= len(val_dataloader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}')

    # 損失が減少していた場合はそのまま、減少していなかった場合は最良の重みをロード
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pt')
    else:
        model.load_state_dict(torch.load('best_model.pt'))
# %%
model.load_state_dict(torch.load('best_model.pt'))
# %%
model.eval()

outputs_list = []
labels_list = []

for dt in test_dataloader:
    inputs = dt['img']
    hidden = dt['hidden']
    labels = dt['label']
    inputs, hidden, labels = inputs.to(device), hidden.to(device), labels.to(device)
    outputs = model(inputs, hidden)

    outputs_list.append(outputs.detach().cpu().numpy())
    labels_list.append(labels.detach().cpu().numpy())

# numpy arrayに変換
outputs_array = np.concatenate(outputs_list, axis=0)
labels_array = np.concatenate(labels_list, axis=0)

print("Output array shape:", outputs_array.shape)
print("Labels array shape:", labels_array.shape)

# %%
np.average(outputs_array[:,0])
# %%
np.average(outputs_array[:,1])
# %%
np.average(labels_array[:,0])
# %%
np.average(labels_array[:,1])
# %%
outputs_array*labels_array
# %%
outputs_array
# %%

# %%
