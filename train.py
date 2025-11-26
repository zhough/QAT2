import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import model
from torchvision import datasets, transforms
from infer_utils import eval
# 定义数据转换
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.0,), (1/3,))
])

# 加载MNIST数据集
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

# 定义数据加载器
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64*2, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

net = model.Net()
net.to('cuda' if torch.cuda.is_available() else 'cpu')
# 定义优化器
#optimizer = optim.Adam(net.parameters(), lr=0.002)
optimizer = optim.AdamW(net.parameters(), lr=0.001, weight_decay=1e-4)
Epochs = 10
for epoch in range(Epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to('cuda' if torch.cuda.is_available() else 'cpu')
        labels = labels.to('cuda' if torch.cuda.is_available() else 'cpu')
        outputs = net(images)
        loss = F.cross_entropy(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{Epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}，lr={optimizer.param_groups[0]["lr"]:.6f}')


net.save_int8_weights("weight2.npz")

p = eval(200,'weight2.npz')
print(p)
