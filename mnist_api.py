import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms as transforms

from random import random
import os

transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize((0.5,), (0.5,))
])

class NN(nn.Module):
    def __init__(self) -> None:
        super(NN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
        
device = torch.device("cpu")
model = NN().to(device)

script_dir = os.path.dirname(os.path.abspath(__file__))
model.load_state_dict(torch.load(f"{script_dir}/model/mnist_model.pth", map_location=torch.device('cpu')))
model.eval()

def predict(matrix: list[list[float]]) -> int:
    matrix = torch.tensor(matrix, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(matrix)
        _, predicted = torch.max(output, 1)
    return predicted.item()
