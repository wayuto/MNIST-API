import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class NN(nn.Module):
    def __init__(self, dropout_rate=0.4):
        super(NN, self).__init__()
        
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.01, inplace=True)
        )
        self.pool1 = nn.MaxPool2d(2)
        
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ELU(alpha=1.0, inplace=True)
        )
        self.pool2 = nn.MaxPool2d(2)
        
        self.fc_block = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.Mish(inplace=True),
            nn.Dropout(dropout_rate * 0.7),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.5),
            
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.pool1(x)
        x = self.conv_block2(x)
        x = self.pool2(x)
        x = torch.flatten(x, 1)
        x = self.fc_block(x)
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
