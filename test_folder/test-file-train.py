import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import argparse
import os




# 모델 정의
class MySimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.my_seq = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 256),  # 784 -> 256
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    def forward(self, x):
        x = self.my_seq(x)
        return x

# 학습 모듈
def train(args):
    print("저장경로======: ", args.model_dir)
    model = MySimpleNet()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device=device)
    
    tr = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # 데이터셋 다운로드 및 로드
    mnist_train = datasets.MNIST(
                                 root=args.training,
                                 train=True,
                                 download=True,
                                 transform=tr)
    train_loader = DataLoader(mnist_train,
                              batch_size=args.batch_size,
                              shuffle=True)

    # 학습
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr)
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")
    
    # 모델 저장
    model_path = os.path.join(args.model_dir, 'model.pth')
    torch.save(model.state_dict(), model_path)

# 추론 모듈
def model_fn(model_dir):
    model = MySimpleNet()
    with open(os.path.join(model_dir, 'model.pth'), 'rb') as f:
        model.load_state_dict(torch.load(f))
    return model


if __name__ == '__main__':
    # 파라미터 정의
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch-size', type=int,   default=32)
    parser.add_argument('--epochs',     type=int,   default=1)
    parser.add_argument('--lr',         type=float, default=0.001)
    
    # parser.add_argument('--training', type=str, default='opt/ml/input/data/training')
    parser.add_argument('--training',  type=str, default=os.environ.get('SM_CHANNEL_TRAINING'))
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    args = parser.parse_args()

    train(args)
