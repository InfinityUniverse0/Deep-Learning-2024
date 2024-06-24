"""
Main Script
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from trainer import Trainer
from visualization import plot_loss_accuracy, plot_confusion_matrix
import argparse

from models.cnn_base import CNN
from models.resnet import ResNet
from models.densenet import DenseNet
from models.mobilenet import MobileNet

supported_models = [
    'cnn_base',
    'resnet18',
    'resnet34',
    'resnet50',
    'resnet101',
    'resnet152',
    'densenet121',
    'densenet169',
    'densenet201',
    'densenet161',
    'mobilenet'
]

# Parse command line arguments
parser = argparse.ArgumentParser(description='Train a model on CIFAR10')
parser.add_argument('--model', type=str, default='resnet18', choices=supported_models, help='which model to use')
args = parser.parse_args()

# Configuration
DATA_PATH = 'data'
BATCH_SIZE = 64
SAVE_PATH = os.path.join('results', args.model)

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
print('device: {}'.format(device))

# Data Loading & Preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)
train_dataset = datasets.CIFAR10(root=DATA_PATH, train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataset = datasets.CIFAR10(root=DATA_PATH, train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
classes = train_dataset.classes
print('Classes: {}'.format(classes))

# Model & Optimization
if args.model == 'cnn_base':
    model = CNN(num_classes=len(classes)).to(device)
elif args.model == 'resnet18':
    model = ResNet(num=18, num_classes=len(classes)).to(device)
elif args.model == 'resnet34':
    model = ResNet(num=34, num_classes=len(classes)).to(device)
elif args.model == 'resnet50':
    model = ResNet(num=50, num_classes=len(classes)).to(device)
elif args.model == 'resnet101':
    model = ResNet(num=101, num_classes=len(classes)).to(device)
elif args.model == 'resnet152':
    model = ResNet(num=152, num_classes=len(classes)).to(device)
elif args.model == 'densenet121':
    model = DenseNet(num=121, num_classes=len(classes)).to(device)
elif args.model == 'densenet169':
    model = DenseNet(num=169, num_classes=len(classes)).to(device)
elif args.model == 'densenet201':
    model = DenseNet(num=201, num_classes=len(classes)).to(device)
elif args.model == 'densenet161':
    model = DenseNet(num=161, num_classes=len(classes)).to(device)
elif args.model == 'mobilenet':
    model = MobileNet(num_classes=len(classes)).to(device)
else:
    raise NotImplementedError

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
print(model)

# Training
num_epochs = 50
trainer = Trainer(model, criterion, optimizer, device)
train_loss_list, train_accuracy_list, valid_loss_list, valid_accuracy_list = trainer.train(
    train_loader, valid_loader=test_loader, num_epochs=num_epochs
)

# Evaluation
test_loss, test_accuracy, conf_mat = trainer.evaluate(test_loader)

# Save Model
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)
torch.save(model.state_dict(), os.path.join(SAVE_PATH, 'model.pth'))
print('Model Saved')

# Visualization
plot_loss_accuracy(train_loss_list, valid_loss_list, train_accuracy_list, valid_accuracy_list, save_path=SAVE_PATH)
plot_confusion_matrix(conf_mat, classes, normalize=True, save_path=SAVE_PATH)
