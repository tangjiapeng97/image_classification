import argparse
import time
import datetime
from pathlib import Path
import numpy as np
import random
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR, MultiStepLR, LambdaLR
from torch.utils.data import DataLoader

from models import resnet
from models.efficientnet import EfficientNet
from models import se_resnet

from datasets import dataset

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        # output = F.log_softmax(x, dim=1)
        return x


def train(args, model, device, data_loader_train, criterion, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(data_loader_train):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: [{}/{}]\tStep: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, args.epochs, batch_idx, len(data_loader_train),
                100. * batch_idx / len(data_loader_train), loss.item()))


def test(model, device, data_loader_test, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for data, target in data_loader_test:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1)  # get the index of the max log-probability
            correct += torch.eq(target, pred).sum().item()
            y_true += target.cpu().numpy().tolist()
            y_pred += pred.cpu().numpy().tolist()
            # print(y_true, y_pred)

    test_loss /= len(data_loader_test.dataset)
    print(classification_report(y_true, y_pred, digits=4))
    print('\nTest Loss: {:.6f}\tAccuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(data_loader_test.dataset),
        100. * correct / len(data_loader_test.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Train Image Classification Example')
    parser.add_argument('--train-batch-size', default=128, type=int, help='input batch size for training')
    parser.add_argument('--test-batch-size', default=128, type=int, help='input batch size for testing')
    parser.add_argument('--epochs', default=50, type=int, help='number of epochs to train')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--milestones', default=[35,45], nargs='+',type=int, help='milestones of learning rate decay')
    parser.add_argument('--warm-up', action='store_true', default=False, help='whether to use warm up')
    parser.add_argument('--warm-up-epochs', default=5, type=int, help='warm_up epochs')
    parser.add_argument('--weight-decay', default=0, type=float, help='weight decay')
    parser.add_argument('--num-workers', default=0, type=int, help='number of subprocesses to use for data loading')
    parser.add_argument('--seed', default=42, type=int, help='random seed')
    parser.add_argument('--log-interval', default=10, type=int, help='interval of logging training status')
    parser.add_argument('--output-dir', default='checkpoint', type=str, help='path to save model')
    args = parser.parse_args()

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                    ])
    
    # transform = {
    #     'train': transforms.Compose([
    #                 transforms.RandomResizedCrop(224),
    #                 transforms.RandomHorizontalFlip(),
    #                 transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
    #                 transforms.RandomGrayscale(p=0.2),
    #                 transforms.ToTensor(),
    #                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    #                 ]),
    #     'test': transforms.Compose([
    #                 transforms.Resize(224),
    #                 transforms.ToTensor(),
    #                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    #                 ])
    # }
    
    dataset_train = datasets.MNIST('../data', train=True, download=True, transform=transform)
    dataset_test = datasets.MNIST('../data', train=False, download=False, transform=transform)
    # dataset_train = dataset.IMG_DATASET('./data', train=True, transform=transform['train'])
    # dataset_test = dataset.IMG_DATASET('./data', train=False, transform=transform['test'])

    data_loader_train = DataLoader(dataset_train, args.train_batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    data_loader_test = DataLoader(dataset_test, args.test_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=False)

    # define the network
    # num_classes = 10
    # # efficientnet
    # model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=num_classes)
    # # se_resnet
    # model = se_resnet.se_resnet50(pretrained=True)
    # model.fc = nn.Linear(model.fc.in_features, num_classes)
    # # resnet
    # model = resnet.resnet50(pretrained=True)
    # model.conv1 = nn.Conv2d(1, model.conv1.out_channels, kernel_size=7, stride=2, padding=3, bias=False)
    # model.fc = nn.Linear(model.fc.in_features, num_classes)
    # # test
    # input = torch.rand((1,3,224,224))
    # output = model(input)
    # print(output.shape)

    model = Net()
    model.to(device)

    train_criterion = nn.CrossEntropyLoss(reduction='mean')
    test_criterion = nn.CrossEntropyLoss(reduction='sum')

    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "fc" in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "fc" not in n and p.requires_grad],
            "lr": args.lr * 0.1,
        },
    ]
    optimizer = optim.Adam(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    
    if not args.warm_up:
        # multistep_lr
        scheduler = MultiStepLR(optimizer, milestones=args.milestones, gamma=0.1)
    else:
        # warm_up_with_multistep_lr
        warm_up_with_multistep_lr = lambda epoch: epoch / args.warm_up_epochs if epoch <= args.warm_up_epochs else 0.1**len([m for m in args.milestones if m <= epoch])
        scheduler = LambdaLR(optimizer, lr_lambda=warm_up_with_multistep_lr)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Start training")
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        print(epoch, optimizer.param_groups[0]['lr'])
        train(args, model, device, data_loader_train, train_criterion, optimizer, epoch)
        test(model, device, data_loader_test, test_criterion)
        torch.save(model.state_dict(), output_dir / f'checkpoint_{epoch}.pth')
        scheduler.step()
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == '__main__':
    main()