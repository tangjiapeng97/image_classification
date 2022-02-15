import argparse
import time
import datetime
from pathlib import Path
import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, DistributedSampler
import torch.distributed as dist

import util.misc as utils
from models import resnet

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
            with torch.no_grad():
                if args.distributed:
                    dist.all_reduce(loss)
                    loss /= utils.get_world_size()
            print('Train Epoch: [{}/{}]\tStep: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, args.epochs, batch_idx, len(data_loader_train),
                100. * batch_idx / len(data_loader_train), loss.item()))


def test(args, model, device, data_loader_test, criterion):
    model.eval()
    test_loss = 0
    total_correct = 0
    with torch.no_grad():
        for data, target in data_loader_test:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            pred = output.argmax(dim=1, keepdim=True)
            batch_correct = pred.eq(target.view_as(pred)).sum()
            if args.distributed:
                dist.all_reduce(loss)
                dist.all_reduce(batch_correct)
            test_loss += loss.item()
            total_correct += batch_correct.item()
    
    test_loss /= len(data_loader_test.dataset)

    print('\nTest Loss: {:.6f}\tAccuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, total_correct, len(data_loader_test.dataset),
        100. * total_correct / len(data_loader_test.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Train Image Classification Example')
    parser.add_argument('--train-batch-size', default=64, type=int, help='input batch size for training')
    parser.add_argument('--test-batch-size', default=64, type=int, help='input batch size for testing')
    parser.add_argument('--epochs', default=10, type=int, help='number of epochs to train')
    parser.add_argument('--lr', default=1.0, type=float, help='learning rate')
    parser.add_argument('--lr-drop', default=1, type=int, help='period of learning rate decay')
    parser.add_argument('--weight-decay', default=0, type=float, help='weight decay')
    parser.add_argument('--num-workers', default=4, type=int, help='number of subprocesses to use for data loading')
    parser.add_argument('--seed', default=42, type=int, help='random seed')
    parser.add_argument('--log-interval', default=10, type=int, help='interval of logging training status')
    parser.add_argument('--output-dir', default='checkpoint', type=str, help='path to save model')
    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', type=str, help='url used to set up distributed training')
    args = parser.parse_args()

    utils.init_distributed_mode(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset_train = datasets.MNIST('../data', train=True, download=True, transform=transform)
    dataset_test = datasets.MNIST('../data', train=False, download=False, transform=transform)

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_test = DistributedSampler(dataset_test, shuffle=False)
    else:
        sampler_train = RandomSampler(dataset_train)
        sampler_test = SequentialSampler(dataset_test)

    data_loader_train = DataLoader(dataset_train, args.train_batch_size, sampler=sampler_train, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    data_loader_test = DataLoader(dataset_test, args.test_batch_size, sampler=sampler_test, num_workers=args.num_workers, pin_memory=True, drop_last=False)

    #define the network
    num_classes = 10
    ## efficientnet
    # model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=num_classes)
    ## se_resnet
    # model = se_resnet.se_resnet50(pretrained=True)
    # model.fc = nn.Linear(model.fc.in_features, num_classes)
    ## resnet
    # model = resnet.resnet50(pretrained=True)
    # model.conv1 = nn.Conv2d(1, model.conv1.out_channels, kernel_size=7, stride=2, padding=3, bias=False)
    # model.fc = nn.Linear(model.fc.in_features, num_classes)
    ## test
    # input = torch.rand((1,3,224,224))
    # output = model(input)
    # print(output.shape)

    model = Net()
    model.to(device)

    train_criterion = nn.CrossEntropyLoss(reduction='mean')
    test_criterion = nn.CrossEntropyLoss(reduction='sum')

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "fc" in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "fc" not in n and p.requires_grad],
            "lr": args.lr * 0.1,
        },
    ]
    optimizer = optim.Adadelta(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=args.lr_drop, gamma=0.1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Start training")
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train(args, model, device, data_loader_train, train_criterion, optimizer, epoch)
        test(args, model, device, data_loader_test, test_criterion)
        utils.save_on_master(model_without_ddp.state_dict(), output_dir / f'checkpoint_{epoch}.pth')
        scheduler.step()
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == '__main__':
    main()