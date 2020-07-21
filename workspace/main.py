import fire
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from warmup_scheduler import GradualWarmupScheduler
import os
import time
import json
from pprint import pprint
import random
import argparse
from torch.utils.tensorboard import SummaryWriter

from utils import progress_bar, parse_args, get_model_name, select_model, select_optimizer, select_scheduler
from datetime import datetime


def main(**kwargs):
    start = datetime.now()
    print('\n[+] Parse arguments')
    args, kwargs = parse_args(kwargs)
    pprint(args)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0  # best test accuracy

    print('\n[+] Create log dir')
    model_name = get_model_name(args)
    log_dir = os.path.join('./runs', model_name)
    os.makedirs(os.path.join(log_dir, 'model'))
    json.dump(kwargs, open(os.path.join(log_dir, 'kwargs.json'), 'w'))
    writer = SummaryWriter(log_dir=log_dir)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    print('\n[+] Create network')
    net = select_model(args)
    net = net.to(device)
    optimizer = select_optimizer(args, net)
    scheduler = select_scheduler(args, optimizer)
    # originally used scheduler
    # def func(epoch):
    #     if epoch < 150:
    #         return 0.1
    #     elif epoch < 250:
    #         return 0.01
    #     else:
    #         return 0.001
    # scheduler = LambdaLR(optimizer, lr_lambda=func)
    # if args.warmup:
    #     scheduler = GradualWarmupScheduler(
    #                 optimizer,
    #                 multiplier=1,
    #                 total_epoch=5,
    #                 after_scheduler=scheduler
    #             )
    criterion = nn.CrossEntropyLoss()
    if args.use_cuda:
        net = net.cuda()
        criterion = criterion.cuda()

    # Data
    print('\n[+] Load dataset')

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=args.num_workers)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    # Training
    def train(epoch):
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        total_steps = len(trainloader)
        steps = 0

        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

            scheduler.step(epoch - 1 + float(batch_idx+1) / total_steps)
            print(scheduler.get_lr())

        writer.add_scalar('train/learning_rate', optimizer.param_groups[0]['lr'], global_step=epoch)
        writer.add_scalar('train/acc1', correct / total, global_step=epoch)
        writer.add_scalar('train/loss', train_loss / (batch_idx+1), global_step=epoch)

    def test(epoch):
        nonlocal best_acc
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

        writer.add_scalar('valid/acc1', correct/total, global_step=epoch)
        writer.add_scalar('valid/loss', test_loss/(batch_idx+1), global_step=epoch)

        # Save checkpoint.
        acc = 100.*correct/total
        if acc > best_acc:
            best_acc = acc
            print('Saving..')
            state = {
                'net': net.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            torch.save(state, os.path.join(log_dir, "model", "model.pt"))

    print('\n[+] Start training')
    if torch.cuda.device_count() > 1:
        print('\n[+] Use {} GPUs'.format(torch.cuda.device_count()))
        net = nn.DataParallel(net)

    for epoch in range(args.epochs):
        train(epoch)
        test(epoch)
    print("Train Started at {}".format(start))
    finish = datetime.now()
    print("Train Finished at {}".format(finish))
    print("Elapsed Time: {}".format(finish - start))
    print("best_accuracy:", best_acc)


if __name__ == "__main__":
    fire.Fire(main())
