import fire
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
from warmup_scheduler import GradualWarmupScheduler
import os
import time
import json
from pprint import pprint
import random
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from torchvision.transforms.functional import to_pil_image
from utils import progress_bar, parse_args, get_model_name, select_model, select_optimizer, select_scheduler, \
    get_train_transform, get_test_transform, get_dataset
from datetime import datetime
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


def main(**kwargs):
    start = datetime.now()
    print('\n[+] Parse arguments')
    args, kwargs = parse_args(kwargs)
    pprint(args)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0  # best test accuracy

    print('\n[+] Create log dir')
    model_name = get_model_name(args)
    log_dir = os.path.join('./runs', args.exp_name, model_name)
    os.makedirs(os.path.join(log_dir, 'model'))
    json.dump(kwargs, open(os.path.join(log_dir, 'kwargs.json'), 'w'))
    writer = SummaryWriter(log_dir=log_dir)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    print('\n[+] Create network')
    net = select_model(args)
    print(net)
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

    if args.warmup:
        scheduler = GradualWarmupScheduler(
                    optimizer,
                    multiplier=1.5,
                    total_epoch=5,
                    after_scheduler=scheduler
                )

    if args.dataset == "comic":
        weights = []
        for title in sorted(os.listdir("./data/comic/")):
            title_path = os.path.join("./data/comic/", title)
            weights.append(1/len(os.listdir(title_path)))  # データの逆数
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights).to(device))
    else:
        criterion = nn.CrossEntropyLoss()
    if args.use_cuda:
        net = net.cuda()
        criterion = criterion.cuda()

    # Data
    print('\n[+] Load dataset')
    transform_train = get_train_transform(args, net, log_dir)
    transform_val = get_test_transform(args, net)
    transform_test = get_test_transform(args, net)
    trainset = get_dataset(args, transform_train, args.train_split)

    # save_image(trainset[0][0], "tmp/rand.png")
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    valset = get_dataset(args, transform_val, 'val')
    valloader = torch.utils.data.DataLoader(
        valset, batch_size=100, shuffle=False, num_workers=args.num_workers)

    testset = get_dataset(args, transform_test, 'test')
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=args.num_workers)
    print("Train:", len(trainset))
    print("Val:", len(valset))
    print("Test", len(testset))
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    if device == 'cuda':
        cudnn.benchmark = True

    # Training
    def train(epoch):
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        total_steps = len(trainloader)

        for batch_idx, (inputs, targets) in enumerate(trainloader):
            # if batch_idx == 0:
            #     save_image(inputs[0], "tmp/{}.png".format(epoch))
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

    def valid(epoch):
        nonlocal best_acc
        net.eval()
        valid_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(valloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                valid_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                progress_bar(batch_idx, len(valloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % (valid_loss / (batch_idx + 1), 100. * correct / total, correct, total))

        writer.add_scalar('valid/acc1', correct / total, global_step=epoch)
        writer.add_scalar('valid/loss', valid_loss / (batch_idx + 1), global_step=epoch)

        # Save checkpoint.
        acc = 100. * correct / total
        if acc > best_acc:
            best_acc = acc
            print('Saving..')
            if torch.cuda.device_count() > 1:
                state = {
                    'net': net.module.state_dict(),
                    'acc': acc,
                    'epoch': epoch,
                }
            else:
                state = {
                    'net': net.state_dict(),
                    'acc': acc,
                    'epoch': epoch,
                }
            torch.save(state, os.path.join(log_dir, "model", "model.pt"))

        return acc

    best_test_acc = 0

    def test(epoch):
        nonlocal best_test_acc
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

        writer.add_scalar('test/acc1', correct/total, global_step=epoch)
        writer.add_scalar('test/loss', test_loss/(batch_idx+1), global_step=epoch)

        acc = 100.*correct/total
        best_test_acc = max(best_test_acc, acc)

        return acc

    print('\n[+] Start training')
    if torch.cuda.device_count() > 1:
        print('\n[+] Use {} GPUs'.format(torch.cuda.device_count()))
        net = nn.DataParallel(net)

    test_last_acc = 0
    for epoch in range(args.epochs):
        train(epoch)
        valid(epoch)
        test_last_acc = test(epoch)

    writer.add_scalar('test/best_acc', best_test_acc)
    writer.add_scalar('test/last_acc', test_last_acc)

    # val best を用いて検証
    model = select_model(args)
    model = model.to(device)
    state = torch.load(os.path.join(log_dir, "model", "model.pt"))
    model.load_state_dict(state['net'])
    preds = []

    if 'cifar' in args.dataset:
        labels = testset.targets
    elif 'svhn' in args.dataset:
        labels = testset.labels
    elif 'comic' in args.dataset:
        labels = testset.targets

    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            preds += predicted.detach().cpu().numpy().tolist()

    print("best epoch:", state['epoch'])
    print(confusion_matrix(labels, preds))
    print(classification_report(labels, preds))

    print("Train Started at {}".format(start))
    finish = datetime.now()
    print("Train Finished at {}".format(finish))
    print("Elapsed Time: {}".format(finish - start))
    print("val best_accuracy:", best_acc)
    print("test best accuracy:", best_test_acc)

    return best_acc, test_last_acc, best_test_acc


if __name__ == "__main__":
    # fire.Fire(main)
    l = []
    N = 3
    for _ in range(N):
        l.append(fire.Fire(main)[2])
    print(l)
    ave = sum(l)/N
    print(ave)
    s = sum((x-ave)**2 for x in l)/N
    print(s**0.5)
