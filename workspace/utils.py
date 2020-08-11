'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset
import collections
import pickle as cp
from models import *
from transforms_range import *

from sklearn.model_selection import StratifiedShuffleSplit

DATASET_PATH = './data/'
current_epoch = 0


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time


def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH * current / total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width - int(TOTAL_BAR_LENGTH) - len(msg) - 3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width - int(TOTAL_BAR_LENGTH / 2) + 2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current + 1, total))

    if current < total - 1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def format_time(seconds):
    days = int(seconds / 3600 / 24)
    seconds = seconds - days * 3600 * 24
    hours = int(seconds / 3600)
    seconds = seconds - hours * 3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds * 1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


def dict_to_namedtuple(d):
    Args = collections.namedtuple('Args', sorted(d.keys()))

    for k, v in d.items():
        if type(v) is dict:
            d[k] = dict_to_namedtuple(v)

        elif type(v) is str:
            try:
                d[k] = eval(v)
            except:
                d[k] = v

    args = Args(**d)
    return args


def parse_args(kwargs):
    # combine with default args
    kwargs['dataset'] = kwargs['dataset'] if 'dataset' in kwargs else 'cifar10'
    kwargs['num_classes'] = kwargs['num_classes'] if 'num_classes' in kwargs else 10
    kwargs['network'] = kwargs['network'] if 'network' in kwargs else 'wresnet28_2'
    kwargs['optimizer'] = kwargs['optimizer'] if 'optimizer' in kwargs else 'sgd'
    kwargs['lr'] = kwargs['lr'] if 'lr' in kwargs else 0.1
    kwargs['seed'] = kwargs['seed'] if 'seed' in kwargs else None
    kwargs['use_cuda'] = kwargs['use_cuda'] if 'use_cuda' in kwargs else True
    kwargs['use_cuda'] = kwargs['use_cuda'] and torch.cuda.is_available()
    kwargs['num_workers'] = kwargs['num_workers'] if 'num_workers' in kwargs else 4
    kwargs['scheduler'] = kwargs['scheduler'] if 'scheduler' in kwargs else 'cosine'
    kwargs['batch_size'] = kwargs['batch_size'] if 'batch_size' in kwargs else 128
    kwargs['epochs'] = kwargs['epochs'] if 'epochs' in kwargs else 200
    kwargs['ind_train_epochs'] = kwargs['ind_train_epochs'] if 'ind_train_epochs' in kwargs else 120
    kwargs['warmup'] = kwargs['warmup'] if 'warmup' in kwargs else False
    kwargs['auto_augment'] = kwargs['auto_augment'] if 'auto_augment' in kwargs else False
    kwargs['auto_augment_ind_train'] =kwargs['auto_augment_ind_train'] if 'auto_augment_ind_train' in kwargs else False
    kwargs['rand_augment'] = kwargs['rand_augment'] if 'rand_augment' in kwargs else False
    kwargs['augment_path'] = kwargs['augment_path'] if 'augment_path' in kwargs else None
    kwargs['mag'] = kwargs['mag'] if 'mag' in kwargs else 10  # [0, 30]
    kwargs['tinit'] = kwargs['tinit'] if 'tinit' in kwargs else 0.05
    kwargs['tfin'] = kwargs['tfin'] if 'tfin' in kwargs else 0.05

    # to named tuple
    args = dict_to_namedtuple(kwargs)
    return args, kwargs


def get_model_name(args):
    from datetime import datetime
    now = datetime.now()
    date_time = now.strftime("%B_%d_%H:%M:%S")
    model_name = '__'.join([date_time, args.network, str(args.seed)])
    return model_name


def select_model(args):
    if args.network == "wresnet28_2":
        net = WideResNet(depth=28, widen_factor=2, dropout_rate=0.0, num_classes=args.num_classes)
    elif args.network == "wresnet28_10":
        net = WideResNet(depth=28, widen_factor=10, dropout_rate=0.0, num_classes=args.num_classes)
    elif args.network == "RegNetX_200MF":
        net = RegNetX_200MF()
    else:  # cifer10以外のデータセットを使う場合はクラス数の指定に注意
        net = WideResNet(depth=28, widen_factor=2, dropout_rate=0.0, num_classes=args.num_classes)

    # net = VGG('VGG19')
    # net = ResNet18()
    # net = PreActResNet18()
    # net = GoogLeNet()
    # net = DenseNet121()
    # net = ResNeXt29_2x64d()
    # net = MobileNet()
    # net = MobileNetV2()
    # net = DPN92()
    # net = ShuffleNetG2()
    # net = SENet18()
    # net = ShuffleNetV2(1)
    # net = EfficientNetB0()
    # net = RegNetX_200MF()
    # net = PyramidNet('cifar10', depth=116, alpha=200, num_classes=10, bottleneck=True)  # originally depth=272
    # net = WideResNet(depth=28, widen_factor=2, dropout_rate=0.0, num_classes=10)  # 'wresnet28_2or10'
    # net = WideResNet(depth=40, widen_factor=2, dropout_rate=0.0, num_classes=10)  # 'wresnet40_2'

    return net


def select_optimizer(args, net):
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr,
                                    momentum=0.9, weight_decay=5e-4)
    elif args.optimizer == 'rms':
        # optimizer = torch.optim.RMSprop(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-5)
        optimizer = torch.optim.RMSprop(net.parameters(), lr=args.lr)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    else:
        raise Exception('Unknown Optimizer')
    return optimizer


def select_scheduler(args, optimizer):
    if not args.scheduler or args.scheduler == 'None':
        return None
    elif args.scheduler == 'clr':
        return torch.optim.lr_scheduler.CyclicLR(optimizer, 0.01, 0.015, mode='triangular2', step_size_up=250000,
                                                 cycle_momentum=False)
    elif args.scheduler == 'exp':
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9999283, last_epoch=-1)
    elif args.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0.)
    else:
        raise Exception('Unknown Scheduler')


def get_train_transform(args, model, log_dir=None):
    if args.rand_augment:
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            RandAugment(n=1, m=4),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            CutoutDefault(length=16),
        ])

    elif args.auto_augment:
        assert args.dataset == 'cifar10'

        from tdga_augment import tdga_augment
        if args.augment_path:
            transform = cp.load(open(args.augment_path, 'rb'))
            os.system('cp {} {}'.format(
                args.augment_path, os.path.join(log_dir, 'augmentation.cp')))
        else:
            transform = tdga_augment(args, model, log_dir=log_dir)
            if log_dir:
                cp.dump(transform, open(os.path.join(log_dir, 'augmentation.cp'), 'wb'))
    elif args.auto_augment_ind_train:  # 個体でミニデータを用いて学習する
        assert args.dataset == 'cifar10'

        from tdga_augment_ind_train import tdga_augment
        if args.augment_path:
            transform = cp.load(open(args.augment_path, 'rb'))
            os.system('cp {} {}'.format(
                args.augment_path, os.path.join(log_dir, 'augmentation.cp')))
        else:
            transform = tdga_augment(args, log_dir=log_dir)
            if log_dir:
                cp.dump(transform, open(os.path.join(log_dir, 'augmentation.cp'), 'wb'))
    elif args.dataset == 'cifar10':
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            CutoutDefault(length=16),
        ])

    elif args.dataset == 'imagenet':
        resize_h, resize_w = model.img_size[0], int(model.img_size[1] * 1.875)
        transform = transforms.Compose([
            transforms.Resize([resize_h, resize_w]),
            transforms.RandomCrop(model.img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])

    else:
        raise Exception('Unknown Dataset')

    print(transform)

    return transform


def get_test_transform(args, model):
    if args.dataset == 'cifar10':
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    elif args.dataset == 'imagenet':
        resize_h, resize_w = model.img_size[0], int(model.img_size[1] * 1.875)
        val_transform = transforms.Compose([
            transforms.Resize([resize_h, resize_w]),
            transforms.ToTensor()
        ])

    else:
        raise Exception('Unknown Dataset')

    return val_transform


def split_dataset(args, dataset, k):
    # load dataset
    X = list(range(len(dataset)))
    Y = dataset.targets

    # split to k-fold
    assert len(X) == len(Y)

    def _it_to_list(_it):
        return list(zip(*list(_it)))

    sss = StratifiedShuffleSplit(n_splits=k, random_state=args.seed, test_size=0.1)
    Dm_indexes, Da_indexes = _it_to_list(sss.split(X, Y))

    return Dm_indexes, Da_indexes


def get_dataset(args, transform, split='train'):
    assert split in ['train', 'val', 'test', 'trainval']

    if args.dataset == 'cifar10':
        train = split in ['train', 'val', 'trainval']
        dataset = torchvision.datasets.CIFAR10(DATASET_PATH,
                                               train=train,
                                               transform=transform,
                                               download=True)

        if split in ['train', 'val']:
            split_path = os.path.join(DATASET_PATH,
                                      'cifar-10-batches-py', 'train_val_index.cp')

            if not os.path.exists(split_path):
                [train_index], [val_index] = split_dataset(args, dataset, k=1)
                split_index = {'train': train_index, 'val': val_index}
                cp.dump(split_index, open(split_path, 'wb'))

            split_index = cp.load(open(split_path, 'rb'))
            dataset = Subset(dataset, split_index[split])

    elif args.dataset == 'imagenet':
        dataset = torchvision.datasets.ImageNet(DATASET_PATH,
                                                split=split,
                                                transform=transform,
                                                download=(split is 'val'))

    else:
        raise Exception('Unknown dataset')

    return dataset
