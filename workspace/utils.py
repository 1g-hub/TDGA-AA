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
from torch.utils.data.dataset import ConcatDataset
from comic_dataset import ComicDataset
from fourscene_dataset import FourSceneDataset
import collections
import pickle as cp
from models import *
from transforms_range_prob import *

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
    kwargs['exp_name'] = kwargs['exp_name'] if 'exp_name' in kwargs else 'hoge'
    kwargs['dataset'] = kwargs['dataset'] if 'dataset' in kwargs else 'cifar10'
    kwargs['num_classes'] = kwargs['num_classes'] if 'num_classes' in kwargs else 10
    kwargs['network'] = kwargs['network'] if 'network' in kwargs else 'wresnet28_2'
    kwargs['optimizer'] = kwargs['optimizer'] if 'optimizer' in kwargs else 'sgd'
    kwargs['lr'] = kwargs['lr'] if 'lr' in kwargs else 0.1
    kwargs['weight_decay'] = kwargs['weight_decay'] if 'weight_decay' in kwargs else 5e-4
    kwargs['seed'] = kwargs['seed'] if 'seed' in kwargs else None
    kwargs['use_cuda'] = kwargs['use_cuda'] if 'use_cuda' in kwargs else True
    kwargs['use_cuda'] = kwargs['use_cuda'] and torch.cuda.is_available()
    kwargs['num_workers'] = kwargs['num_workers'] if 'num_workers' in kwargs else 4
    kwargs['scheduler'] = kwargs['scheduler'] if 'scheduler' in kwargs else 'cosine'
    kwargs['batch_size'] = kwargs['batch_size'] if 'batch_size' in kwargs else 128
    kwargs['epochs'] = kwargs['epochs'] if 'epochs' in kwargs else 200
    kwargs['pre_train_epochs'] = kwargs['pre_train_epochs'] if 'pre_train_epochs' in kwargs else 200
    kwargs['warmup'] = kwargs['warmup'] if 'warmup' in kwargs else False
    kwargs['auto_augment'] = kwargs['auto_augment'] if 'auto_augment' in kwargs else False
    kwargs['rand_augment'] = kwargs['rand_augment'] if 'rand_augment' in kwargs else False
    kwargs['augment_path'] = kwargs['augment_path'] if 'augment_path' in kwargs else None
    kwargs['ckpt_path'] = kwargs['ckpt_path'] if 'ckpt_path' in kwargs else None
    kwargs['mag'] = kwargs['mag'] if 'mag' in kwargs else 5  # [0, 30]
    kwargs['tinit'] = kwargs['tinit'] if 'tinit' in kwargs else 0.05
    kwargs['tfin'] = kwargs['tfin'] if 'tfin' in kwargs else 0.05
    kwargs['B'] = kwargs['B'] if 'B' in kwargs else 16
    kwargs['Np'] = kwargs['Np'] if 'Np' in kwargs else 64
    kwargs['Ng'] = kwargs['Ng'] if 'Ng' in kwargs else 30
    kwargs['train_split'] = kwargs['train_split'] if 'train_split' in kwargs else 'train'
    kwargs['prob_mul'] = kwargs['prob_mul'] if 'prob_mul' in kwargs else 1
    kwargs['select_gamma'] = kwargs['select_gamma'] if 'select_gamma' in kwargs else 'nonrest'
    kwargs['allele_max'] = kwargs['allele_max'] if 'allele_max' in kwargs else 1
    kwargs['denom_of_gamma'] = kwargs['denom_of_gamma'] if 'denom_of_gamma' in kwargs else 6.31

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
    if args.dataset == "fourscene-comic":
        assert args.ckpt_path
        wrn_model = WideResNet(depth=28, widen_factor=2, dropout_rate=0.0, num_classes=args.num_classes)
        data = torch.load(args.ckpt_path)
        wrn_model.load_state_dict(data['net'])
        wrn_model.eval()
        net = LSTMFourScene(wrn_model=wrn_model)
    elif args.network == "wresnet28_2":
        net = WideResNet(depth=28, widen_factor=2, dropout_rate=0.0, num_classes=args.num_classes)
    elif args.network == "wresnet28_10":
        net = WideResNet(depth=28, widen_factor=10, dropout_rate=0.0, num_classes=args.num_classes)
    elif args.network == "pyramidnet":  # only for cifer-10
        net = PyramidNet('cifar10', depth=272, alpha=200, num_classes=args.num_classes,
                         bottleneck=True)  # originally depth=272 alpha=200
    else:
        net = WideResNet(depth=28, widen_factor=2, dropout_rate=0.0, num_classes=args.num_classes)

    return net


def select_optimizer(args, net):
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr,
                                    momentum=0.9, weight_decay=args.weight_decay)
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
            RandAugment(n=1, m=args.mag),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            CutoutDefault(length=16),
        ])

    elif args.auto_augment:
        assert args.dataset == 'cifar10' or args.dataset == 'cifar100' or 'svhn' in args.dataset or "comic" in args.dataset

        from tdga_augment import tdga_augment

        if args.augment_path:
            transform = cp.load(open(args.augment_path, 'rb'))
            os.system('cp {} {}'.format(
                args.augment_path, os.path.join(log_dir, 'augmentation.cp')))
        else:
            transform = tdga_augment(args, model, log_dir=log_dir)
            if log_dir:
                cp.dump(transform, open(os.path.join(log_dir, 'augmentation.cp'), 'wb'))
    elif args.dataset == 'cifar10' or args.dataset == 'cifar100' or 'svhn' in args.dataset:
        if args.dataset == 'cifar10' or 'svhn' in args.dataset:
            MEAN, STD = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        elif args.dataset == 'cifar100':
            MEAN, STD = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)

        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
            # CutoutDefault(length=16),
        ])

    elif args.dataset == 'imagenet':
        resize_h, resize_w = model.img_size[0], int(model.img_size[1] * 1.875)
        transform = transforms.Compose([
            transforms.Resize([resize_h, resize_w]),
            transforms.RandomCrop(model.img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])

    elif "comic" in args.dataset:
        MEAN, STD = (0.8017, 0.8015, 0.8015), (0.2930, 0.2930, 0.2930)
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            # Senga(prob=0.5, mag=1),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
            # CutoutDefault(length=32),
        ])
    else:
        raise Exception('Unknown Dataset')

    print(transform)

    return transform


def get_test_transform(args, model):
    if args.dataset == 'cifar10' or args.dataset == 'cifar100' or 'svhn' in args.dataset:
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

    elif "comic" in args.dataset:
        MEAN, STD = (0.8017, 0.8015, 0.8015), (0.2930, 0.2930, 0.2930)
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
    else:
        raise Exception('Unknown Dataset')

    return val_transform


def split_dataset(args, dataset, k):
    # load dataset
    X = list(range(len(dataset)))

    if 'cifar' in args.dataset:
        Y = dataset.targets
    elif 'svhn' in args.dataset:
        Y = dataset.labels
    elif 'comic' in args.dataset:
        Y = dataset.targets
    else:
        raise Exception('Unknown dataset')

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

    elif args.dataset == 'cifar100':
        train = split in ['train', 'val', 'trainval']
        dataset = torchvision.datasets.CIFAR100(DATASET_PATH,
                                                train=train,
                                                transform=transform,
                                                download=True
                                                )
        if split in ['train', 'val']:
            split_path = os.path.join(DATASET_PATH,
                                      'cifar-100-python', 'train_val_index.cp')

            if not os.path.exists(split_path):
                [train_index], [val_index] = split_dataset(args, dataset, k=1)
                split_index = {'train': train_index, 'val': val_index}
                cp.dump(split_index, open(split_path, 'wb'))

            split_index = cp.load(open(split_path, 'rb'))
            dataset = Subset(dataset, split_index[split])

    elif args.dataset == 'svhn-core':
        train = 'train' if split in ['train', 'val', 'trainval'] else 'test'
        dataset = torchvision.datasets.SVHN(os.path.join(DATASET_PATH, 'svhn-core-python'),
                                            split=train,
                                            transform=transform,
                                            download=True
                                            )
        if split in ['train', 'val']:
            split_path = os.path.join(DATASET_PATH,
                                      'svhn-core-python', 'train_val_index.cp')

            if not os.path.exists(split_path):
                [train_index], [val_index] = split_dataset(args, dataset, k=1)
                split_index = {'train': train_index, 'val': val_index}
                cp.dump(split_index, open(split_path, 'wb'))

            split_index = cp.load(open(split_path, 'rb'))
            dataset = Subset(dataset, split_index[split])

    elif args.dataset == 'comic':
        train = 'train' if split in ['train', 'trainval'] else 'test'
        dataset = ComicDataset(os.path.join(DATASET_PATH, 'comic'),
                               split=train,
                               transform=transform
                               )

    elif args.dataset == 'fourscene-comic':
        train = 'train' if split in ['train', 'trainval'] else 'test'
        dataset = FourSceneDataset(os.path.join(DATASET_PATH, 'comic'),
                                   split=train,
                                   transform=transform
                                   )

    # elif args.dataset == 'svhn-full':
    #     train = 'train' if split in ['train', 'val', 'trainval'] else 'test'
    #     if train == 'train':
    #         trainset = torchvision.datasets.SVHN(os.path.join(DATASET_PATH, 'svhn-full-python'),
    #                                              split=train,
    #                                              transform=transform,
    #                                              download=True
    #                                              )
    #         extraset = torchvision.datasets.SVHN(os.path.join(DATASET_PATH, 'svhn-full-python'),
    #                                              split='extra',
    #                                              transform=transform,
    #                                              download=True
    #                                              )
    #         dataset = ConcatDataset([trainset, extraset])
    #     else:
    #         dataset = torchvision.datasets.SVHN(DATASET_PATH,
    #                                             split=train,
    #                                             transform=transform,
    #                                             download=True
    #                                             )
    #     if split in ['train', 'val']:
    #         split_path = os.path.join(DATASET_PATH,
    #                                   'svhn-full-python', 'train_val_index.cp')
    #
    #         if not os.path.exists(split_path):
    #             [train_index], [val_index] = split_dataset(args, dataset, k=1)
    #             split_index = {'train': train_index, 'val': val_index}
    #             cp.dump(split_index, open(split_path, 'wb'))
    #
    #         split_index = cp.load(open(split_path, 'rb'))
    #         dataset = Subset(dataset, split_index[split])

    elif args.dataset == 'imagenet':
        dataset = torchvision.datasets.ImageNet(DATASET_PATH,
                                                split=split,
                                                transform=transform,
                                                download=(split is 'val'))

    else:
        raise Exception('Unknown dataset')

    return dataset
