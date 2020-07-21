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
import collections
from models import *


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
    kwargs['print_step'] = kwargs['print_step'] if 'print_step' in kwargs else 2000
    kwargs['val_step'] = kwargs['val_step'] if 'val_step' in kwargs else 2000
    kwargs['scheduler'] = kwargs['scheduler'] if 'scheduler' in kwargs else 'cosine'
    kwargs['batch_size'] = kwargs['batch_size'] if 'batch_size' in kwargs else 128
    kwargs['start_step'] = kwargs['start_step'] if 'start_step' in kwargs else 0
    kwargs['max_step'] = kwargs['max_step'] if 'max_step' in kwargs else 64000
    kwargs['epochs'] = kwargs['epochs'] if 'epochs' in kwargs else 200
    kwargs['warmup'] = kwargs['warmup'] if 'warmup' in kwargs else False
    kwargs['auto_augment'] = kwargs['auto_augment'] if 'auto_augment' in kwargs else False
    kwargs['augment_path'] = kwargs['augment_path'] if 'augment_path' in kwargs else None
    kwargs['mag'] = kwargs['mag'] if 'mag' in kwargs else None
    kwargs['tinit'] = kwargs['tinit'] if 'tinit' in kwargs else None
    kwargs['tfin'] = kwargs['tfin'] if 'tfin' in kwargs else None

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

    print(net)
    return net


def select_optimizer(args, net):
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr,
                              momentum=0.9, weight_decay=5e-4)
    elif args.optimizer == 'rms':
        #optimizer = torch.optim.RMSprop(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-5)
        optimizer = torch.optim.RMSprop(net.parameters(), lr=args.lr)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    else:
        raise Exception('Unknown Optimizer')
    return optimizer


def select_scheduler(args, optimizer):
    if not args.scheduler or args.scheduler == 'None':
        return None
    elif args.scheduler =='clr':
        return torch.optim.lr_scheduler.CyclicLR(optimizer, 0.01, 0.015, mode='triangular2', step_size_up=250000, cycle_momentum=False)
    elif args.scheduler =='exp':
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9999283, last_epoch=-1)
    elif args.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0.)
    else:
        raise Exception('Unknown Scheduler')
