import copy
import json
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import random
import torchvision.transforms as transforms
from deap import base
from deap import creator
from deap import tools
from tdga.td_selection import ThermoDynamicalSelection
from analyzer import Analyzer
import numpy as np

from torch.utils.data import Subset
from sklearn.model_selection import StratifiedShuffleSplit


# from transforms_range import *
#
# DEFAULT_CANDIDATES = augment_list()

from transforms_range_prob import *

DEFAULT_CANDIDATES = augment_list()

# DEFAULT_CANDIDATES = BAYES_DEFAULT_CANDIDATES

from utils import *


def validate_child_train(args, dataset, Dm_indx, Da_indx, transform):  # individualを元に学習
    net = select_model(args)

    criterion = nn.CrossEntropyLoss()
    optimizer = select_optimizer(args, net)
    train_epoch = args.ind_train_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_epoch, eta_min=0.)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = None
    if device:
        net = net.to(device)
        criterion = criterion.to(device)

    elif args.use_cuda:
        net = net.cuda()
        criterion = criterion.cuda()

    dataset.transform = transform
    Dm_subset = Subset(dataset, Dm_indx)  # train dataset
    trainloader = torch.utils.data.DataLoader(
        Dm_subset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    for epoch in range(train_epoch):
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        total_steps = len(trainloader)

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
                         % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

            scheduler.step(epoch - 1 + float(batch_idx + 1) / total_steps)
            print(scheduler.get_lr())

    Da_subset = Subset(dataset, Da_indx)  # valid dataset
    val_loader = torch.utils.data.DataLoader(
        Da_subset, batch_size=100, shuffle=False, num_workers=args.num_workers)
    net.eval()
    valid_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            valid_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(val_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (valid_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    val_res = {
        'loss': valid_loss / (batch_idx + 1),
        'acc': correct / total,
    }
    return val_res


def update_transform_prob(pop, transform_prob):  # 世代の適応度情報から各操作の適用確率を更新する
    f_bar = sum(ind.fitness.values[0] for ind in pop)/len(pop)
    f_l = [[] for _ in range(len(pop[0]))]
    for ind in pop:
        for i in range(len(ind)):
            if ind[i]:
                f_l[i].append(ind.fitness.values[0])

    for i in range(len(pop[0])):
        if f_l[i]:
            fi_bar = sum(f_l[i])/len(f_l[i])
            transform_prob[i] *= fi_bar / f_bar

    sum_p = sum(transform_prob)
    transform_prob = [p/sum_p for p in transform_prob]  # 総和を1に
    return transform_prob


def ind_to_subpolicy(individual, transform_candidates, transform_prob, mag):  # individual から subpolicy に変換する
    subpolicy = []

    ### USE RANDOM CHOICE
    for allele, op in zip(individual, transform_candidates):
        if allele:
            subpolicy.append(op(prob=1, mag=mag))

    subpolicy = transforms.Compose([
        transforms.RandomChoice(subpolicy),
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])

    ### USE PROB
    # for allele, op, prob in zip(individual, transform_candidates, transform_prob):
    #     if allele:
    #         subpolicy.append(op(prob=prob, mag=mag))
    #
    # subpolicy = transforms.Compose([
    #     *subpolicy,
    #     transforms.Resize(32),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])
    return subpolicy


def search_subpolicies_tdga(args, transform_candidates, dataset, Dm_indx, Da_indx, B, log_dir):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    toolbox = base.Toolbox()

    allele_max = 1  # 対立遺伝子の最大値
    toolbox.register("attr_bool", random.randint, 0, allele_max)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, len(transform_candidates))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evaluate_ind(individual):
        subpolicy = ind_to_subpolicy(individual, transform_candidates, transform_prob, args.mag)
        return validate_child_train(args, dataset, Dm_indx, Da_indx, subpolicy)['acc'],

    toolbox.register("evaluate", evaluate_ind)
    toolbox.register("mate", tools.cxUniform, indpb=0.5)

    def mutChangeBit(individual, indpb):
        for i in range(len(individual)):
            if random.random() < indpb:
                individual[i] = random.randint(0, allele_max)

        return individual,

    # toolbox.register("mutate", mutChangeBit, indpb=0.06)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.06)

    Np = 8
    Ngen = 20
    t_init = args.tinit
    t_fin = args.tfin
    tds = ThermoDynamicalSelection(Np=Np, t_init=t_init, t_fin=t_fin, Ngen=Ngen, is_compress=False)
    toolbox.register("select", tds.select)
    pop = toolbox.population(n=Np)
    CXPB, MUTPB, NGEN = 1, 1, Ngen

    transform_prob = [1/len(transform_candidates)]*len(transform_candidates)

    print("Start of evolution")
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    print("  Evaluated %i individuals" % len(pop))

    print([ind.fitness.values[0] for ind in pop])
    analizer = Analyzer()
    for g in range(NGEN):
        print("-- Generation %i --" % g)
        elite = tools.selBest(pop, 1)
        elite = list(map(toolbox.clone, elite))
        offspring = list(map(toolbox.clone, pop))

        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):

            if random.random() < CXPB:
                toolbox.mate(ind1, ind2)
                del ind1.fitness.values
                del ind2.fitness.values

        gen = pop + offspring  # 2Np
        for mutant in gen:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        gen += elite

        invalid_ind = [ind for ind in gen if not ind.fitness.valid]
        print("  Evaluated %i individuals" % len(invalid_ind))
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        selected = toolbox.select(gen, k=Np)
        pop[:] = selected

        print("Pop:", pop)
        print("Fitnesses", [ind.fitness.values[0] for ind in pop])

        analizer.add_pop(list(map(toolbox.clone, pop)))

        transform_prob = update_transform_prob(pop, transform_prob)

    os.makedirs(os.path.join(log_dir, 'figures'), exist_ok=True)
    analizer.plot_entropy_matrix(file_name=os.path.join(log_dir, 'figures/entropy.png'))
    analizer.plot_stats(file_name=os.path.join(log_dir, 'figures/stats.png'))
    subpolicies = []
    print("Final Pop:", pop)
    print("Fitnesses", [ind.fitness.values[0] for ind in pop])
    best_ind = toolbox.select(pop, B)
    # best_ind = tools.selBest(pop, B)
    print("best_ind", best_ind)
    for ind in best_ind:
        subpolicy = []
        for allele, op, prob in zip(ind, transform_candidates, transform_prob):
            if allele:
                subpolicy.append(op(prob=prob, mag=args.mag))
                # subpolicy.append(op(prob=1, mag=args.mag))
        subpolicy = transforms.Compose([
            ## baseline augmentation
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            ## policy
            transforms.RandomChoice(subpolicy),
            # *subpolicy,
            ## to tensor
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            CutoutDefault(length=16),
        ])
        subpolicies.append((subpolicy, ind.fitness.values[0]))
    with open(os.path.join(log_dir, 'subpolicies.txt'), mode='a', encoding="utf_8") as f:
        for subpolicy in subpolicies:
            f.writelines(str(subpolicy) + "\n")
        f.write("\n")
        f.writelines(str(transform_prob))
    print(subpolicies)

    return subpolicies


def process_fn(args_str, dataset, Dm_indx, Da_indx, transform_candidates, B, log_dir):
    kwargs = json.loads(args_str)
    args, kwargs = parse_args(kwargs)

    # search sub policy
    subpolicies = search_subpolicies_tdga(args, transform_candidates, dataset, Dm_indx, Da_indx, B, log_dir)

    return [subpolicy[0] for subpolicy in subpolicies]


def tdga_augment(args, transform_candidates=None, B=8, log_dir=None):  # B: 最終的にとる個体数
    args_str = json.dumps(args._asdict())
    dataset = get_dataset(args, None, 'trainval')

    transform_candidates = DEFAULT_CANDIDATES

    if args.dataset == "cifar10":
    # split
        Dm_indexes, Da_indexes = split_dataset(args, dataset, 10)  # train_dataとval_dataを分割
        Dm_indexes = Da_indexes[0]  # 5000こずつ
        Da_indexes = Da_indexes[1]
    else:
        Dm_indexes, Da_indexes = None, None

    transform = process_fn(args_str, dataset, Dm_indexes, Da_indexes, transform_candidates, B, log_dir)

    transform = transforms.RandomChoice(transform)

    return transform
