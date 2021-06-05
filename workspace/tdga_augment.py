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
import math

from torch.utils.data import Subset
from datetime import datetime

# from transforms import *

# BAYES_DEFAULT_CANDIDATES = [
#     ShearXY,
#     TranslateXY,
#     Rotate,
#     AutoContrast,
#     Invert,
#     Equalize,
#     Solarize,
#     Posterize,
#     Contrast,
#     Color,
#     Brightness,
#     Sharpness,
#     Cutout,
#     # SamplePairing,
# ]

from transforms_range_prob import *

DEFAULT_CANDIDATES = augment_list()

# DEFAULT_CANDIDATES = BAYES_DEFAULT_CANDIDATES

from utils import *


def train_child(args, net, dataset, subset_indx):
    # device = None
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    optimizer = select_optimizer(args, net)
    scheduler = select_scheduler(args, optimizer)

    if args.dataset == "comic":
        weights = []
        for title in sorted(os.listdir("./data/comic/")):
            title_path = os.path.join("./data/comic/", title)
            weights.append(1/len(os.listdir(title_path)))  # データの逆数
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights).to(device))
    else:
        criterion = nn.CrossEntropyLoss()

    if "comic" in args.dataset:
        MEAN, STD = (0.8017, 0.8015, 0.8015), (0.2930, 0.2930, 0.2930)
        dataset.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
    else:
        dataset.transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])

    subset = Subset(dataset, subset_indx)
    trainloader = torch.utils.data.DataLoader(
        subset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    if device:
        net = net.to(device)
        criterion = criterion.to(device)

    elif args.use_cuda:
        net = net.cuda()
        criterion = criterion.cuda()

        if torch.cuda.device_count() > 1:
            print('\n[+] Use {} GPUs'.format(torch.cuda.device_count()))
            net = nn.DataParallel(net)

    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    for epoch in range(args.pre_train_epochs):
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

    _train_res = {
        'loss': train_loss / (batch_idx + 1),
        'acc': correct / total,
    }
    return _train_res


def validate_child(args, net, dataset, subset_indx, transform):
    criterion = nn.CrossEntropyLoss()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = None
    if device:
        net = net.to(device)
        criterion = criterion.to(device)

    elif args.use_cuda:
        net = net.cuda()
        criterion = criterion.cuda()

    dataset.transform = transform
    subset = Subset(dataset, subset_indx)
    val_loader = torch.utils.data.DataLoader(
        subset, batch_size=100, shuffle=False, num_workers=args.num_workers)
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


def get_next_subpolicy(transform_candidates, op_per_subpolicy=2):
    n_candidates = len(transform_candidates)
    subpolicy = []

    for i in range(op_per_subpolicy):
        indx = random.randrange(n_candidates)
        prob = random.random()
        mag = random.random()
        subpolicy.append(transform_candidates[indx](prob, mag))

    subpolicy = transforms.Compose([
        *subpolicy,
        transforms.Resize(32),
        transforms.ToTensor()])

    return subpolicy


def search_subpolicies(args, transform_candidates, child_model, dataset, Da_indx, B, log_dir):
    subpolicies = []

    for b in range(B):
        subpolicy = get_next_subpolicy(transform_candidates)
        val_res = validate_child(args, child_model, dataset, Da_indx, subpolicy)
        subpolicies.append((subpolicy, val_res[2]))

    return subpolicies


def ind_to_subpolicy(args, individual, transform_candidates, allele_max, mag):  # individual から subpolicy に変換する
    subpolicy = []
    for allele, op in zip(individual, transform_candidates):
        if allele:
            # subpolicy.append(op(prob=1 / sum(individual), mag=mag))
            magnitude = mag if allele_max == 1 else allele
            subpolicy.append(op(prob=args.prob_mul/len(transform_candidates), mag=magnitude))

    if "comic" in args.dataset:
        MEAN, STD = (0.8017, 0.8015, 0.8015), (0.2930, 0.2930, 0.2930)
        subpolicy = transforms.Compose([
            *subpolicy,
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD), ])
    else:
        subpolicy = transforms.Compose([
            *subpolicy,
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])
    return subpolicy


def search_subpolicies_tdga(args, transform_candidates, child_model, dataset, Dm_indx, Da_indx, B, log_dir, select_gamma, allele_max, denom_of_gamma):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    toolbox = base.Toolbox()

    # allele_max = 1  # 対立遺伝子の最大値
    toolbox.register("attr_bool", random.randint, 0, allele_max)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, len(transform_candidates))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evaluate_ind(individual):
        subpolicy = ind_to_subpolicy(args, individual, transform_candidates, allele_max, args.mag)

        gamma = 1
        if select_gamma == "rest5": 
            gamma = sum(individual) <= 5
        elif select_gamma == "down":
            c_denom = math.e if allele_max == 1 else denom_of_gamma
            gamma = min(1, math.log(len(individual)*allele_max  + 1 - sum(individual))/c_denom) 

        return gamma * validate_child(args, child_model, dataset, Da_indx, subpolicy)['acc'],  # val acc@1

    toolbox.register("evaluate", evaluate_ind)
    toolbox.register("mate", tools.cxUniform, indpb=0.5)

    def mutChangeBit(individual, indpb):
        for i in range(len(individual)):
            if random.random() < indpb:
                individual[i] = random.randint(0, allele_max)

        return individual,

    # toolbox.register("mutate", mutChangeBit, indpb=0.06)
    toolbox.register("mutate", tools.mutFlipBit, indpb=1/len(transform_candidates))

    Np = args.Np
    Ngen = args.Ng
    t_init = args.tinit
    t_fin = args.tfin
    fuzzy = args.fuzzy
    tds = ThermoDynamicalSelection(Np=Np, t_init=t_init, t_fin=t_fin, Ngen=Ngen, is_compress=False, allele_max=allele_max)
    toolbox.register("select", tds.select)
    pop = toolbox.population(n=Np)
    CXPB, MUTPB, NGEN = 1, 1, Ngen
    print("Start of evolution")
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    print("  Evaluated %i individuals" % len(pop))

    print([ind.fitness.values[0] for ind in pop])
    analizer = Analyzer(log_dir)
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

        selected_pop, selected_H = toolbox.select(gen, k=Np, fuzzy=fuzzy)
        pop[:] = selected_pop

        print("Pop:", pop)
        print("Fitnesses", [ind.fitness.values[0] for ind in pop])

        analizer.add_pop(list(map(toolbox.clone, pop)))
        analizer.add_entropy(list(map(toolbox.clone, selected_H)))

        # fits = [ind.fitness.values[0] for ind in pop]

    print("Final Pop:", pop)
    print("Fitnesses", [ind.fitness.values[0] for ind in pop])

    best_ind, best_H = toolbox.select(pop, B, fuzzy=fuzzy, mul_lambda=mul_lambda)
    analizer.set_best_entropy(list(map(toolbox.clone, best_H)))
    # best_ind = tools.selBest(pop, B)
    print("best_ind", best_ind)

    os.makedirs(os.path.join(log_dir, 'figures'), exist_ok=True)
    analizer.plot_entropy_matrix(file_name=os.path.join(log_dir, 'figures/entropy.png'))
    analizer.plot_entropy_num_transforms(file_name=os.path.join(log_dir, 'figures/num_transforms.png'), applied_pop=best_ind)
    analizer.plot_stats(file_name=os.path.join(log_dir, 'figures/stats.png'))

    subpolicies = []

    for ind in best_ind:
        subpolicy = []
        for allele, op in zip(ind, transform_candidates):
            if allele:
                # subpolicy.append(op(prob=1/sum(ind), mag=args.mag))
                magnitude = args.mag if allele_max == 1 else allele
                subpolicy.append(op(prob=args.prob_mul/len(transform_candidates), mag=magnitude))

        if "comic" in args.dataset:
            MEAN, STD = (0.8017, 0.8015, 0.8015), (0.2930, 0.2930, 0.2930)
            subpolicy = transforms.Compose([
                ## baseline augmentation
                transforms.RandomHorizontalFlip(),
                ## policy
                *subpolicy,
                ## to tensor
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD),
                CutoutDefault(length=32),
            ])
        else:
            subpolicy = transforms.Compose([
                ## baseline augmentation
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                ## policy
                *subpolicy,
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
    print(subpolicies)

    return subpolicies


def process_fn(args_str, model, dataset, Dm_indx, Da_indx, transform_candidates, log_dir):
    kwargs = json.loads(args_str)
    args, kwargs = parse_args(kwargs)
    _transform = []

    print('[+] Pre-Training strated')
    pre_start = datetime.now()
    print('num of sub-policies:', args.B)

    # train child model
    child_model = copy.deepcopy(model)
    train_res = train_child(args, child_model, dataset, Dm_indx)

    print("GA Search Started")
    ga_start = datetime.now()
    # search sub policy
    subpolicies = search_subpolicies_tdga(args, transform_candidates, child_model, dataset, Dm_indx, Da_indx, args.B, log_dir, args.select_gamma, args.allele_max, args.denom_of_gamma)
    finish = datetime.now()
    print("Pre-Train Elapsed Time: {}".format(ga_start - pre_start))
    print("Search Elapsed Time: {}".format(finish - ga_start))
    return [subpolicy[0] for subpolicy in subpolicies]


def tdga_augment(args, model, transform_candidates=None, log_dir=None):
    args_str = json.dumps(args._asdict())
    dataset = get_dataset(args, None, 'trainval')

    # torch.multiprocessing.set_start_method('spawn', force=True)

    # transform_candidates = BAYES_DEFAULT_CANDIDATES
    transform_candidates = DEFAULT_CANDIDATES

    # split
    Dm_indexes, Da_indexes = split_dataset(args, dataset, 1)  # train_dataとval_dataを分割
    # Dm_indexes = [list(range(4000))]
    # Da_indexes = [list(range(500))]
    # print(len(Dm_indexes[0]))
    # print(len(Da_indexes[0]))
    # print(type(Dm_indexes))
    transform = process_fn(args_str, model, dataset, Dm_indexes[0], Da_indexes[0], transform_candidates, log_dir)

    transform = transforms.RandomChoice(transform)

    return transform
