import copy
import json
import time
import torch
import torch.nn as nn
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
from concurrent.futures import ProcessPoolExecutor

from transforms import *

BAYES_DEFAULT_CANDIDATES = [
    ShearXY,
    TranslateXY,
    Rotate,
    AutoContrast,
    Invert,
    Equalize,
    Solarize,
    Posterize,
    Contrast,
    Color,
    Brightness,
    Sharpness,
    Cutout,
    # SamplePairing,
]

from transforms_range import augment_list
DEFAULT_CANDIDATES = augment_list()

# DEFAULT_CANDIDATES = BAYES_DEFAULT_CANDIDATES

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from utils import *


def train_child(args, model, dataset, subset_indx):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    optimizer = select_optimizer(args, model)
    scheduler = select_scheduler(args, optimizer)
    criterion = nn.CrossEntropyLoss()

    dataset.transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor()])
    subset = Subset(dataset, subset_indx)
    data_loader = get_inf_dataloader(args, subset)

    if device:
        model = model.to(device)
        criterion = criterion.to(device)

    elif args.use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()

        if torch.cuda.device_count() > 1:
            print('\n[+] Use {} GPUs'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)

    start_t = time.time()
    for step in range(args.start_step, args.max_step):
        batch = next(data_loader)
        _train_res = train_step(args, model, optimizer, scheduler, criterion, batch, step, None, device)

        if step % args.print_step == 0:
            print('\n[+] Training step: {}/{}\tElapsed time: {:.2f}min\tLearning rate: {}\tDevice: {}'.format(
                step, args.max_step,(time.time()-start_t)/60, optimizer.param_groups[0]['lr'], device))

            print('  Acc@1 : {:.3f}%'.format(_train_res[0].data.cpu().numpy()[0]*100))
            print('  Acc@5 : {:.3f}%'.format(_train_res[1].data.cpu().numpy()[0]*100))
            print('  Loss : {}'.format(_train_res[2].data))

    return _train_res


def validate_child(args, model, dataset, subset_indx, transform, device=None):
    criterion = nn.CrossEntropyLoss()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device:
        model = model.to(device)
        criterion = criterion.to(device)

    elif args.use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    dataset.transform = transform
    subset = Subset(dataset, subset_indx)
    data_loader = get_dataloader(args, subset, pin_memory=False)

    return validate(args, model, criterion, data_loader, 0, None, device)


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


def ind_to_subpolicy(individual, transform_candidates, allele_max, mag):  # individual から subpolicy に変換する
    subpolicy = []
    for allele, op in zip(individual, transform_candidates):
        if allele:
            subpolicy.append(op(mag=mag))

    subpolicy = transforms.Compose([
        *subpolicy,
        transforms.Resize(32),
        transforms.ToTensor()])
    return subpolicy


global_step = 0  # 探索を何回したか


def search_subpolicies_tdga(args, transform_candidates, child_model, dataset, Da_indx, B, log_dir):
    global global_step
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    toolbox = base.Toolbox()

    allele_max = 1  # 対立遺伝子の最大値
    toolbox.register("attr_bool", random.randint, 0, allele_max)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, len(transform_candidates))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evaluate_ind(individual):
        subpolicy = ind_to_subpolicy(individual, transform_candidates, allele_max, args.mag)
        return validate_child(args, child_model, dataset, Da_indx, subpolicy, device)[0].cpu().numpy(),  # val acc@1

    toolbox.register("evaluate", evaluate_ind)
    toolbox.register("mate", tools.cxUniform, indpb=0.5)

    def mutChangeBit(individual, indpb):
        for i in range(len(individual)):
            if random.random() < indpb:
                individual[i] = random.randint(0, allele_max)

        return individual,

    # toolbox.register("mutate", mutChangeBit, indpb=0.02)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.06)

    Np = 16
    Ngen = 50
    t_init = args.tinit
    t_fin = args.tfin
    tds = ThermoDynamicalSelection(Np=Np, t_init=t_init, t_fin=t_fin, Ngen=Ngen, is_compress=False)
    toolbox.register("select", tds.select)
    pop = toolbox.population(n=Np)
    CXPB, MUTPB, NGEN = 1, 1, Ngen
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

        # fits = [ind.fitness.values[0] for ind in pop]
    os.makedirs(os.path.join(log_dir, 'figures'), exist_ok=True)
    analizer.plot_entropy_matrix(file_name=os.path.join(log_dir, 'figures/entropy_{}.png'.format(global_step)))
    analizer.plot_stats(file_name=os.path.join(log_dir, 'figures/stats_{}.png'.format(global_step)))
    subpolicies = []
    print("Final Pop:",  pop)
    print("Fitnesses", [ind.fitness.values[0] for ind in pop])
    best_ind = toolbox.select(pop, B)
    # best_ind = tools.selBest(pop, B)
    print("best_ind", best_ind)
    for ind in best_ind:
        subpolicy = []
        for allele, op in zip(ind, transform_candidates):
            if allele:
                subpolicy.append(op(mag=args.mag))
        subpolicy = transforms.Compose([
            ## baseline augmentation
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            ## policy
            *subpolicy,
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ## to tensor
            transforms.ToTensor()])
        subpolicies.append((subpolicy, ind.fitness.values[0]))
    with open(os.path.join(log_dir, 'subpolicies.txt'), mode='a', encoding="utf_8") as f:
        f.writelines("Search Phase {}\n".format(global_step))
        for subpolicy in subpolicies:
            f.writelines(str(subpolicy) + "\n")
        f.write("\n")
    print(subpolicies)

    global_step += 1
    return subpolicies



def process_fn(args_str, model, dataset, Dm_indx, Da_indx, transform_candidates, B, log_dir):
    kwargs = json.loads(args_str)
    args, kwargs = parse_args(kwargs)
    _transform = []

    print('[+] Pre-Training strated')

    # train child model
    child_model = copy.deepcopy(model)
    train_res = train_child(args, child_model, dataset, Dm_indx)

    # search sub policy
    subpolicies = search_subpolicies_tdga(args, transform_candidates, child_model, dataset, Da_indx, B, log_dir)

    _transform.extend([subpolicy[0] for subpolicy in subpolicies])

    return _transform


def tdga_augment(args, model, transform_candidates=None, B=8, log_dir=None):
    args_str = json.dumps(args._asdict())
    dataset = get_dataset(args, None, 'trainval')
    transform, futures = [], []

    torch.multiprocessing.set_start_method('spawn', force=True)

    if not transform_candidates:
        if not args.ga:
            transform_candidates = BAYES_DEFAULT_CANDIDATES
        else:
            transform_candidates = DEFAULT_CANDIDATES

    # split
    Dm_indexes, Da_indexes = split_dataset(args, dataset, 1)  # train_dataとval_dataを分割
    transform = process_fn(args_str, model, dataset, Dm_indexes, Da_indexes, transform_candidates, B, log_dir)

    transform = transforms.RandomChoice(transform)

    return transform
