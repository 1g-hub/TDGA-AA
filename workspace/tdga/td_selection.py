from math import log
from collections import defaultdict
from typing import Callable


class ThermoDynamicalSelection:
    def __init__(self, Np: int, t_init: float, t_fin: float = None, Ngen: int = 0, scheduler: Callable[[float], float] = None, is_compress: bool = False):
        """
        :param Np: 初期温度
        :param Ngen: 終了世代
        :param t_init: 初期温度
        :param t_fin: 終了温度 (スケジューラを与える場合には不要)
        :param scheduler: 温度スケジューラ
        :param is_compress:  座標圧縮用フラグ
        """
        self.Np = Np
        self.t_init = t_init
        self.t_fin = t_fin
        self.Ngen = Ngen
        self.temperature = t_init
        self.log_value_ = [0] * (Np + 1)
        for i in range(1, self.Np + 1):
            self.log_value_[i] = log(i)
        # 各遺伝子座の対立遺伝子の個数カウント. self.num_of_gene[i][j] := 遺伝子座iにおける対立遺伝子jの個数
        self.num_of_gene_ = None
        self.scheduler_ = scheduler if scheduler else None
        self.is_compress_ = is_compress
        self.generation = 0

    def select(self, individuals, k):  # 入力個体数:2*Np+1 出力個体数:k
        if self.is_compress_:  # 同個体の圧縮
            individuals = self.compress_(individuals)

        self.num_of_gene_ = [defaultdict(int) for _ in range(len(individuals[0]))]
        selected_individuals = [None] * k
        min_index = None
        E_sum = 0
        for i in range(k):  # k個体の選択
            F_min = float('inf')  # 自由エネルギー最小化なので最初は十分に大きな値
            for j in range(len(individuals)):  # 各個体から加えたときに F=<E>-HT の最も小さくなる個体を探す
                Hj = self.entropy_(i + 1, individuals[j])
                E_bar = (E_sum + -getattr(individuals[j], "fitness").wvalues[0]) / (i + 1)
                F = E_bar - Hj * self.temperature
                if F < F_min:
                    F_min = F
                    min_index = j
            selected_individuals[i] = individuals[min_index]
            E_sum += -getattr(individuals[min_index], "fitness").wvalues[0]
            self.update_num_of_gene_(individuals[min_index])

        self.generation += 1
        self.update_temperature_()
        return selected_individuals

    def update_num_of_gene_(self, individual):  # num_of_gene更新
        for k, allele in enumerate(individual):
            self.num_of_gene_[k][allele] += 1

    def update_temperature_(self):  # 温度更新
        if self.scheduler_:
            self.temperature = self.scheduler_(self.temperature)
        else:  # T = Tmax^(1-t)*Tmin^(t)
            assert self.t_fin != None and self.Ngen, "set the t_fin and Ngen value when scheduler is not specified"
            assert self.t_fin != 0, "cannot set t_fin value to 0"
            t = self.generation/self.Ngen
            self.temperature = pow(self.t_init, 1-t) * pow(self.t_fin, t)

    def entropy_(self, target_num, candidate):  # num_of_geneと候補個体からエントロピーを計算する
        Hall = 0
        for k, allele in enumerate(candidate):
            H1 = self.log_value_[target_num]
            self.num_of_gene_[k][allele] += 1
            for nk in self.num_of_gene_[k].values():
                H1 += -(nk * self.log_value_[nk]) / target_num
            Hall += H1
            self.num_of_gene_[k][allele] -= 1

        return Hall

    @staticmethod
    def compress_(individuals):  # 同一の遺伝子を持つ個体を圧縮する．
        compressed = []
        exist = set()
        for ind in individuals:
            if not str(ind) in exist:
                exist.add(str(ind))
                compressed.append(ind)

        return compressed
