from math import log, log2, exp
from collections import defaultdict
from typing import Callable


class ThermoDynamicalSelection:
    def __init__(self, Np: int, t_init: float, t_fin: float = None, Ngen: int = 0, scheduler: Callable[[float], float] = None, is_compress: bool = False, allele_max : int = 1):
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
        self.log_value_ = [0.0] * (Np + 1)
        for i in range(1, self.Np + 1):
            self.log_value_[i] = log(i)
        # 各遺伝子座の対立遺伝子の個数カウント. self.num_of_gene[i][j] := 遺伝子座iにおける対立遺伝子jの個数
        self.num_of_gene_ = None
        self.scheduler_ = scheduler if scheduler else None
        self.is_compress_ = is_compress
        self.generation = 0
        
        self.allele_max = allele_max
        self.e_value_ = [[0.0] * (self.allele_max + 1) for _ in range(self.allele_max+1)]

    def select(self, individuals, k, fuzzy=False):  # 入力個体数:2*Np+1 出力個体数:k
        if self.is_compress_:  # 同個体の圧縮
            individuals = self.compress_(individuals)

        self.num_of_gene_ = [defaultdict(int) for _ in range(len(individuals[0]))]
        selected_individuals = [None] * k
        min_index = None
        E_sum = 0
        selected_Hl = None
        for i in range(k):  # k個体の選択
            if fuzzy: self.calc_fuzzy_table_(i + 1)
            F_min = float('inf')  # 自由エネルギー最小化なので最初は十分に大きな値
            Hl = []
            for j in range(len(individuals)):  # 各個体から加えたときに F=<E>-HT の最も小さくなる個体を探す
                Hj = self.entropy_(i + 1, individuals[j]) if not fuzzy else self.fuzzy_entropy_(individuals[j])
                Hl.append(Hj)
                E_bar = (E_sum + -getattr(individuals[j], "fitness").wvalues[0]) / (i + 1)
                F = E_bar - sum(Hj) * self.temperature
                if F < F_min:
                    F_min = F
                    min_index = j
            if i == k-1: selected_Hl = Hl[min_index]
            selected_individuals[i] = individuals[min_index]
            E_sum += -getattr(individuals[min_index], "fitness").wvalues[0]
            self.update_num_of_gene_(individuals[min_index])

        self.generation += 1
        self.update_temperature_()
        return selected_individuals, selected_Hl

    def update_num_of_gene_(self, individual):  # num_of_gene更新
        for k, allele in enumerate(individual):
            self.num_of_gene_[k][allele] += 1

    def update_temperature_(self):  # 温度更新
        if self.scheduler_:
            self.temperature = self.scheduler_(self.temperature)
        else:  # T = Tmax^(1-t)*Tmin^(t)
            assert self.t_fin is not None and self.Ngen, "set the t_fin and Ngen value when scheduler is not specified"
            assert self.t_fin != 0, "cannot set t_fin value to 0"
            t = self.generation/self.Ngen
            self.temperature = pow(self.t_init, 1-t) * pow(self.t_fin, t)

    def entropy_(self, target_num, candidate):  # num_of_geneと候補個体からエントロピーを計算する
        Hall = []
        for k, allele in enumerate(candidate):
            H1 = self.log_value_[target_num]
            self.num_of_gene_[k][allele] += 1
            for nk in self.num_of_gene_[k].values():
                H1 += -(nk * self.log_value_[nk]) / target_num
            Hall.append(H1)
            self.num_of_gene_[k][allele] -= 1

        return Hall

    def calc_fuzzy_table_(self, target_num):
        d_value = [[0.0] * (self.allele_max + 1) for _ in range(self.allele_max+1)]
        u_value = [[0.0] * (self.allele_max + 1) for _ in range(self.allele_max+1)]

        for i in range(self.allele_max+1):
            for j in range(self.allele_max+1):
                d_value[i][j] = ((i - j) / self.allele_max) ** 2
                u_value[i][j] = exp((-1)*d_value[i][j]*target_num)
                e = 0
                e += (-1) * u_value[i][j] * log2(u_value[i][j]) if u_value[i][j] > 0 else 0
                e += (-1) * (1 - u_value[i][j]) * log2(1 - u_value[i][j]) if u_value[i][j] < 1 else 0 
                self.e_value_[i][j] = e

    def fuzzy_entropy_(self, candidate):
        Hall = []
        for k, allele in enumerate(candidate):
            H1 = 0
            self.num_of_gene_[k][allele] += 1
            for ki, vi in self.num_of_gene_[k].items():
                if vi == 0: continue
                for kj, vj in self.num_of_gene_[k].items():
                    H1 += vj * self.e_value_[ki][kj]
            Hall.append(H1)
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
