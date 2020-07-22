import numpy as np
from math import log
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class Analyzer:
    def __init__(self):
        self.populations = []
        self.entropy_matrix = None
        

    def add_pop(self, pop):
        self.populations.append(pop)

    def add_stat(self, max_val, min_val, mean_val, optimum_val=None):
        self.max.append(max_val)
        self.min.append(min_val)
        self.mean.append(mean_val)
        if optimum_val:
            self.optimum.append(optimum_val)

    def make_entropy_matrix_(self):
        N_gen = len(self.populations)
        Np = len(self.populations[0])
        N_locus = len(self.populations[0][0])

        entropy_matrix = [[0]*N_locus for _ in range(N_gen)]  # entropy_matrix[i][j] := 世代iにおける遺伝子座jのエントロピー

        for g in range(N_gen):
            pop = np.array(self.populations[g])  # 1つの世代
            for l in range(N_locus):  # 遺伝子座毎にカウント
                H1 = log(Np)
                for nk in np.bincount(pop[:, l]).tolist():
                    if nk == 0: continue  # 個数0のエントロピーは0
                    H1 += -nk*log(nk)/Np

                entropy_matrix[g][l] = H1

        self.entropy_matrix = np.array(entropy_matrix)

    def plot_entropy_matrix(self, file_name):
        self.make_entropy_matrix_()
        matrix = self.entropy_matrix
        X1, X2 = np.mgrid[1:len(matrix)+1, 1:len(matrix[0])+1]

        fig = plt.figure()
        ax = Axes3D(fig)
        ax.set_xlabel("Generation")
        ax.set_ylabel("Locus")
        ax.set_zlabel("Entropy")
        ax.view_init(elev=30, azim=-20)
        ax.plot_surface(X1, X2, matrix, rcount=1000, ccount=1000, cmap='cividis', antialiased=True)
        plt.savefig(file_name)

    def plot_stats(self, file_name, optimum_val=None):
        pop_max = []
        pop_min = []
        pop_mean = []

        for i in range(len(self.populations)):
            fits = [ind.fitness.values[0] for ind in self.populations[i]]
            length = len(fits)
            mean = sum(fits) / length

            pop_max.append(max(fits))
            pop_min.append(min(fits))
            pop_mean.append(mean)

        if optimum_val:
            pop_optimum = [optimum_val]*len(self.populations)
        fig = plt.figure()
        gen = np.arange(1, len(self.populations)+1)
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        p1 = plt.plot(gen, pop_max, linestyle="solid")
        p2 = plt.plot(gen, pop_min, linestyle="dotted")
        p3 = plt.plot(gen, pop_mean, linestyle="dashdot")
        if optimum_val:
            p4 = plt.plot(gen, pop_optimum, linestyle="dashed")
            plt.legend((p4[0], p1[0], p3[0], p2[0]), ("optimum", "max", "mean", "min"))
        else:
            plt.legend((p1[0], p3[0], p2[0]), ("max", "mean", "min"))
        plt.savefig(file_name)
