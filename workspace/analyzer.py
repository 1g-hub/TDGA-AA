import numpy as np
from math import log
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os


class Analyzer:
    def __init__(self, log_dir):
        self.populations = []
        self.entropy_matrix = None
        self.log_dir = log_dir

    def add_pop(self, pop):
        self.populations.append(pop)

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

    def plot_entropy_matrix(self, file_name, applied_pop):
        self.make_entropy_matrix_()
        matrix = self.entropy_matrix
        X1, X2 = np.mgrid[1:len(matrix)+1, 1:len(matrix[0])+1]

        l = [
            "AutoContrast",
            "Equalize",
            "Invert",
            "Rotate",
            "Posterize",
            "Solarize",
            "SolarizeAdd",
            "Color",
            "Contrast",
            "Brightness",
            "Sharpness",
            "ShearX",
            "ShearY",
            "Cutout",
            "TranslateX",
            "TranslateY",
            # TODO : New filter 
            "GaussianNoise",
            "LineArt", 
            "BalloonAdd",
            "Offset",
        ]

        fig = plt.figure()
        ax = Axes3D(fig)
        ax.set_xlabel("Generation")
        ax.set_ylabel("Locus")
        ax.set_zlabel("Entropy")
        ax.view_init(elev=30, azim=-20)
        ax.plot_surface(X1, X2, matrix, rcount=1000, ccount=1000, cmap='cividis', antialiased=True)

        # plt.yticks(range(1, len(matrix[0]) + 1), l)
        # plt.yticks(rotation=-60)

        plt.savefig(file_name)

        fig = plt.figure()
        X = np.linspace(1, len(l), len(l))
        pop = self.populations[-1]
        Y_num = [0]*len(l)
        for i in range(len(l)):
            cnt = 0
            for j in range(len(pop)):
                cnt += pop[j][i]
            Y_num[i] = cnt

        Y_num_applied = [0]*len(l)
        for i in range(len(l)):
            cnt = 0
            for j in range(len(applied_pop)):
                cnt += applied_pop[j][i]
            Y_num_applied[i] = cnt

        # Y_num = np.array([39, 29, 15, 30, 5, 8, 16, 39, 27, 30, 35, 38, 36, 37, 40, 39]) # t:0.02 5回
        # Y_num = np.array([30, 0, 0, 1, 0, 0, 0, 19, 0, 0, 16, 22, 22, 30, 32, 28])  # t:0.002 5回
        Y_entropy = matrix[-1]

        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.plot(X, Y_entropy, linewidth=2, color="red", linestyle="solid", marker="o", markersize=8, label='Entropy')
        ax2.bar(X, Y_num, label='Number of operations')
        ax2.bar(X, Y_num_applied)
        ax1.set_zorder(2)
        ax2.set_zorder(1)
        ax1.patch.set_alpha(0)
        ax1.legend(bbox_to_anchor=(0, 1), loc='upper left', borderaxespad=0.5, fontsize=10)
        ax2.legend(bbox_to_anchor=(0, 0.9), loc='upper left', borderaxespad=0.5, fontsize=10)
        ax1.set_xlabel('Operation')
        ax1.set_xticklabels(X, rotation=90)
        ax1.set_ylabel('Entropy')
        ax2.set_ylabel('Number of operations')

        plt.xticks(range(1, len(l)+1), l)
        plt.xticks(rotation=-90)
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, 'figures/num_transforms.png'))

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
