import matplotlib
import matplotlib.pyplot as plt
import numpy as np


filename = "rest5"

x = [0.005, 0.006, 0.007, 0.008, 0.009, 0.01]
length = len(x)

y = [0.9536, 0.9526, 0.9566, 0.9556, 0.9527, 0.9546]
y_std = [0.0061, 0.0158, 0.0037, 0.0024, 0.0049, 0.0050]

p0 = plt.errorbar(x, y, 
                  yerr=y_std, 
                  capsize=5, 
                  markersize=8, 
                  ecolor='black', 
                  markeredgecolor="black", 
                  color='b')

y_baseline = [0.9428] * length
y_baseline_g = [0.9414] * length
y_baseline_s = [0.9402] * length
y_baseline_b = [0.9456] * length
y_baseline_k = [0.9468] * length

p1 = plt.plot(x, y_baseline, linestyle="dotted")
p2 = plt.plot(x, y_baseline_g, linestyle="dashdot")
p3 = plt.plot(x, y_baseline_s, linestyle="dashdot")
p4 = plt.plot(x, y_baseline_b, linestyle="dashdot")
p5 = plt.plot(x, y_baseline_k, linestyle="dashdot")

plt.legend((p0[0], p1[0], p2[0], p3[0], p4[0], p5[0]), 
           ("TDGA AA", 
            "Baseline", 
            "Baseline(Gaussian Noise)", 
            "Baseline(Senga)", 
            "Baseline(Balloon Add)", 
            "Baseline(Koma Split)"))

plt.xlabel("Temperature of TDGA")
plt.ylabel("Test Accuracy")
plt.savefig("exp_change_temperature_comic"+ filename + ".png")
