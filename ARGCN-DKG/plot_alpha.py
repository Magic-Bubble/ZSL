# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

def plot(x, y1, title):
    plt.figure(1)
    plt.title(title)
    plt.plot(x, y1[0], 'bo-', linewidth=2.0, ms=5, label='hit@1 w.o res')
    plt.plot(x, y1[1], 'ms-', linewidth=2.0, ms=5, label='hit@2 w.o res')
    plt.plot(x, y1[2], 'g*-', linewidth=2.0, ms=5, label='hit@5 w.o res')
    plt.plot(x, y1[3], 'y^-', linewidth=2.0, ms=5, label='hit@10 w.o res')
    plt.plot(x, y1[4], 'cx-', linewidth=2.0, ms=5, label='hit@20 w.o res')
    plt.xlim(0, 11)
    plt.xlabel('alpha * (0.1)')
    plt.ylabel('Acc(%)')
    plt.grid()

x = np.linspace(1, 11, 11)

# 2-hops, 横坐标：alpha, 纵坐标：hit@k

y_2hops = np.array([[24.62, 37.44, 56.10, 68.20, 77.39],
                    [25.02, 38.00, 56.73, 68.76, 77.87],
                    [25.28, 38.43, 57.28, 69.23, 78.31],
                    [25.44, 38.74, 57.70, 69.64, 78.69],
                    [25.50, 38.92, 58.00, 69.98, 79.00],
                    [25.53, 39.03, 58.25, 70.25, 79.26],
                    [25.52, 39.04, 58.40, 70.45, 79.46],
                    [25.49, 38.96, 58.46, 70.59, 79.63],
                    [25.39, 38.78, 58.44, 70.65, 79.72],
                    [25.22, 38.59, 58.34, 70.63, 79.73],
                    [25.02, 38.31, 58.15, 70.53, 79.70]]).T

plt.ylim(22, 81)
plot(x, y_2hops, '2-hops')
plt.savefig('./multi-alpha.png')
