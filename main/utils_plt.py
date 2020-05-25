import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
import numpy as np
import sys
from scipy.signal import savgol_filter

#python utils_plt.py data/horizontal.npy

def plt_lines(filename = 'data/ch_2.npy', data=None):
    if data is None:
        data = np.load(open(filename, 'rb'))
    y = data[:, 2]
    #x = data[:, 0]
    x = np.arange(0, y.shape[0])
    fig= plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x, y)
    y1 = savgol_filter(y, 51, 3)
    ax.plot(x, y1)
    #depth
    ax.set(ylim=(0, np.max(y)*1.2))
    #x-axis
    #ax.set(xlim=(-1000, 1000))
    plt.show()

def xz(filename = 'data/ch_2.npy', data=None):
    if data is None:
        data = np.load(open(filename, 'rb'))
    y = data[:, 2]
    x = data[:, 0]
    fig= plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x, y)
    y1 = savgol_filter(y, 51, 3)
    ax.plot(x, y1)
    #depth
    ax.set(ylim=(0, np.max(y)*1.2))
    #x-axis
    ax.set(xlim=(-1000, 1000))
    plt.show()

def visualize(filename = 'data/ch_2.npy', data=None, annotate=False, lines=False, axs=0):
    if data is None:
        data = np.load(open(filename, 'rb'))
    data = data.reshape(-1, 3)
    n_data = data.shape[0]
    data = data[:n_data]
    print(data.shape)
    fig = plt.figure()
    ax = Axes3D(fig)
    X, Y, Z = np.arange(0, data[:, 0].shape[0]) , list(data[:, 1*(axs==1)]), list(data[:, 2])
    Y1 = savgol_filter(Y, 51, 3)
    Z1 = savgol_filter(Z, 51, 3)
    data = list(zip(X, Y1, Z1))
    ax.scatter3D(X, Y1, Z1)
#   plt.xlim(-300, 300)
#   plt.ylim(-200, 200)
#   plt.zlim(0, 1700)
    #plt.gca().set_aspect('equal', adjustable='box')
    def connectpoints(x,y):
        x1, x2, x3 = x[0], x[1], x[2]
        y1, y2, y3 = y[0], y[1], y[2]
    #     ax.plot([VecStart_x[i], VecEnd_x[i]], [VecStart_y[i],VecEnd_y[i]],zs=[VecStart_z[i],VecEnd_z[i]])
        ax.plot([x1,y1], [x2, y2], zs=[x3, y3])
    arr=[]
    for i in range(0, len(data)-1):
        arr.append([i, i+1])
    if lines is True:
        for tmp in arr:
            connectpoints(data[tmp[0]], data[tmp[1]])
    ax.set(ylim=(-4000, 4000), zlim=(0, 8000))
    #ax.set(zlim=(0, 8000))
    plt.show()

def visualize_3D(filename = 'data/ch_2.npy', data=None, annotate=False, lines=False):
    if data is None:
        data = np.load(open(filename, 'rb'))
    data = data.reshape(-1, 3)
    n_data = data.shape[0]
    data = data[:n_data]
    print(data.shape)
    fig = plt.figure()
    ax = Axes3D(fig)
    X, Y, Z =  list(data[:, 0]), list(data[:, 1]), list(data[:, 2])
    #X1 = savgol_filter(X, 51, 3)
    Y1 = savgol_filter(Y, 51, 3)
    Z1 = savgol_filter(Z, 51, 3)
    data = list(zip(X, Y1, Z1))
    ax.scatter3D(X, Y1, Z1)
#   plt.xlim(-300, 300)
#   plt.ylim(-200, 200)
#   plt.zlim(0, 1700)
    def connectpoints(x,y):
        x1, x2, x3 = x[0], x[1], x[2]
        y1, y2, y3 = y[0], y[1], y[2]
    #     ax.plot([VecStart_x[i], VecEnd_x[i]], [VecStart_y[i],VecEnd_y[i]],zs=[VecStart_z[i],VecEnd_z[i]])
        ax.plot([x1,y1], [x2, y2], zs=[x3, y3])
    arr=[]
    for i in range(0, len(data)-1):
        arr.append([i, i+1])
    if lines is True:
        for tmp in arr:
            connectpoints(data[tmp[0]], data[tmp[1]])
    #ax.set(xlim=(-1000, 1000), ylim=(-2000, 1000), zlim=(0, 8000))
    ax.set(xlim=(-4000, 4000), ylim=(-4000, 4000), zlim=(0, 8000))
    plt.show()


if __name__ == '__main__':
    if len(sys.argv)==1:
        plt_lines()
        xz()
        visualize(lines=True, axs=0) #x-0, y-1, z-2
        visualize_3D(lines=True) #x-0, y-1, z-2
    else:
        plt_lines(filename=sys.argv[1])
        xz(filename=sys.argv[1])
        visualize(filename=sys.argv[1], axs=0) #x-0, y-1, z-2
        visualize_3D(filename=sys.argv[1]) #x-0, y-1, z-2
