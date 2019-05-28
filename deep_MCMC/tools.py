import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.axes3d as p3
from IPython.display import HTML
import matplotlib.animation as animation

def draw_config(x, ax=None, dimercolor='blue', alpha=0.7, draw_labels=False, rm=1.1, title=None, nsolvent=36, d=4.):
    n = nsolvent+2
    X = x.reshape(((nsolvent+2), 2))
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim((-d, d))
    ax.set_ylim((-d, d))
    circles = []
    circles.append(ax.add_patch(plt.Circle(X[0], radius=0.5*rm, color=dimercolor, alpha=alpha)))
    circles.append(ax.add_patch(plt.Circle(X[1], radius=0.5*rm, color=dimercolor, alpha=alpha)))
    for x_ in X[2:]:
        circles.append(ax.add_patch(plt.Circle(x_, radius=0.5*rm, color='grey', alpha=alpha)))
    ax.add_patch(plt.Rectangle((-d, -d), 2*d, 0.5*rm, color='grey', linewidth=0))
    ax.add_patch(plt.Rectangle((-d, d-0.5*rm), 2*d, 0.5*rm, color='grey', linewidth=0))
    ax.add_patch(plt.Rectangle((-d, -d), 0.5*rm, 2*d, color='grey', linewidth=0))
    ax.add_patch(plt.Rectangle((d-0.5*rm, -d), 0.5*rm, 2*d, color='grey', linewidth=0))
    if draw_labels:
        for i in range(n):
            ax.text(X[i,0], X[i,1], str(i))
    if title is not None:
        ax.set_title(title)
    ax.set_aspect('equal')


class MakeVideoFromMCMC:

    def __init__(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlim(-3, 3)
        self.ax.set_ylim(-3, 3)
        self.ax.set_zlim(-3, 3)
        self.sct, = self.ax.plot([], [], [], "o", markersize=2)

    def update(self, ifrm, xa, ya, za):
        self.sct.set_data(xa[ifrm], ya[ifrm])
        self.sct.set_3d_properties(za[ifrm])
        return self.ax

    def make(self, markov_chain):

        x_chain = []
        y_chain = []
        z_chain = []

        for count in range(len(markov_chain)):
            x_conf = []
            y_conf = []
            z_conf = []
            for i_part in range(5):
                x_conf.append(markov_chain[count, i_part, 0])
                y_conf.append(markov_chain[count, i_part, 1])
                z_conf.append(markov_chain[count, i_part, 2])
            x_chain.append(np.array(x_conf))
            y_chain.append(np.array(y_conf))
            z_chain.append(np.array(z_conf))

        ani = animation.FuncAnimation(self.fig, self.update, len(markov_chain), fargs=(x_chain, y_chain, z_chain), interval=200)
        return ani


