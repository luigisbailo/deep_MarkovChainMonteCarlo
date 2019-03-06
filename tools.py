import matplotlib.pyplot as plt


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
