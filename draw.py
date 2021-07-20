import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def get_color(atom):
    if atom == 'F':
        return 'green'
    if atom == 'C':
        return 'black'
    if atom == 'H':
        return 'cyan'
    if atom == 'O':
        return 'red'
    if atom == 'N':
        return 'blue'

def draw(st_dist):
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    xs = st_dist.x
    ys = st_dist.y
    zs = st_dist.z
    ax.scatter(xs, ys, zs, color=[get_color(i) for i in st_dist.atom], s=100)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    
    for ind0 in st_dist.index:
        series = st_dist.loc[ind0]
        atom0 = series.atom
        ax.text3D(series.x, series.y,series.z, str(ind0), color="magenta", fontsize=14)
        for ind1, dist in zip(series.nn_indices[:4], series.nn_distances[:4]):
            atom1 = st_dist.loc[ind1].atom
            if ('H' == atom0) or ('H' == atom1):
                threash = 1.35
            else:
                threash = 1.82
            if dist < threash:
                x_line = np.linspace(series.x, st_dist.loc[ind1].x, 3)
                y_line = np.linspace(series.y, st_dist.loc[ind1].y, 3)
                z_line = np.linspace(series.z, st_dist.loc[ind1].z, 3)
                ax.plot3D(x_line, y_line, z_line, 'gray')
                
            
            

    plt.show()