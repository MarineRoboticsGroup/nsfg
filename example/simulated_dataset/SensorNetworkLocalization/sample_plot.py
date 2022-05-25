import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend import Legend

import matplotlib
# matplotlib.rcParams.update({'font.size': 16})
from scipy.stats import gaussian_kde

if __name__ == '__main__':
    np.random.seed(1)
    data = np.loadtxt("raw_data", delimiter=',', dtype=str)[1:, :]
    data = data.astype(float)
    # the second last col is ID
    sorted_idx = np.argsort(data[:, -2])
    sorted_data = data[sorted_idx]
    print(sorted_data[0])

    known_marker = "^"
    unknown_marker = "o"
    # unknown_marker = "^"

    sorted_color = ['r', 'b', 'g', 'tab:purple', 'tab:orange','tab:pink', 'tab:brown',  'tab:olive', 'yellow',
                    'tab:cyan', 'black']

    known_idx = np.where(sorted_data[:, -1] == 1)[0]
    print("Known indices are ", known_idx)
    unknown_idx = np.where(sorted_data[:, -1] == 0)[0]
    print("Unknown indices are ", unknown_idx)

    # plotting the ground truthpy
    known_lgd = []
    unknown_lgd = []

    fig, axs = plt.subplots()
    color_cnt = 0
    known_labels = []
    unknown_labels = []
    known_plot = []
    unknown_plot = []

    # including observations
    adj_mat = np.zeros((len(sorted_data), len(sorted_data)))
    adj_mat[0, [5, 4, 3]] = 1
    adj_mat[1, [2, 3]] = 1
    adj_mat[2, [1, 5, 4]] = 1
    adj_mat[3, [1, 0]] = 1
    adj_mat[4, [0, 2]] = 1
    adj_mat[5, [0, 2]] = 1

    data_dict = {}

    print(adj_mat - adj_mat.T)

    assert np.all(np.abs(adj_mat - adj_mat.T) < 1e-5) == True


    run_dir = "run1"

    samples = np.loadtxt(f"{run_dir}/step0.sample")

    sample_num = 5000
    if len(samples) < sample_num:
        selected_row = np.random.choice(len(samples), size = sample_num)
        down_samples = samples[selected_row, :]
    else:
        down_samples = samples
    cur_dim = 0
    dim_inc = 2
    for i in unknown_idx:
        x = down_samples[:, cur_dim]
        y = down_samples[:, cur_dim + 1]
        unknown_labels.append(str(i+1))
        handle = axs.scatter(x, y, color=sorted_color[i], s=2)
        unknown_lgd.append(handle)

        cur_dim += dim_inc

    for i, line in enumerate(sorted_data):
        x, y, id, ifknow = line
        id = int(id)
        data_dict[i] = {'x': x, 'y': y, 'known': ifknow, "connect": np.where(adj_mat[i] == 1)[0]}
        if color_cnt >= len(sorted_color):
            c = "black"
        else:
            c = sorted_color[color_cnt]
            color_cnt += 1
        if ifknow:
            known_labels.append(str(id))
            handle = axs.scatter([x], [y], marker=known_marker, c=c, s=60)
            known_lgd.append(handle)
        else:
            # unknown_labels.append(str(id))
            handle = axs.scatter([x], [y], marker=known_marker, c='black', s=60)
            # unknown_lgd.append(handle)

        for target in data_dict[i]['connect']:
            if target < i:
                t_x = data_dict[target]['x']
                t_y = data_dict[target]['y']
                axs.plot([t_x, x], [t_y, y], '--', color='grey')

    # lgd1 = plt.legend(known_lgd, loc = "upper right")
    axs.legend(known_lgd, known_labels, bbox_to_anchor=(.2, .0), loc='lower center', frameon=False, title="Known")
    leg = Legend(axs, unknown_lgd, unknown_labels, bbox_to_anchor=(.8, .0), loc='lower center', frameon=False,
                 title="Unknown")
    axs.add_artist(leg)

    axs.set_xlabel("x")
    axs.set_ylabel("y")
    xlim = [-1.2, 1.8]
    ylim = [-1.2, 1.8]
    axs.set_xlim(xlim)
    axs.set_ylim(ylim)
    axs.set_aspect('equal', 'box')
    plt.subplots_adjust(right=.8)
    plt.savefig("sample.png", dpi=300, bbox_to_anchor="tight")
    plt.show()

    yxratio = 1/3
    fig2 = plt.figure(figsize=(5,5*yxratio*len(unknown_idx)))
    # fig = plt.figure()
    gs = fig2.add_gridspec(len(unknown_idx),1, hspace=0.5, wspace=0.05)
    # axs = fig.subplots(len(folders), len(step_nums),sharex="row",sharey="row")
    axs2 = gs.subplots(squeeze=True)

    #
    # fig2, axs2 = plt.subplots(4, 1)
    cur_dim = 0
    dim_inc = 2
    for i in unknown_idx:
        x = down_samples[:, cur_dim]
        # w = .1
        # n = math.ceil((x.max() - x.min()) / w)
        axs2[i].hist(x, bins=40, density=True)
        cur_dim += dim_inc
        axs2[i].set_xlabel(f'$x_{i+1}$', color=sorted_color[i])
        axs2[i].get_xaxis().set_label_coords(0.5, -0.25)
        axs2[i].set_ylabel(f"Density")
        axs2[i].get_yaxis().set_label_coords(-0.1, 0.5)

        density = gaussian_kde(x)
        xs = np.linspace(min(x)-.3, max(x)+.3, 400)
        density.covariance_factor = lambda: .04
        density._compute_covariance()
        axs2[i].plot(xs, density(xs))
    fig2.savefig(f"{run_dir}/hist.png", dpi = 300, bbox_inches = "tight")