import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend import Legend

from sampler.NestedSampling import GlobalNestedSampler
from slam.Variables import R2Variable, VariableType
from factors.Factors import UncertainR2RangeGaussianLikelihoodFactor, UncertainUnaryR2RangeGaussianPriorFactor
import random
from scipy.spatial import distance_matrix as dist_mat_fun
from slam.FactorGraphSimulator import factor_graph_to_string
import multiprocessing as mp

from utils.Functions import NumpyEncoder
from utils.Visualization import plot_2d_samples

if __name__ == '__main__':
    np.random.seed(1)
    random.seed(1)

    data = np.loadtxt("raw_data", delimiter=',', dtype=str)[1:,:]
    data = data.astype(float)
    # the second last col is ID
    sorted_idx = np.argsort(data[:, -2])
    sorted_data = data[sorted_idx]
    print(sorted_data[0])

    known_marker = "^"
    unknown_marker = "o"

    sorted_color = ['r', 'b', 'g', 'tab:purple', 'tab:orange', 'yellow', 'tab:brown', 'tab:pink',  'tab:olive',  'tab:cyan','black']

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
    adj_mat[2, [1,5, 4]] = 1
    adj_mat[3, [1, 0]] = 1
    adj_mat[4, [0, 2]] = 1
    adj_mat[5, [0, 2]] = 1

    data_dict = {}

    print(adj_mat-adj_mat.T)

    assert np.all(np.abs(adj_mat-adj_mat.T) < 1e-5) == True

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
            handle = axs.scatter([x], [y], marker = known_marker, c = c)
            known_lgd.append(handle)
        else:
            unknown_labels.append(str(id))
            handle = axs.scatter([x], [y], marker = unknown_marker, c = c)
            unknown_lgd.append(handle)

        for target in data_dict[i]['connect']:
            if target < i:
                t_x = data_dict[target]['x']
                t_y = data_dict[target]['y']
                axs.plot([t_x, x],[t_y, y], '--', color = 'grey')

    # lgd1 = plt.legend(known_lgd, loc = "upper right")
    axs.legend(known_lgd, known_labels, bbox_to_anchor=(.98, 1.0), loc='upper left', frameon = False, title="Known")
    leg = Legend(axs, unknown_lgd, unknown_labels, bbox_to_anchor=(.98, .5), loc='upper left', frameon=False, title="Unknown")
    axs.add_artist(leg)
    axs.set_xlabel("x")
    axs.set_ylabel("y")
    xlim = [-1.2, 1.8]
    ylim = [-1.2, 1.8]
    axs.set_xlim(xlim)
    axs.set_ylim(ylim)
    axs.set_aspect('equal', 'box')
    plt.subplots_adjust(right = .8)
    plt.savefig("gt.png", dpi = 300, bbox_to_anchor = "tight")
    plt.show()

    
    # generating factor graphs
    unobsv_sigma = .3
    sigma = .02

    # dist mat
    meas = np.loadtxt("measurements", dtype = str)[1:, :].astype(float)
    noisy_dist_mat = np.zeros_like(adj_mat)
    for i, line in enumerate(meas):
        id1, id2, distance = line
        id1 = int(id1 - 1)
        id2 = int(id2 - 1)
        assert adj_mat[id1, id2] == 1
        noisy_dist_mat[id1, id2], noisy_dist_mat[id2, id1] = distance, distance

    factors = []
    var2truth = {}
    vars = []
    for i in unknown_idx:
        assert data_dict[i]['known'] == False
        v = R2Variable(str(i))
        vars.append(v)
        var2truth[v] = np.array([data_dict[i]['x'], data_dict[i]['y']])

    manual_partition_llk = []
    auto_partition_factors = []

    for i in unknown_idx:
        for j in range(len(sorted_data) - 1, i, -1):
            if j in known_idx:
                if adj_mat[i, j] == 1:
                    factor = UncertainUnaryR2RangeGaussianPriorFactor(vars[i],
                                                                      np.array([data_dict[j]['x'], data_dict[j]['y']]),
                                                                      noisy_dist_mat[i, j],
                                                                      sigma,
                                                                      True,
                                                                      unobsv_sigma)
                    factors.append(factor)
                    auto_partition_factors.append(factor)
                else:
                    factor = UncertainUnaryR2RangeGaussianPriorFactor(vars[i],
                                                                      np.array([data_dict[j]['x'], data_dict[j]['y']]),
                                                                      noisy_dist_mat[i, j],
                                                                      sigma,
                                                                      False,
                                                                      unobsv_sigma)
                    factors.append(factor)
                    manual_partition_llk.append(factor)
            else:
                if adj_mat[i, j] == 1:
                    factor = UncertainR2RangeGaussianLikelihoodFactor(vars[i], vars[j],
                                                                      noisy_dist_mat[i, j],
                                                                      sigma,
                                                                      True,
                                                                      unobsv_sigma)
                    factors.append(factor)
                    auto_partition_factors.append(factor)
                else:
                    factor = UncertainR2RangeGaussianLikelihoodFactor(vars[i], vars[j],
                                                                      noisy_dist_mat[i, j],
                                                                      sigma,
                                                                      False,
                                                                      unobsv_sigma)
                    factors.append(factor)
                    manual_partition_llk.append(factor)

    fg = open('factor_graph2.fg','w+')
    lines = factor_graph_to_string(vars,factors, var2truth)
    fg.write(lines)
    fg.close()

    plot_args = {'xlim': xlim, 'ylim': ylim, 'fig_size': (8, 8),
                 'truth_label_offset': (3, -3), 'show_plot': False, 'equal_axis': True}

    i = 0
    run_cnt = 1
    run_dir  = f"run{run_cnt}"
    while True:
        if not os.path.exists(run_dir):
            os.mkdir(run_dir)
            break
        else:
            run_cnt += 1
            run_dir  = f"run{run_cnt}"

    step_file_prefix = f'{run_dir}/step{i}'
    observed_nodes = vars
    # ready to solve the problem!
    # solver = GlobalNestedSampler(nodes=vars, factors=auto_partition_factors, manually_partitioned_llh_factors=manual_partition_llk)
    solver = GlobalNestedSampler(nodes=vars, factors=factors, xlim=xlim, ylim=ylim)
    res_summary = {}
    parallel_config = {'cpu_frac': .8, 'queue_size': 12}
    plot_args = plot_args
    dlogz = .01
    live_pt = 1000
    step_timer = []
    more_configs = {'adapt_live_pt': False,
                    'dlogz': dlogz, 'sample': 'rwalk', 'use_grad_u': 'False'}
    start = time.time()
    if parallel_config is None:
        sample_arr = solver.sample(live_points=live_pt, res_summary=res_summary,
                                   **more_configs)
    else:
        if 'cpu_frac' in parallel_config:
            pool = mp.Pool(int(mp.cpu_count() * parallel_config['cpu_frac']))
        else:
            pool = mp.Pool(int(mp.cpu_count() * .5))
        sample_arr = solver.sample(live_points=live_pt, pool=pool, queue_size=parallel_config['queue_size'],
                                   res_summary=res_summary, **more_configs)
    end = time.time()

    with open(f'{step_file_prefix}.summary', 'w+') as smr_fp:
        smr_fp.write(json.dumps(res_summary, cls=NumpyEncoder))

    cur_sample = {}
    cur_dim = 0
    for var in observed_nodes:
        cur_sample[var] = sample_arr[:, cur_dim:cur_dim + var.dim]
        cur_dim += var.dim

    step_timer.append(end - start)
    print(f"step {i} time: {step_timer[-1]} sec, "
          f"total time: {sum(step_timer)}")

    file = open(f"{step_file_prefix}_ordering", "w+")
    file.write(" ".join([var.name for var in observed_nodes]))
    file.close()

    X = np.hstack([cur_sample[var] for var in observed_nodes])
    np.savetxt(fname=step_file_prefix + '.sample', X=X)

    plot_2d_samples(samples_mapping=cur_sample,
                    truth={variable: pose for variable, pose in
                           var2truth.items() if variable in observed_nodes},
                    truth_factors={factor for factor in auto_partition_factors if
                                   set(factor.vars).issubset(observed_nodes)},
                    file_name=f"{step_file_prefix}.png", title=f'Step {i}',
                    **plot_args)

    file = open(f"{run_dir}/step_timing", "w+")
    file.write(" ".join(str(t) for t in step_timer))
    file.close()