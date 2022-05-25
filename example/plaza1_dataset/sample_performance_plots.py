import json
import random
from typing import List

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from example.slam.manhattan_waterworld.process_gtsam import getMeans
from factors.Factors import AmbiguousDataAssociationFactor, PriorFactor, KWayFactor, BinaryFactor, \
    SE2RelativeGaussianLikelihoodFactor, OdomFactor

from slam.RunBatch import group_nodes_factors_incrementally

from slam.FactorGraphSimulator import read_factor_graph_from_file
from slam.Variables import Variable, VariableType, R2Variable, SE2Variable
import os
import pandas as pd
from sampler.sampler_utils import JointFactor
import sklearn.metrics.pairwise

from utils.Functions import array_order_to_dict
from utils.Statistics import mmd, MMDb, MMDu2
import seaborn as sns
from matplotlib.ticker import MaxNLocator, ScalarFormatter

from utils.Visualization import plot_2d_samples


def reorder_samples(ref_order: List[Variable],
                    sample_order: List[Variable],
                    samples: np.ndarray):
    # res = np.zeros_like(samples)
    # cur_dim = 0
    # for var in ref_order:
    #     var_idx = sample_order.index(var)
    #     if var_idx == 0:
    #         sample_dim = 0
    #     else:
    #         sample_dim = np.sum([it.dim for it in sample_order[:var_idx]])
    #     res[:,cur_dim:cur_dim+var.dim] = samples[:,sample_dim:sample_dim+var.dim]
    #     cur_dim += var.dim
    # return res
    res = []
    cur_dim = 0
    for var in ref_order:
        var_idx = sample_order.index(var)
        if var_idx == 0:
            sample_dim = 0
        else:
            sample_dim = np.sum([it.dim for it in sample_order[:var_idx]])
        res.append(samples[:, sample_dim:sample_dim + 2])
        cur_dim += var.dim
    return np.hstack(res)

if __name__ == '__main__':

    # m_s = 12
    f_size = 14
    m_s = 10

    plot_scatter = False
    plot_perf = True
    compute_traj_div_perf = True

    map_rmse = True
    mean_rmse = False

    all_scatter = True

    mmd_func = MMDb
    MMD_type = 'MMDb'
    kernel_scale = 5
    threshold_dist = .5


    plt.rc('xtick', labelsize=f_size)
    plt.rc('ytick', labelsize=f_size)


    ada20_folder = "Plaza1ADA0.2EFG"
    reference_folder = "Plaza1EFG"
    ada40_folder = "Plaza1ADA0.4EFG"
    ada60_folder = "Plaza1ADA0.6EFG"
    # nfisam_folder = "run1"
    # unif_folder = "dyn_unif1"


    # plt.rc('xtick', labelsize=f_size)
    # plt.rc('ytick', labelsize=f_size)
    setup_folder = "RangeOnlyDataset"
    # setup_folder = "test_samplers/loop_clos_odom_factor_graphs/2_loop_clos"
    # setup_folder = "test_samplers/5PosesGP_NF"

    plot_dir = f"{setup_folder}/figures_rev"
    if (os.path.exists(plot_dir)):
        pass
    else:
        os.mkdir(plot_dir)

    # case_dirs = [f"{setup_folder}/{f}" for f in os.listdir(setup_folder) if os.path.isdir(f"{setup_folder}/{f}")]
    # seed_nums = np.arange(10)
    # case_dirs = [f"{setup_folder}/test_{i}" for i in seed_nums]


    fg_name = "factor_graph_trim.fg"

    folder2lgd = {}
    folder2lgd[ada40_folder] = '40% ADA'
    folder2lgd[ada60_folder] = '60% ADA'
    folder2lgd[reference_folder] = '00% ADA'
    folder2lgd[ada20_folder] = '20% ADA'
    # folder2lgd[nfisam_folder] = 'NFiSAM'
    # folder2lgd[unif_folder] = 'NS(UnifPr)'

    folder2linestyle = {}
    folder2linestyle[ada40_folder] = 'dashdot'
    folder2linestyle[reference_folder] = 'solid'
    folder2linestyle[ada60_folder] = 'dashed'
    folder2linestyle[ada20_folder] = 'dotted'
    # folder2linestyle[nfisam_folder] = ':'
    # folder2linestyle[unif_folder] = '--'

    folder2color = {}
    folder2color[ada40_folder] = 'g'
    folder2color[reference_folder] = 'r'
    folder2color[ada60_folder] = 'b'
    folder2color[ada20_folder] = 'k'
    # folder2color[nfisam_folder] = 'm'
    # folder2color[unif_folder] = 'y'

    # solver2linestyle = {}
    # solver2linestyle[folder2lgd[ada40_folder]] = (3, 1.25, 1.5, 1.25)
    # solver2linestyle[folder2lgd[reference_folder]] = ''
    # solver2linestyle[folder2lgd[ada60_folder]] = (4, 1.5)
    # solver2linestyle[folder2lgd[ada20_folder]] = (1, 1)

    solver2color = {}
    solver2color[folder2lgd[ada40_folder]] = 'g'
    solver2color[folder2lgd[reference_folder]] = 'r'
    solver2color[folder2lgd[ada60_folder]] = 'b'
    solver2color[folder2lgd[ada20_folder]] = 'k'
    # solver2color[folder2lgd[nfisam_folder]] = 'm'
    # solver2color[folder2lgd[unif_folder]] = 'y'

    solver2marker = {}
    solver2marker[folder2lgd[ada40_folder]] = 'o'
    solver2marker[folder2lgd[reference_folder]] = 'X'
    solver2marker[folder2lgd[ada60_folder]] = '^'
    solver2marker[folder2lgd[ada20_folder]] = 'P'

    case_all_step_data = [] #Seed, Step, ADA, Time, RMSE

    all_folders = [reference_folder, ada20_folder, ada40_folder, ada60_folder]#]
    baseline_folders = [reference_folder, ada20_folder, ada40_folder, ada60_folder]#]
    if all_scatter:
        scatter_folders = all_folders
    else:
        scatter_folders = [ada40_folder, reference_folder]
    sample_num = 10000

    # mmd_func = MMDb
    # MMD_type = 'MMDb'
    # kernel_scale = 10

    color_list = ['m','darkorange','black','y','c','b','g','r']

    # xlim=[-50, 220]
    # xlim=[-20, 70]
    #
    # ylims=np.array([[30, 130],[0, 170],[-30, 190],[-70,190],[-10,200],[-100,210],[-30,130],[50,190],[-30,120],[30,180]])

    seed_nums = np.arange(1, 7)
    # case_dirs = [f"{setup_folder}/dyn{i}" for i in seed_nums]

    xlim = (-65, 65)
    ylims = [(-65, 65)] * len(seed_nums)
    # ylims = np.array([xlim]*len(seed_nums))


    y_lens = [ylim[1] - ylim[0] for ylim in ylims]
    y_len = sum(y_lens)
    x_len = xlim[1] - xlim[0]
    y_x_ratio = y_len / x_len

    tot_steps = 29

    if plot_scatter:
        axs_arr = []
        figs_arr = []
        for t_step in range(tot_steps):
            fig_scale = 2
            scatter_fig = plt.figure(figsize=(fig_scale * len(seed_nums), fig_scale * len(scatter_folders)))
            gs = scatter_fig.add_gridspec(len(scatter_folders), len(seed_nums),
                                          # width_ratios=[1]*len(scatter_folders),height_ratios = y_lens,
                                          hspace=0.05, wspace=0.05)
            axs = gs.subplots(sharex=True, sharey='row')
            figs_arr.append(scatter_fig)
            axs_arr.append(axs)

        for k, case_folder in enumerate(scatter_folders):
            case_path = f"{setup_folder}/{case_folder}"
            fg_file = f"{case_path}/{fg_name}"
            nodes, truth, factors = read_factor_graph_from_file(fg_file)
            nodes_factors_by_step = group_nodes_factors_incrementally(
                nodes=nodes, factors=factors, incremental_step=1)
            colors = {}
            for i, node in enumerate(nodes):
                if node.type == VariableType.Pose:
                    colors[node] = 'grey'
                else:
                    colors[node] = color_list[i % len(color_list)]
            for t_step in range(tot_steps):
                fig = figs_arr[t_step]
                axs = axs_arr[t_step]
                for i, seed in enumerate(seed_nums):
                    dir = f"{case_path}/dyn{seed}"
                    step = t_step
                    ax = axs[k, i]
                    sample_file = f"{dir}/step{step}.sample"
                    order_file = f"{dir}/step{step}_ordering"
                    if os.path.exists(sample_file):
                        samples = np.loadtxt(sample_file)
                        order = Variable.file2vars(order_file=order_file)
                        if (samples.shape[0] >= sample_num):
                            downsampling_indices = np.array(
                                random.sample(list(range(samples.shape[0])), sample_num))
                            trimmed = samples[downsampling_indices, :]
                        else:
                            print(f"{dir} has {samples.shape[0]} samples at step {step}.")
                            trimmed = samples
                        part_color = {key: colors[key] for key in order}
                        ax = plot_2d_samples(ax=ax, samples_array=trimmed, variable_ordering=order,
                                             show_plot=False, equal_axis=False, colors=part_color, marker_size=.1,
                                             xlabel=None, ylabel=None,
                                             xlim=xlim, ylim=ylims[k])
                        for node in order:
                            if node.name == "L0":
                                dx, dy = -5, 5
                            else:
                                dx, dy = 5, -5
                            if len(truth[node]) == 2:
                                x, y = truth[node][:2]
                                color = "blue"
                                marker = "x"
                                ax.scatter(x, y, s=20, marker=marker, color=color, linewidths=1.0)
                                ax.text(x + dx, y + dy, s=node.name, fontsize=f_size - 3)
                            # else:
                            #     x, y, th = truth[node][:3]
                            #     color = "red"
                            #     marker = mpl.markers.MarkerStyle(marker='$\u2193$')  # Downwards arrow in Unicode: ↓
                            #     marker._transform = marker.get_transform().rotate_deg(90 + th * 180 / np.pi)
                            # ax.plot([x], [y], c=color, markersize=2,mar marker="+")
                            # ax.scatter(x, y, s=90,facecolors='none',edgecolors='r')

                        for factor in factors:
                            if not isinstance(factor, PriorFactor) and (set(factor.vars).issubset(set(order))):
                                if isinstance(factor, KWayFactor):
                                    var1 = factor.root_var
                                    var2s = factor.child_vars
                                    for var2 in var2s:
                                        x1, y1 = truth[var1][:2]
                                        x2, y2 = truth[var2][:2]
                                        ax.plot([x1, x2], [y1, y2], linestyle='--', dashes=(5, 5), c='red',
                                                linewidth=.2)
                                # elif isinstance(factor, SE2RelativeGaussianLikelihoodFactor):
                                #     var1, var2 = factor.vars
                                #     x1, y1 = truth[var1][:2]
                                #     x2, y2 = truth[var2][:2]
                                #     id_diff = abs(int(var1.name[1:]) - int(var2.name[1:]))
                                #     if id_diff == 1:
                                #         ax.plot([x1, x2], [y1, y2], c='lime', linewidth=.5)
                                #     else:
                                #         ax.plot([x1, x2], [y1, y2], c='red', linewidth=.5)
                                elif isinstance(factor, BinaryFactor):
                                    if isinstance(factor, OdomFactor):
                                        c = 'lime'
                                        lw = .5
                                    else:
                                        c = 'k'
                                        lw = .2
                                    var1, var2 = factor.vars
                                    x1, y1 = truth[var1][:2]
                                    x2, y2 = truth[var2][:2]
                                    ax.plot([x1, x2], [y1, y2], c=c, linewidth=lw)
                        # ax.axis("equal")
                    else:
                        ax.set_xlim(xlim)
                        ax.set_ylim(ylims[k])
                        ax.text(10, 25, 'No solution', fontsize=10)
                    # if i == 0:
                    ax.set_xlabel(f"Seed {seed}", fontsize=f_size + 4)
                    ax.get_yaxis().set_label_coords(-.4, 0.5)
                    # if k == len(case_dirs)-1:
                    ax.set_ylabel(folder2lgd[case_folder], fontsize=f_size + 4)
                # scatter_fig.savefig(f"{plot_dir}/scatter_final_step.png",dpi=300,bbox_inches="tight")
                fig.suptitle(f'Time step {t_step}', fontsize=f_size + 4, y=0.91)
        for i, fig in enumerate(figs_arr):
            axs = axs_arr[i]
            for ax in axs.flat:
                ax.label_outer()
            fig.savefig(f"{plot_dir}/all_scatter_step{i}.png", dpi=300,
                        bbox_inches="tight")

    if compute_traj_div_perf:
        for k, case_folder in enumerate(scatter_folders):
            case_path = f"{setup_folder}/{case_folder}"
            fg_file = f"{case_path}/{fg_name}"
            nodes, truth, factors = read_factor_graph_from_file(fg_file)
            nodes_factors_by_step = group_nodes_factors_incrementally(
                nodes=nodes, factors=factors, incremental_step=1)
            colors = {}
            for i, node in enumerate(nodes):
                if node.type == VariableType.Pose:
                    colors[node] = 'grey'
                else:
                    colors[node] = color_list[i % len(color_list)]
            folder2timing = {}
            if compute_traj_div_perf:
                for i, seed in enumerate(seed_nums):
                    # for i, folder in enumerate(all_folders):
                    dir = f"{case_path}/dyn{seed}"
                    if os.path.exists(f"{dir}/timing"):
                        folder2timing[case_folder] = np.loadtxt(f"{dir}/timing").tolist()
                    elif os.path.exists(f"{dir}/step_timing"):
                        folder2timing[case_folder] = np.loadtxt(f"{dir}/step_timing").tolist()
                    else:
                        print(dir)
                        raise ValueError("No timing files.")
                    cur_factors = []
                    for t_step in range(tot_steps):
                        _, step_factors = nodes_factors_by_step[t_step]
                        cur_factors += step_factors
                        step = t_step
                        sample_file = None
                        if os.path.exists(f"{dir}/step{step}.sample"):
                            sample_file = f"{dir}/step{step}.sample"
                        elif os.path.exists(f"{dir}/step{step}"):
                            sample_file = f"{dir}/step{step}"
                        else:
                            print("No samples!")
                        order_file = f"{dir}/step{step}_ordering"
                        samples = np.loadtxt(sample_file)
                        order = Variable.file2vars(order_file=order_file)

                        true_xy = []
                        for var in order:
                            true_xy.append(truth[var][0])
                            true_xy.append(truth[var][1])
                        true_xy = np.array(true_xy)

                        if map_rmse:
                            jf = JointFactor(cur_factors, order)
                            assert set(order) == set(jf.vars)
                            log_pdf = jf.log_pdf(samples)
                            map_idx = np.argmax(log_pdf)
                            map_point = samples[map_idx, :]

                            plot_2d_samples(samples_array=samples[map_idx:map_idx+1, :], variable_ordering=order,
                                            show_plot=True, equal_axis=False,  marker_size=10,
                                            title=f"step {t_step}")

                            map_trans = []
                            cur_dim = 0
                            for var in order:
                                map_trans.append(map_point[cur_dim])
                                map_trans.append(map_point[cur_dim + 1])
                                cur_dim += var.dim
                            map_trans = np.array(map_trans)
                            diff_xy = map_trans - true_xy
                            rmse = np.sqrt(np.mean(diff_xy ** 2))
                        else:
                            samples = reorder_samples(ref_order=order, sample_order=order, samples=samples)
                            assert (samples.shape[1] % 2 == 0)
                            # m = mmd_func(samples, ref_samples,
                            #              kernel_scale * np.sqrt(samples.shape[1]))
                            # print(f"MMD of {var.name} for {folder2lgd[folder]} at step {step}: {m}")
                            mean_xy = np.mean(samples, axis=0)
                            diff_xy = mean_xy - true_xy
                            rmse = np.sqrt(np.mean(diff_xy ** 2))
                        cur_res = [seed, step, folder2lgd[case_folder], folder2timing[case_folder][step], rmse]
                        print(f"Current result: {cur_res}")
                        case_all_step_data.append(cur_res)
        columns = ["Seed", "Step", "ADA", "Time", "RMSE"]
        dd_df = pd.DataFrame(case_all_step_data, columns=columns)
        dd_df.to_csv(f"{plot_dir}/compute_traj_div_perf.txt", index=False)
    elif plot_perf:
        dd_df = pd.read_csv(f"{plot_dir}/compute_traj_div_perf.txt")
    if dd_df is not None and plot_perf:
        dd_df.head()
        var_names = ["Seed", "Step", "ADA"]
        target_names = ["Time", "RMSE"]
        label_names = ['Time (sec)', 'RMSE (m)']
        aspect_ratio = 1/2
        fig_scale = 2
        fig = plt.figure(figsize=( 2 *fig_scale *  aspect_ratio, len(label_names) *fig_scale * 2))
        gs = fig.add_gridspec(len(label_names),1,  hspace=0.05, wspace=0.05)#, top=.9, bottom=.1, right=.95, left=.05)
        axs = gs.subplots(sharex='col')

        row = 0
        label_dx = -0.13
        # plot ADA
        # data = dd_df.query("Solver != 'GTSAM'")
        data = dd_df

        selects = [21,24]
        select_y = .5
        bot, top = 0.5, 30
        for j, target in enumerate(target_names):
            ax = axs[j]
            # if j >0:
                # ax.yaxis.tick_right()
                # ax.yaxis.set_label_position("right")
            if target == "RMSE":
                ax = sns.lineplot(ax=ax, data=data, x="Step", y=target, hue="ADA",
                                  markersize=int(m_s / 4))
                for select in selects:
                    ax.plot(select, bot, '+', c="red")
                    ax.text(select-1.5, bot+.05, f"{select}", fontsize = f_size - 6)
                ax.set_ylim(bot, top)

            else:
                ax = sns.lineplot(ax=ax, data=data, x="Step", y=target, hue="ADA",
                                  markersize=int(m_s / 4))
            ax.set(yscale="log")
            ax.set_xticks([0, 10, 20, 30])
            ax.tick_params(labelsize=f_size - 4)
            ax.set_ylabel(label_names[j], fontsize=f_size - 2)
            ax.set_xlabel("Time Step", fontsize=f_size - 2)

            # for folder in scatter_folders:
            #     # ada_data = dd_df.query(f"ADA == '{folder2lgd[folder]}'")
            #     if target == "RMSE":
            #         ax=sns.lineplot(ax=ax, data=data, x="Step", y=target,hue="ADA", label=folder2lgd[folder],color=solver2color[folder2lgd[folder]],
            #                   linestyle=folder2linestyle[folder],
            #                   markersize=int(m_s/4))
            #     else:
            #         ax=sns.lineplot(ax=ax, data=data, x="Step", y=target,hue="ADA", label=folder2lgd[folder], color=solver2color[folder2lgd[folder]],
            #                   linestyle=folder2linestyle[folder],
            #                   markersize=int(m_s/4))

            # sns.barplot(ax=ax,x="Solver", y=target, data=data)
            # ax.set_ylabel(label_names[j], fontsize=f_size - 2)
            # ax.set_xlabel("Solver", fontsize=f_size - 2)
            # ax.get_yaxis().set_label_coords(label_dx, 0.5)
        handles, labels = axs[1].get_legend_handles_labels()
        # axs[0].get_legend().set_visible(False)
        axs[1].get_legend().set_visible(False)
        # fig.legend(handles, labels, loc="upper center", ncol=len(all_folders), prop={'size': 9},
        #            bbox_to_anchor=(.5, 1.0))
        # axs[0].set_yticks([1e-3,1e-2,1e-1,1,10,100,1000])
        # axs[1].set_yticks([.1,1,10])
        # axs[2].set_yticks([.01,.1])
        # for ax in axs.flat:
        #     ax.label_outer()
        plt.show()
        fig.savefig(f"{plot_dir}/traj_diversity_performance_grid.png", dpi=300, bbox_inches='tight')
