import random
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from factors.Factors import PriorFactor, KWayFactor, BinaryFactor, OdomFactor
from slam.FactorGraphSimulator import read_factor_graph_from_file
from utils.Visualization import plot_2d_samples
from slam.Variables import Variable, VariableType
import os

if __name__ == '__main__':

    # m_s = 12
    f_size = 12
    plt.rc('xtick', labelsize=f_size)
    plt.rc('ytick', labelsize=f_size)
    bound = 70
    xlim = [-bound, bound]
    ylim = [-bound, bound]
    yxratio = (ylim[1]-ylim[0]) /(xlim[1]-xlim[0])
    # plt.rc('xtick', labelsize=f_size)
    # plt.rc('ytick', labelsize=f_size)
    # setup_folder = "test_samplers/RO_NF_NoisyOdom_STD2"
    setup_folder = "RangeOnlyDataset"
    # setup_folder = "test_samplers/5PosesGP_NF"
    case_folder = "Plaza1EFG"
    # case_folder = "Plaza1ADA0.6EFG"
    # case_folder = "test_9"
    # case_folder = "pada0.4_r0.5_odom0.01_mada3"
    # gtsam_folder = "gtsam"
    reference_folder = "dyn1"
    smc_folder = "sir1"
    # nuts_folder = "nuts1"
    # unif_folder = "dyn_unif1"
    # nfisam_folder = "run1"

    folder2lgd = {}
    folder2lgd[smc_folder] = "SIR"
    # folder2lgd[nuts_folder] = "NUTS"
    # folder2lgd[reference_folder] = "Known DA"
    folder2lgd[reference_folder] = "NSFG"
    # folder2lgd[gtsam_folder] = "GTSAM"
    # folder2lgd[unif_folder] = "NS(UnifPr)"
    # folder2lgd[nfisam_folder] = "NFiSAM"

    case_dir = f"{setup_folder}/{case_folder}"
    ref_dir = f"{setup_folder}/{case_folder}/{reference_folder}"
    # gtsam_dir = f"{setup_folder}/{case_folder}/{gtsam_folder}"
    # nf_dir = f"{setup_folder}/{case_folder}/{smc_folder}"

    plot_dir = f"{setup_folder}/{case_folder}/figures_rev"
    if(os.path.exists(plot_dir)):
        pass
    else:
        os.mkdir(plot_dir)
    fg_file = "factor_graph_trim.fg"
    nodes, truth, factors = read_factor_graph_from_file(
        f"{case_dir}/{fg_file}")

    # folders = [reference_folder,unif_folder,nfisam_folder, smc_folder, nuts_folder, gtsam_folder]
    folders = [reference_folder, smc_folder]
    # folders = [nuts_folder]

    # step_nums = [5,10,15]
    step_nums = [5, 15, 19, 20, 21,24]
    # step_nums = [2,3]
    # step_nums = [5]

    sample_num = 10000

    color_list = ['m','darkorange','b','y','c','r']
    colors = {}
    for i, node in enumerate(nodes):
        if node.type == VariableType.Pose:
            colors[node] = 'grey'
        else:
            colors[node] = color_list[i%len(color_list)]

    # fig = plt.figure(figsize=(5*len(step_nums),5*len(folders)*yxratio))
    fig = plt.figure(figsize=(5*len(step_nums),5*len(folders)*yxratio))
    # fig = plt.figure()
    gs = fig.add_gridspec(len(folders),len(step_nums), hspace=0.05, wspace=0.05)
    # axs = fig.subplots(len(folders), len(step_nums),sharex="row",sharey="row")
    axs = gs.subplots(sharex=True,sharey=True,squeeze=False)

    for i, folder in enumerate(folders):
        for j, step in enumerate(step_nums):
            dir = f"{setup_folder}/{case_folder}/{folder}"
            sample_file = f"{dir}/step{step}.sample"
            if not os.path.exists(sample_file):
                sample_file = f"{dir}/step{step}"
            order_file = f"{dir}/step{step}_ordering"
            ax = axs[i, j]

            if os.path.exists(sample_file):
                samples = np.loadtxt(sample_file)
                order = Variable.file2vars(order_file=order_file)
                if (samples.shape[0] >= sample_num):
                    downsampling_indices = np.array(random.sample(list(range(samples.shape[0])), sample_num))
                    trimmed = samples[downsampling_indices, :]
                else:
                    print(f"{folder} has fewer samples than others at step {step}.")
                    trimmed = samples

                # plt.axis("equal")
                part_color = {key:colors[key] for key in order}
                # plt.xlim([-10, 30])
                ax = plot_2d_samples(ax=ax, samples_array=trimmed, variable_ordering=order,
                                     show_plot=False, equal_axis=True, colors=part_color,marker_size=.7,xlabel=None,ylabel=None,
                                     xlim=xlim, ylim=ylim)

                for factor in factors:
                    if not isinstance(factor, PriorFactor) and (set(factor.vars).issubset(set(order))):
                        if isinstance(factor, KWayFactor):
                            var1 = factor.root_var
                            var2s = factor.child_vars
                            for var2 in var2s:
                                x1, y1 = truth[var1][:2]
                                x2, y2 = truth[var2][:2]
                                ax.plot([x1, x2], [y1, y2],linestyle='--', dashes=(5, 5), c='red', linewidth=.2)
                        elif isinstance(factor, BinaryFactor):
                            if isinstance(factor, OdomFactor):
                                c='lime'
                                lw = 1.0
                            else:
                                c='k'
                                lw = .2
                            var1, var2 = factor.vars
                            x1, y1 = truth[var1][:2]
                            x2, y2 = truth[var2][:2]
                            ax.plot([x1, x2], [y1, y2], c=c, linewidth=lw)


                for node in order:
                    if len(truth[node]) == 2:
                        x, y = truth[node][:2]
                        color = "black"
                        marker = "x"
                        if node.name == "L0":
                            dx, dy = -5, 5
                        else:
                            dx, dy = 5, -5
                        ax.text(x + dx, y + dy, s=node.name, fontsize=f_size)
                        ax.scatter(x, y, s=50, marker=marker, color=color, linewidths=1.0)
                    else:
                        x, y, th = truth[node][:3]
                        dx, dy = 5, -5
                        color = "black"
                        # marker = mpl.markers.MarkerStyle(marker='$\u2193$')  # Downwards arrow in Unicode: ↓
                        # marker._transform = marker.get_transform().rotate_deg(90 + th * 180 / np.pi)
                        marker = "+"
                        ax.scatter(x, y, s=15, marker=marker, color=color, linewidths=1.0)

                    # ax.scatter(x, y, s=5, marker=marker, color=color, linewidths=.2)
                    # ax.plot([x], [y], c=color, markersize=2,mar marker="+")

                # ax.axis("equal")
            else:
                ax.text(25, 60, 'No solution', fontsize=30)

            # if i == 0:
            ax.set_xlabel(f"Time step {step}",fontsize=f_size+4)
            # if step == max(step_nums):
            ax.set_ylabel(folder2lgd[folder],fontsize=f_size+4)

            # plt.xlabel('x(m)',fontsize=f_size+2)
            # plt.ylabel('y(m)',fontsize=f_size+2)
            # plt.xlim((-10, 30))
            # plt.ylim((-8, 23))

            # plt.xlim((-15, 35))
            # plt.ylim((-12, 25))

            # if folder == reference_folder:
            #     plt.savefig(f"{plot_dir}/small_case_step{step}_dynesty.png",bbox_inches='tight')
            # elif folder == gtsam_folder:
            #     plt.savefig(f"{plot_dir}/small_case_step{step}_Max-mixture.png",bbox_inches='tight')
            # elif folder == smc_folder:
            #     plt.savefig(f"{plot_dir}/small_case_step{step}_NF-iSAM.png",bbox_inches='tight')
            # elif folder == nuts_folder:
            #     plt.savefig(f"{plot_dir}/small_case_step{step}_Caesar.png",bbox_inches='tight')
            # plt.show()
            # plt.close()
    for ax in axs.flat:
        ax.label_outer()
        ax.set_aspect('equal', 'box')
    fig.tight_layout()
    fig.savefig(f"{plot_dir}/scatter_ns_sir.png", dpi=300, bbox_inches="tight")
    # fig.show()
