import os

from sampler.ParticleFilter import GlobalParticleFilter, pf_run_batch, SIRPF_run_batch

if __name__ == '__main__':
    run_file_dir = os.path.dirname(os.path.realpath(__file__))

    parent_dirs = ["RangeOnlyDataset"]

    for i, parent_dir in enumerate(parent_dirs):
        # xlim = xlims[i]
        # ylim = ylims[i]
        # plot_args = {'xlim': xlim, 'ylim': ylim, 'fig_size': (8, 8),
        #              'truth_label_offset': (3, -3), 'show_plot': False, 'equal_axis': True}
        plot_args = {'fig_size': (8, 8),
                     'truth_label_offset': (3, -3), 'show_plot': False, 'equal_axis': True}
        case_dirs = [f"{parent_dir}/Plaza1EFG"]# for dir in os.listdir(parent_dir) if os.path.isdir(f"{parent_dir}/{dir}")]
        # case_dirs = [f"{parent_dir}/{dir}" for dir in os.listdir(parent_dir) if os.path.isdir(f"{parent_dir}/{dir}")]
        for case_dir in case_dirs:
            data_file = 'factor_graph_trim.fg'
            data_format = 'fg'
            SIRPF_run_batch(400000, case_dir, data_file, data_format,
                         incremental_step=1, resample_rate = 0,
                         plot_args=plot_args)
            #
            # try:
            # except Exception as e:
            #     print(str(e))
