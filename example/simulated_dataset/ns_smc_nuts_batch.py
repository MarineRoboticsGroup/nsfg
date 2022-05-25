import os

from sampler.SMCSampler import smc_run_batch

from sampler.NUTSampler import nuts_run_batch

from sampler.NestedSampling import dynesty_run_batch

if __name__ == '__main__':
    run_file_dir = os.path.dirname(__file__)

    parent_dirs = ["SingleRobotRangeOnly"]

    xlims = [(-100, 250)]
    ylims = [(-100, 250)]

    for i, parent_dir in enumerate(parent_dirs):
        xlim = xlims[i]
        ylim = ylims[i]
        plot_args = {'xlim': xlim, 'ylim': ylim, 'fig_size': (8, 8),
                     'truth_label_offset': (3, -3), 'show_plot': False, 'equal_axis': True}
        case_dirs = [f"{parent_dir}/{dir}" for dir in os.listdir(parent_dir) if
                     os.path.isdir(f"{parent_dir}/{dir}")]
        for case_dir in case_dirs:
            data_files = [name for name in os.listdir(case_dir) if name[-2:] == 'fg']
            data_format = 'fg'
            for data_file in data_files:
                try:
                    # NSFG
                    dynesty_run_batch(1000, case_dir, data_file, data_format,
                                      parallel_config={'cpu_frac': 1.0, 'queue_size': 12},
                                      incremental_step=1, prior_cov_scale=0.1,
                                      plot_args=plot_args,
                                      dynamic_ns=False,
                                      adapt_live_pt=False,
                                      dlogz=.01,
                                      use_grad_u=False
                                      )
                except Exception as e:
                    print(str(e))
                try:
                    # NS(UnifPr): a vanilla sampler using nested sampling with uniform distributions as priors
                    dynesty_run_batch(1000, case_dir, data_file, data_format,
                                      parallel_config={'cpu_frac': 1.0, 'queue_size': 12},
                                      incremental_step=1, prior_cov_scale=0.1,
                                      plot_args=plot_args,
                                      dynamic_ns=False,
                                      adapt_live_pt=False,
                                      dlogz=.01,
                                      xlim=xlim,
                                      ylim=ylim,
                                      use_grad_u=False
                                      )
                except Exception as e:
                    print(str(e))
                try:
                    #NUTS
                    nuts_run_batch(500, case_dir, data_file, data_format,
                                   incremental_step=1, nuts_config={'tune': 2000,
                                                                     'chains': 12,
                                                                     'cores': 8,
                                                                    'target_accept':.9
                                                                     }, prior_cov_scale=0.1,
                                   plot_args=plot_args)
                except Exception as e:
                    print(str(e))
                try:
                    #SMC
                    smc_run_batch(500, xlim, ylim, case_dir, data_file, data_format, incremental_step=1,
                                  prior_cov_scale=0.1, plot_args=plot_args,
                                  smc_config={'parallel': True, 'cores': 8})
                except Exception as e:
                    print(str(e))