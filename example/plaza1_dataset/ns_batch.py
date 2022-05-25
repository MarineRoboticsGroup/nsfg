import os
from sampler.NestedSampling import dynesty_run_batch

if __name__ == '__main__':
    run_file_dir = os.path.dirname(__file__)
    # cases = ["Plaza2ADA0.6","Plaza2ADA0.4","Plaza2ADA0.2","Plaza2"]
    for _ in range(6):
        cases = ["Plaza1","Plaza1ADA0.2","Plaza1ADA0.4","Plaza1ADA0.6"]
        for case in cases:
            case_dir = os.path.join(run_file_dir,f"RangeOnlyDataset/{case}EFG")
            data_file = 'factor_graph_trim.fg'
            data_format = 'fg'
            dynesty_run_batch(1000, case_dir, data_file, data_format,
                              incremental_step=1, parallel_config={'cpu_frac': 0.8, 'queue_size': 64}, prior_cov_scale=0.1,
                              plot_args={'truth_label_offset': (3, -3), 'show_plot': False})
