import os
import time
from typing import List

import numpy as np
from matplotlib import pyplot as plt
from theano import tensor as tt
import pymc3 as pm
from sampler.sampler_utils import JointFactor
from sampler.theano_functions import LogJFWithGrad
from factors.Factors import Factor, BinaryFactorMixture, KWayFactor
from slam.RunBatch import graph_file_parser, group_nodes_factors_incrementally
from slam.Variables import Variable
from utils.Visualization import plot_2d_samples
import arviz as az


class GlobalMCMCSampler(object):
    def __init__(self, nodes: List[Variable], factors: List[Factor]):
        self._dim = sum([var.dim for var in nodes])
        self._nodes = nodes
        self._joint_factor = JointFactor(factors=factors, vars=nodes)
        self._logjf = LogJFWithGrad(self._joint_factor, self._dim)
        self._dftrace = None
        self._trace = None

    def sample(self, draws:int = 500, tune:int = 500, chains:int = 4, cores:int = 4, discard_tuned_samples:bool = True,
               x_lim = None, y_lim = None, **kwargs):

        if x_lim is None or y_lim is None:
            with pm.Model() as pm_model:
                pm.DensityDist('joint_factor', logp=self._logjf, shape=(self._dim,))
                trace = pm.sample(draws=draws, tune=tune, chains=chains, cores=cores, discard_tuned_samples=discard_tuned_samples,**kwargs)
                self._dftrace = pm.trace_to_dataframe(trace)
                self._trace = az.from_pymc3(trace)
        else:
            RVs = []
            with pm.Model() as pm_model:
                cur_idx = 0
                for var in self._nodes:
                    # we assume var is with dim 2 or 3
                    RVs.append(pm.Uniform(f'x{cur_idx}', lower=x_lim[0], upper=x_lim[1]))
                    cur_idx += 1
                    RVs.append(pm.Uniform(f'x{cur_idx}', lower=y_lim[0], upper=y_lim[1]))
                    cur_idx += 1
                    if var.dim == 3 and var.circular_dim_list[2]:
                        RVs.append(pm.Uniform(f'x{cur_idx}', lower=-np.pi, upper=np.pi))
                        cur_idx += 1
                X = tt.as_tensor_variable(RVs)
                pm.DensityDist('joint_factor', lambda v: self._logjf(v), observed={'v': X}, shape=(self._dim,))
                trace = pm.sample(draws=draws, tune=tune, chains=chains, cores=cores, discard_tuned_samples=discard_tuned_samples,**kwargs)
                self._dftrace = pm.trace_to_dataframe(trace)
                self._trace = az.from_pymc3(trace)
        return self._dftrace.values


def nuts_run_batch(draws, case_dir, data_file, data_format,
                  incremental_step=1, nuts_config = None, prior_cov_scale=0.1, plot_args=None, trace_plot_var = None):
    data_dir = os.path.join(case_dir, data_file)
    nodes, truth, factors = graph_file_parser(data_file=data_dir, data_format=data_format, prior_cov_scale=prior_cov_scale)

    nodes_factors_by_step = group_nodes_factors_incrementally(
        nodes=nodes, factors=factors, incremental_step=incremental_step)

    run_count = 1
    while os.path.exists(f"{case_dir}/nuts{run_count}"):
        run_count += 1
    os.mkdir(f"{case_dir}/nuts{run_count}")
    run_dir = f"{case_dir}/nuts{run_count}"
    print("create run dir: "+run_dir)

    num_batches = len(nodes_factors_by_step)
    observed_nodes = []
    observed_factors = []
    step_timer = []
    step_list = []

    mixture_factor2weights = {}

    for i in range(num_batches):
        step_nodes, step_factors = nodes_factors_by_step[i]
        observed_nodes += step_nodes
        observed_factors += step_factors
        for factor in step_factors:
            if isinstance(factor, BinaryFactorMixture):
                mixture_factor2weights[factor] = []
        solver = GlobalMCMCSampler(nodes=observed_nodes, factors=observed_factors)
        step_list.append(i)
        step_file_prefix = f"{run_dir}/step{i}"
        start = time.time()
        if nuts_config is not None:
            sample_arr = solver.sample(draws=draws, **nuts_config)
        else:
            sample_arr = solver.sample(draws=draws)
        end = time.time()
        cur_sample = {}
        cur_dim = 0
        for var in observed_nodes:
            cur_sample[var] = sample_arr[:,cur_dim:cur_dim + var.dim]
            cur_dim += var.dim

        step_timer.append(end - start)
        print(f"step {i}/{num_batches} time: {step_timer[-1]} sec, "
              f"total time: {sum(step_timer)}")

        file = open(f"{step_file_prefix}_ordering", "w+")
        file.write(" ".join([var.name for var in observed_nodes]))
        file.close()

        X = np.hstack([cur_sample[var] for var in observed_nodes])
        np.savetxt(fname=step_file_prefix+'.sample', X=X)

        plot_2d_samples(samples_mapping=cur_sample,
                     truth={variable: pose for variable, pose in
                            truth.items() if variable in observed_nodes},
                     truth_factors={factor for factor in observed_factors if
                                    set(factor.vars).issubset(observed_nodes)},
                     file_name=f"{step_file_prefix}.png", title=f'Step {i}',
                     **plot_args)

        file = open(f"{run_dir}/step_timing", "w+")
        file.write(" ".join(str(t) for t in step_timer))
        file.close()
        file = open(f"{run_dir}/step_list", "w+")
        file.write(" ".join(str(s) for s in step_list))
        file.close()

        tmp_fig, tmp_ax = plt.subplots()
        tmp_ax.plot(step_list, step_timer, 'go--')
        tmp_ax.set_ylabel("Time (sec)")
        tmp_fig.savefig(f"{run_dir}/step_timing.png", bbox_inches="tight")

        az.to_netcdf(solver._trace, filename=run_dir + f'/step{i}_trace.netcdf')
        if trace_plot_var is not None:
            # plt.figure(figsize=(4, 1))
            fig, axs = plt.subplots(1, 2, figsize=(8, 2), squeeze=False)
            plt.subplots_adjust(wspace=.3)
            az.plot_trace(solver._trace, var_names=trace_plot_var['var_names'],divergences=None,coords=trace_plot_var['coords'], axes=axs)
            axs[0,0].set_title('')
            axs[0,1].set_title('')
            f_size = 12
            label = trace_plot_var['label']
            axs[0,0].set_xlabel(f'{label}', fontsize=f_size)
            axs[0,1].set_xlabel('Samples per chain', fontsize=f_size)
            axs[0,0].set_ylabel('KDE', fontsize=f_size)
            axs[0,1].set_ylabel(label, fontsize=f_size)
            plt.rcParams.update({'font.size': f_size-2})
            plt.savefig(run_dir + f'/step{i}_trace.png', dpi=300, bbox_inches="tight")

        if mixture_factor2weights:
            # write updated hypothesis weights
            hypo_file = open(run_dir + f'/step{i}.hypoweights', 'w+')
            plt.figure()
            for factor, weights in mixture_factor2weights.items():
                hypo_weights = factor.posterior_weights(cur_sample)
                line = ' '.join([var.name for var in factor.vars]) + ' : ' + ','.join(
                    [str(w) for w in hypo_weights])
                hypo_file.writelines(line+'\n')
                weights.append(hypo_weights)
                for i_w in range(len(hypo_weights)):
                    plt.plot(np.arange(i+1-len(weights), i+1), np.array(weights)[:, i_w],'-o',
                             label=f"H{i_w}at{factor.observer_var.name}" if not isinstance(factor, KWayFactor) else
                             f"{factor.observer_var.name} to {factor.observed_vars[i_w].name}")
            hypo_file.close()
            plt.legend()
            plt.xlabel('Step')
            plt.ylabel('Hypothesis weights')
            plt.savefig(run_dir + f'/step{i}_hypoweights.png', dpi=300)