import json
import multiprocessing as mp
import os
import random
import time
from typing import List

import numpy as np
from dynesty import utils as dyfunc
from matplotlib import pyplot as plt

from sampler.sampler_utils import JointLikelihoodForNestedSampler, JointFactorForNestedSampler, \
    JointFactorForParticleFilter
from factors.Factors import Factor, BinaryFactorMixture, KWayFactor, OdomFactor, PriorFactor, BinaryFactor
from slam.RunBatch import graph_file_parser, group_nodes_factors_incrementally
from slam.Variables import Variable
from utils.Functions import NumpyEncoder
from utils.Visualization import plot_2d_samples


class GlobalParticleFilter(object):
    def __init__(self, nodes: List[Variable], factors: List[Factor], xlim:list=None,ylim:list=None, *args, **kwargs):
        self._dim = sum([var.dim for var in nodes])
        if xlim is not None and ylim is not None:
        # very naive particle filter proposes prior from uniform distribution
            raise NotImplementedError
        else:
            self._joint_factor = JointFactorForParticleFilter(factors=factors, variable_pattern=nodes, *args, **kwargs)

    def sample(self, num_sample: int) -> np.ndarray:
        """
        particle filtering with naive proposals
        """
        print(f" Dim of problem: {self._dim}")
        ns_start = time.time()

        proposal = self._joint_factor.naive_proposal(num_sample)  # samples
        weights = np.exp(self._joint_factor.loglike(proposal))  # unnormalized weights
        weights = weights / sum(weights) # unnormalized weights
        samples = dyfunc.resample_equal(proposal, weights)

        # print("re-sample cov: ", cov)
        ns_end = time.time()
        print("Sampling time: " + str(ns_end - ns_start) + " sec")
        return samples

class SIRPF(object):
    """
    Sampling importance resampling
    See Algorithm 4: SIR particle filter in
    Arulampalam, M. Sanjeev, et al.
    "A tutorial on particle filters for online nonlinear/non-Gaussian Bayesian tracking."
    IEEE Transactions on signal processing 50.2 (2002): 174-188.
    """
    def __init__(self, num_sample: int, dim_cap: int = 200):
        self.n = num_sample
        self.vars = []
        self.var2slice = {}
        self.samples = np.zeros((self.n, dim_cap))
        self.weights = np.ones(self.n) / self.n

    @property
    def dim(self):
        return sum([v.dim for v in self.vars])

    def update_var_sample(self, s: np.ndarray, var: Variable):
        cur_dim = self.dim
        new_slice = cur_dim + np.arange(var.dim)
        self.var2slice[var] = new_slice
        self.vars.append(var)
        self.samples[:, new_slice] = s

    def one_step_update(self, factors: List[Factor], nodes: List[Variable], resample = True):
        # there must be at most one new robot pose and odometry factor
        assert len(factors) > 0 and len(nodes) > 0
        priors = []
        odom = []
        bllh = [] # binary likelihood factors that are not odometry
        multi = [] # multi-modal factors
        for f in factors:
            if isinstance(f, OdomFactor):
                odom.append(f)
            elif isinstance(f, PriorFactor):
                priors.append(f)
            elif isinstance(f, BinaryFactor):
                bllh.append(f)
        assert len(odom) < 2

        # init temp weight list
        log_weights = np.zeros(self.n)

        # check if new dims exceed the current sample array
        new_vars = set(nodes).intersection(self.vars)
        new_dim = 0
        for v in new_vars:
            new_dim += v.dim
        new_dim += self.dim
        if new_dim >= self.samples.shape[1]:
            # expand the sample array to two-fold columns
            self.samples = np.hstack((self.samples, np.zeros((self.n, self.samples.shape[1]))))

        for f in priors:
            assert len(f.vars) == 1
            var = f.vars[0]
            if var in self.vars:
                # this prior will be treated as a weight function
                log_weights += f.log_pdf(self.samples[:, self.var2slice[var]])
            else:
                # new samples from prior factor
                s = f.sample(self.n)
                self.update_var_sample(s, var)

        for f in odom:
            var1, var2 = f.vars
            if var1 in self.vars and var2 not in self.vars:
                # prediction
                var = var2
                var1sample = self.samples[:, self.var2slice[var1]]
                s = f.sample(var1=var1sample)
                self.update_var_sample(s, var)
            elif var1 not in self.vars and var2 in self.vars:
                # prediction
                var = var1
                var2sample = self.samples[:, self.var2slice[var2]]
                s = f.sample(var2=var2sample)
                self.update_var_sample(s, var)
            elif var1 in self.vars:
                raise ValueError("Both ends of the odom factor have been sampled.")
            else:
                raise ValueError("Both ends of the odom factor have not been sampled.")

        for f in bllh:
            var1, var2 = f.vars
            if var1 in self.vars and var2 not in self.vars:
                # prediction
                var = var2
                var1sample = self.samples[:, self.var2slice[var1]]
                s = f.sample(var1=var1sample)
                self.update_var_sample(s, var)
            elif var1 not in self.vars and var2 in self.vars:
                # prediction
                var = var1
                var2sample = self.samples[:, self.var2slice[var2]]
                s = f.sample(var2=var2sample)
                self.update_var_sample(s, var)
            elif var1 in self.vars:
                # Both ends of the odom factor have been sampled.
                slices = np.concatenate([self.var2slice[var1], self.var2slice[var2]])
                log_weights += f.log_pdf(self.samples[:, slices])
            else:
                raise ValueError("Both ends of the odom factor have not been sampled.")

        for f in multi:
            assert set(f.vars).issubset(self.vars)
            slices = np.concatenate([self.var2slice[v] for v in f.vars])
            log_weights += f.log_pdf(self.samples[:, slices])

        weights = self.weights * np.exp(log_weights)
        weights = weights / sum(weights)
        self.weights = weights
        if resample:
            # resampling
            self.samples = dyfunc.resample_equal(self.samples, weights)
            self.weights = np.ones(self.n) / self.n

def pf_run_batch(num_sample, case_dir, data_file, data_format,
                 incremental_step=1, selected_steps = None, prior_cov_scale=0.1, plot_args=None,
                 xlim = None,
                 ylim = None,
                 **kwargs):
    data_dir = os.path.join(case_dir, data_file)
    nodes, truth, factors = graph_file_parser(data_file=data_dir, data_format=data_format, prior_cov_scale=prior_cov_scale)

    nodes_factors_by_step = group_nodes_factors_incrementally(
        nodes=nodes, factors=factors, incremental_step=incremental_step)

    run_count = 1
    while os.path.exists(f"{case_dir}/pf{run_count}"):
        run_count += 1
    os.mkdir(f"{case_dir}/pf{run_count}")
    run_dir = f"{case_dir}/pf{run_count}"
    print("create run dir: "+run_dir)
    print("saving config of sampling")
    with open(run_dir+'/config.json', 'w') as fp:
        json.dump(kwargs, fp)

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
        if selected_steps is None or i in selected_steps:
            solver = GlobalParticleFilter(nodes=observed_nodes, factors=observed_factors, xlim=xlim, ylim=ylim)
            step_list.append(i)
            step_file_prefix = f"{run_dir}/step{i}"
            start = time.time()
            sample_arr = solver.sample(num_sample=num_sample)
            end = time.time()

            cur_sample = {}
            cur_dim = 0
            for var in observed_nodes:
                cur_sample[var] = sample_arr[:, cur_dim:cur_dim + var.dim]
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

def SIRPF_run_batch(num_sample, case_dir, data_file, data_format,
              incremental_step=1, selected_steps = None, prior_cov_scale=0.1, plot_args=None, resample_rate = 1,
              xlim = None,
              ylim = None,
              **kwargs):
    data_dir = os.path.join(case_dir, data_file)
    nodes, truth, factors = graph_file_parser(data_file=data_dir, data_format=data_format, prior_cov_scale=prior_cov_scale)

    dim_cap = sum([v.dim for v in nodes])

    nodes_factors_by_step = group_nodes_factors_incrementally(
        nodes=nodes, factors=factors, incremental_step=incremental_step)

    run_count = 1
    while os.path.exists(f"{case_dir}/sir{run_count}"):
        run_count += 1
    os.mkdir(f"{case_dir}/sir{run_count}")
    run_dir = f"{case_dir}/sir{run_count}"
    print("create run dir: "+run_dir)
    print("saving config of sampling")
    with open(run_dir+'/config.json', 'w') as fp:
        json.dump(kwargs, fp)

    num_batches = len(nodes_factors_by_step)
    observed_nodes = []
    observed_factors = []
    step_timer = []
    step_list = []

    mixture_factor2weights = {}
    solver = SIRPF(num_sample=num_sample, dim_cap=dim_cap)

    for i in range(num_batches):
        step_nodes, step_factors = nodes_factors_by_step[i]
        observed_nodes += step_nodes
        observed_factors += step_factors
        resample_state = False
        if selected_steps is None or i in selected_steps:
            if resample_rate > 0:
                if i % resample_rate == 0:
                    resample_state = True
            start = time.time()
            solver.one_step_update(nodes=step_nodes, factors=step_factors, resample=resample_state)
            step_list.append(i)
            step_file_prefix = f"{run_dir}/step{i}"
            if resample_state:
                sample_arr = solver.samples
            else:
                sample_arr = dyfunc.resample_equal(solver.samples, solver.weights)
            end = time.time()

            cur_sample = {}
            cur_dim = 0
            for var in observed_nodes:
                cur_sample[var] = sample_arr[:, cur_dim:cur_dim + var.dim]
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