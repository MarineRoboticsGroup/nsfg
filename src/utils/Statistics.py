import numpy as np
import TransportMaps.Distributions as dist
from typing import Tuple, List, Dict

from slam.Variables import Variable
from scipy.stats import circmean
from geometry.TwoDimension import Rot2, Point2, SE2Pose
from slam.Variables import SE2Variable, R2Variable
from scipy.spatial.distance import pdist, squareform

def mmd(samples1: np.ndarray, samples2: np.ndarray, k_sigma2: float = 1.0
        ) -> np.ndarray:
    """
    Compute maximum mean discrepancy between two sets of samples
    """
    m = samples1.shape[0]
    n = samples2.shape[0]
    dim = samples1.shape[1]
    gaussian = dist.GaussianDistribution(mu=np.zeros(dim),
                                         sigma=np.identity(dim) * k_sigma2)
    E1 = 0.0
    for i in range(m):
        for j in range(m):
            if j != i:
                E1 += gaussian.pdf(samples1[[i], :] - samples1[[j], :])
    E1 /= m * (m - 1)

    E2 = 0.0
    for i in range(n):
        for j in range(n):
            if j != i:
                E2 += gaussian.pdf(samples2[[i], :] - samples2[[j], :])
    E2 /= n * (n - 1)

    E3 = 0.0
    for i in range(m):
        for j in range(n):
            E3 += gaussian.pdf(samples1[[i], :] - samples2[[j], :])
    E3 /= m * n

    res = np.sqrt((E1 + E2 - 2.0 * E3) / gaussian.pdf(np.zeros(dim)))
    return res

def gaussian_displacement_factor_graph_with_equal_dim(
        variables: List[Variable],
        displacements: Dict[Tuple[Variable, Variable],
                            Tuple[np.ndarray, np.ndarray]],
        priors: Dict[Variable, Tuple[np.ndarray, np.ndarray]]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Obtain the standard representation of a Gaussian factor graph, where
        all variables have the same dimensionality and all likelihood factors
        are displacements
    :param variables: variables in the distribution
    :param displacements: the displacements from variable to variable
        (variable_from, variable_to): (mean, sigma)
    :param priors: the priors of variables
        variable: (mean, sigma)
    :return: the mean and sigma of the distribution
    """
    # Build index mapping
    indices = {}
    index = 0
    for var in variables:
        indices[var] = (index, index + var.dim)
        index += var.dim

    # Add all prior factors
    dim_tot = sum(var.dim for var in variables)
    Lambda = np.zeros((dim_tot, dim_tot))
    h = np.zeros(dim_tot)
    for var in priors.keys():
        i_start, i_end = indices[var]
        mean_loc, cov_loc = priors[var]
        Lambda_loc = np.linalg.inv(cov_loc)
        h_loc = Lambda_loc @ mean_loc
        h[i_start:i_end] += h_loc
        Lambda[i_start:i_end, i_start:i_end] += Lambda_loc

    # Add all displacement factors
    for var_from, var_to in displacements.keys():
        i_start, i_end = indices[var_from]
        j_start, j_end = indices[var_to]
        mean_loc, cov_loc = displacements[(var_from, var_to)]
        Lambda_loc = np.linalg.inv(cov_loc)
        h_loc = Lambda_loc @ mean_loc
        Lambda[i_start:i_end, i_start:i_end] += Lambda_loc
        Lambda[j_start:j_end, j_start:j_end] += Lambda_loc
        Lambda[i_start:i_end, j_start:j_end] -= Lambda_loc
        Lambda[j_start:j_end, i_start:i_end] -= Lambda_loc
        h[i_start:i_end] -= h_loc
        h[j_start:j_end] += h_loc

    # Conversion to the standard form
    Sigma = np.linalg.inv(Lambda)
    mu = Sigma @ h
    return mu, Sigma


def rmse(samples1: np.ndarray, samples2: np.ndarray) -> float:
    """
    Compute RMSE between two sets of samples
    """
    if samples1.shape != samples2.shape:
        raise ValueError("Dimensionality of sets of samples do not match")
    return np.sqrt(np.sum((samples1 - samples2) ** 2) / samples1.size)


def sample_mean(samples: np.ndarray, var_ordering: List[Variable]) -> (np.ndarray, Dict):
    """
    Compute mean of samples
    """
    circular_dim_list = []
    for var in var_ordering:
        circular_dim_list += var.circular_dim_list
    aug_clique_dim = samples.shape[-1]
    means = np.zeros(aug_clique_dim)
    circular_indices = np.where(circular_dim_list)[0]
    euclidean_indices = np.setdiff1d(np.arange(aug_clique_dim), circular_indices)
    means[circular_indices] = circmean(samples[:, circular_indices], high=np.pi, low=-np.pi, axis=0)
    means[euclidean_indices] = np.mean(samples[:, euclidean_indices], axis=0)

    var2mean = {}
    cur_dim = 0
    for var in var_ordering:
        var2mean[var] = means[cur_dim:cur_dim + var.dim]
        cur_dim += var.dim

    return means, var2mean


def sample_frechet_mean(samples: np.ndarray, var_ordering: List[Variable]) -> (np.ndarray, Dict):
    # TODO: implement after adding manifolds to variables
    raise NotImplementedError


def geodesic_distance(var2point1: Dict[Variable, np.ndarray],
                      var2point2: Dict[Variable, np.ndarray]):
    err = 0
    for var in var2point1:
        pt1 = var2point1[var]
        pt2 = var2point2[var]
        if isinstance(var, SE2Variable):
            err += sum((SE2Pose(*pt1) / SE2Pose(*pt2)).log_map() ** 2)
        elif isinstance(var, R2Variable):
            err += sum((pt1 - pt2) ** 2)
        else:
            raise ValueError('Unknown variable type.')
    return np.sqrt(err)

def compute_score_ksd(X, joint_factor):
    score = joint_factor.grad_x_log_pdf(X)
    return score
    c_dims = []
    for var in joint_factor.vars:
        c_dims += var.circular_dim_list
    masks = np.invert(c_dims)
    masks = np.array(masks, dtype=float)
    return masks * score


def translation_distance(var2point1: Dict[Variable, np.ndarray],
                      var2point2: Dict[Variable, np.ndarray]):
    err = 0
    for var in var2point1:
        pt1 = var2point1[var]
        pt2 = var2point2[var]
        if isinstance(var, (SE2Variable, R2Variable)):
            err += sum((pt1[:2] - pt2[:2]) ** 2)
        else:
            raise ValueError('Unknown variable type.')
    return np.sqrt(err/len(var2point1))

def Gaussian_kernel_stein_discrepancy(joint_factor, kernel_precision, samples, nboot=10):
    X = samples
    P = kernel_precision
    n = X.shape[0]
    score = compute_score_ksd(X, joint_factor)
    XX = squareform(pdist(X, 'mahalanobis', VI=P))
    KXX = np.exp(-XX**2/2)
    raw_ksd = np.zeros((n,n))
    off_ksd = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            grad_SE_dxi = -P @ (X[i:i+1].T-X[j:j+1].T)
            # dk_dxi = KXX[i,j] * grad_SE
            grad_SE_dxj = -grad_SE_dxi
            p1 = np.dot(score[i], score[j])
            p2 = np.dot(score[i], grad_SE_dxj.flatten())
            p3 = np.dot(score[j], grad_SE_dxi.flatten())
            p4 = np.trace(grad_SE_dxi @ grad_SE_dxj.T+P)
            raw_ksd[i, j] = (p1 + p2 + p3 + p4) * KXX[i, j]
            if i != j:
                off_ksd[i,j] = raw_ksd[i, j]
    ustats = np.sum(off_ksd)/(n*(n-1))
    vstats = np.sum(raw_ksd)/n**2
    bootstrap = np.zeros(nboot)
    for i in range(nboot):
        w = (np.random.multinomial(n, np.ones(n)/n)/n).reshape((-1,1))
        bootstrap[i] = ((w.T - 1.0/n) @ off_ksd @ (w - 1.0/n))[0][0]
    p_u = len(np.where(bootstrap>=ustats)[0]) / nboot
    print("ustats, p_u, vstats: ", " ".join([str(ustats), str(p_u), str(vstats)]))
    return ustats, p_u, off_ksd, vstats