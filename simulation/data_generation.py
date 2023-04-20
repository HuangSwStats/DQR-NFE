import igraph as ig
import cvxpy as cp
import numpy as np
import pandas as pd
import scipy
from scipy.stats import norm
from scipy import stats
import matplotlib.pyplot as plt
import random
import warnings

warnings.filterwarnings("ignore")



# Network data generation
def Graph_generate(N):
    np.random.seed(1)
    g = ig.Graph.SBM(n=N, pref_matrix=[[.2, .02, .02], [.02, .2, .02], [.02, .02, .2]],
                     block_sizes=[N // 3, N // 3, N // 3])
    g.layout_grid()
    ig.plot(g, "tmp.png", bbox=(1200, 1200), opacity=0.9, margin=(10, 10, 10, 10))

    Adj_trix = g.get_adjacency()
    Adj_arr = pd.DataFrame(Adj_trix)

    D = np.diag(Adj_arr.apply(lambda x: x.sum(), axis=1))
    L = D - Adj_arr.values

    n = N // 2
    L_split = MatrixSplit(L)

    # training sample adjacency matrix
    Ls = np.zeros(shape=(n, n))
    Ls_list = []
    for m in L_split:
        L_m = matrix_tri(m)[0]
        Ls_list.append(L_m)
    Ls[0:n // 3, 0:n // 3] = Ls_list[0]
    Ls[n // 3:2 * n // 3, n // 3:2 * n // 3] = Ls_list[1]
    Ls[2 * n // 3:n, 2 * n // 3:n] = Ls_list[2]
    Ls[0:n // 3, n // 3:2 * n // 3], Ls[n // 3:2 * n // 3, 0:n // 3] = Ls_list[3], Ls_list[3].T
    Ls[n // 3:2 * n // 3, 2 * n // 3:n], Ls[2 * n // 3:n, n // 3:2 * n // 3] = Ls_list[4], Ls_list[4].T
    Ls[0:n // 3, 2 * n // 3:n], Ls[2 * n // 3:n, 0:n // 3] = Ls_list[5], Ls_list[5].T

    # prediction sample adjacency matrix
    Lp = np.zeros(shape=(n, n))
    Lp_list = []
    for m in L_split:
        L_m = matrix_tri(m)[1]
        Lp_list.append(L_m)
    # print(Lp_list)
    Lp[0:n // 3, 0:n // 3] = Lp_list[0]
    Lp[n // 3:2 * n // 3, n // 3:2 * n // 3] = Lp_list[1]
    Lp[2 * n // 3:n, 2 * n // 3:n] = Ls_list[2]
    Lp[0:n // 3, n // 3:2 * n // 3], Lp[n // 3:2 * n // 3, 0:n // 3] = Lp_list[3], Lp_list[3].T
    Lp[n // 3:2 * n // 3, 2 * n // 3:n], Lp[2 * n // 3:n, n // 3:2 * n // 3] = Lp_list[4], Lp_list[4].T
    Lp[0:n // 3, 2 * n // 3:n], Lp[2 * n // 3:n, 0:n // 3] = Lp_list[5], Lp_list[5].T

    # Off-diagonal matrix
    Lt = np.zeros(shape=(n, n))
    Lt_list = []
    for m in L_split:
        L_m = matrix_tri(m)[2]
        Lt_list.append(L_m)
    # print(Lt_list)
    Lt[0:n // 3, 0:n // 3] = Lt_list[0]
    Lt[n // 3:2 * n // 3, n // 3:2 * n // 3] = Lt_list[1]
    Lt[2 * n // 3:n, 2 * n // 3:n] = Lt_list[2]
    Lt[0:n // 3, n // 3:2 * n // 3], Lt[n // 3:2 * n // 3, 0:n // 3] = Lt_list[3], Lt_list[3].T
    Lt[n // 3:2 * n // 3, 2 * n // 3:n], Lt[2 * n // 3:n, n // 3:2 * n // 3] = Lt_list[4], Lt_list[4].T
    Lt[0:n // 3, 2 * n // 3:n], Lt[2 * n // 3:n, 0:n // 3] = Lt_list[5], Lt_list[5].T

    return Ls, Lp + 0.01 * np.eye(L.shape[0] // 2), Lt


def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)


def MatrixSplit(L):
    c = len(L)
    return [L[0:c // 3, 0:c // 3], L[c // 3:2 * c // 3, c // 3:2 * c // 3], L[2 * c // 3:c, 2 * c // 3:c], \
            L[0:c // 3, c // 3:2 * c // 3], L[c // 3:2 * c // 3, 2 * c // 3:c], L[0:c // 3, 2 * c // 3:c]]


def matrix_tri(M):
    c = len(M)
    return [M[0:c // 2, 0:c // 2], M[c // 2:c, c // 2:c], M[0:c // 2, c // 2:c]]


# DGP process
def DGP(Time, N, ta, seed_num, sigam, distribution_type):
    """
    :param Time: Generation time length of data
    :param N: number of sample
    :param ta: quantile level
    :return: panel data
    """
    np.random.seed(seed_num)
    eps_list = np.random.randn(N)
    epss_matrix = np.zeros(N * Time).reshape(N, Time)
    eps_matrix = np.zeros(N * Time).reshape(N, Time)
    alpha = np.repeat(np.array([-1, 1, 2]), N // 3) + np.random.randn(N) * np.sqrt(sigam)
    global Talpha
    Talpha = np.concatenate([alpha[N//6:N//3], alpha[N//2:2*N//3], alpha[5*N//6:]], axis=0)

    if distribution_type == 'Normal':
        for t in range(0, Time):
            epss = np.random.randn(N)
            eps_list = 0.7 * eps_list + 0.2 * epss
            epss_matrix[:, t] = epss
            eps_matrix[:, t] = eps_list
        miu = np.random.randn(N) + np.mean(epss_matrix, axis=1)
        X = np.repeat(miu, Time).reshape(N, Time) + eps_matrix
        Y = np.zeros(N)
        Y_matrix = np.zeros(N * Time).reshape(N, Time)
        Y_matrix[:, 0] = Y
        for t in range(1, Time):
            Y = alpha + 0.5 * Y + 0.3 * X[:, t] + np.repeat(stats.norm.ppf(ta), N) + np.random.randn(N)
            Y_matrix[:, t] = Y
        return Y_matrix, X

    if distribution_type == 'T':
        for t in range(0, Time):
            epss = np.random.randn(N)
            eps_list = 0.7 * eps_list + 0.2 * epss
            epss_matrix[:, t] = epss
            eps_matrix[:, t] = eps_list
        miu = np.random.randn(N) + np.mean(epss_matrix, axis=1)
        X = np.repeat(miu, Time).reshape(N, Time) + eps_matrix
        Y = np.zeros(N)
        Y_matrix = np.zeros(N * Time).reshape(N, Time)
        Y_matrix[:, 0] = Y
        for t in range(1, Time):
            Y = alpha + 0.5 * Y + 0.3 * X[:, t] + np.repeat(stats.t(df=3).ppf(ta), N) + np.random.standard_t(df=3, size=N)
            Y_matrix[:, t] = Y
        return Y_matrix, X

    if distribution_type == 'Normal_h':
        for t in range(0, Time):
            epss = np.random.randn(N)
            eps_list = 0.7 * eps_list + 0.2 * epss
            epss_matrix[:, t] = epss
            eps_matrix[:, t] = eps_list
        miu = np.random.randn(N) + np.mean(epss_matrix, axis=1)
        X = np.repeat(miu, Time).reshape(N, Time) + eps_matrix
        Y = np.zeros(N)
        Y_matrix = np.zeros(N * Time).reshape(N, Time)
        Y_matrix[:, 0] = Y
        for t in range(1, Time):
            Y = alpha + 0.5 * Y + 0.3 * X[:, t] + np.repeat(stats.norm.ppf(ta), N) + (0.5*X[:, t])*np.random.randn(N)
            Y_matrix[:, t] = Y
        return Y_matrix, X

    if distribution_type == 'T_h':
        for t in range(0, Time):
            epss = np.random.randn(N)
            eps_list = 0.7 * eps_list + 0.2 * epss
            epss_matrix[:, t] = epss
            eps_matrix[:, t] = eps_list
        miu = np.random.randn(N) + np.mean(epss_matrix, axis=1)
        X = np.repeat(miu, Time).reshape(N, Time) + eps_matrix
        Y = np.zeros(N)
        Y_matrix = np.zeros(N * Time).reshape(N, Time)
        Y_matrix[:, 0] = Y
        for t in range(1, Time):
            Y = alpha + 0.5 * Y + 0.3 * X[:, t] + np.repeat(stats.t(df=3).ppf(ta), N) + (0.5*X[:, t])*np.random.standard_t(df=3, size=N)
            Y_matrix[:, t] = Y
        return Y_matrix, X