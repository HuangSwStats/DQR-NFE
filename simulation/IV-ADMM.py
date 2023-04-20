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
from sklearn.cluster import KMeans
import statsmodels.formula.api as smf
from scipy import stats
from scipy.stats import norm

import time
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

    Lp = np.zeros(shape=(n, n))
    Lp_list = []
    for m in L_split:
        L_m = matrix_tri(m)[1]
        Lp_list.append(L_m)

    Lp[0:n // 3, 0:n // 3] = Lp_list[0]
    Lp[n // 3:2 * n // 3, n // 3:2 * n // 3] = Lp_list[1]
    Lp[2 * n // 3:n, 2 * n // 3:n] = Ls_list[2]
    Lp[0:n // 3, n // 3:2 * n // 3], Lp[n // 3:2 * n // 3, 0:n // 3] = Lp_list[3], Lp_list[3].T
    Lp[n // 3:2 * n // 3, 2 * n // 3:n], Lp[2 * n // 3:n, n // 3:2 * n // 3] = Lp_list[4], Lp_list[4].T
    Lp[0:n // 3, 2 * n // 3:n], Lp[2 * n // 3:n, 0:n // 3] = Lp_list[5], Lp_list[5].T

    Lt = np.zeros(shape=(n, n))
    Lt_list = []
    for m in L_split:
        L_m = matrix_tri(m)[2]
        Lt_list.append(L_m)

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
    Talpha = np.concatenate([alpha[N // 6:N // 3], alpha[N // 2:2 * N // 3], alpha[5 * N // 6:]], axis=0)

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
            Y = alpha + 0.5 * Y + 0.3 * X[:, t] + np.repeat(stats.t(df=3).ppf(ta), N) + np.random.standard_t(df=3,
                                                                                                             size=N)
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
            Y = alpha + 0.5 * Y + 0.3 * X[:, t] + np.repeat(stats.norm.ppf(ta), N) + (0.5 * X[:, t]) * np.random.randn(
                N)
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
            Y = alpha + 0.5 * Y + 0.3 * X[:, t] + np.repeat(stats.t(df=3).ppf(ta), N) + (
                        0.5 * X[:, t]) * np.random.standard_t(df=3, size=N)
            Y_matrix[:, t] = Y
        return Y_matrix, X


def quantile_loss(u, tau):
    return 0.5 * cp.abs(u) + (tau - 0.5) * u


def L2_penilty(alpha, L):
    return cp.quad_form(alpha, L)


def L1_penilty(alpha):
    return cp.norm1(alpha)


def Ridge_penilty(alpha):
    return cp.pnorm(alpha, p=2) ** 2


def QRL1_fn(Y, Y_1, X, X_1, moment, beta, alpha, IV_hat, ta, lambd, alp):
    return Objective_fn(Y, Y_1, X, X_1, moment, beta, alpha, IV_hat, ta) \
           + lambd * L1_penilty(alp)


def QRL2_fn(Y, Y_1, X, X_1, moment, beta, alpha, IV_hat, ta, lambd, alp):
    return Objective_fn(Y, Y_1, X, X_1, moment, beta, alpha, IV_hat, ta) \
           + lambd * Ridge_penilty(alp)


def Objective_fn(Y, Y_1, X, X_1, moment, beta, alpha, IV_hat, ta):
    return cp.sum(quantile_loss(Y - moment * Y_1 - beta * X - IV_hat * X_1 - alpha, tau=ta))


def QRNC_fn(Y, Y_1, X, X_1, moment, beta, alpha, IV_hat, ta, lambd, alp, L):
    return Objective_fn(Y, Y_1, X, X_1, moment, beta, alpha, IV_hat, ta) \
           + lambd * L2_penilty(alp, L)


def MSE(Y, Y_1, X, moment, beta, alpha, ta, N, T, insample):
    I_list = np.diag(np.repeat(1, X.shape[1]))
    if insample == False:
        X_out = X.reshape(N * (T - time_mask - 1), 1, order='F')
    else:
        X_out = X.reshape((N * time_mask), 1, order='F')
    e = np.repeat(1, X.shape[1])
    Z = np.dot(cp.kron(I_list, cp.reshape(cp.vec(alpha), (len(X), 1))), cp.reshape(e, (X.shape[1], 1)))
    u = Y - moment * Y_1 - beta * X_out - Z
    loss = cp.sum(quantile_loss(u, tau=ta)).value
    return 1 / (X.shape[0] * X.shape[1]) * loss


def Variance(Y, Y_1, X, moment, beta, alpha, N, T, insample):
    I_list = np.diag(np.repeat(1, X.shape[1]))
    if insample == False:
        X_out = X.reshape(N * (T - time_mask - 1), 1, order='F')
    else:
        X_out = X.reshape((N * time_mask), 1, order='F')
    e = np.repeat(1, X.shape[1])
    Z = np.dot(cp.kron(I_list, cp.reshape(cp.vec(alpha), (len(X), 1))), cp.reshape(e, (X.shape[1], 1)))
    u = Y - moment * Y_1 - beta * X_out - Z
    return 1 / (X.shape[0] * X.shape[1]) * np.var(u.value)


def split_sample(Y, N, insample):
    """
    Separate the in-sample and out-sample of training set for out-of-sample prediction
    : param Y: input sample matrix
    : param N: total sample range entered
    : return: returns the split sample box
    """
    N = 2 * N
    if insample == True:
        Y1, Y2, Y3 = Y[:N // 6, :], Y[N // 3:N // 2, :], Y[-N // 3:-N // 6, :]
        return np.vstack([Y1, Y2, Y3])
    if insample == False:
        Y1, Y2, Y3 = Y[N // 6:N // 3, :], Y[N // 2:2 * N // 3, :], Y[-N // 6:, :]
        return np.vstack([Y1, Y2, Y3])


# model estimation

# DQR-NFE estimation via DCP
def QRNC_model_fit(Y_DGP, X_DGP, split_ratio, N, T, ta, Ls, LT, Ltr):

    """
    : param Y: original sample matrix Y
    : param X: original sample matrix X
    : param split_ratio: the split ratio of training and testing in the time dimension
    : param Ls: network Laplacian for training sample
    ：param LT: network Laplacian for testing sample
    ：param Ltr: network Laplacian between testing sample and training sample
    : return: training sample matrix y_in, x_in;  Test sample matrix y_out, x_out
    """

    N = N // 2
    Y = Y_DGP[:, 1:]
    X = X_DGP[:, 1:]
    Y_1 = Y_DGP[:, :-1]
    X_1 = X_DGP[:, :-1]
    # training data
    global time_mask
    Y_in_all = split_sample(Y, N, True)
    X_in_all = split_sample(X, N, True)
    X_1_in_all = split_sample(X_1, N, True)
    Y_1_in_all = split_sample(Y_1, N, True)
    time_mask = int(Y_in_all.shape[1] * (1 - split_ratio))
    Y_in = Y_in_all[:, :time_mask].reshape(N * time_mask, 1, order='F')
    X_in = X_in_all[:, :time_mask].reshape(N * time_mask, 1, order='F')
    X_in_mse = X_in_all[:, :time_mask]
    X_1_in = X_1_in_all[:, :time_mask].reshape(N * time_mask, 1, order='F')
    Y_1_in = Y_1_in_all[:, :time_mask].reshape(N * time_mask, 1, order='F')
    I_list = np.diag(np.repeat(1, time_mask))
    e = np.repeat(1, time_mask)
    # validation data
    Y_out, X_out, Y_1_out = Y_in_all[:, time_mask:], X_in_all[:, time_mask:], Y_1_in_all[:, time_mask:]
    Y_out = Y_out.reshape(N * (T - time_mask - 1), 1, order='F')
    Y_1_out = Y_1_out.reshape(N * (T - time_mask - 1), 1, order='F')
    # test data
    Y_out_all, X_out_all, Y_1_out_all = split_sample(Y, N, False), split_sample(X, N, False), split_sample(Y_1, N,
                                                                                                           False)
    Y_out_all, X_out_all, Y_1_out_all = Y_out_all[:, time_mask:], X_out_all[:, time_mask:], Y_1_out_all[:, time_mask:]
    Y_test = Y_out_all.reshape(N * (T - time_mask - 1), 1, order='F')
    Y_1_test = Y_1_out_all.reshape(N * (T - time_mask - 1), 1, order='F')

    # model estimation
    # Selection of first-order lag term parameter
    moment_list = np.arange(0.1, 0.9, 0.1)
    IV_list, beta_list, betaBaseline_list = [], [], []
    Z_hat_list = np.zeros(N * len(moment_list)).reshape(N, len(moment_list))
    for i in range(0, len(moment_list)):
        mom = moment_list[i]
        beta = cp.Variable(1)
        alpha = cp.Variable(N)
        Z_hat = np.dot(cp.kron(I_list, cp.reshape(cp.vec(alpha), (N, 1))), cp.reshape(e, (time_mask, 1)))
        # Z_hat = cp.reshape(cp.vec(np.repeat(alpha, time_mask)), (time_mask, 1))
        IV_hat = cp.Variable(1)
        problem = cp.Problem(
            cp.Minimize(Objective_fn(Y_in, Y_1_in, X_in, X_1_in, mom, beta, Z_hat, IV_hat, ta)))  # 先选择滞后阶系数，再做修正
        problem.solve(solver='ECOS')
        IV_list.append(IV_hat.value[0])
        beta_list.append(beta.value[0])
        Z_hat_list[:, i] = alpha.value

    opt_mom = moment_list[IV_list.index(min(IV_list))]
    beta = cp.Variable(1)
    alpha = cp.Variable(N)
    I_list = np.diag(np.repeat(1, time_mask))
    e = np.repeat(1, time_mask)
    Z_hat = np.dot(cp.kron(I_list, cp.reshape(cp.vec(alpha), (N, 1))), cp.reshape(e, (time_mask, 1)))  #
    IV_hat = 0
    lambd = cp.Parameter(nonneg=True)
    problem = cp.Problem(cp.Minimize(QRNC_fn(Y_in, Y_1_in, X_in, X_1_in, opt_mom,
                                             beta, Z_hat, IV_hat, ta, lambd, alpha, Ls)))
    global lambd_values
    lambd_values = np.arange(0.01, 1, 0.1)
    train_errors, train_vars, test_errors, test_vars = [], [], [], []
    beta_list, Z_matrix = [], np.zeros(N * len(lambd_values)).reshape(N, len(lambd_values))
    start = time.time()
    for v in range(0, len(lambd_values)):
        lambd.value = lambd_values[v]
        problem.solve(solver='ECOS')
        opt_beta = beta.value[0]
        opt_Z = alpha.value
        beta_list.append(opt_beta)
        Z_matrix[:, v] = opt_Z

        # test model
        mse = MSE(Y_out, Y_1_out, X_out, opt_mom, opt_beta, opt_Z, ta, N, T, insample=False)
        var = Variance(Y_out, Y_1_out, X_out, opt_mom, opt_beta, opt_Z, N, T, insample=False)
        test_errors.append(mse)
        test_vars.append(var)
        mse = MSE(Y_in, Y_1_in, X_in_mse, opt_mom, opt_beta, opt_Z, ta, N, T, insample=True)
        var = Variance(Y_in, Y_1_in, X_in_mse, opt_mom, opt_beta, opt_Z, N, T, insample=True)
        train_errors.append(mse)
        train_vars.append(var)
    end = time.time()
    diff = end - start

    # find opt_lambda
    optional_beta = beta_list[test_errors.index(min(test_errors))]
    optional_alpha = Z_matrix[:, test_errors.index(min(test_errors))]

    train_error = train_errors[test_errors.index(min(test_errors))]
    test_error = test_errors[test_errors.index(min(test_errors))]
    train_var = train_vars[test_errors.index(min(test_errors))]
    test_var = test_vars[test_errors.index(min(test_errors))]

    # out-of-sample prediction
    optional_alphaPre = -np.dot(np.dot(np.linalg.inv(LT), Ltr), np.array(optional_alpha).reshape(N, 1))
    optional_alphaPre = optional_alphaPre[:, 0]  # 解决维数不为（N,）的赋值问题
    mse = MSE(Y_test, Y_1_test, X_out_all, opt_mom, optional_beta, optional_alphaPre, ta, N, T, insample=False)
    varOut = Variance(Y_test, Y_1_test, X_out_all, opt_mom, optional_beta, optional_alphaPre, N, T, insample=False)

    # return opt_mom, opt_beta, opt_Z, opt_betaBaseline, mse
    return train_error, test_error, train_var, test_var, opt_mom, optional_beta, optional_alpha, mse, varOut, diff



# DQR-L1 model estimation
def QR_L1_model_fit(Y_DGP, X_DGP, split_ratio, N, T, ta):

    N = N // 2
    Y = Y_DGP[:, 1:]
    X = X_DGP[:, 1:]
    Y_1 = Y_DGP[:, :-1]
    X_1 = X_DGP[:, :-1]
    # training data
    global time_mask
    Y_in_all = split_sample(Y, N, True)
    X_in_all = split_sample(X, N, True)
    X_1_in_all = split_sample(X_1, N, True)
    Y_1_in_all = split_sample(Y_1, N, True)
    time_mask = int(Y_in_all.shape[1] * (1 - split_ratio))
    Y_in = Y_in_all[:, :time_mask].reshape(N * time_mask, 1, order='F')
    X_in = X_in_all[:, :time_mask].reshape(N * time_mask, 1, order='F')
    X_in_mse = X_in_all[:, :time_mask]
    X_1_in = X_1_in_all[:, :time_mask].reshape(N * time_mask, 1, order='F')
    Y_1_in = Y_1_in_all[:, :time_mask].reshape(N * time_mask, 1, order='F')
    I_list = np.diag(np.repeat(1, time_mask))
    e = np.repeat(1, time_mask)
    # validation data
    Y_out, X_out, Y_1_out = Y_in_all[:, time_mask:], X_in_all[:, time_mask:], Y_1_in_all[:, time_mask:]
    Y_out = Y_out.reshape(N * (T - time_mask - 1), 1, order='F')
    Y_1_out = Y_1_out.reshape(N * (T - time_mask - 1), 1, order='F')
    # test data
    Y_out_all, X_out_all, Y_1_out_all = split_sample(Y, N, False), split_sample(X, N, False), split_sample(Y_1, N,
                                                                                                           False)
    Y_out_all, X_out_all, Y_1_out_all = Y_out_all[:, time_mask:], X_out_all[:, time_mask:], Y_1_out_all[:, time_mask:]
    Y_test = Y_out_all.reshape(N * (T - time_mask - 1), 1, order='F')
    Y_1_test = Y_1_out_all.reshape(N * (T - time_mask - 1), 1, order='F')

    # model estimation
    moment_list = np.arange(0.1, 0.9, 0.1)
    IV_list, beta_list, betaBaseline_list = [], [], []
    Z_hat_list = np.zeros(N * len(moment_list)).reshape(N, len(moment_list))
    for i in range(0, len(moment_list)):
        mom = moment_list[i]
        beta = cp.Variable(1)
        alpha = cp.Variable(N)
        Z_hat = np.dot(cp.kron(I_list, cp.reshape(cp.vec(alpha), (N, 1))), cp.reshape(e, (time_mask, 1)))
        # Z_hat = cp.reshape(cp.vec(np.repeat(alpha, time_mask)), (time_mask, 1))
        IV_hat = cp.Variable(1)
        problem = cp.Problem(
            cp.Minimize(Objective_fn(Y_in, Y_1_in, X_in, X_1_in, mom, beta, Z_hat, IV_hat, ta)))  # 先选择滞后阶系数，再做修正
        problem.solve(solver='ECOS')
        IV_list.append(IV_hat.value[0])
        beta_list.append(beta.value[0])
        Z_hat_list[:, i] = alpha.value

    # find opt mom
    opt_mom = moment_list[IV_list.index(min(IV_list))]
    beta = cp.Variable(1)
    alpha = cp.Variable(N)
    I_list = np.diag(np.repeat(1, time_mask))
    e = np.repeat(1, time_mask)
    Z_hat = np.dot(cp.kron(I_list, cp.reshape(cp.vec(alpha), (N, 1))), cp.reshape(e, (time_mask, 1)))
    IV_hat = 0
    lambd = cp.Parameter(nonneg=True)
    problem = cp.Problem(cp.Minimize(QRL1_fn(Y_in, Y_1_in, X_in, X_1_in, opt_mom,
                                             beta, Z_hat, IV_hat, ta, lambd, alpha)))
    # global lambd_values
    lambd_values = np.arange(0.01, 1, 0.1)
    train_errors, train_vars, test_errors, test_vars = [], [], [], []
    beta_list, Z_matrix = [], np.zeros(N * len(lambd_values)).reshape(N, len(lambd_values))
    for v in range(0, len(lambd_values)):
        lambd.value = lambd_values[v]
        problem.solve(solver='ECOS')
        opt_beta = beta.value[0]
        opt_Z = alpha.value
        beta_list.append(opt_beta)
        Z_matrix[:, v] = opt_Z

        # test model
        mse = MSE(Y_out, Y_1_out, X_out, opt_mom, opt_beta, opt_Z, ta, N, T, insample=False)
        var = Variance(Y_out, Y_1_out, X_out, opt_mom, opt_beta, opt_Z, N, T, insample=False)
        test_errors.append(mse)
        test_vars.append(var)
        mse = MSE(Y_in, Y_1_in, X_in_mse, opt_mom, opt_beta, opt_Z, ta, N, T, insample=True)
        var = Variance(Y_in, Y_1_in, X_in_mse, opt_mom, opt_beta, opt_Z, N, T, insample=True)
        train_errors.append(mse)
        train_vars.append(var)
    # find opt_lambda
    optional_beta = beta_list[test_errors.index(min(test_errors))]
    optional_alpha = Z_matrix[:, test_errors.index(min(test_errors))]

    train_error = train_errors[test_errors.index(min(test_errors))]
    test_error = test_errors[test_errors.index(min(test_errors))]
    train_var = train_vars[test_errors.index(min(test_errors))]
    test_var = test_vars[test_errors.index(min(test_errors))]

    # out-of-sample prediction
    optional_alphaPre = np.repeat(np.mean(optional_alpha), N)
    mse = MSE(Y_test, Y_1_test, X_out_all, opt_mom, optional_beta, optional_alphaPre, ta, N, T, insample=False)
    varOut = Variance(Y_test, Y_1_test, X_out_all, opt_mom, optional_beta, optional_alphaPre, N, T, insample=False)

    return train_error, test_error, train_var, test_var, opt_mom, optional_beta, optional_alpha, mse, varOut


def QR_L2_model_fit(Y_DGP, X_DGP, split_ratio, N, T, ta):

    N = N // 2
    Y = Y_DGP[:, 1:]
    X = X_DGP[:, 1:]
    Y_1 = Y_DGP[:, :-1]
    X_1 = X_DGP[:, :-1]
    # training data
    global time_mask
    Y_in_all = split_sample(Y, N, True)
    X_in_all = split_sample(X, N, True)
    X_1_in_all = split_sample(X_1, N, True)
    Y_1_in_all = split_sample(Y_1, N, True)
    time_mask = int(Y_in_all.shape[1] * (1 - split_ratio))
    Y_in = Y_in_all[:, :time_mask].reshape(N * time_mask, 1, order='F')
    X_in = X_in_all[:, :time_mask].reshape(N * time_mask, 1, order='F')
    X_in_mse = X_in_all[:, :time_mask]
    X_1_in = X_1_in_all[:, :time_mask].reshape(N * time_mask, 1, order='F')
    Y_1_in = Y_1_in_all[:, :time_mask].reshape(N * time_mask, 1, order='F')
    I_list = np.diag(np.repeat(1, time_mask))
    e = np.repeat(1, time_mask)
    # validation data
    Y_out, X_out, Y_1_out = Y_in_all[:, time_mask:], X_in_all[:, time_mask:], Y_1_in_all[:, time_mask:]
    Y_out = Y_out.reshape(N * (T - time_mask - 1), 1, order='F')
    Y_1_out = Y_1_out.reshape(N * (T - time_mask - 1), 1, order='F')
    # test data
    Y_out_all, X_out_all, Y_1_out_all = split_sample(Y, N, False), split_sample(X, N, False), split_sample(Y_1, N,
                                                                                                           False)
    Y_out_all, X_out_all, Y_1_out_all = Y_out_all[:, time_mask:], X_out_all[:, time_mask:], Y_1_out_all[:, time_mask:]
    Y_test = Y_out_all.reshape(N * (T - time_mask - 1), 1, order='F')
    Y_1_test = Y_1_out_all.reshape(N * (T - time_mask - 1), 1, order='F')

    # model estimation
    moment_list = np.arange(0.1, 0.9, 0.1)
    IV_list, beta_list, betaBaseline_list = [], [], []
    Z_hat_list = np.zeros(N * len(moment_list)).reshape(N, len(moment_list))
    for i in range(0, len(moment_list)):
        mom = moment_list[i]
        beta = cp.Variable(1)
        alpha = cp.Variable(N)
        Z_hat = np.dot(cp.kron(I_list, cp.reshape(cp.vec(alpha), (N, 1))), cp.reshape(e, (time_mask, 1)))
        # Z_hat = cp.reshape(cp.vec(np.repeat(alpha, time_mask)), (time_mask, 1))
        IV_hat = cp.Variable(1)
        problem = cp.Problem(
            cp.Minimize(Objective_fn(Y_in, Y_1_in, X_in, X_1_in, mom, beta, Z_hat, IV_hat, ta)))  # 先选择滞后阶系数，再做修正
        problem.solve(solver='ECOS')
        IV_list.append(IV_hat.value[0])
        beta_list.append(beta.value[0])
        Z_hat_list[:, i] = alpha.value

    # find opt mom
    opt_mom = moment_list[IV_list.index(min(IV_list))]
    beta = cp.Variable(1)
    alpha = cp.Variable(N)
    I_list = np.diag(np.repeat(1, time_mask))
    e = np.repeat(1, time_mask)
    Z_hat = np.dot(cp.kron(I_list, cp.reshape(cp.vec(alpha), (N, 1))), cp.reshape(e, (time_mask, 1)))
    IV_hat = 0
    lambd = cp.Parameter(nonneg=True)
    problem = cp.Problem(cp.Minimize(QRL2_fn(Y_in, Y_1_in, X_in, X_1_in, opt_mom,
                                             beta, Z_hat, IV_hat, ta, lambd, alpha)))
    # global lambd_values
    lambd_values = np.arange(0.01, 1, 0.1)
    train_errors, train_vars, test_errors, test_vars = [], [], [], []
    beta_list, Z_matrix = [], np.zeros(N * len(lambd_values)).reshape(N, len(lambd_values))
    for v in range(0, len(lambd_values)):
        lambd.value = lambd_values[v]
        problem.solve(solver='ECOS')
        opt_beta = beta.value[0]
        opt_Z = alpha.value
        beta_list.append(opt_beta)
        Z_matrix[:, v] = opt_Z

        # test model
        mse = MSE(Y_out, Y_1_out, X_out, opt_mom, opt_beta, opt_Z, ta, N, T, insample=False)
        var = Variance(Y_out, Y_1_out, X_out, opt_mom, opt_beta, opt_Z, N, T, insample=False)
        test_errors.append(mse)
        test_vars.append(var)
        mse = MSE(Y_in, Y_1_in, X_in_mse, opt_mom, opt_beta, opt_Z, ta, N, T, insample=True)
        var = Variance(Y_in, Y_1_in, X_in_mse, opt_mom, opt_beta, opt_Z, N, T, insample=True)
        train_errors.append(mse)
        train_vars.append(var)

    # find opt_lambda
    optional_beta = beta_list[test_errors.index(min(test_errors))]
    optional_alpha = Z_matrix[:, test_errors.index(min(test_errors))]

    train_error = train_errors[test_errors.index(min(test_errors))]
    test_error = test_errors[test_errors.index(min(test_errors))]
    train_var = train_vars[test_errors.index(min(test_errors))]
    test_var = test_vars[test_errors.index(min(test_errors))]

    # out-of-sample prediction
    optional_alphaPre = np.repeat(np.mean(optional_alpha), N)
    mse = MSE(Y_test, Y_1_test, X_out_all, opt_mom, optional_beta, optional_alphaPre, ta, N, T, insample=False)
    varOut = Variance(Y_test, Y_1_test, X_out_all, opt_mom, optional_beta, optional_alphaPre, N, T, insample=False)

    return train_error, test_error, train_var, test_var, opt_mom, optional_beta, optional_alpha, mse, varOut


def AlphaMSE(alphaMartix, alphaTrue, ta, dis_type):
    df = pd.DataFrame(alphaMartix)
    if dis_type[:-2] == 'Normal':
        alphaT = alphaTrue + np.repeat(stats.norm.ppf(ta), len(alphaTrue))
        mean_vactor = df.apply(lambda x: x.mean(), axis=1)
        var_vactor = df.apply(lambda x: x.var(), axis=1)
        bias = np.mean(np.square(mean_vactor - alphaT))
        vars = np.mean(var_vactor)
        return bias + vars

    if dis_type[:-2] == 'T':
        alphaT = alphaTrue + np.repeat(stats.t(df=3).ppf(ta), len(alphaTrue))
        mean_vactor = df.apply(lambda x: x.mean(), axis=1)
        var_vactor = df.apply(lambda x: x.var(), axis=1)
        bias = np.mean(np.square(mean_vactor - alphaT))
        vars = np.mean(var_vactor)
        return bias + vars

    if dis_type == 'Normal':
        alphaT = alphaTrue + np.repeat(stats.norm.ppf(ta), len(alphaTrue))
        mean_vactor = df.apply(lambda x: x.mean(), axis=1)
        var_vactor = df.apply(lambda x: x.var(), axis=1)
        bias = np.mean(np.square(mean_vactor - alphaT))
        vars = np.mean(var_vactor)
        return bias + vars

    if dis_type == 'T':
        alphaT = alphaTrue + np.repeat(stats.t(df=3).ppf(ta), len(alphaTrue))
        mean_vactor = df.apply(lambda x: x.mean(), axis=1)
        var_vactor = df.apply(lambda x: x.var(), axis=1)
        bias = np.mean(np.square(mean_vactor - alphaT))
        vars = np.mean(var_vactor)
        return bias + vars

def BetaMSE(betaVactor, betaTrue):
    betaTrueV = np.repeat(betaTrue, turns_num)
    bias = np.mean(np.square(betaVactor - betaTrueV))
    vars = np.var(betaVactor)
    return bias + vars


# DQR-NFE estimation via IV-ADMM
def QRNC_ADMM_model_fit(Y_DGP, X_DGP, split_ratio, N, T, ta, Ls, LT, Ltr):

    """
    : param Y: original sample matrix Y
    : param X: original sample matrix X
    : param split_ratio: the split ratio of training and testing in the time dimension
    : param Ls: network Laplacian for training sample
    ：param LT: network Laplacian for testing sample
    ：param Ltr: network Laplacian between testing sample and training sample
    : return: training sample matrix y_in, x_in;  Test sample matrix y_out, x_out
    """

    N = N // 2
    Y = Y_DGP[:, 1:]
    X = X_DGP[:, 1:]
    Y_1 = Y_DGP[:, :-1]
    X_1 = X_DGP[:, :-1]
    # training data
    global time_mask
    Y_in_all = split_sample(Y, N, True)
    X_in_all = split_sample(X, N, True)
    X_1_in_all = split_sample(X_1, N, True)
    Y_1_in_all = split_sample(Y_1, N, True)
    time_mask = int(Y_in_all.shape[1] * (1 - split_ratio))
    Y_in = Y_in_all[:, :time_mask].reshape(N * time_mask, 1, order='F')
    X_in = X_in_all[:, :time_mask].reshape(N * time_mask, 1, order='F')
    X_in_mse = X_in_all[:, :time_mask]
    X_1_in = X_1_in_all[:, :time_mask].reshape(N * time_mask, 1, order='F')
    Y_1_in = Y_1_in_all[:, :time_mask].reshape(N * time_mask, 1, order='F')
    I_list = np.diag(np.repeat(1, time_mask))
    e = np.repeat(1, time_mask)
    # validation data
    Y_out, X_out, Y_1_out = Y_in_all[:, time_mask:], X_in_all[:, time_mask:], Y_1_in_all[:, time_mask:]
    Y_out = Y_out.reshape(N * (T - time_mask - 1), 1, order='F')
    Y_1_out = Y_1_out.reshape(N * (T - time_mask - 1), 1, order='F')
    # test data
    Y_out_all, X_out_all, Y_1_out_all = split_sample(Y, N, False), split_sample(X, N, False), split_sample(Y_1, N,
                                                                                                           False)
    Y_out_all, X_out_all, Y_1_out_all = Y_out_all[:, time_mask:], X_out_all[:, time_mask:], Y_1_out_all[:, time_mask:]
    Y_test = Y_out_all.reshape(N * (T - time_mask - 1), 1, order='F')
    Y_1_test = Y_1_out_all.reshape(N * (T - time_mask - 1), 1, order='F')

    # model estimation
    # Selection of first-order lag term parameter  (IV step)
    moment_list = np.arange(0.1, 0.9, 0.1)
    IV_list, beta_list, betaBaseline_list = [], [], []
    Z_hat_list = np.zeros(N * len(moment_list)).reshape(N, len(moment_list))
    for i in range(0, len(moment_list)):
        mom = moment_list[i]
        beta = cp.Variable(1)
        alpha = cp.Variable(N)
        Z_hat = np.dot(cp.kron(I_list, cp.reshape(cp.vec(alpha), (N, 1))), cp.reshape(e, (time_mask, 1)))
        IV_hat = cp.Variable(1)
        problem = cp.Problem(
            cp.Minimize(Objective_fn(Y_in, Y_1_in, X_in, X_1_in, mom, beta, Z_hat, IV_hat, ta)))
        problem.solve(solver='ECOS')
        IV_list.append(IV_hat.value[0])
        beta_list.append(beta.value[0])
        Z_hat_list[:, i] = alpha.value
    opt_beta = beta_list[IV_list.index(min(IV_list))]
    opt_Z = Z_hat_list[:, IV_list.index(min(IV_list))]

    opt_mom = moment_list[IV_list.index(min(IV_list))]
    beta_int = opt_beta
    alpha_int = opt_Z
    I_list = np.diag(np.repeat(1, time_mask))
    e = np.repeat(1, time_mask)
    eta_int = np.dot(np.kron(I_list, np.reshape(alpha_int, (N, 1))), np.reshape(e, (time_mask, 1)))
    global lambd_values
    lambd_values = np.arange(0.01, 1, 0.1)
    train_errors, train_vars, test_errors, test_vars = [], [], [], []
    beta_list, Z_matrix = [], np.zeros(N * len(lambd_values)).reshape(N, len(lambd_values))
    Aut_int = np.repeat(0.5, N*time_mask).reshape(-1, 1)
    xi = 0.001
    max_inter = 100
    Z_int = model_error(Y_in, Y_1_in, X_in, opt_mom, beta_int, eta_int)

    # ADMM-step
    start = time.time()
    for v in range(0, len(lambd_values)):
        opt_beta, opt_eta, errors_list = ADMM_DQR_NFE(Y_in, Y_1_in, X_in, opt_mom, beta_int, eta_int, Aut_int, Z_int,
                                                      ta, v, Ls, xi, max_inter, N)  # ADMM
        opt_Z = opt_eta
        beta_list.append(opt_beta)
        Z_matrix[:, v] = opt_Z[:,0]

        # model test
        mse = MSE(Y_out, Y_1_out, X_out, opt_mom, opt_beta, opt_Z, ta, N, T, insample=False)
        var = Variance(Y_out, Y_1_out, X_out, opt_mom, opt_beta, opt_Z, N, T, insample=False)
        test_errors.append(mse)
        test_vars.append(var)
        mse = MSE(Y_in, Y_1_in, X_in_mse, opt_mom, opt_beta, opt_Z, ta, N, T, insample=True)
        var = Variance(Y_in, Y_1_in, X_in_mse, opt_mom, opt_beta, opt_Z, N, T, insample=True)
        train_errors.append(mse)
        train_vars.append(var)
    end = time.time()
    diff = end-start

    # find opt_lambda
    optional_beta = beta_list[test_errors.index(min(test_errors))]
    optional_alpha = Z_matrix[:, test_errors.index(min(test_errors))]

    train_error = train_errors[test_errors.index(min(test_errors))]
    test_error = test_errors[test_errors.index(min(test_errors))]
    train_var = train_vars[test_errors.index(min(test_errors))]
    test_var = test_vars[test_errors.index(min(test_errors))]

    # out-of-sample prediction
    optional_alphaPre = -np.dot(np.dot(np.linalg.inv(LT), Ltr), np.array(optional_alpha).reshape(N, 1))
    optional_alphaPre = optional_alphaPre[:, 0]
    mse = MSE(Y_test, Y_1_test, X_out_all, opt_mom, optional_beta, optional_alphaPre, ta, N, T, insample=False)
    varOut = Variance(Y_test, Y_1_test, X_out_all, opt_mom, optional_beta, optional_alphaPre, N, T, insample=False)

    return train_error, test_error, train_var, test_var, opt_mom, optional_beta, optional_alpha, mse, varOut, diff


def model_error(Y, Y_1, X, mom, beta, eta):
    try:
        W = Y - mom * Y_1 - beta * X - eta
    except:
        I_list = np.diag(np.repeat(1, time_mask))
        e = np.repeat(1, time_mask)
        eta = np.dot(np.kron(I_list, np.reshape(eta, (-1, 1))), np.reshape(e, (-1, 1)))
        W = Y - mom * Y_1 - beta * X - eta
    return W


def eta_step(Y_in, Y_1_in, X_in, opt_mom, beta_int, eta_int, Aut_int, Z_int,
             lambd, Ls, xi, N, time_mask):

    T_sum_matrix = np.kron(np.repeat(1, time_mask), np.eye(N))
    Aut_T = np.dot(T_sum_matrix, Aut_int)
    E_T = xi * np.dot(T_sum_matrix,
                      Z_int - model_error(Y_in, Y_1_in, X_in, opt_mom, beta_int, eta_int*0))
    eta = np.dot(np.linalg.inv(2 * lambd * Ls + xi * np.eye(N)*time_mask), (Aut_T + E_T))
    return eta


def beta_step(Y_in, Y_1_in, X_in, opt_mom, beta_int, eta_int, Aut_int, Z_int, xi):
    I_list = np.diag(np.repeat(1, time_mask))
    e = np.repeat(1, time_mask)
    eta_int = np.dot(np.kron(I_list, np.reshape(eta_int, (-1, 1))), np.reshape(e, (-1, 1)))
    Y_tilde = xi * (Z_int - model_error(Y_in, Y_1_in, X_in, opt_mom, beta_int*0, eta_int)) - Aut_int
    beta = np.dot(np.dot(xi * np.linalg.inv(np.dot(X_in.T, X_in)), X_in.T), Y_tilde)
    return beta


def Z_step(Y_in, Y_1_in, X_in, opt_mom, beta_int, eta_int, Aut_int, ta, xi):
    I_list = np.diag(np.repeat(1, time_mask))
    e = np.repeat(1, time_mask)
    eta_int = np.dot(np.kron(I_list, np.reshape(eta_int, (-1, 1))), np.reshape(e, (-1, 1)))
    z_wilde = Aut_int / xi + model_error(Y_in, Y_1_in, X_in, opt_mom, beta_int, eta_int) \
              - np.kron(np.ones((len(Y_in), 1)), (ta - 0.5) / xi)
    z_max = np.max(
        np.concatenate(((np.abs(z_wilde) - np.ones((len(Y_in), 1)) * 1 / (2 * xi)), np.zeros((len(Y_in), 1))), axis=1),
        axis=1).reshape(-1, 1)
    z_sign = np.sign(z_wilde)
    Z = np.multiply(z_sign, z_max)
    return Z


def Aut_step(Y_in, Y_1_in, X_in, opt_mom, beta_int, eta_int, Aut_int, Z_int, xi):
    I_list = np.diag(np.repeat(1, time_mask))
    e = np.repeat(1, time_mask)
    eta_int = np.dot(np.kron(I_list, np.reshape(eta_int, (-1, 1))), np.reshape(e, (-1, 1)))
    Aut_k = Aut_int - xi * (Z_int - model_error(Y_in, Y_1_in, X_in, opt_mom, beta_int, eta_int))
    return Aut_k


def ADMM_DQR_NFE(Y_in, Y_1_in, X_in, opt_mom, beta_int, eta_int, Aut_int, Z_int,
                 ta, lambd, Ls, xi, max_inter, N):
    '''
   :param Y_in: The vectors of N * T dimension
   :param Y_1_in: First-order lag vector of Y
   :param X_in: The vectors of N * T dimension
   :param X_1_in: First-order lag vector of X
   :param opt_mom: Optimal lag order coefficient
   :param beta_int: The initial beta coefficient
   :param eta_int: The initial eta coefficient is a vector of NT * 1
   :param Aut_int: ADMM first-order multiplier
   :param ta: Quantile level
   :param lambd: Tuning parameter
   :param Ls: Matrix within the sample
   :param xi: Tuning parameter for ADMM
   :param max_inter: Maximum Iterations
   :return: Optimal estimators for  beta and eta
   '''
    errors_z, errors_beta, errors_eta = 10, 10, 10
    time_mask = len(Y_in) // N
    i = 0
    while ((errors_z >= 1 * 0.001) or (errors_beta >= 1 * 0.001)
           or (errors_eta >= 1 * 0.001)) and (i < max_inter):
        eta_new = eta_step(Y_in, Y_1_in, X_in, opt_mom, beta_int, eta_int, Aut_int, Z_int,
                           ta, lambd, Ls, xi, max_inter, N, time_mask)
        beta_new = beta_step(Y_in, Y_1_in, X_in, opt_mom, beta_int, eta_new, Aut_int, Z_int, xi)
        Z_new = Z_step(Y_in, Y_1_in, X_in, opt_mom, beta_new, eta_new, Aut_int, ta, xi)
        Aut_new = Aut_step(Y_in, Y_1_in, X_in, opt_mom, beta_new, eta_new, Aut_int, Z_new, xi)
        errors_z = xi * np.mean(np.square(Aut_new - Aut_int))
        errors_beta = np.mean(np.square(beta_new - beta_int))
        errors_eta = np.mean(np.square(eta_new - eta_int[:len(eta_new)]))
        beta_int, eta_int, Aut_int = beta_new, eta_new, Aut_new
        i += 1
    return beta_int[0], eta_int, [errors_z, errors_beta, errors_eta]


class Simulation_Sigma():

    def __init__(self, N, Time, tau_list, turns_num, sigma, dis_type, Ls, LT, Ltr):

        self.turns_num = turns_num
        self.N = N
        self.Time = Time
        self.sigma = sigma
        self.tau_list = tau_list
        self.dis_type = dis_type
        self.df_loss = pd.DataFrame(columns=['ADMM_train_loss', 'NC_train_loss', 'L1_train_loss', 'L2_train_loss',
                                             'ADMM_test_loss', 'NC_test_loss', 'L1_test_loss', 'L2_test_loss',
                                             'OutPre_ADMM', 'OutPre_NC', 'OutPre_L1', 'OutPre_L2', 'time_ADMM', 'time_IP', 'N', 'T'],
                                    index=tau_list)
        self.Ls, self.LT, self.Ltr = Ls, LT, Ltr

    def estimate(self):
        print(self.LT.shape)
        print(np.linalg.inv(self.LT))

        for tt, tau in enumerate(self.tau_list):
            seed_list = np.arange(0, self.turns_num, 1)  # 设置一列随机种子
            QR_train_loss, QR_test_loss, QR_predict_loss, QR_beta, QR_alpha = [], [], [], [], np.zeros(
                [N // 2, self.turns_num])
            NC_train_loss, NC_test_loss, NC_predict_loss, NC_beta, NC_alpha = [], [], [], [], np.zeros(
                [N // 2, self.turns_num])
            L1_train_loss, L1_test_loss, L1_predict_loss, L1_beta, L1_alpha = [], [], [], [], np.zeros(
                [N // 2, self.turns_num])
            L2_train_loss, L2_test_loss, L2_predict_loss, L2_beta, L2_alpha = [], [], [], [], np.zeros(
                [N // 2, self.turns_num])
            ADMM_TIME, IP_TIME = [], []
            print("quantile level...", tau, self.N, self.Time, 'sigma:', self.sigma, 'turns_num:', self.turns_num, 'dis_type:',
                  self.dis_type)
            for s in range(0, len(seed_list)):
                Y, X = DGP(Time=Time, N=N, ta=tau, seed_num=seed_list[s], sigam=self.sigma,
                           distribution_type=self.dis_type)
                train_er1, test_er1, train_var1, test_var1, momQR, betaQR, alphaQR, MseOutQR, VarQR, time1 \
                    = QRNC_ADMM_model_fit(Y, X, 0.2, self.N, self.Time, tau, self.Ls, self.LT, self.Ltr)
                train_er, test_er, train_var, test_var, momNC, betaNC, alphaNC, MseOutNC, VarNC, time2 \
                    = QRNC_model_fit(Y, X, 0.2, self.N, self.Time, tau, self.Ls, self.LT, self.Ltr)
                train_er_L1, test_er_L1, train_var_L1, test_var_L1, mom_L1, beta_L1, alpha_L1, MseOutL1, VarL1 \
                    = QR_L1_model_fit(Y, X, 0.2, self.N, self.Time, tau)
                train_er_L2, test_er_L2, train_var_L2, test_var_L2, mom_L2, beta_L2, alpha_L2, MseOutL2, VarL2 \
                    = QR_L2_model_fit(Y, X, 0.2, self.N, self.Time, tau)

                QR_train_loss.append(train_er1)
                QR_test_loss.append(test_er1)
                QR_predict_loss.append(MseOutQR)
                ADMM_TIME.append(time1)

                NC_train_loss.append(train_er)
                NC_test_loss.append(test_er)
                NC_predict_loss.append(MseOutNC)
                IP_TIME.append(time2)

                L1_train_loss.append(train_er_L1)
                L1_test_loss.append(test_er_L1)
                L1_predict_loss.append(MseOutL1)

                L2_train_loss.append(train_er_L2)
                L2_test_loss.append(test_er_L2)
                L2_predict_loss.append(MseOutL2)


            QR_train_loss_all, NC_train_loss_all, L1_train_loss_all, L2_train_loss_all, \
            QR_test_loss_all, NC_test_loss_all, L1_test_loss_all, L2_train_loss_all, \
            QR_predict_loss_all, NC_predict_loss_all, L1_predict_loss, L2_predict_loss, ADMM_TIME, IP_TIME = \
                np.mean(QR_train_loss), np.mean(NC_train_loss), np.mean(L1_train_loss), np.mean(L2_train_loss), \
                np.mean(QR_test_loss), np.mean(NC_test_loss), np.mean(L1_test_loss), np.mean(L2_test_loss), \
                np.mean(QR_predict_loss), np.mean(NC_predict_loss), np.mean(L1_predict_loss), np.mean(
                    L2_predict_loss), np.mean(ADMM_TIME), np.mean(IP_TIME)


            Loss_list = [QR_train_loss_all, NC_train_loss_all, L1_train_loss_all, L2_train_loss_all,
                         QR_test_loss_all, NC_test_loss_all, L1_test_loss_all, L2_train_loss_all,
                         QR_predict_loss_all, NC_predict_loss_all, L1_predict_loss, L2_predict_loss, ADMM_TIME, IP_TIME, self.N, self.Time]

            self.df_loss.iloc[tt, :] = Loss_list

        self.df_loss.to_csv(r'data/OutPreADMM_LOSS_%s_%s_%s_%s_%s.csv' % (
        self.N, self.Time, self.sigma, self.turns_num, self.dis_type))


if __name__ == '__main__':
    global turns_num
    turns_num: int = 20
    LOSS_DF = pd.DataFrame()
    PARM_DF = pd.DataFrame()
    for N_turn in range(120, 120*5, 120):
        N = N_turn
        Time = N_turn //2
        dis_type = 'Normal'
        tau_list = np.arange(0.5, 0.6, 0.1)
        sigma = 0.5
        sum_all = pd.DataFrame()
        sum_all_para = pd.DataFrame()
        Ls, LT, Ltr = Graph_generate(N)
        sim = Simulation_Sigma(N, Time, tau_list, turns_num, sigma, dis_type, Ls, LT, Ltr)
        sim.estimate()
        pre_loss = pd.read_csv(
            r'data/OutPreADMM_LOSS_%s_%s_%s_%s_%s.csv' % (N, Time, sigma, turns_num, dis_type))
        if len(LOSS_DF) == 0:
            LOSS_DF = pre_loss
        else:
            LOSS_DF = pd.concat([LOSS_DF, pre_loss])

        print(LOSS_DF)
        LOSS_DF.to_csv(r'data/ADMM_TIME_LOSS_%s_%s.csv' % (dis_type, turns_num), index=False)

        print(N, Time, dis_type, turns_num, '------------------ADMM-simulation-------------------')







