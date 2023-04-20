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

def quantile_loss(u, tau):
    return 0.5 * cp.abs(u) + (tau - 0.5) * u


def L2_penilty(alpha, L):
    return cp.quad_form(alpha, L)


def L1_penilty(alpha):
    return cp.norm1(alpha)


def Ridge_penilty(alpha):
    return cp.pnorm(alpha, p=2)**2


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


def MSE(Y, Y_1, X, moment, beta, alpha, ta, N,T,insample):
    I_list = np.diag(np.repeat(1, X.shape[1]))
    if insample == False:
        X_out = X.reshape(N*(T-time_mask-1), 1, order='F')
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
        X_out = X.reshape(N*(T-time_mask-1), 1, order='F')
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
    N = 2*N
    if insample ==True:
        Y1, Y2, Y3 = Y[:N // 6, :], Y[N // 3:N // 2, :], Y[-N // 3:-N // 6, :]
        return np.vstack([Y1, Y2, Y3])
    if insample==False:
        Y1, Y2, Y3 = Y[N // 6:N // 3, :], Y[N // 2:2*N// 3, :], Y[-N // 6:, :]
        return np.vstack([Y1, Y2, Y3])


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


# model estimation functions

# DQR-NFE model estimation
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

    N = N // 2  # Screening out half of the samples for training and verification
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
    Y_out_all, X_out_all, Y_1_out_all = split_sample(Y, N, False), split_sample(X, N, False), split_sample(Y_1, N,                                                                                                    False)
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
        IV_hat = cp.Variable(1)
        problem = cp.Problem(
            cp.Minimize(Objective_fn(Y_in, Y_1_in, X_in, X_1_in, mom, beta, Z_hat, IV_hat, ta)))
        problem.solve(solver='ECOS')
        IV_list.append(IV_hat.value[0])
        beta_list.append(beta.value[0])
        Z_hat_list[:, i] = alpha.value

    opt_mom = moment_list[IV_list.index(min(IV_list))]
    beta = cp.Variable(1)
    alpha = cp.Variable(N)
    I_list = np.diag(np.repeat(1, time_mask))
    e = np.repeat(1, time_mask)
    Z_hat = np.dot(cp.kron(I_list, cp.reshape(cp.vec(alpha), (N, 1))), cp.reshape(e, (time_mask, 1)))
    IV_hat = 0
    lambd = cp.Parameter(nonneg=True)
    problem = cp.Problem(cp.Minimize(QRNC_fn(Y_in, Y_1_in, X_in, X_1_in, opt_mom,
                                             beta, Z_hat, IV_hat, ta, lambd, alpha, Ls)))
    global lambd_values
    lambd_values = np.arange(0.01, 3, 0.1)
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
    optional_alphaPre = -np.dot(np.dot(np.linalg.inv(LT), Ltr), np.array(optional_alpha).reshape(N, 1))
    optional_alphaPre = optional_alphaPre[:, 0]
    mse = MSE(Y_test, Y_1_test, X_out_all, opt_mom, optional_beta, optional_alphaPre, ta, N, T, insample=False)
    varOut = Variance(Y_test, Y_1_test, X_out_all, opt_mom, optional_beta, optional_alphaPre, N, T, insample=False)

    return train_error, test_error, train_var, test_var, opt_mom, optional_beta, optional_alpha, mse, varOut


# DQR-FE model estimation
def QR_model_fit(Y_DGP, X_DGP, split_ratio, N, T, ta):

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

    # test data
    Y_out, X_out, Y_1_out = Y_in_all[:, time_mask:], X_in_all[:, time_mask:], Y_1_in_all[:, time_mask:]
    Y_out = Y_out.reshape(N * (T - time_mask - 1), 1, order='F')
    Y_1_out = Y_1_out.reshape(N * (T - time_mask - 1), 1, order='F')
    Y_out_all, X_out_all, Y_1_out_all = split_sample(Y, N, False), split_sample(X, N, False), split_sample(Y_1, N,                                                                                                      False)
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
    opt_mom = moment_list[IV_list.index(min(IV_list))]
    opt_beta = beta_list[IV_list.index(min(IV_list))]
    opt_Z = Z_hat_list[:, IV_list.index(min(IV_list))]

    # model test
    train_errors, train_var, test_errors, test_var = [], [], [], []
    mse = MSE(Y_out, Y_1_out, X_out, opt_mom, opt_beta, opt_Z, ta, N, T, insample=False)
    var = Variance(Y_out, Y_1_out, X_out, opt_mom, opt_beta, opt_Z, N, T, insample=False)
    test_errors.append(mse)
    test_var.append(var)
    mse = MSE(Y_in, Y_1_in, X_in_mse, opt_mom, opt_beta, opt_Z, ta, N, T, insample=True)
    var = Variance(Y_in, Y_1_in, X_in_mse, opt_mom, opt_beta, opt_Z, N, T, insample=True)
    train_errors.append(mse)
    train_var.append(var)
    out_Z = np.repeat(np.mean(opt_Z), N)

    MseOut = MSE(Y_test, Y_1_test, X_out_all, opt_mom, opt_beta, out_Z, ta, N, T, insample=False)
    varOut = Variance(Y_test, Y_1_test, X_out_all, opt_mom, opt_beta, out_Z, N, T, insample=False)
    return train_errors, test_errors, train_var, test_var, opt_mom, opt_beta, opt_Z, MseOut, varOut


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

    lambd_values = np.arange(0.01, 3, 0.1)
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


# DQR-L2 model estimation
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
    lambd_values = np.arange(0.01, 3, 0.1)
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