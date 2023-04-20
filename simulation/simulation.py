import igraph as ig
import cvxpy as cp
import numpy as np
import pandas as pd
import scipy
from scipy.stats import norm
from scipy import stats
import matplotlib.pyplot as plt
from data_generation import *
from model_function import *
import random
import warnings

warnings.filterwarnings("ignore")


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

# The out-of-sample performance of each model when sigma increases
class Simulation_Sigma():

    def __init__(self, N, Time, tau_list, turns_num, sigma, dis_type, Ls, LT, Ltr):

        self.turns_num = turns_num
        self.N = N
        self.Time = Time
        self.sigma = sigma
        self.tau_list = tau_list
        self.dis_type = dis_type
        self.df_loss = pd.DataFrame(columns=['QR_train_loss', 'NC_train_loss', 'L1_train_loss', 'L2_train_loss',
                                             'QR_test_loss', 'NC_test_loss', 'L1_test_loss', 'L2_test_loss',
                                             'OutPre_QR', 'OutPre_NC', 'OutPre_L1', 'OutPre_L2'],
                                    index=tau_list)
        self.df_parm = pd.DataFrame(columns=['QR_betaMSE', 'NC_betaMSE', 'L1_betaMSE', 'L2_betaMSE',
                                             'QR_alphaMSE', 'NC_alphaMSE', 'L1_alphaMSE', 'L2_alphaMSE'], index=tau_list)
        self.Ls, self.LT, self.Ltr = Ls, LT, Ltr

    def estimate(self):
        print(self.LT.shape)
        print(np.linalg.inv(self.LT))

        for tt, tau in enumerate(self.tau_list):
            seed_list = np.arange(0, self.turns_num, 1)
            QR_train_loss, QR_test_loss, QR_predict_loss, QR_beta, QR_alpha = [], [], [], [], np.zeros(
                [N // 2, self.turns_num])
            NC_train_loss, NC_test_loss, NC_predict_loss, NC_beta, NC_alpha = [], [], [], [], np.zeros(
                [N // 2, self.turns_num])
            L1_train_loss, L1_test_loss, L1_predict_loss, L1_beta, L1_alpha = [], [], [], [], np.zeros([N // 2, self.turns_num])
            L2_train_loss, L2_test_loss, L2_predict_loss, L2_beta, L2_alpha = [], [], [], [], np.zeros([N // 2, self.turns_num])
            print("quantile level...", tau, self.N, self.Time, 'sigma:', self.sigma, 'turns_num:', self.turns_num, 'dis_type:', self.dis_type)
            for s in range(0, len(seed_list)):
                Y, X = DGP(Time=Time, N=N, ta=tau, seed_num=seed_list[s], sigam=self.sigma, distribution_type=self.dis_type)
                train_er1, test_er1, train_var1, test_var1, momQR, betaQR, alphaQR, MseOutQR, VarQR \
                    = QR_model_fit(Y, X, 0.2, self.N, self.Time, tau)
                train_er, test_er, train_var, test_var, momNC, betaNC, alphaNC, MseOutNC, VarNC \
                    = QRNC_model_fit(Y, X, 0.2, self.N, self.Time, tau, self.Ls, self.LT, self.Ltr)
                train_er_L1, test_er_L1, train_var_L1, test_var_L1, mom_L1, beta_L1, alpha_L1, MseOutL1, VarL1 \
                    = QR_L1_model_fit(Y, X, 0.2,  self.N, self.Time, tau)
                train_er_L2, test_er_L2, train_var_L2, test_var_L2, mom_L2, beta_L2, alpha_L2, MseOutL2, VarL2 \
                    = QR_L2_model_fit(Y, X, 0.2, self.N, self.Time, tau)

                QR_train_loss.append(train_er1)
                QR_test_loss.append(test_er1)
                QR_predict_loss.append(MseOutQR)
                QR_beta.append(betaQR)
                QR_alpha[:, s] = alphaQR
                NC_train_loss.append(train_er)
                NC_test_loss.append(test_er)
                NC_predict_loss.append(MseOutNC)
                NC_beta.append(betaNC)
                NC_alpha[:, s] = alphaNC
                L1_train_loss.append(train_er_L1)
                L1_test_loss.append(test_er_L1 )
                L1_predict_loss.append(MseOutL1)
                L1_beta.append(beta_L1)
                L1_alpha[:, s] = alpha_L1
                L2_train_loss.append(train_er_L2 )
                L2_test_loss.append(test_er_L2)
                L2_predict_loss.append(MseOutL2)
                L2_beta.append(beta_L2)
                L2_alpha[:, s] = alpha_L2

            # Calculate the results after multiple simulations
            QR_train_loss_all, NC_train_loss_all, L1_train_loss_all, L2_train_loss_all,\
            QR_test_loss_all, NC_test_loss_all, L1_test_loss_all, L2_train_loss_all,\
            QR_predict_loss_all, NC_predict_loss_all, L1_predict_loss, L2_predict_loss = \
                np.mean(QR_train_loss), np.mean(NC_train_loss), np.mean(L1_train_loss), np.mean(L2_train_loss),\
                np.mean(QR_test_loss), np.mean(NC_test_loss), np.mean(L1_test_loss), np.mean(L2_test_loss), \
                np.mean(QR_predict_loss), np.mean(NC_predict_loss), np.mean(L1_predict_loss), np.mean(L2_predict_loss)

            QR_betaMSE, NC_betaMSE, L1_betaMSE, L2_betaMSE, QR_alphaMSE, NC_alphaMSE, L1_alphaMSE, L2_alphaMSE \
                = BetaMSE(QR_beta, 0.3), BetaMSE(NC_beta, 0.3), BetaMSE(L1_beta, 0.3), BetaMSE(L2_beta, 0.3), \
                  AlphaMSE(QR_alpha, Talpha, tau, self.dis_type), AlphaMSE(NC_alpha, Talpha, tau, self.dis_type), \
                  AlphaMSE(L1_alpha, Talpha, tau, self.dis_type), AlphaMSE(L2_alpha, Talpha, tau, self.dis_type)

            Loss_list = [QR_train_loss_all, NC_train_loss_all, L1_train_loss_all, L2_train_loss_all,
                         QR_test_loss_all, NC_test_loss_all, L1_test_loss_all, L2_train_loss_all,
                         QR_predict_loss_all, NC_predict_loss_all, L1_predict_loss, L2_predict_loss]
            Para_list = [QR_betaMSE, NC_betaMSE, L1_betaMSE, L2_betaMSE,
                         QR_alphaMSE, NC_alphaMSE, L1_alphaMSE, L2_alphaMSE]

            self.df_loss.iloc[tt, :] = Loss_list
            self.df_parm.iloc[tt, :] = Para_list

        self.df_loss.to_csv(r'data/OutPre_LOSS&bench_%s_%s_%s_%s_%s.csv' % (self.N, self.Time, self.sigma, self.turns_num, self.dis_type))
        self.df_parm.to_csv(r'data/OutPre_PARM&bench_%s_%s_%s_%s_%s.csv' % (self.N, self.Time, self.sigma, self.turns_num, self.dis_type))



if __name__ == '__main__':
    global turns_num
    turns_num: int = 10 # Set the number of repeated tests
    N = 150 # Number of samples
    Time = 40  # Time length
    dis_type = 'Normal'  # error distribution
    tau_list = np.arange(0.6, 0.9, 0.1)
    sigma_list = np.arange(0.1, 1.1, 0.1)  # sigma (noise) setting
    sum_all = pd.DataFrame()
    sum_all_para = pd.DataFrame()
    Ls, LT, Ltr = Graph_generate(N)
    for sigma in sigma_list:
        sim = Simulation_Sigma(N, Time, tau_list, turns_num, sigma, dis_type, Ls, LT, Ltr)
        sim.estimate()
        pre_loss = pd.read_csv(r'data/OutPre_LOSS&bench_%s_%s_%s_%s_%s.csv' % (N, Time, sigma, turns_num, dis_type))
        para_loss = pd.read_csv(r'data/OutPre_PARM&bench_%s_%s_%s_%s_%s.csv' % (N, Time, sigma, turns_num, dis_type))
        pre_loss.loc['col_sum'] = pre_loss.apply(lambda x: x.sum())
        para_loss.loc['col_sum'] = para_loss.apply(lambda x: x.sum())
        sum_list = pre_loss.iloc[-1, 1:]
        sum_list_para = para_loss.iloc[-1, 1:]
        if len(sum_all) == 0:
            sum_all = sum_list
            sum_all_para = sum_list_para
        else:
            sum_all = pd.concat([sum_all, sum_list], axis=1)
            sum_all_para = pd.concat([sum_all_para, sum_list_para], axis=1)
        print(sum_all)
        print(sum_all_para)
    sum_all.columns = sigma_list
    sum_all_para.columns = sigma_list
    sum_all.to_csv(r'data/sigma_OutPre_LOSS&bench_sim_%s_%s_%s_%s.csv' % (N, Time, dis_type, turns_num))
    sum_all_para.to_csv(r'data/sigma_OutPre_PARM&bench_sim_%s_%s_%s_%s.csv' % (N, Time, dis_type, turns_num))

    print(N, Time, dis_type, turns_num, 'sigma_sim')