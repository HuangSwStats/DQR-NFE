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
import random
import datetime

warnings.filterwarnings("ignore")


def MSE(Y, Y_1, X, moment, beta, alpha, ta, N, predict_length):
    I_list = np.diag(np.repeat(1, predict_length))
    e = np.repeat(1, predict_length)
    Z = np.dot(cp.kron(I_list, cp.reshape(cp.vec(alpha), (N, 1))), cp.reshape(e, (predict_length, 1)))
    u = Y - moment * Y_1 - X @ beta - Z
    loss = cp.sum(quantile_loss(u, tau=ta)).value
    return 1 / (N * predict_length) * loss

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
    :param Y: Input sample matrix
    :param N: Input sample number
    :param insample: Forecast type (in-sample or out-of-sample prediction)
    :return: The split sample dataframe
    """

    N = 2*N
    if insample ==True:
        Y1, Y2, Y3 = Y[:N // 6, :], Y[N // 3:N // 2, :], Y[-N // 3:-N // 6, :]
        return np.vstack([Y1, Y2, Y3])
    if insample==False:
        Y1, Y2, Y3 = Y[N // 6:N // 3, :], Y[N // 2:2*N// 3, :], Y[-N // 6:, :]
        return np.vstack([Y1, Y2, Y3])

# Define the functions required for DQR-NFE and benchmark models estimation
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

def quantile_loss(u, tau):
    return 0.5 * cp.abs(u) + (tau - 0.5) * u

def Objective_fn(Y, Y_1, X, X_1, moment, beta, alpha, IV_hat, ta):
    return cp.sum(quantile_loss(Y - moment * Y_1 - X @ beta - IV_hat * X_1 - alpha, tau=ta))

def QRNC_fn(Y, Y_1, X, X_1, moment, beta, alpha, IV_hat, ta, lambd, alp, L):
    return Objective_fn(Y, Y_1, X, X_1, moment, beta, alpha, IV_hat, ta) \
           + lambd * L2_penilty(alp, L)

def QRNC_model_fit(Y_in, Y_1_in, X_in, X_1_in,
                   Y_test, Y_1_test, X_out_all, N, N_pre, T, time_mask, ta, Ls, LT, Ltr):
    # DQR-NFE model we proposed
    # N is the number of sample for training and N_pre is for test
    # time_mask is the length of time for training
    I_list = np.diag(np.repeat(1, time_mask))
    e = np.repeat(1, time_mask)
    predict_length = T - time_mask

    # model estimation
    moment_list = np.arange(0.1, 0.9, 0.05)
    IV_list, beta_list, betaBaseline_list = [], [], []
    Z_hat_list = np.zeros(N * len(moment_list)).reshape(N, len(moment_list))
    for i in range(0, len(moment_list)):
        mom = moment_list[i]
        beta = cp.Variable((8, 1), name='beta')
        alpha = cp.Variable(N)
        Z_hat = np.dot(cp.kron(I_list, cp.reshape(cp.vec(alpha), (N, 1))), cp.reshape(e, (time_mask, 1)))
        IV_hat = cp.Variable(1)
        problem = cp.Problem(
            cp.Minimize(Objective_fn(Y_in, Y_1_in, X_in, X_1_in, mom, beta, Z_hat, IV_hat, ta)))
        problem.solve(solver='ECOS')
        IV_list.append(IV_hat.value)
        beta_list.append(beta.value)
        Z_hat_list[:, i] = alpha.value

    opt_mom = moment_list[IV_list.index(min(IV_list))]
    beta = cp.Variable((8, 1), name='beta')
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
        opt_beta = beta.value
        opt_Z = alpha.value
        beta_list.append(opt_beta)
        Z_matrix[:, v] = opt_Z

        # test model
        mse = MSE(Y_in, Y_1_in, X_in, opt_mom, opt_beta, opt_Z, ta, N, time_mask)
        test_errors.append(mse)

    optional_beta = beta_list[test_errors.index(min(test_errors))]
    optional_alpha = Z_matrix[:, test_errors.index(min(test_errors))]
    test_error = test_errors[test_errors.index(min(test_errors))]

    # out-of-sample prediction
    optional_alphaPre = -np.dot(np.dot(np.linalg.inv(LT), Ltr), np.array(optional_alpha).reshape(N, 1))
    optional_alphaPre = optional_alphaPre[:, 0]
    mse = MSE(Y_test, Y_1_test, X_out_all, opt_mom, optional_beta, optional_alphaPre, ta, N_pre, predict_length)

    return test_error, opt_mom, optional_beta, optional_alpha, mse


def QR_model_fit(Y_in, Y_1_in, X_in, X_1_in,
                   Y_test, Y_1_test, X_out_all, N, N_pre, T, time_mask, ta):

    # DQR-FE model used as baseline
    I_list = np.diag(np.repeat(1, time_mask))
    e = np.repeat(1, time_mask)
    predict_length = T - time_mask

    moment_list = np.arange(0.1, 0.9, 0.05)
    IV_list, beta_list, betaBaseline_list = [], [], []
    Z_hat_list = np.zeros(N * len(moment_list)).reshape(N, len(moment_list))
    for i in range(0, len(moment_list)):
        mom = moment_list[i]
        beta = cp.Variable((8, 1), name='beta')
        alpha = cp.Variable(N)
        Z_hat = np.dot(cp.kron(I_list, cp.reshape(cp.vec(alpha), (N, 1))), cp.reshape(e, (time_mask, 1)))
        IV_hat = cp.Variable(1)
        problem = cp.Problem(
            cp.Minimize(Objective_fn(Y_in, Y_1_in, X_in, X_1_in, mom, beta, Z_hat, IV_hat, ta)))  # 先选择滞后阶系数，再做修正
        problem.solve(solver='ECOS')
        IV_list.append(IV_hat.value)
        beta_list.append(beta.value)
        Z_hat_list[:, i] = alpha.value

    opt_mom = moment_list[IV_list.index(min(IV_list))]
    optional_beta = beta_list[IV_list.index(min(IV_list))]
    optional_alpha = Z_hat_list[:, IV_list.index(min(IV_list))]
    optional_alpha = optional_alpha.reshape(-1, 1)
    test_error = MSE(Y_in, Y_1_in, X_in, opt_mom, optional_beta, optional_alpha, ta, N, time_mask)


    # out-of-sample prediction
    optional_alphaPre = np.repeat(np.mean(optional_alpha), N_pre)
    mse = MSE(Y_test, Y_1_test, X_out_all, opt_mom, optional_beta, optional_alphaPre, ta, N_pre, predict_length)

    return test_error, opt_mom, optional_beta, optional_alpha, mse


def QRL1_model_fit(Y_in, Y_1_in, X_in, X_1_in,
                   Y_test, Y_1_test, X_out_all, N, N_pre, T, time_mask, ta):
    # DQR-L1 model used as baseline
    I_list = np.diag(np.repeat(1, time_mask))
    e = np.repeat(1, time_mask)
    predict_length = T - time_mask

    moment_list = np.arange(0.1, 0.9, 0.05)
    IV_list, beta_list, betaBaseline_list = [], [], []
    Z_hat_list = np.zeros(N * len(moment_list)).reshape(N, len(moment_list))
    for i in range(0, len(moment_list)):
        mom = moment_list[i]
        beta = cp.Variable((8, 1), name='beta')
        alpha = cp.Variable(N)
        Z_hat = np.dot(cp.kron(I_list, cp.reshape(cp.vec(alpha), (N, 1))), cp.reshape(e, (time_mask, 1)))
        IV_hat = cp.Variable(1)
        problem = cp.Problem(
            cp.Minimize(Objective_fn(Y_in, Y_1_in, X_in, X_1_in, mom, beta, Z_hat, IV_hat, ta)))  # 先选择滞后阶系数，再做修正
        problem.solve(solver='ECOS')
        IV_list.append(IV_hat.value)
        beta_list.append(beta.value)
        Z_hat_list[:, i] = alpha.value

    opt_mom = moment_list[IV_list.index(min(IV_list))]
    beta = cp.Variable((8, 1), name='beta')
    alpha = cp.Variable(N)
    I_list = np.diag(np.repeat(1, time_mask))
    e = np.repeat(1, time_mask)
    Z_hat = np.dot(cp.kron(I_list, cp.reshape(cp.vec(alpha), (N, 1))), cp.reshape(e, (time_mask, 1)))
    IV_hat = 0
    lambd = cp.Parameter(nonneg=True)
    problem = cp.Problem(cp.Minimize(QRL1_fn(Y_in, Y_1_in, X_in, X_1_in, opt_mom,
                                             beta, Z_hat, IV_hat, ta, lambd, alpha)))
    global lambd_values
    lambd_values = np.arange(0.01, 3, 0.1)
    train_errors, train_vars, test_errors, test_vars = [], [], [], []
    beta_list, Z_matrix = [], np.zeros(N * len(lambd_values)).reshape(N, len(lambd_values))
    for v in range(0, len(lambd_values)):
        lambd.value = lambd_values[v]
        problem.solve(solver='ECOS')
        opt_beta = beta.value
        opt_Z = alpha.value
        beta_list.append(opt_beta)
        Z_matrix[:, v] = opt_Z

        mse = MSE(Y_in, Y_1_in, X_in, opt_mom, opt_beta, opt_Z, ta, N, time_mask)
        test_errors.append(mse)
    optional_beta = beta_list[test_errors.index(min(test_errors))]
    optional_alpha = Z_matrix[:, test_errors.index(min(test_errors))]
    test_error = test_errors[test_errors.index(min(test_errors))]

    # 样本外预测
    optional_alphaPre = np.repeat(np.mean(optional_alpha), N_pre)
    mse = MSE(Y_test, Y_1_test, X_out_all, opt_mom, optional_beta, optional_alphaPre, ta, N_pre, predict_length)

    return test_error, opt_mom, optional_beta, optional_alpha, mse



def QRL2_model_fit(Y_in, Y_1_in, X_in, X_1_in,
                   Y_test, Y_1_test, X_out_all, N, N_pre, T, time_mask, ta):

    # DQR-L2 model used as baseline
    I_list = np.diag(np.repeat(1, time_mask))
    e = np.repeat(1, time_mask)
    predict_length = T - time_mask

    moment_list = np.arange(0.1, 0.9, 0.05)
    IV_list, beta_list, betaBaseline_list = [], [], []
    Z_hat_list = np.zeros(N * len(moment_list)).reshape(N, len(moment_list))
    for i in range(0, len(moment_list)):
        mom = moment_list[i]
        beta = cp.Variable((8, 1), name='beta')
        alpha = cp.Variable(N)
        Z_hat = np.dot(cp.kron(I_list, cp.reshape(cp.vec(alpha), (N, 1))), cp.reshape(e, (time_mask, 1)))
        IV_hat = cp.Variable(1)
        problem = cp.Problem(
            cp.Minimize(Objective_fn(Y_in, Y_1_in, X_in, X_1_in, mom, beta, Z_hat, IV_hat, ta)))
        problem.solve(solver='ECOS')
        IV_list.append(IV_hat.value)
        beta_list.append(beta.value)
        Z_hat_list[:, i] = alpha.value

    opt_mom = moment_list[IV_list.index(min(IV_list))]
    beta = cp.Variable((8, 1), name='beta')
    alpha = cp.Variable(N)
    I_list = np.diag(np.repeat(1, time_mask))
    e = np.repeat(1, time_mask)
    Z_hat = np.dot(cp.kron(I_list, cp.reshape(cp.vec(alpha), (N, 1))), cp.reshape(e, (time_mask, 1)))
    IV_hat = 0
    lambd = cp.Parameter(nonneg=True)
    problem = cp.Problem(cp.Minimize(QRL2_fn(Y_in, Y_1_in, X_in, X_1_in, opt_mom,
                                             beta, Z_hat, IV_hat, ta, lambd, alpha)))
    global lambd_values
    lambd_values = np.arange(0.01, 3, 0.1)
    train_errors, train_vars, test_errors, test_vars = [], [], [], []
    beta_list, Z_matrix = [], np.zeros(N * len(lambd_values)).reshape(N, len(lambd_values))
    for v in range(0, len(lambd_values)):
        lambd.value = lambd_values[v]
        problem.solve(solver='ECOS')
        opt_beta = beta.value
        opt_Z = alpha.value
        beta_list.append(opt_beta)
        Z_matrix[:, v] = opt_Z

        mse = MSE(Y_in, Y_1_in, X_in, opt_mom, opt_beta, opt_Z, ta, N, time_mask)
        test_errors.append(mse)

    optional_beta = beta_list[test_errors.index(min(test_errors))]
    optional_alpha = Z_matrix[:, test_errors.index(min(test_errors))]
    test_error = test_errors[test_errors.index(min(test_errors))]

    optional_alphaPre = np.repeat(np.mean(optional_alpha), N_pre)
    mse = MSE(Y_test, Y_1_test, X_out_all, opt_mom, optional_beta, optional_alphaPre, ta, N_pre, predict_length)

    return test_error, opt_mom, optional_beta, optional_alpha, mse

def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)


def protected_division(x1, x2):
    """Closure of division (x1/x2) for zero denominator."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x2) > _protected_ratio, np.divide(x1, x2), 1.)


def clean_data(df):
    codes = np.unique(df.code.tolist())
    code_list = []
    for code, d in df.groupby(by=['code']):
        if len(d) < 9:
            code_list.append(code)
        else:
            pass
    codes = list(set(codes) - set(code_list))
    df1 = df[df['code'].isin(codes)]
    df1 = df1.sort_values(by=['TradingDate', 'code'], ascending=[True, True])
    return df1

if __name__ == '__main__':

    df_sum_all = pd.DataFrame()  # The dataframe storing the final test results
    times = 2   # Number of repeated experiments (Select 50 samples randomly each time as the prediction target outside the sample).
    for rs in np.arange(1, times):
        print('===================send', rs, '=======================')

        # Closure of division (x1/x2) for zero denominator.
        _protected_ratio = 0.00000001

        df = pd.read_csv(r'data/fund_data.csv')
        df['TradingDate'] = pd.to_datetime(df['TradingDate'])
        df_matrix = pd.read_csv(r'data/adj_Matrix_20200630.csv', encoding='GBK')

        print(df)
        mpd = []
        # Data standardization
        for lab, df1 in df.groupby(by=['TradingDate']):
            df2 = df1.copy()
            df2['Turn'] = protected_division(df1['AvgTurnoverRate'] - df1['AvgTurnoverRate'].mean(),
                                             df1['AvgTurnoverRate'].std())
            df2['Volume'] = protected_division(df1['Volume'] - df1['Volume'].mean(), df1['Volume'].std())
            df2['MarketValue'] = protected_division(df1['MarketValue'] - df1['MarketValue'].mean(),
                                                    df1['MarketValue'].std())
            df2['NAV'] = protected_division(df1['NAV'] - df1['NAV'].mean(), df1['NAV'].std())
            df2['ReturnAccumulativeNAV'] = protected_division(
                df1['ReturnAccumulativeNAV'] - df1['ReturnAccumulativeNAV'].mean(), df1['ReturnAccumulativeNAV'].std())
            df2['PurchaseHouseholders'] = protected_division(
                df1['PurchaseHouseholders'] - df1['PurchaseHouseholders'].mean(), df1['PurchaseHouseholders'].std())
            df2['NetSales'] = protected_division(df1['NetSales'] - df1['NetSales'].mean(), df1['NetSales'].std())
            df2['Interest'] = protected_division(df1['Interest'] - df1['Interest'].mean(), df1['Interest'].std())
            mpd.append(df2)
        data = pd.concat(mpd)
        data['code'] = [x + '.OF' for x in data.Symbol.astype(str)]
        print(data)

        # Divide training set and back-test set
        data = data[
            ['TradingDate', 'code', 'Ipostpbdt', 'y', 'ChangeRatio', 'IV', 'Turn', 'Volume', 'MarketValue', 'NAV',
             'ReturnAccumulativeNAV',
             'PurchaseHouseholders', 'NetSales', 'Interest']]

        data = clean_data(data)

        code_list = np.unique(data['code'])
        code_predict = code_list[:50]                    # 50 samples for prediction
        code_train = code_list[50:]
        data_predict = data[data['code'].isin(code_predict)]
        data_train = data[data['code'].isin(code_train)]

        data_predict = data_predict[data_predict['TradingDate'] == '2020-11-30']
        data_predict = data_predict.sort_values(by=['TradingDate', 'code'], ascending=[True, True])
        data_train = data_train[data_train['TradingDate'] < '2020-11-30']
        data_train = data_train.sort_values(by=['TradingDate', 'code'], ascending=[True, True])

        data_predict.to_csv(r'data/predict_panel_data.csv', index=False)
        data_train.to_csv(r'data/train_panel_data.csv', index=False)

        # Reconstruction of fund network
        code_list = np.unique(data_train['code']).tolist() + np.unique(data_predict['code']).tolist()
        code_list = code_list + ['证券代码']
        code_train = np.unique(data_train['code']).tolist() + ['证券代码']
        df_matrix_train = df_matrix[code_train]
        df_matrix = df_matrix[code_list]
        df_matrix = df_matrix[df_matrix['证券代码'].isin(code_list)]

        # Adjust the code order of adjacency matrix
        df_matrix_1 = df_matrix[df_matrix['证券代码'].isin(code_predict)].sort_values(by='证券代码', ascending=True)
        df_matrix_2 = df_matrix[df_matrix['证券代码'].isin(code_train)].sort_values(by='证券代码', ascending=True)
        df_matrix = pd.concat([df_matrix_2, df_matrix_1], axis=0).set_index('证券代码')
        df_matrix_train = df_matrix_train[df_matrix_train['证券代码'].isin(code_train)]. \
            sort_values(by='证券代码', ascending=True).set_index('证券代码')

        # Adjust the weight coefficient of train matrix
        for i in range(0, len(df_matrix_train)):
            for j in range(0, len(df_matrix_train.columns)):
                if df_matrix_train.iloc[i, j] != 0:
                    df_matrix_train.iloc[i, j] = -1
                else:
                    continue
        print(df_matrix_train)

        # Adjust the matrix of predict
        for i in range(0, len(df_matrix)):
            for j in range(0, len(df_matrix.columns)):
                if df_matrix.iloc[i, j] != 0:
                    df_matrix.iloc[i, j] = -1
                else:
                    continue

        # Adjust the diagonal elements of the matrix of predict
        for i in range(0, len(df_matrix)):
            for j in range(0, len(df_matrix.columns)):
                if i == j:
                    df_matrix.iloc[i, j] = -np.sum(df_matrix.iloc[i, :])
                else:
                    continue

        # Adjust the diagonal elements of the matrix of train
        for i in range(0, len(df_matrix_train)):
            for j in range(0, len(df_matrix_train.columns)):
                if i == j:
                    df_matrix_train.iloc[i, j] = -np.sum(df_matrix_train.iloc[i, :])
                else:
                    continue

        print(df_matrix)
        # Storage network Laplace matrix
        df_matrix.to_csv(r'data/L_matrix_panel_predict.csv', index=False)
        df_matrix_train.to_csv(r'data/L_matrix_panel_train.csv', index=False)

        # read data
        df_train = pd.read_csv(r'data/train_panel_data.csv')
        df_predict = pd.read_csv(r'data/predict_panel_data.csv')
        Y_in = np.mat(df_train['y']).reshape(-1, 1)
        Y_1_in = np.mat(df_train['ChangeRatio']).reshape(-1, 1)
        X_in = np.mat(df_train.iloc[:, -8:].values)
        X_1_in = np.mat(df_train['IV']).reshape(-1, 1)
        Y_test = np.mat(df_predict['y']).reshape(-1, 1)
        Y_1_test = np.mat(df_predict['ChangeRatio']).reshape(-1, 1)
        X_out_all = np.mat(df_predict.iloc[:, -8:].values)

        # read network structure
        L_predict = pd.read_csv(r'data/L_matrix_panel_predict.csv')
        L_train = pd.read_csv(r'data/L_matrix_panel_train.csv')
        Ls = L_train.values
        Lp = L_predict.values[-50:, -50:]
        Ltr = L_predict.values[-50:, :-50]
        N = len(np.unique(df_train['code'].tolist()))
        T = 9
        time_mask = len(df_train)//N
        N_pre = len(np.unique(df_predict['code'].tolist()))


        ta_list = np.arange(0.05, 1, 0.05)  # selection for quantile levels
        df_result_mse = pd.DataFrame(columns=['tau', 'DQR-FE', 'DQR-L1', 'DQR-L2', 'DQR-NFE'])
        df_result_trainError = pd.DataFrame(columns=['tau', 'DQR-FE', 'DQR-L1', 'DQR-L2', 'DQR-NFE'])
        df_result_mse['tau'] = ta_list
        df_result_trainError['tau'] = ta_list

        for i, ta in enumerate(ta_list):
            print(i, ta)
            test_error, opt_mom, optional_beta, optional_alpha, mse = \
                QR_model_fit(Y_in, Y_1_in, X_in, X_1_in,
                               Y_test, Y_1_test, X_out_all, N, N_pre, T, time_mask, ta)

            test_error1, opt_mom1, optional_beta1, optional_alpha1, mse1 = \
                QRL1_model_fit(Y_in, Y_1_in, X_in, X_1_in,
                               Y_test, Y_1_test, X_out_all, N, N_pre, T, time_mask, ta)

            test_error2, opt_mom2, optional_beta2, optional_alpha2, mse2 = \
                QRL2_model_fit(Y_in, Y_1_in, X_in, X_1_in,
                               Y_test, Y_1_test, X_out_all, N, N_pre, T, time_mask, ta)

            test_error3, opt_mom3, optional_beta3, optional_alpha3, mse3 = \
                QRNC_model_fit(Y_in, Y_1_in, X_in, X_1_in,
                               Y_test, Y_1_test, X_out_all, N, N_pre, T, time_mask, ta, Ls, Lp, Ltr)

            mse_list = [mse, mse1, mse2, mse3]
            loss_list = [test_error, test_error1, test_error2, test_error3]
            df_result_mse.iloc[i, 1:] = mse_list
            df_result_trainError.iloc[i, 1:] = loss_list


        df_result_mse.to_csv(r'data/result_preError_%s.csv' % (rs))   # Out-of-sample prediction errors of each test
        df_result_trainError.to_csv(r'data/result_trainError_%s.csv' % (rs)) # Out-of-sample train errors of each test
        df_sum = df_result_mse.apply(lambda x:x.sum())[-4:]
        if len(df_sum_all) == 0:
            df_sum_all = df_sum
        else:
            df_sum_all = pd.concat([df_sum_all, df_sum], axis=1)

    df_sum_all.to_csv(r'data/SumResult_preError_%s_times.csv' % (times))  # Out-of-sample prediction results for t times





