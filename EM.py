import numpy as np
from copy import deepcopy
import scipy.special

def randomParams(num):
    param = np.random.rand(num)
    param = param / np.sum(param)
    return param

def EM(X, W, Z, max_step, epsilon, try_times):
    N = len(X)
    P_z = np.zeros((N, Z))
    ThetaOut = np.zeros((Z, W))
    lambdaOut = np.zeros(Z)
    Q_Value = 0
    oldQ_Value = 0    
    # calculate log(\sum_{j=1}^{|W|}X_{ij}!)
    sumXijFactorial = scipy.special.factorial(np.sum(X, axis=1), exact=True).astype('float')
    # shape(1,N)
    log_sumXijFactorial = np.log(sumXijFactorial)
    # calculate \sum_{j=1}^{|W|}log(X_{ij}!)
    logXijFactorial = np.log(scipy.special.factorial(X, exact=True).astype('float'))
    # shape(1,N)
    sum_logXijFactorial = np.sum(logXijFactorial, axis=1)
    for tryTime in range(try_times):
        print('try_time:%d'%(tryTime + 1))
        lambda_k = randomParams(Z)
        Theta = np.zeros((Z, W))
        for k in range(Z):
            Theta[k] = randomParams(W)
        for step in range(max_step):
            print('step:%d'%(step + 1))
            oldTheta = deepcopy(Theta)
            oldLambda_k = deepcopy(lambda_k)
            # E-step
            for i in range(N):
                # calculate \prod_{j=1}^{|W|} theta_{k,j}^{X_ij}
                Pi_theta = np.power(Theta, X[i])
                # calculate P(z_i=k)
                P_z_k = lambda_k * np.prod(Pi_theta, axis=1)
                P_z_k[np.isnan(P_z_k)] = 0
                P_z[i] = P_z_k / np.sum(P_z_k)
            P_z[np.isnan(P_z)] = 0
            P_z[P_z == np.inf] = 0
            P_z[P_z == -np.inf] = 0
            # M-step
            Theta = P_z.T @ X
            sumTheta = np.sum(Theta, axis=1).reshape((Z, 1))
            Theta = Theta / sumTheta
            lambda_k = np.sum(P_z, axis=0) / N
            # check stop condition
            deltaLambda = np.linalg.norm(lambda_k - oldLambda_k) / np.linalg.norm(oldLambda_k)
            deltaTheta = np.linalg.norm(Theta - oldTheta) / np.linalg.norm(oldTheta)
            if (max(deltaLambda, deltaTheta) < epsilon):
                print(step)
                break
        # calculate log(\lambda_k), shape=(1,Z)
        lambda_k[lambda_k == 0] = 1e-300
        log_lambda_k = np.log(lambda_k)
        # calculate \sum_{j=1}^{|W|}X_{ij}log(Theta_{k,j})
        Theta[Theta == 0] = 1e-300
        logTheta = np.log(Theta)
        # shape=(N,Z)
        sum_XijlogTheta = X @ logTheta.T
        # calculate Q-value
        Q_Value += np.sum(P_z @ log_lambda_k)
        Q_Value += np.sum(P_z.T @ log_sumXijFactorial)
        Q_Value -= np.sum(P_z.T @ sum_logXijFactorial)
        Q_Value += np.sum(P_z * sum_XijlogTheta)
        if (tryTime == 0):
            ThetaOut = deepcopy(Theta)
            lambdaOut = deepcopy(lambda_k)
            oldQ_Value = deepcopy(Q_Value)
        else:
            if (Q_Value / oldQ_Value > 1):
                ThetaOut = deepcopy(Theta)
                lambdaOut = deepcopy(lambda_k)
                oldQ_Value = deepcopy(Q_Value)
    return ThetaOut, lambdaOut