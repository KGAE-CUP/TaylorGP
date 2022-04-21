
from sklearn.metrics import mean_squared_error
from sympy import symbols
import numpy as np
import time

from sklearn.gaussian_process import GaussianProcessRegressor,GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
def cal_gplearn_master(fileNum):
    print('fileNum=',fileNum)
    repeatNum = 1
    for ii in range(repeatNum):
        startTime = time.time()
        # np.random.seed(ii)

        x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25, x26, x27, x28, x29 = symbols("x0,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18,x19,x20,x21,x22,x23,x24,x25,x26,x27,x28,x29 ")
        _x = [x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24,
              x25, x26, x27, x28, x29]
        X_Y = np.loadtxt(r"D:\PYcharm_program\Test_everything\ML_DataSet\\" + str(fileNum) + ".tsv", dtype=np.float, skiprows=1)
        # X_Y = np.loadtxt(r"D:\PYcharm_program\Test_everything\AIFeynman\BenchMark_" + str(fileNum) + ".tsv", dtype=np.float,skiprows=1)
        # X_Y = np.loadtxt(r"D:\PYcharm_program\Test_everything\Bench_0.15\BenchMark_" + str(fileNum) + ".tsv", dtype=np.float,skiprows=1)
        # hsplit函数可以水平分隔数组，该函数有两个参数，第 1 个参数表示待分隔的数组， 第 2 个参数表示要将数组水平分隔成几个小数组
        np.random.shuffle(X_Y)
        X, Y = np.split(X_Y, (-1,), axis=1)
        _split = int(X.shape[0] * 0.3)
        print("ML已知百分之100预测剩下百分之0")
        train_X = X
        train_y = Y
        # train_X = X
        # train_y = Y
        test_X = X[_split:]
        test_y = Y[_split:]
        '''
        def f(x):
            """The function to predict."""
            return x * np.sin(x)
        
        # ----------------------------------------------------------------------
        #  First the noiseless case
        X = np.atleast_2d([1., 3., 5., 6., 7., 8.]).T
        
        # Observations
        y = f(X).ravel()
        
        # Mesh the input space for evaluations of the real function, the prediction and
        # its MSE
        x = np.atleast_2d(np.linspace(0, 10, 1000)).T
        '''


        # Instantiate a Gaussian Process model
        kernel = [C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2)),DotProduct() + WhiteKernel(),None]
        '''
        if fileNum in [7,8,10,11,12,13,14,15]:
            for kern in kernel:
                gp = GaussianProcessClassifier(kernel=kern, n_restarts_optimizer=9)

                # Fit to data using Maximum Likelihood Estimation of the parameters
                train_y=train_y.astype('int')
                gp.fit(train_X, train_y)

                # Make the prediction on the meshed x-axis (ask for MSE as well)
                # print(gp.predict(test_X))
                # print("="*30)
                # print(gp.predict_proba(test_X))
                fitness = float('inf')
                fitness = min(mean_squared_error(gp.predict(test_X), test_y, squared=False), fitness)  # RMSE
                print('rmse_fitness: ', fitness)
        else:        
        '''

        for kern in kernel:
            gp = GaussianProcessRegressor(kernel=kern, n_restarts_optimizer=9)

            # Fit to data using Maximum Likelihood Estimation of the parameters
            gp.fit(train_X, train_y)

            # Make the prediction on the meshed x-axis (ask for MSE as well)
            # y_pred, sigma = gp.predict(test_X, return_std=True)
            fitness = float('inf')
            fitness = min(mean_squared_error(gp.predict(test_X), test_y, squared=False), fitness)  # RMSE
            print('rmse_fitness: ', fitness)
        endTime = time.time()
        print(str(fileNum)+".csv "+"cost Time : "+ str(endTime-startTime)+"s")
if __name__ == '__main__':
    for ii in range(17):
        cal_gplearn_master(ii)
#
# # Plot the function, the prediction and the 95% confidence interval based on
# # the MSE
# plt.figure()
# plt.plot(test_X, test_y, 'r:', label=r'$f(x) = x\,\sin(x)$')
# plt.plot(train_X, train_y, 'r.', markersize=10, label='Observations')
# plt.plot(test_X, y_pred, 'b-', label='Prediction')
# plt.fill(np.concatenate([test_X, test_X[::-1]]),
#          np.concatenate([y_pred - 1.9600 * sigma,
#                         (y_pred + 1.9600 * sigma)[::-1]]),
#          alpha=.5, fc='b', ec='None', label='95% confidence interval')
# plt.xlabel('$x$')
# plt.ylabel('$f(x)$')
# plt.ylim(-10, 20)
# plt.legend(loc='upper left')
#
# # ----------------------------------------------------------------------
# # now the noisy case
# X = np.linspace(0.1, 9.9, 20)
# X = np.atleast_2d(X).T
#
# # Observations and noise
# y = f(X).ravel()
# dy = 0.5 + 1.0 * np.random.random(y.shape)
# noise = np.random.normal(0, dy)
# y += noise
#
# # Instantiate a Gaussian Process model
# gp = GaussianProcessRegressor(kernel=kernel, alpha=dy ** 2,
#                               n_restarts_optimizer=10)
#
# # Fit to data using Maximum Likelihood Estimation of the parameters
# gp.fit(X, y)
#
# # Make the prediction on the meshed x-axis (ask for MSE as well)
# y_pred, sigma = gp.predict(x, return_std=True)
#
# # Plot the function, the prediction and the 95% confidence interval based on
# # the MSE
# plt.figure()
# plt.plot(x, f(x), 'r:', label=r'$f(x) = x\,\sin(x)$')
# plt.errorbar(X.ravel(), y, dy, fmt='r.', markersize=10, label='Observations')
# plt.plot(x, y_pred, 'b-', label='Prediction')
# plt.fill(np.concatenate([x, x[::-1]]),
#          np.concatenate([y_pred - 1.9600 * sigma,
#                         (y_pred + 1.9600 * sigma)[::-1]]),
#          alpha=.5, fc='b', ec='None', label='95% confidence interval')
# plt.xlabel('$x$')
# plt.ylabel('$f(x)$')
# plt.ylim(-10, 20)
# plt.legend(loc='upper left')
#
# plt.show()

