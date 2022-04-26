import numpy as np
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error  # 均方误差
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel


#  前后两个数值之间进行interloop_times次插值，真实数据放在最前
def expand_data_by_nearby(X, Y, function, interloop_times, need_refit=True):
    if len(X.shape) != 2:
        #  error
        return None, None

    if need_refit:
        function.fit(X, Y)

    new_X = np.zeros((X.shape[0] + (X.shape[0] - 1) * interloop_times, X.shape[1]), dtype=X.dtype)
    new_X[0:X.shape[0]] = X
    for i in range(X.shape[0] - 1):
        gap = X[i + 1] - X[i]
        for j in range(interloop_times):
            new_X[j + i * interloop_times + X.shape[0]] = gap * (j + 1) / (interloop_times + 1) + X[i]
    return new_X, function.predict(new_X)


#  每个维度最大最小值之间进行均匀插值，真实数据放在最前，最后扩充到amount_in_need个数据
def expand_data_by_range(X, Y, function, amount_in_need, need_refit=True):
    if len(X.shape) != 2:
        #  error
        return None, None
    if X.shape[0] >= amount_in_need:
        return X, Y
    if need_refit:
        function.fit(X, Y)

    new_X = np.zeros((amount_in_need, X.shape[1]), dtype=X.dtype)
    new_X[0:X.shape[0]] = X
    new_X = new_X.transpose()
    XT = X.transpose()
    Xmax = np.max(XT, axis=1)
    Xmin = np.min(XT, axis=1)
    gap = Xmax - Xmin
    linspace = np.linspace(0, 1, amount_in_need - X.shape[0] + 2)
    for i in range(len(Xmin)):
        new_X[i][X.shape[0]: amount_in_need] = gap[i] * linspace[1:-1] + Xmin[i]

    new_X = new_X.transpose()

    return new_X, function.predict(new_X)


def choose_func_und_expand(X, Y, choosing_rate, amound_in_need):
    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    kernels_gauss = [C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2)), DotProduct() + WhiteKernel(), None]
    functions = [LinearRegression(), XGBRegressor(), RandomForestRegressor(),  # SVR(),
                 KernelRidge(kernel=kernels[0]), KernelRidge(kernel=kernels[1]),
                 KernelRidge(kernel=kernels[2]), KernelRidge(kernel=kernels[3]),
                 GaussianProcessRegressor(kernel=kernels_gauss[0], n_restarts_optimizer=9),
                 GaussianProcessRegressor(kernel=kernels_gauss[1], n_restarts_optimizer=9),
                 GaussianProcessRegressor(kernel=kernels_gauss[2], n_restarts_optimizer=9)]
    result = -1
    fitness = float('inf')
    for i in range(len(functions)):
        choose_amount = int(X.shape[0] * choosing_rate)
        choose_amount = max(0, choose_amount)
        choose_amount = min(choose_amount, X.shape[0])
        functions[i].fit(X[0: choose_amount], Y[0:choose_amount])
        cur_fitness = mean_squared_error(functions[i].predict(X), Y, squared=False)
        if cur_fitness < fitness:
            result = i
            fitness = cur_fitness
    return expand_data_by_range(X, Y, functions[result], amound_in_need, need_refit=False)


if __name__ == '__main__':
    X_Y = np.loadtxt(r"/home/yxgao/sr/comp/TaylorGP/data/example.tsv", dtype=np.float64, skiprows=1)
    # hsplit函数可以水平分隔数组，该函数有两个参数，第 1 个参数表示待分隔的数组， 第 2 个参数表示要将数组水平分隔成几个小数组
    # X,Y=np.hsplit(X_Y,2)
    print(X_Y.shape)
    # np.random.shuffle(X_Y)
    X, Y = np.split(X_Y, (-1,), axis=1)
    X = np.reshape(X, (10, 2))
    Y = np.reshape(Y, (10, 2))
    m = 100000
    newX, newY = choose_func_und_expand(X, Y, 1, m)
    print(newX[m - 1])
