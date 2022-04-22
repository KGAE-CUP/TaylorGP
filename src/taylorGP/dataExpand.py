import numpy as np
from sklearn.linear_model import LinearRegression

#  前后两个数值之间进行interloop_times次插值，真实数据放在最前
def expand_data_by_nearby(X, Y, function, interloop_times):
    if len(X.shape) != 2:
        #  error
        return None, None

    function.fit(X, Y)

    new_X = np.zeros((X.shape[0] + (X.shape[0] - 1) * interloop_times, X.shape[1]), dtype=X.dtype)
    new_X[0:X.shape[0]] = X
    for i in range(X.shape[0] - 1):
        gap = X[i + 1] - X[i]
        for j in range(interloop_times):
            new_X[j + i * interloop_times + X.shape[0]] = gap * (j + 1) / (interloop_times + 1) + X[i]
    return new_X, function.predict(new_X)

#  每个维度最大最小值之间进行均匀插值，真实数据放在最前，最后扩充到amount_in_need个数据
def expand_data_by_range(X, Y, function, amount_in_need):
    if len(X.shape) != 2:
        #  error
        return None, None
    if X.shape[0] >= amount_in_need:
        return X,Y

    function.fit(X, Y)

    new_X = np.zeros((amount_in_need, X.shape[1]), dtype=X.dtype)
    new_X[0:X.shape[0]] = X
    new_X = new_X.transpose()
    XT = X.transpose()
    Xmax = np.max(XT,axis=1)
    Xmin = np.min(XT,axis=1)
    gap = Xmax - Xmin
    linspace = np.linspace(0,1,amount_in_need - X.shape[0] + 2)
    for i in range(len(Xmin)):
        new_X[i][X.shape[0] : amount_in_need] = gap[i] * linspace[1:-1] + Xmin[i]

    new_X = new_X.transpose()

    return new_X, function.predict(new_X)

if __name__ == '__main__':
    X_Y = np.loadtxt(r"/home/yxgao/sr/comp/TaylorGP/data/example.tsv",dtype=np.float64,skiprows=1)
    # hsplit函数可以水平分隔数组，该函数有两个参数，第 1 个参数表示待分隔的数组， 第 2 个参数表示要将数组水平分隔成几个小数组
    # X,Y=np.hsplit(X_Y,2)
    print(X_Y.shape)
    #np.random.shuffle(X_Y)
    X,Y = np.split(X_Y, (-1,), axis=1)
    X = np.reshape(X,(10,2))
    Y = np.reshape(Y,(10,2))
    m = 100000
    newX, newY = expand_data_by_range(X,Y,LinearRegression(),m)
    print(newX[m-1])