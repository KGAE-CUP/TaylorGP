# from feat import Feat
import ffx
import time
import numpy as np
import sys
import argparse
def cal_gplearn_master(fileNum):
    print('fileNum=',fileNum)
    average_fitness = 0
    repeat = 30
    time_start1 = time.time()
    for repeatNum in range(repeat):
        X_Y = np.loadtxt(r"Bench_0.15/BenchMark_" + str(fileNum) + ".tsv",dtype=np.float,skiprows=1)
        # hsplit函数可以水平分隔数组，该函数有两个参数，第 1 个参数表示待分隔的数组， 第 2 个参数表示要将数组水平分隔成几个小数组
        # X,Y=np.hsplit(X_Y,2)
        np.random.shuffle(X_Y)
        X,Y = np.split(X_Y, (-1,), axis=1)
        _split = int(X.shape[0]*0.75)
        train_X = X[:_split]
        train_y = Y[:_split]

        test_X = X[_split:]
        test_y = Y[_split:]

        models = ffx.run(train_X, train_y, test_X, test_y, ["predictor_a", "predictor_b"],verbose=True)
        # print('')
        '''
        FFX = ffx.FFXRegressor()
        FFX.fit(train_X, train_y)
        print("Prediction:", FFX.predict(test_X))
        print("Score:", FFX.score(test_X, test_y))       
        '''

    time_end1 = time.time()
    print('average time cost', (time_end1 - time_start1) / 3600 / repeat, 'hour')
    # print('average_fitness = ',average_fitness/repeat)



if __name__ == '__main__':
    # for i in range(45):
    sys.setrecursionlimit(300)  # 设置求导最大递归深度，防止程序卡主或崩溃
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--fileNum', default=20, type=int)
    args = argparser.parse_args()
    cal_gplearn_master(args.fileNum)
    '''
    numList = []
    for ii in range(19):
        p = multiprocessing.Process(target=cal_gplearn_master, args=(ii,))
        numList.append(p)  # 将子进程对象添加在列表中
        p.start()  # 启动进程
    for i in numList:
        i.join()  # 每一个子进程执行结束后，开始下一个循环，待子进程全部执行完毕，再执行主进程
    print("Process end.")    
    '''
