import time

from sympy import *
import numpy as np
from sklearn.metrics import mean_squared_error
# import timeout_decorator
import copy
import itertools
from .getCombinatorics import get_combinatorics, get_combinatorics_byk

CountACC = 0.0


def Global():
    global CountACC


x, y, z, v, w, a, b, c, d = symbols("x,y,z,v,w,a,b,c,d")
x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21,x22, x23, x24, x25, x26, x27, x28, x29, x30, x31, x32, x33, x34, x35, x36, x37, x38, x39, x40, x41, x42, \
x43, x44, x45, x46, x47, x48, x49,x50, x51, x52, x53, x54, x55, x56, x57, x58, x59, x60, x61, x62, x63, x64, x65, x66, x67, x68, x69, x70, \
x71, x72, x73, x74, x75, x76, x77, x78, x79,x80, x81, x82, x83, x84, x85, x86, x87, x88, x89, x90, x91, x92, x93, x94, x95, x96, x97, x98, x99, x100 = symbols(
    "x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21,x22, x23, x24, x25, x26, x27, x28, x29, x30, x31, x32, x33, x34, x35, x36, x37, x38, x39, x40, x41, x42,\
      x43, x44, x45, x46, x47, x48, x49,x50, x51, x52, x53, x54, x55, x56, x57, x58, x59, x60, x61, x62, x63, x64, x65, x66, x67, x68, x69, x70,\
      x71, x72, x73, x74, x75, x76, x77, x78, x79,x80, x81, x82, x83, x84, x85, x86, x87, x88, x89, x90, x91, x92, x93, x94, x95, x96, x97, x98, x99, x100 ")

class Point:
    name = 'Select k+1 Points to calculate Taylor Series'

    def __init__(self, in1=0, in2=0, in3=0, in4=0, in5=0, target=0, expansionPoint=2., varNum=1):
        self.varNum = varNum
        self.in1 = in1
        self.in2 = in2
        self.in3 = in3
        self.in4 = in4
        self.in5 = in5
        self.target = target
        self.expansionPoint = expansionPoint

    def __lt__(self, other):
        if self.varNum == 1:
            return abs(self.in1 - self.expansionPoint) < abs(other.in1 - self.expansionPoint)
        elif self.varNum == 2:
            return (self.in1 - self.expansionPoint) ** 2 + (self.in2 - self.expansionPoint) ** 2 < (
                    other.in1 - self.expansionPoint) ** 2 + (other.in2 - self.expansionPoint) ** 2
        elif self.varNum == 3:
            return (self.in1 - self.expansionPoint) ** 2 + (self.in2 - self.expansionPoint) ** 2 + (
                    self.in3 - self.expansionPoint) ** 2 < (other.in1 - self.expansionPoint) ** 2 + (
                           other.in2 - self.expansionPoint) ** 2 + (other.in3 - self.expansionPoint) ** 2
        elif self.varNum == 4:
            return (self.in1 - self.expansionPoint) ** 2 + (self.in2 - self.expansionPoint) ** 2 + (
                    self.in3 - self.expansionPoint) ** 2 + (self.in4 - self.expansionPoint) ** 2 < (
                           other.in1 - self.expansionPoint) ** 2 + (other.in2 - self.expansionPoint) ** 2 + (
                           other.in3 - self.expansionPoint) ** 2 + (other.in4 - self.expansionPoint) ** 2
        elif self.varNum == 5:
            return (self.in1 - self.expansionPoint) ** 2 + (self.in2 - self.expansionPoint) ** 2 + (
                    self.in3 - self.expansionPoint) ** 2 + (self.in4 - self.expansionPoint) ** 2 + (
                           self.in5 - self.expansionPoint) ** 2 < (other.in1 - self.expansionPoint) ** 2 + (
                           other.in2 - self.expansionPoint) ** 2 + (other.in3 - self.expansionPoint) ** 2 + (
                           other.in4 - self.expansionPoint) ** 2 + (other.in5 - self.expansionPoint) ** 2


class Metrics:
    name = 'Good calculator'

    def __init__(self, fileName=0, dataSet=None, model=None, f=None, classNum=8, varNum=1):
        self.model = model
        self.f_taylor = 0
        self.f_low_taylor = 0
        self.fileName = fileName
        self.dataSet = dataSet
        self.classNum = classNum
        self.x, self.y, self.z, self.v, self.w = symbols("x,y,z,v,w")
        self.x0, self.x1, self.x2, self.x3, self.x4 = symbols("x0,x1,x2,x3,x4")
        _x = [x0, x1, x2, x3, x4, x5, x6, x7, x8, x9]
        self._x = _x[:varNum]
        self.count = 0
        self.mediumcount = 0
        self.supercount = 0
        self.count1 = 0
        self.mediumcount1 = 0
        self.supercount1 = 0
        self.count2 = 0
        self.mediumcount2 = 0
        self.supercount2 = 0
        self.tempVector = np.zeros((1, 6, 126))
        self.varNum = varNum
        self.di_jian_flag = False
        self.parity_flag = False
        self.bias = 0.
        self.nmse = float("inf")
        self.low_nmse = float("inf")
        self.mse_log = float("inf")
        self.Y_log = None
        self.b_log = None
        self.Taylor_log = None
        self.f_taylor_log = None
        self.A = None
        self.midpoint = None
        self.Y_left, self.Y_right = None, None
        self.X_left, self.X_right = None, None
        X_Y = dataSet
        self.expantionPoint = copy.deepcopy(X_Y[0])
        self.mmm = X_Y.shape[0] - 1
        np.random.shuffle(X_Y)
        change = True
        for i in range(self.mmm):
            if (X_Y[i] == self.expantionPoint).all():
                X_Y[[i, -1], :] = X_Y[[-1, i], :]
                break

        X, Y = np.split(X_Y, (-1,), axis=1)
        self.X = copy.deepcopy(X)
        _X = []
        len = X.shape[1]
        for i in range(len):
            X, temp = np.split(X, (-1,), axis=1)
            temp = temp.reshape(-1)
            _X.extend([temp])
        _X.reverse()
        self._X, self.X_Y, self.Y = _X, X_Y, Y.reshape(-1)
        self.f0_log, self.Y_log = np.log(X_Y[0][-1]), np.log(abs(self.Y))
        self.b, self.b_log = (self.Y - self.expantionPoint[-1])[:-1], (self.Y_log - self.f0_log)[:-1]
        self.nihe_flag = False
        self._mid_left, self._mid_right = 0, 0
        self._x_left, self._x_right = 0, 0
        try:
            if varNum == 1:
                self.taylor, self.expantionPointa0, self.expantionPointf0, self.X0, self.Y = self._getData_1var()
                self._X = [self.X0]
            elif varNum == 2:
                self.taylor = self._getData_xvar(2)
            elif varNum == 3:
                self.taylor = self._getData_xvar(3)
            elif varNum == 4:
                self.taylor = self._getData_xvar(4)
            elif varNum == 5:
                self.taylor = self._getData_xvar(5)
            elif varNum == 6:
                self.taylor = self._getData_xvar(6)
            else:
                self.taylor = np.array([1] * 10000)
        except BaseException:#防止程序因矩阵非满秩矩阵而报错停止
            print('metrix error')
            self.taylor = np.array([1] * 10000)
        self.f_taylor = self._getTaylorPolynomial(varNum=varNum)

    def _getData_1var(self, k=18, taylorNum=18):
        mmm = self.X.shape[0] - 1
        b = [0.0] * mmm
        X_Y = self.dataSet
        a0 = X_Y[int(X_Y.shape[0] / 2)][0]
        f0 = X_Y[int(X_Y.shape[0] / 2)][1]
        np.random.shuffle(X_Y)
        X, Y = np.hsplit(X_Y, 2)
        for i in range(mmm):  #
            if X[i] == a0:
                X[[i, -1], :] = X[[-1, i], :]  #
                Y[[i, -1], :] = Y[[-1, i], :]  #
                break
        self.X = X
        X, Y = X.reshape(-1), Y.reshape(-1)  #
        self.Y = Y
        A = np.zeros((mmm, mmm))
        for i in range(mmm):
            b[i] = Y[i] - f0

        for i in range(mmm):
            for j in range(mmm):
                A[i][j] = ((X[i] - a0) ** (j + 1))
        Taylor = np.linalg.solve(A, b)
        Taylor = np.insert(Taylor, 0, f0)  #
        return Taylor.tolist()[:taylorNum], a0, f0, X, Y

    # yxGao
    def _getData_xvar(self, n):
        start = time.time()
        mmm = self.X.shape[0] - 1  #
        X = np.zeros((n, mmm))
        for i in range(n):
            X[i] = self._X[i][:mmm] - self.expantionPoint[i]
        # @yxgao taylorNum = ?
        # @yxgao m = ?
        combine_number = get_combinatorics(n, mmm)
        A = np.ones((mmm, mmm))
        for i in range(mmm):
            for j in range(n):
                A[i][0:] *= X[j][0:] ** combine_number[i][n - 1 - j]
        A = A.transpose()
        Taylor = np.linalg.solve(A, self.b)
        # Taylor_log = np.linalg.solve(A, self.b_log)
        Taylor = np.insert(Taylor, 0, self.expantionPoint[-1])  #
        # self.Taylor_log = np.insert(Taylor_log, 0, self.f0_log)[:]
        self.A = A
        end = time.time()
        print("time=", end - start)
        if n == 1:
            TaylorNum = 18
        elif n == 2:
            TaylorNum = 91
        elif n == 3:
            TaylorNum = 455
        elif n == 4:
            TaylorNum = 1820
        else:
            TaylorNum = 6188
        return Taylor.tolist()[:TaylorNum]  # TaylorNum后期再改

    def judge_Low_polynomial(self, lowLine=7, varNum=1):
        return not self.low_nmse > 1e-5

    def _cal_f_taylor_lowtaylor(self, k, taylor_log_flag=False):
        varNum = self.varNum
        if taylor_log_flag:
            Taylor = self.Taylor_log
        else:
            Taylor = self.taylor
        if varNum == 1:
            f = str(Taylor[0])
            for i in range(1, k):
                if Taylor[i] > 0:
                    f += '+' + str(Taylor[i]) + '*' + '(x0-' + str(self.expantionPointa0) + ')**' + str(i)
                elif Taylor[i] < 0:
                    f += str(Taylor[i]) + '*' + '(x0-' + str(self.expantionPointa0) + ')**' + str(i)
            f_taylor = sympify(f)
            f_taylor = f_taylor.expand()
            f_split = str(f_taylor).split()
            if taylor_log_flag == False:
                try:
                    self.bias = float(f_split[-2] + f_split[-1])
                except BaseException:
                    self.bias = 0.
            print(f_taylor)
            return f_taylor, sympify(str(f_taylor).split('*x0**7')[-1])
        else:
            taylorNum = len(Taylor)
            # print("real_taylorNum:",taylorNum)
            f = str(Taylor[0])
            ret = get_combinatorics_byk(varNum, taylorNum, k)
            newRange = min(taylorNum - 1, len(ret))
            print(newRange)
            for i in range(newRange):
                if Taylor[i + 1] > 0:
                    f += '+' + str(Taylor[i + 1])
                    for j in range(varNum):
                        f += '*(x' + str(j) + '-' + str(self.expantionPoint[j]) + ')**' + str(ret[i][varNum - 1 - j])
                elif Taylor[i + 1] < 0:
                    f += str(Taylor[i + 1])
                    for j in range(varNum):
                        f += '*(x' + str(j) + '-' + str(self.expantionPoint[j]) + ')**' + str(ret[i][varNum - 1 - j])
            f_taylor = sympify(f)
            f_taylor = f_taylor.expand()
            # print("f_taylor: ",f_taylor)
            f_split = str(f_taylor).split()
            if taylor_log_flag == False:
                try:
                    self.bias = float(f_split[-2] + f_split[-1])
                except BaseException:
                    self.bias = 0.
            return f_taylor

    def _getTaylorPolynomial(self, varNum=1):
        Taylor = self.taylor
        if varNum == 1:
            self.f_taylor, self.f_low_taylor = self._cal_f_taylor_lowtaylor(k=14)
            y_pred = self._calY(self.f_taylor)
            y_low_pred = self._calY(self.f_low_taylor)
            nmse = mean_squared_error(self.Y, y_pred)
            low_nmse = mean_squared_error(self.Y, y_low_pred)
            print('NMSE of Taylor polynomal：', nmse)
            print('NMSE of Low order Taylor polynomial：', low_nmse)
            self.nmse = nmse
            self.low_nmse = low_nmse
            return self.f_taylor
        else:
            count1 = 2
            count2 = 3
            if self.varNum == 2:
                count1 = 5
                count2 = 9
            elif self.varNum == 3:
                count1 = 4
                count2 = 8
            elif self.varNum == 4:
                count1 = 3
                count2 = 7
            elif self.varNum == 5:
                count1 = 3
                count2 = 6
            elif self.varNum == 6:
                count1 = 3
                count2 = 5
            for k in range(1, count1):
                test_f_k = self._cal_f_taylor_lowtaylor(k)
                test_y_pred = np.array(self._calY(test_f_k))
                test_nmse = mean_squared_error(self.Y, test_y_pred)
                print('NMSE expanded to order k，k=', k, 'nmse=', test_nmse)
                if test_nmse < self.low_nmse:
                    self.low_nmse = test_nmse
                    self.f_low_taylor = test_f_k
            self.nmse = self.low_nmse
            self.f_taylor = self.f_low_taylor
            try:
                for k in range(count1, count2):
                    test_f_k = self._cal_f_taylor_lowtaylor(k)
                    test_y_pred = self._calY(test_f_k)
                    test_nmse = mean_squared_error(self.Y, test_y_pred)
                    print('NMSE expanded to order k，k=', k, 'nmse=', test_nmse)
                    if test_nmse < self.nmse:
                        self.nmse = test_nmse
                        self.f_taylor = test_f_k
            except BaseException:

                print('sympify error')
            try:
                self.f_taylor_log = self._cal_f_taylor_lowtaylor(k=8, taylor_log_flag=True)
            except BaseException:
                self.f_taylor_log = 0
                print('f_taylor_log error')
            # y_pred = self._calY(self.f_taylor)
            # y_low_pred = self._calY(self.f_low_taylor)
            # nmse = mean_squared_error(self.Y, y_pred)
            # low_nmse = mean_squared_error(self.Y, y_low_pred)
            print('NMSE of Taylor polynomal：', self.nmse)
            print('NMSE of Low order Taylor polynomial：', self.low_nmse)
            return self.f_taylor

    def _calY(self, f, _x=None, X=None):
        y_pred = []
        len1, len2 = 0, 0
        if _x is None:
            _x = [x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21,
                  x22, x23, x24, x25, x26, x27, x28, x29,x30, x31, x32, x33, x34, x35, x36, x37, x38, x39, x40, x41, x42, x43, x44, x45, x46, x47, x48, x49,
                  x50, x51, x52, x53, x54, x55, x56, x57, x58, x59, x60, x61, x62, x63, x64, x65, x66, x67, x68, x69, x70, x71, x72, x73, x74, x75, x76, x77, x78, x79,
                  x80, x81, x82, x83, x84, x85, x86, x87, x88, x89, x90, x91, x92, x93, x94, x95, x96, x97, x98, x99,x100 ]
        if X is None:
            X = self._X
            len2 = self.varNum
        else:
            len2 = len(X)
        len1 = X[0].shape[0]
        for i in range(len1):
            _sub = {}
            for j in range(len2):
                _sub.update({_x[j]: X[j][i]})
            y_pred.append(f.evalf(subs=_sub))
        return y_pred

    '''
    @timeout_decorator.timeout(10, use_signals=False)
    def cal_critical_point(self, fx, x):
        if self.varNum == 2:
            return solve([fx[0], fx[1]], [x[0], x[1]])    
    '''

    def judge_Bound(self):
        if self.nihe_flag == False:
            y_bound, var_bound = [], []
            _X = copy.deepcopy(self._X)
            for i in range(len(_X)):
                _X[i].sort()
                var_bound.extend([_X[i][0], _X[i][-1]])
            _Y = self.Y.reshape(-1)
            _Y.sort()
            y_bound.extend([_Y[0], _Y[-1]])
            return [y_bound, var_bound]
        else:
            y_bound, var_bound = [], []
            _X = copy.deepcopy(self._X)
            for i in range(len(_X)):
                _X[i].sort()
                var_bound.extend([_X[i][0], _X[i][-1]])
            Y = copy.deepcopy(self.Y)
            Y.sort()
            y_bound.extend([Y[0], Y[-1]])

            f_diff = []
            for i in range(len(self._X)):
                f_diff.append(sympify(diff(self.f_taylor, self._x[i])))
            '''
            try:
                critical_point = self.cal_critical_point(f_diff, self._x[:len(self._X)])
            except BaseException:
                critical_point = None
            if critical_point is not None:
                for c in critical_point:
                    if 'I' not in str(c) and not any(
                            [c[0] < var_bound[[i][0]] and c[1] > var_bound[i][1] for i in range(len(c))]):
                        _sub = {}
                        for i in range(len(c)):
                            _sub.update({self._x[i]: c[i]})
                        y_bound.append(self.f_taylor.evalf(subs=_sub))
                        print('Critical Point', c)            
            '''

            y_bound.sort()
            return [[y_bound[0], y_bound[-1]], var_bound]

    def judge_monotonicity(self, Num=1):
        Increase, Decrease = False, False
        X, Y = copy.deepcopy(self.X), copy.deepcopy(self.Y)
        Y_index = np.argsort(Y, axis=0)
        Y_index = Y_index.reshape(-1)
        for i in range(1, Y_index.shape[0]):
            Increase_flag = not any([(X[Y_index[i]][j] < X[Y_index[i - 1]][j]) for j in range(X.shape[1])])
            if Increase_flag:
                Increase = True
            Decrease_flag = not any([(X[Y_index[i]][j] > X[Y_index[i - 1]][j]) for j in range(X.shape[1])])
            if Decrease_flag:
                Decrease = True
        if Increase == True and Decrease == False:
            if Num == 1:
                print('Monotone increasing function！！！')
            else:
                print('concave function')
            return 1
        elif Increase == False and Decrease == True:
            if Num == 1:
                print('Monotone decreasing function！！！')
            else:
                print('convex function')
            return 2
        if Num == 1:
            print('Non increasing and non decreasing function！！！')
        else:
            print(' concavity and convexity')
        return -1

    def judge_program_monotonicity(self, Num=1):  # 适合任意一维和多维的情况
        Increase, Decrease = False, False
        f_ = diff(self.f_taylor, self.x, Num)  # 求f的Num阶导数
        for x_ in self.X0:
            if f_.evalf(subs={x: x_}) >= 0:
                Increase = True
            if f_.evalf(subs={x: x_}) <= 0:
                Decrease = True
        if Increase == True and Decrease == False:
            if Num == 1:
                print('Monotone increasing function！！！')
            else:
                print('concave function')
            return 1
        elif Increase == False and Decrease == True:
            if Num == 1:
                self.di_jian_flag = True
                print('Monotone decreasing function！！！')
            else:
                print('convex function')
            return 2
        if Num == 1:
            print('Non increasing non decreasing function！！！')
        else:
            print('no concavity and convexity')
        return -1

    def judge_concavityConvexity(self):
        return self.judge_monotonicity(Num=2)

    def cal_power_expr(self, expr):
        expr = expr.split('*')
        j = 0
        for i in range(len(expr)):
            if expr[j] == '':
                expr.pop(j)
            else:
                j += 1
        count = 0
        for i in range(1, len(expr) - 1):
            if 'x' in expr[i] and expr[i + 1].isdigit():
                count += int(expr[i + 1])
            elif 'x' in expr[i] and 'x' in expr[i + 1]:
                count += 1
        if 'x' in expr[-1]:
            count += 1
        return count

    def judge_parity(self):
        '''
                odd function：1
                even function：2
        '''
        print("奇偶性判别")
        if self.nihe_flag:
            if self.bias != 0:
                f_taylor = str(self.f_taylor).split()[:-2]
            else:
                f_taylor = str(self.f_taylor).split()
            Y = self.Y - self.bias
            f_odd, f_even = '', ''
            print(self.f_taylor)
            print(f_taylor)
            if self.cal_power_expr(f_taylor[0]) % 2 == 1:
                f_odd += f_taylor[0]
            else:
                f_even += f_taylor[0]
            for i in range(2, len(f_taylor), 2):
                if self.cal_power_expr(f_taylor[i - 1] + f_taylor[i]) % 2 == 1:
                    f_odd += f_taylor[i - 1] + f_taylor[i]
                else:
                    f_even += f_taylor[i - 1] + f_taylor[i]
            if f_even == '':
                f_even = '0'
            elif f_odd == '':
                f_odd = '0'
            f_odd, f_even = sympify(f_odd), sympify(f_even)
            Jishu, Oushu, nmse_odd, nmse_even = False, False, 0, 0
            y_pred = self._calY(f_odd)
            nmse_odd = mean_squared_error(Y, y_pred)
            y_pred = self._calY(f_even)
            nmse_even = mean_squared_error(Y, y_pred)
            print('NMSE of parity function', nmse_odd, nmse_even, sep='\n')
            if nmse_odd < 0.01:
                Jishu = True
            if nmse_even < 0.01:
                Oushu = True
            if Jishu == True and Oushu == False:
                print('odd function！！！')
                self.parity_flag = True
                return 1
            elif Jishu == False and Oushu == True:
                self.parity_flag = True
                print('even function！！！')
                return 2
        print('non odd non even function！！！')
        return -1

    def judge_program_parity(self):
        Jishu, Oushu = False, False
        f = self.f_taylor
        for x_ in self.X0:
            if abs(f.evalf(subs={x: -1 * x_}) + f.evalf(subs={x: x_})) < 0.001:
                Jishu = True
            elif abs(f.evalf(subs={x: -1 * x_}) - f.evalf(subs={x: x_})) < 0.001:
                Oushu = True
            else:
                print('non odd non even function！！！')
                return -1
        if Jishu == True and Oushu == False:
            print('odd function！！！')
            return 1
        elif Jishu == False and Oushu == True:
            print('even function！！！')
            return 2

    def _cal_add_separability(self, multi_flag=False, expantionpoint=None):
        f_taylor = str(copy.deepcopy(self.f_taylor))
        for i in range(len(self._x_right)):
            f_taylor = f_taylor.replace(str(self._x_right[i]), str(self._mid_right[i])) + '-(' + str(self.bias) + ')'
        self.f_left_taylor = f_taylor
        f_taylor = str(copy.deepcopy(self.f_taylor))
        for i in range(len(self._x_left)):
            f_taylor = f_taylor.replace(str(self._x_left[i]), str(self._mid_left[i]))
        self.f_right_taylor = f_taylor

        self.f_left_taylor = sympify(self.f_left_taylor)
        self.f_right_taylor = sympify(self.f_right_taylor)
        Y_left = self._calY(self.f_left_taylor, self._x_left, self._X_left)
        Y_right = self._calY(self.f_right_taylor, self._x_right, self._X_right)
        if multi_flag:
            f_taylor = str(copy.deepcopy(self.f_taylor))
            for i in range(len(self._x)):
                f_taylor = f_taylor.replace(str(self._x[i]), str(expantionpoint[i]))
            self.f_mid = eval(f_taylor)
        len_Y = len(Y_left)
        Y_left, Y_right = np.array(Y_left), np.array(Y_right)
        self.Y_left, self.Y_right = Y_left.reshape(len_Y, 1), Y_right.reshape(len_Y, 1)
        return None

    def judge_additi_separability(self, f_taylor=None, taylor_log_flag=False):
        for i in range(len(self._x) - 1):
            _x = copy.deepcopy(self._x)
            _x_left = _x.pop(i)
            _x_right = _x
            if f_taylor == None:
                f_taylor = self.f_taylor
            f_taylor_split = str(f_taylor).split()
            f_temp = []
            for subF in f_taylor_split:
                if any([str(_x_left) in subF and str(_x_right[j]) in subF for j in range(len(_x_right))]):
                    f_temp = f_temp[:-1]
                else:
                    f_temp.append(subF)

            f = ''
            for temp in f_temp:
                f += temp
            f = sympify(f)
            y_pred = self._calY(f, self._x, self._X)
            if taylor_log_flag:
                nmse = mean_squared_error(self.Y_log, y_pred)
            else:
                nmse = mean_squared_error(self.Y, y_pred)
            if nmse > 0.001:
                continue
            else:
                print('addition separable')
                return True

    def judge_multi_separability(self):
        '''multiplicative separability discrimination'''
        _expantionpoint = copy.deepcopy(self.expantionPoint)
        if self.expantionPoint[-1] == 0:
            _expantionpoint = _expantionpoint + 0.2
        for i in range(len(self._x) - 1):
            f_taylor = self.f_taylor
            _x = copy.deepcopy(self._x)
            self._x_left = [_x[i]]
            _x.pop(i)
            self._x_right = _x
            expantionpoint = copy.deepcopy(_expantionpoint).tolist()
            self._mid_left = [expantionpoint[i]]
            expantionpoint.pop(i)
            self._mid_right = expantionpoint[:-1]
            _X = copy.deepcopy(self._X)
            self._X_left = [_X.pop(i)]
            try:
                a = _X[0].shape[0]
            except BaseException:
                _X = [_X]
            self._X_right = _X
            self._cal_add_separability(multi_flag=True, expantionpoint=expantionpoint)
            nmse = mean_squared_error(self.Y, self.Y_left * self.Y_right / self.f_mid)
            if nmse < 0.001:
                print('multiplication separable')
                print('multi_nmse = ', nmse)
                return True
            else:
                try:
                    return self.judge_additi_separability(f_taylor=self.f_taylor_log, taylor_log_flag=True)
                except BaseException:
                    return False

        return False

    def change_Y(self, Y):
        if Y is None:
            return None
        if self.parity_flag:
            if abs(self.bias) > 1e-5:
                Y -= self.bias
        if self.di_jian_flag:
            return Y * (-1)
        else:
            return Y


class Metrics2(Metrics):

    def __init__(self, f_taylor, _x, X, Y):
        self.f_taylor = f_taylor
        self.f_low_taylor = None
        self.x0, self.x1, self.x2, self.x3, self.x4 = symbols("x0,x1,x2,x3,x4")
        self._x = _x
        self.bias, self.low_nmse = 0., 0.
        self.varNum = X.shape[1]
        self.Y_left, self.Y_right, self.Y_right_temp = None, None, None
        self.X_left, self.X_right = None, None
        self.midpoint = None
        self.parity_flag = False
        self.di_jian_flag = False
        self.expantionPoint = np.append(copy.deepcopy(X[0]), Y[0][0])
        self.X = copy.deepcopy(X)
        _X = []
        len = X.shape[1]
        for i in range(len):
            X, temp = np.split(X, (-1,), axis=1)
            temp = temp.reshape(-1)
            _X.extend([temp])
        _X.reverse()
        self._X, self.Y = _X, Y.reshape(-1)
        self.b = (self.Y - self.expantionPoint[-1])[:-1]
        y_pred = self._calY(f_taylor, self._x, self._X)
        self.nihe_flag = False
        if mean_squared_error(self.Y, y_pred) < 0.01:
            self.nihe_flag = True
        self._mid_left, self._mid_right = 0, 0
        self._x_left, self._x_right = 0, 0

    def judge_Low_polynomial(self):
        f_taylor = str(self.f_taylor).split()
        try:
            self.bias = float(f_taylor[-2] + f_taylor[-1])
        except BaseException:
            self.bias = 0.
        f_low_taylor = ''
        if self.cal_power_expr(f_taylor[0]) <= 4:
            f_low_taylor += f_taylor[0]
        for i in range(2, len(f_taylor), 2):
            if self.cal_power_expr(f_taylor[i - 1] + f_taylor[i]) <= 4:
                f_low_taylor += f_taylor[i - 1] + f_taylor[i]
        self.f_low_taylor = sympify(f_low_taylor)
        print(f_low_taylor)
        y_pred_low = self._calY(self.f_low_taylor, self._x, self._X)
        self.low_nmse = mean_squared_error(self.Y, y_pred_low)
        if self.low_nmse < 1e-5:
            return True
        else:
            return False

    def judge_Bound(self):
        y_bound, var_bound = [], []
        _X = copy.deepcopy(self._X)
        for i in range(len(_X)):
            _X[i].sort()
            var_bound.extend([_X[i][0], _X[i][-1]])
        _Y = self.Y.reshape(-1)
        _Y.sort()
        y_bound.extend([_Y[0], _Y[-1]])
        return [y_bound, var_bound]

    def change_XToX(self, _X):
        len1 = len(_X)
        len2 = len(_X[0])
        X = np.array(_X[0])
        X = X.reshape(len(_X[0]), 1)
        for i in range(1, len1):
            temp = np.array(_X[i]).reshape(len2, 1)
            X = np.concatenate((X, temp), axis=1)
        return X

    def _cal_add_separability(self, multi_flag=False, expantionpoint=None):
        f_taylor = str(copy.deepcopy(self.f_taylor))
        for i in range(len(self._x_right)):
            f_taylor = f_taylor.replace(str(self._x_right[i]), str(self._mid_right[i])) + '-(' + str(self.bias) + ')'
        self.f_left_taylor = f_taylor
        f_taylor = str(copy.deepcopy(self.f_taylor))
        for i in range(len(self._x_left)):
            f_taylor = f_taylor.replace(str(self._x_left[i]), str(self._mid_left[i]))
        self.f_right_taylor = f_taylor

        self.f_left_taylor = sympify(self.f_left_taylor)
        self.f_right_taylor = sympify(self.f_right_taylor)
        Y_left = self._calY(self.f_left_taylor, self._x_left, self._X_left)
        Y_right = self._calY(self.f_right_taylor, self._x_right, self._X_right)
        self.X_left = self.change_XToX(self._X_left)
        self.X_right = self.change_XToX(self._X_right)
        if multi_flag:
            f_taylor = str(copy.deepcopy(self.f_taylor))
            for i in range(len(self._x)):
                f_taylor = f_taylor.replace(str(self._x[i]), str(expantionpoint[i]))
            self.f_mid = eval(f_taylor)
        len_Y = len(Y_left)
        Y_left, Y_right = np.array(Y_left), np.array(Y_right)
        Y = self.Y.reshape(len_Y, 1)
        self.Y_left = Y_left.reshape(len_Y, 1)
        try:
            if multi_flag:
                self.Y_right = Y_right.reshape(len_Y, 1)
                self.Y_right_temp = Y / self.Y_left
            else:
                self.Y_right = Y - self.Y_left
        except BaseException:
            self.Y_right_temp = Y_right.reshape(len_Y, 1)
        return None

    def judge_additi_separability(self, f_taylor=None, taylor_log_flag=False):
        '''additive separability discrimination'''
        for i in range(len(self._x) - 1):
            _x = copy.deepcopy(self._x)
            _x_left = _x.pop(i)
            _x_right = _x
            if f_taylor == None:
                f_taylor = self.f_taylor
            f_taylor_split = str(f_taylor).split()
            f_temp = []
            for subF in f_taylor_split:
                if any([str(_x_left) in subF and str(_x_right[j]) in subF for j in range(len(_x_right))]):
                    f_temp = f_temp[:-1]
                else:
                    f_temp.append(subF)
            f = ''
            for temp in f_temp:
                f += temp
            f = sympify(f)
            y_pred = self._calY(f, self._x, self._X)
            if taylor_log_flag:
                nmse = mean_squared_error(self.Y_log, y_pred)
            else:
                nmse = mean_squared_error(self.Y, y_pred)
            if nmse > 0.001:
                continue
            else:
                _x = copy.deepcopy(self._x)
                self._x_left = [_x[i]]
                _x.pop(i)
                self._x_right = _x
                expantionpoint = copy.deepcopy(self.expantionPoint).tolist()
                self._mid_left = [expantionpoint[i]]
                expantionpoint.pop(i)
                self._mid_right = expantionpoint[:-1]
                _X = copy.deepcopy(self._X)
                self._X_left = [_X.pop(i)]
                try:
                    a = _X[0].shape[0]
                except BaseException:
                    _X = [_X]
                self._X_right = _X
                self._cal_add_separability()
                print('addition separable')
                return True

    def judge_multi_separability(self):
        '''multiplicative separability discrimination'''
        _expantionpoint = copy.deepcopy(self.expantionPoint)
        if self.expantionPoint[-1] == 0:
            _expantionpoint = _expantionpoint + 0.2
        for i in range(len(self._x) - 1):
            f_taylor = self.f_taylor
            _x = copy.deepcopy(self._x)
            self._x_left = [_x[i]]
            _x.pop(i)
            self._x_right = _x
            expantionpoint = copy.deepcopy(_expantionpoint).tolist()
            self._mid_left = [expantionpoint[i]]
            expantionpoint.pop(i)
            self._mid_right = expantionpoint[:-1]
            _X = copy.deepcopy(self._X)
            self._X_left = [_X.pop(i)]
            try:
                a = _X[0].shape[0]
            except BaseException:
                _X = [_X]
            self._X_right = _X
            self._cal_add_separability(multi_flag=True, expantionpoint=expantionpoint)
            nmse = mean_squared_error(self.Y, self.Y_left * self.Y_right / self.f_mid)
            if nmse < 0.001:
                print('multiplication separable')
                print('multi_nmse = ', nmse)
                self.Y_right = self.Y_right_temp
                return True
            else:
                try:
                    return self.judge_additi_separability(f_taylor=self.f_taylor_log, taylor_log_flag=True)
                except BaseException:
                    return False

        return False

    def judge_parity(self):
        '''
        return：non odd non even function：-1
                odd function：1
                even function：2
        '''
        if self.nihe_flag:
            if self.bias != 0:
                f_taylor = str(self.f_taylor).split()[:-2]
            else:
                f_taylor = str(self.f_taylor).split()
            Y = self.Y - self.bias
            f_odd, f_even = '', ''
            if self.cal_power_expr(f_taylor[0]) % 2 == 1:
                f_odd += f_taylor[0]
            else:
                f_even += f_taylor[0]
            for i in range(2, len(f_taylor), 2):
                if self.cal_power_expr(f_taylor[i - 1] + f_taylor[i]) % 2 == 1:
                    f_odd += f_taylor[i - 1] + f_taylor[i]
                else:
                    f_even += f_taylor[i - 1] + f_taylor[i]
            f_odd, f_even = sympify(f_odd), sympify(f_even)
            Jishu, Oushu, nmse_odd, nmse_even = False, False, 0, 0
            y_pred = self._calY(f_odd, self._x, self._X)
            nmse_odd = mean_squared_error(Y, y_pred)
            y_pred = self._calY(f_even, self._x, self._X)
            nmse_even = mean_squared_error(Y, y_pred)
            print('NMSE of parity function', nmse_odd, nmse_even, sep='\n')
            if nmse_odd < 0.001:
                Jishu = True
            if nmse_even < 0.001:
                Oushu = True
            if Jishu == True and Oushu == False:
                print('Odd function！！！')
                self.parity_flag = True
                return 1
            elif Jishu == False and Oushu == True:
                self.parity_flag = True
                print('even function！！！')
                return 2
        print('non odd non even function！！！')
        return -1


def cal_Taylor_features(varNum, dataSet, Y=None):
    '''qualified_list = [low_high_target_bound, low_high_var_bound,bias,partity,monity]'''
    qualified_list = []
    low_polynomial = False
    loopNum = 0
    Metric = []
    while True:
        metric = Metrics(varNum=varNum, dataSet=dataSet)
        loopNum += 1
        Metric.append(metric)
        if loopNum == 1:
            break
    Metric.sort(key=lambda x: x.nmse)
    metric = Metric[0]
    print('NMSE of polynomial and lower order polynomial after sorting', metric.nmse, metric.low_nmse)
    if metric.nmse < 0.1:
        metric.nihe_flag = True
    else:
        print('Fitting failed')
    if metric.judge_Low_polynomial():
        print('The result is a low order polynomial')
        low_polynomial = True

    '''
    add_seperatity = metric.judge_additi_separability()
    multi_seperatity = metric.judge_multi_separability()

    qualified_list.extend(metric.judge_Bound()) 
    # qualified_list.extend([1,1,1,1])
    qualified_list.append(metric.f_low_taylor)
    qualified_list.append(metric.low_nmse) 
    qualified_list.append(metric.bias)  
    qualified_list.append(metric.judge_parity())
    qualified_list.append(metric.judge_monotonicity())
    # qualified_list.append(metric.di_jian_flag)
    print('qualified_list = ',qualified_list)
    # X,Y = metric.X, metric.change_Y(Y)
    return metric.nihe_flag,low_polynomial,qualified_list,metric.change_Y(Y)     
    '''


if __name__ == '__main__':
    Global()
    # fileName = "D:\PYcharm_program\Test_everything\Bench_0.15\BenchMark_44.tsv"
    fileName = "example.tsv"
    X_Y = np.loadtxt(fileName, dtype=np.float, skiprows=1)
    for i in [1]:
        # cal_Taylor_features(varNum=2, fileName="example.tsv")
        cal_Taylor_features(varNum=2, dataSet=X_Y)
