import numpy as np


def generate_tri(amount):
    triang = np.ones((amount, amount), dtype=int)
    for i in range(2, triang.shape[0]):
        triang[i][1:i] = triang[i - 1][1:i] + triang[i - 1][0:i - 1]
    return triang


def combine(n, m, tri):
    k = min(m, n - m)
    if k > 2:
        return tri[n][min(m, n - m)]
    else:
        sub = 1
        mo = 1
        for i in range(k):
            sub *= n - i
            mo *= i + 1
        return sub / mo


def get_combinatorics(n, lines):
    tri = generate_tri(40)  #
    pre_sum = 0
    cur_sum = combine(n, n - 1, tri)  # init C (n + k - 1 , n - 1)
    cur_k = 1
    ret = np.ones((lines, n), dtype=int)
    for i in range(lines):
        mark = -1
        if i < cur_sum:
            mark = i - pre_sum
        elif i == cur_sum:
            cur_k += 1
            pre_sum = cur_sum
            cur_sum += combine(cur_k + n - 1, n - 1, tri)
            mark = 0
        else:
            print('error')
            # error
        # make cur_k divide into n combox(cur_k + n - 1, n - 1, tri)
        c = np.zeros(n, dtype=int)

        cur_idx = 1
        cp_mk = mark
        for j in range(n - 1):
            while mark >= combine(cur_k + n - 1 - cur_idx, n - 1 - (j + 1), tri):
                mark -= combine(cur_k + n - 1 - cur_idx, n - 1 - (j + 1), tri)
                cur_idx += 1
            c[j + 1] = cur_idx
            cur_idx += 1
        # c[n] = cur_k - np.sum(c[0:n-1])
        res = np.zeros(n, dtype=int)

        res[:-1] = c[1:] - c[:-1] - 1
        # if ( 1 ,2 ,3 ) is selected from
        res[n - 1] = cur_k - np.sum(res[0:n - 1])
        ret[i] = res
    return ret


def get_combinatorics_byk(n, lines, k):
    tri = generate_tri(40)  #
    pre_sum = 0
    cur_sum = combine(n, n - 1, tri)  # init C (n + k - 1 , n - 1)
    totsum = cur_sum
    pre_k = 1
    while pre_k + 1 <= k and totsum + combine(n + pre_k, n - 1, tri) < lines:
        totsum += combine(n + pre_k, n - 1, tri)
        pre_k += 1
    lines = int(min(lines, totsum))
    cur_k = 1
    ret = np.ones((lines, n), dtype=int)
    for i in range(lines):
        mark = -1
        if i < cur_sum:
            mark = i - pre_sum
        elif i == cur_sum:
            cur_k += 1
            pre_sum = cur_sum
            cur_sum += combine(cur_k + n - 1, n - 1, tri)
            mark = 0
        else:
            print('error')
            # error
        # make cur_k divide into n combox(cur_k + n - 1, n - 1, tri)
        c = np.zeros(n, dtype=int)

        cur_idx = 1
        cp_mk = mark
        for j in range(n - 1):
            while mark >= combine(cur_k + n - 1 - cur_idx, n - 1 - (j + 1), tri):
                mark -= combine(cur_k + n - 1 - cur_idx, n - 1 - (j + 1), tri)
                cur_idx += 1
            c[j + 1] = cur_idx
            cur_idx += 1
        # c[n] = cur_k - np.sum(c[0:n-1])
        res = np.zeros(n, dtype=int)

        res[:-1] = c[1:] - c[:-1] - 1
        # if ( 1 ,2 ,3 ) is selected from
        res[n - 1] = cur_k - np.sum(res[0:n - 1])
        ret[i] = res
    return ret
