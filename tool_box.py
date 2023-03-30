import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture


"""
1. 给定障碍物 2 个点，确定 mu sigma 幅值
2. 生成GMM
3. 计算距离：根据GMM的表达式 进行曲线积分
    3.1 start：(x1,y1,z1)  goal: (x2,y2,z2) 
    3.2 z = GMM(x,y)  在障碍物附近的值很大; z1 = z2 = 0
    3.2 计算 曲线积分
"""
def dist_Gaussian(start,goal, mean_list, cov_list, amp):
    step = 10

    x = np.arange(min(start[0], goal[0]),max(start[0], goal[0]), (abs(start[0]-goal[0]))/step)
    y = np.arange(min(start[1], goal[1]),max(start[1], goal[1]), (abs(start[1]-goal[1]))/step)

    dist = 0
    for i in range(step-1):
        z = GaussMixture(np.array([x[i], y[i]]), mean_list, cov_list, amp)
        z_ = GaussMixture(np.array([x[i+1], y[i+1]]), mean_list, cov_list, amp)
        dz = z_ - z
        temp = np.power(x[i+1] - x[i], 2) + np.power(y[i+1] - y[i], 2) + np.power(dz, 2)
        dist += np.sqrt(temp)

    return dist


def GaussMixture(x, mean_list, cov_list, amp):

    k = len(mean_list)
    prob = 0
    for i in range(k):
        prob += (1/k) * Gaussian(x, mean_list[i], cov_list[i])
    return prob * amp

def Gaussian(x,mean,cov):

    """
    这是自定义的高斯分布概率密度函数
    :param x: 输入数据
    :param mean: 均值数组
    :param cov: 协方差矩阵
    :return: x的概率
    """

    dim = np.shape(cov)[0]
    # cov的行列式为零时的措施
    covdet = np.linalg.det(cov + np.eye(dim) * 0.001)
    covinv = np.linalg.inv(cov + np.eye(dim) * 0.001)
    xdiff = (x - mean).reshape((1,dim))
    # 概率密度
    temp1 = np.power(np.power(2*np.pi,dim)*np.abs(covdet),0.5)
    temp2 = np.exp(-0.5*xdiff.dot(covinv).dot(xdiff.T))

    prob = 1.0/temp1 * temp2[0][0]
    return prob

def get_mean_covar(maze, min_sample):
    maze = np.array(maze)
    x = maze[:, 0].reshape(-1, 1)
    y = maze[:, 1].reshape(-1, 1)
    X = np.hstack((x, y))
    model2 = GaussianMixture(n_components=9, covariance_type='spherical')
    model = DBSCAN(eps=0.18, min_samples = min_sample)
    model.fit(X)
    labels = model.labels_
    if filter:
        outx = X[labels != -1]
    else:
        outx = X
    model2.fit(outx)
    covariances = [np.diag([s,s]) for s in model2.covariances_]
    return model2.means_, covariances