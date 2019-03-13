import numpy as np
import time
from sklearn import linear_model, datasets
import matplotlib.pyplot as plt

# 产生数据样本点集合
# 样本点的特征X维度为1维，输出y的维度也为1维
# 输出是在输入的基础上加入了高斯噪声N（0,10）
# 产生的样本点数目为1000个

n_samples = 1000
X, y, coef = datasets.make_regression(n_samples=n_samples,
                                      n_features=1,
                                      n_informative=1,
                                      noise=10,
                                      coef=True,
                                      random_state=0)

# 将上面产生的样本点中的前50个设为异常点（外点）
# 即：让前50个点偏离原来的位置，模拟错误的测量带来的误差
n_outliers = 50
np.random.seed(int(time.time()) % 100)
X[:n_outliers] = 3 + 0.5 * np.random.normal(size=(n_outliers, 1))
y[:n_outliers] = -3 + 0.5 * np.random.normal(size=n_outliers)

# 用普通线性模型拟合X，y
model = linear_model.LinearRegression()
model.fit(X, y)

# 使用RANSAC算法拟合X,y
model_ransac = linear_model.RANSACRegressor(linear_model.LinearRegression())
model_ransac.fit(X, y)
inlier_mask = model_ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)

# 使用一般回归模型和RANSAC算法分别对测试数据做预测
line_X = np.arange(-5, 5)
line_y = model.predict(line_X[:, np.newaxis])
line_y_ransac = model_ransac.predict(line_X[:, np.newaxis])

print("真实数据参数：", coef)
print("线性回归模型参数：", model.coef_)
print("RANSAC算法参数： ", model_ransac.estimator_.coef_)

plt.plot(X[inlier_mask], y[inlier_mask], '.g', label='Inliers')
plt.plot(X[outlier_mask], y[outlier_mask], '.r', label='Outliers')
plt.plot(line_X, line_y, '-k', label='Linear Regression')
plt.plot(line_X, line_y_ransac, '-b', label="RANSAC Regression")
plt.legend(loc='upper left')
plt.show()