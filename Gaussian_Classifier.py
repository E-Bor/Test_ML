import numpy as np
import matplotlib.pyplot as plt
# The simplest implementation of the Gaussian classifier in binary classification problem

np.random.seed(1)

# initial parameters of distributions of two classes
# red class
correlations = 0.8
dispersion1 = 1.0
Math_expect1 = [0, -3]
covariance_matrix1 = [[dispersion1, dispersion1 * correlations], [dispersion1 * correlations, dispersion1]]
# green class
correlations2 = 0.7
dispersion2 = 2.0
Math_expect2 = [0, 3]
covariance_matrix2 = [[dispersion2, dispersion2 * correlations2], [dispersion2 * correlations2, dispersion2]]

# simulation of the training sample in accordance with the multivariate normal distribution
N = 1000
x1 = np.random.multivariate_normal(Math_expect1, covariance_matrix1, N).T
x2 = np.random.multivariate_normal(Math_expect2, covariance_matrix2, N).T

# Calculation of parameters of mathematical expectation and covariance matrices from the generated sample x1 x2
me1 = np.mean(x1.T, axis=0)
me2 = np.mean(x2.T, axis=0)

a = (x1.T - me1).T
covariance_matrix_exp1 = np.array([[np.dot(a[0], a[0]) / N, np.dot(a[0], a[1]) / N],
                [np.dot(a[1], a[0]) / N, np.dot(a[1], a[1]) / N]])

a = (x2.T - me2).T
covariance_matrix_exp2 = np.array([[np.dot(a[0], a[0]) / N, np.dot(a[0], a[1]) / N],
                [np.dot(a[1], a[0]) / N, np.dot(a[1], a[1]) / N]])

# Gaussian classifier model
Py1, L1 = 0.5, 1  # probability of occurrence of classes and misclassification fines
Py2, L2 = 1 - Py1, 1  # probability of occurrence of classes and misclassification fines

func = lambda x, v, m, l, py: np.log(l * py) - 0.5 * (x - m) @ np.linalg.inv(v) @ (x - m).T - 0.5 * np.log(
    np.linalg.det(v))

x = np.array([0, -1])  # input vector (x, y)

# gaussian classifier
a = np.argmax([func(x, covariance_matrix_exp1, me1, L1, Py1), func(x, covariance_matrix_exp2, me2, L2, Py2)])
print(f"vector in class: {'red' if a == 0 else 'green'}")

# create a graph
plt.figure(figsize=(7, 7))
plt.title(f"Correlations: r1 = {correlations}, r2 = {correlations}")
plt.scatter(x1[0], x1[1], s=10, c="red")    # dots from 1st class
plt.scatter(x2[0], x2[1], s=10, c="green")  # dots from 2nd class
plt.scatter(x[0], x[1], s=40, c="blue")     # our dot
plt.show()