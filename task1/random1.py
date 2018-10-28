import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

poisson_1 = np.random.poisson(100, 1000)
poisson_2 = np.random.poisson(100, 1000)
poisson_3 = np.random.poisson(100, 1000)

poisson = plt.figure()
p_ax1 = poisson.add_subplot(131)
p_ax2 = poisson.add_subplot(132)
p_ax3 = poisson.add_subplot(133)
p_ax1.set_title('sample1')
p_ax2.set_title('sample2')
p_ax3.set_title('sample3')
p_ax1.xlabel('Probability distribution')
p_ax1.ylabel('Number')
poisson.suptitle('Poisson distribution')

p_ax1.hist(poisson_1, bins = 20, edgecolor='black')
p_ax2.hist(poisson_2, bins = 20, edgecolor='black')
p_ax3.hist(poisson_3, bins = 20, edgecolor='black')
poisson.show()

gaussian_1 = np.random.normal(100, 10, 1000)
gaussian_2 = np.random.normal(100, 10, 1000)
gaussian_3 = np.random.normal(100, 10, 1000)

gaussian = plt.figure()
g_ax1 = gaussian.add_subplot(131)
g_ax2 = gaussian.add_subplot(132)
g_ax3 = gaussian.add_subplot(133)
g_ax1.set_title('sample1')
g_ax2.set_title('sample2')
g_ax3.set_title('sample3')
g_ax1.xlabel('Probability distribution')
g_ax1.ylabel('Number')
gaussian.suptitle('Gaussian distribution')

g_ax1.hist(gaussian_1, bins = 20, edgecolor='black')
g_ax2.hist(gaussian_2, bins = 20, edgecolor='black')
g_ax3.hist(gaussian_3, bins = 20, edgecolor='black')
gaussian.show()


poisson_1 = poisson_1.astype(float)
arr1 = np.concatenate((poisson_1, gaussian_1))
fig = plt.figure(figsize=(50,50))
fig.suptitle('Histograms-(Q3)')
for i in range(1, 101):
	arr2 = arr1[np.random.randint(1, 2000, 100)]
	ax = fig.add_subplot(10,10,i)
	ax.hist(arr2, bins=10, color='yellow', edgecolor='black')

fig.show()


