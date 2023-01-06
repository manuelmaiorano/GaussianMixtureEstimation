import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from clustering import EM_clustering


max_size = np.array([150,150])
num_classes = 7
max_a = 10
data_size = 20

#np.random.seed(0)

means =  np.random.rand(num_classes, 2) * max_size
As = np.random.rand(num_classes, 2, 2) * max_a


data = np.zeros((num_classes* data_size, 2))
for i in range(num_classes):
    data[i*data_size:(i+1)*data_size, :] = np.random.randn(data_size, 2) @ As[i]  + means[i]

def confidence_matrix(ax, mean, cov):
    try:
        vars, R = np.linalg.eig(cov)
    except np.linalg.LinAlgError:
        return

    ellipse = Ellipse((0, 0), width=1, height=1, facecolor='none', edgecolor = 'red')

    transf = transforms.Affine2D() \
        .scale(*np.sqrt(vars))\
        .rotate(np.arccos(R[0,0])) \
        .translate(*mean)

    
    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

clust = EM_clustering(num_classes, 2)
probs, means, covs = clust.get_distribution(data, 100)
print(probs, means, covs)

for i in range(num_classes):
    plt.scatter(data[i*data_size:(i+1)*data_size, 0], data[i*data_size:(i+1)*data_size, 1] )

plt.scatter(means[:, 0], means[: ,1])

for i, cov in enumerate(covs):
    confidence_matrix(plt.gca(),means[i, :], cov)

plt.show()


