import numpy as np


class EM_clustering:
    def __init__(self, k, dim):
        self.probs = np.ones(k)/k
        self.means = np.random.rand(k, dim)
        self.covs = [np.eye(dim)*100 for _ in range(k)]

    def initialize(self, data):
        range_x = np.max(data[:,0])-np.min(data[:, 0])
        range_y = np.max(data[:,1])-np.min(data[:, 1])
        self.means *= np.array([range_x, range_y])
        self.means += np.array([np.min(data[:, 0]), np.min(data[:, 1])])

    def build_p(self, data):
        k = self.probs.size
        p = np.zeros((k, data.shape[0]))
        for i in range(k):
            a = (data - self.means[i, :][np.newaxis, :])

            p[i, :] =1/(2*np.pi*np.sqrt(np.linalg.det(self.covs[i])))\
                *np.exp(-1/2*np.sum((a@np.linalg.inv(self.covs[i]))*a, 1))*self.probs[i]

                
        p /= np.sum(p, 0)
        return p

    def update(self, data):
        k = self.probs.size
        
        p = self.build_p(data)
        n = np.sum(p, 1)
       
        self.means = (p @ data)/n[:, np.newaxis]

        for i in range(k):
            a = data - self.means[i, :]
            self.covs[i] = (a.T) @ (a*p[i, :][:, np.newaxis]) /n[i]

        self.probs = n.T/data.shape[0]

    def get_distribution(self, data, n_iter):
        
        self.initialize(data)
        for i in range(n_iter):
            self.update(data)

        return self.probs, self.means, self.covs