import numpy as np

class KMeans():
    def __init__(self, n_clusters):
        """
        This class implements the traditional KMeans algorithm with hard assignments:

        https://en.wikipedia.org/wiki/K-means_clustering

        The KMeans algorithm has two steps:

        1. Update assignments
        2. Update the means

        While you only have to implement the fit and predict functions to pass the
        test cases, we recommend that you use an update_assignments function and an
        update_means function internally for the class.

        Use only numpy to implement this algorithm.

        Args:
            n_clusters (int): Number of clusters to cluster the given data into.

        """
        self.n_clusters = n_clusters
        self.means = None

    def euclidean_distances(self, X, Y):  #copied from HW2 code
        euclidean = np.zeros((X.shape[0], Y.shape[0]))
        x_ind = 0
        for row_x in X:
            y_ind = 0
            for row_y in Y:
                euclidean[x_ind, y_ind] = np.sqrt((np.square(row_x - row_y)).sum())
                y_ind += 1
            x_ind += 1
        return euclidean

    def update_assignments(self, features, means):
        distances = self.euclidean_distances(features, means)
        assignments = np.argmin(distances, axis=1)
        return assignments

    def update_means(self, features, indices):
        meanarray = np.zeros((self.n_clusters, features.shape[1]))
        samplenum = np.zeros((self.n_clusters,),dtype=int)
        for ind in range(indices.shape[0]):
            meanarray[indices[ind]] += features[ind]
            samplenum[indices[ind]] += 1
        for i in range(self.n_clusters):
            meanarray[i] /= samplenum[i]

        if 0 in samplenum: #empty cluster edge case - if a cluster empty, assign random mean value from given samples to it
            np.random.seed()
            for i in range(self.n_clusters):
                if samplenum[i] == 0:
                    meanarray[i] = features[np.random.choice(features.shape[0],1)]

        self.means = meanarray
        return

    def fit(self, features):
        """
        Fit KMeans to the given data using `self.n_clusters` number of clusters.
        Features can have greater than 2 dimensions.

        Args:
            features (np.ndarray): array containing inputs of size
                (n_samples, n_features).
        Returns:
            None (saves model - means - internally)
        """
        np.random.seed()
        randmeans = np.random.choice(features.shape[0],self.n_clusters) #randomly select k means from features samples
        self.means = features[randmeans]

        newassignments = self.update_assignments(features, self.means) #returns index of self.means that matches cluster of sample
        self.update_means(features, newassignments) #update means of initial clusters
        oldassignments = []
        while np.array_equal(oldassignments, newassignments) == False:
            oldassignments = newassignments
            newassignments = self.update_assignments(features, self.means) #update cluster assignments based on new means
            self.update_means(features, newassignments) #update means based on new cluster assignments


    def predict(self, features):
        """
        Given features, an np.ndarray of size (n_samples, n_features), predict cluster
        membership labels.

        Args:
            features (np.ndarray): array containing inputs of size
                (n_samples, n_features).
        Returns:
            predictions (np.ndarray): predicted cluster membership for each features,
                of size (n_samples,). Each element of the array is the index of the
                cluster the sample belongs to.
        """
        meandist = self.euclidean_distances(features, self.means)

        predictions = np.argmin(meandist, axis=1)

        return predictions