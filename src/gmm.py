import numpy as np
from src import KMeans
from scipy.stats import multivariate_normal

class GMM():
    def __init__(self, n_clusters, covariance_type):
        """
        This class implements a Gaussian Mixture Model updated using expectation
        maximization.

        The fit and predict functions are implemented for you. To complete the implementation,
        you must implement:
            - _e_step
            - _m_step
            - _log_likelihood

        Args:
            n_clusters (int): Number of Gaussians to cluster the given data into.
            covariance_type (str): Either 'spherical', 'diagonal'. Determines the
                covariance type for the Gaussians in the mixture model.

        """
        self.n_clusters = n_clusters
        allowed_covariance_types = ['spherical', 'diagonal']
        if covariance_type not in allowed_covariance_types:
            raise ValueError(f'covariance_type must be in {allowed_covariance_types}')
        self.covariance_type = covariance_type

        self.means = None
        self.covariances = None
        self.mixing_weights = None
        self.max_iterations = 200

    def fit(self, features):
        """
        Fit GMM to the given data using `self.n_clusters` number of Gaussians.
        Features can have greater than 2 dimensions.

        Args:
            features (np.ndarray): array containing inputs of size
                (n_samples, n_features).
        Returns:
            None (saves model - means, covariances, and mixing weights - internally)
        """
        # 1. Use your KMeans implementation to initialize the means of the GMM.
        kmeans = KMeans(self.n_clusters)
        kmeans.fit(features)
        self.means = kmeans.means

        # 2. Initialize the covariance matrix and the mixing weights
        self.covariances = self._init_covariance(features.shape[-1])

        # 3. Initialize the mixing weights
        self.mixing_weights = np.random.rand(self.n_clusters)
        self.mixing_weights /= np.sum(self.mixing_weights)

        # 4. Compute log_likelihood under initial random covariance and KMeans means.
        prev_log_likelihood = -float('inf')
        log_likelihood = self._overall_log_likelihood(features)

        # 5. While the log_likelihood is increasing significantly, or max_iterations has
        # not been reached, continue EM until convergence.
        n_iter = 0
        while log_likelihood - prev_log_likelihood > 1e-4 and n_iter < self.max_iterations:
            prev_log_likelihood = log_likelihood

            assignments = self._e_step(features)
            self.means, self.covariances, self.mixing_weights = (
                self._m_step(features, assignments)
            )

            log_likelihood = self._overall_log_likelihood(features)
            n_iter += 1

    def predict(self, features):
        """
        Given features, an np.ndarray of size (n_samples, n_features), predict the label
        of each sample (e.g. the index of the Gaussian with the highest posterior for that
        sample).

        Args:
            features (np.ndarray): array containing inputs of size
                (n_samples, n_features).
        Returns:
            predictions (np.ndarray): predicted assigment to each cluster for each sample,
                of size (n_samples,). Each element is which cluster that sample belongs to.
        """
        posteriors = self._e_step(features)
        return np.argmax(posteriors, axis=-1)

    def _e_step(self, features):
        """
        The expectation step in Expectation-Maximization. Given the current class member
        variables self.mean, self.covariance, and self.mixing_weights:
            1. Calculate the log_likelihood of each point under each Gaussian.
            2. Calculate the posterior probability for each point under each Gaussian
            3. Return the posterior probability (assignments).
        
        Arguments:
            features {np.ndarray} -- Features to apply means, covariance, and mixing_weights
                to.

        Returns:
            np.ndarray -- Posterior probabilities to each Gaussian (shape is
                (features.shape[0], self.n_clusters))
        """
        pos_probs = np.zeros((features.shape[0], self.n_clusters))

        for gaussian in range(self.n_clusters): #posterior returns sample size array for each gausian - each column of returned array
            pos_probs[:,gaussian] = self._posterior(features, gaussian)

        return pos_probs

    def _m_step(self, features, assignments):
        """
        Maximization step in Expectation-Maximization. Given the current features and
        assignments, update self.means, self.covariances, and self.mixing_weights. Here,
        you implement the update equations for the means, covariances, and mixing weights.
            1. Update the means with the mu_j update in Slide 24.
            2. Update the mixing_weights with the w_j update in Slide 24
            3. Update the covariance matrix with the sigma_j update in Slide 24.

        Arguments:
            features {np.ndarray} -- Features to update means and covariances, given the
                current assignments.
            assignments {np.ndarray} -- Soft assignments of each point to one of the cluster,
                given by _e_step.

        Returns:
            means -- Updated means
            covariances -- Updated covariances
            mixing_weights -- Updated mixing weights
        """
        responsibilities = assignments.sum(axis=0)

        mixing_weights = responsibilities / features.shape[0]

        means = np.dot(assignments.T, features)
        for column in range(means.shape[1]):
            means[: , column] /= responsibilities.T

        covariances = np.zeros((self.n_clusters, features.shape[1]))
        for g in range(self.n_clusters):
            sub = features - means[g]
            subsqr = np.square(sub)
            prod = subsqr * assignments[:,g][:, None]
            sum = prod.sum(axis=0)
            covariances[g]=sum
        for column in range(covariances.shape[1]):
            covariances[:, column] /= responsibilities.T

        return means, covariances, mixing_weights

    def _init_covariance(self, n_features):
        
        if self.covariance_type == 'spherical':
            return np.random.rand(self.n_clusters)
        elif self.covariance_type == 'diagonal':
            return np.random.rand(self.n_clusters, n_features)

    def _log_likelihood(self, features, k_idx):
        """
        Arguments:
            features {np.ndarray} -- Features to compute multivariate_normal distribution
                on.
            k_idx {int} -- Which Gaussian to use (e.g. use self.means[k_idx], 
                self.covariances[k_idx], self.mixing_weights[k_idx]).

        Returns:
            np.ndarray -- log likelihoods of each feature given a Gaussian.
        """
        logpdf = multivariate_normal.logpdf(features, self.means[k_idx], self.covariances[k_idx])
        logpdf *= self.mixing_weights[k_idx]

        return logpdf

    def _overall_log_likelihood(self, features):
        denom = [
            self._log_likelihood(features, j) for j in range(self.n_clusters)
        ]
        return np.sum(denom)

    def _posterior(self, features, k):
        """
        Computes the posteriors given the log likelihoods for the GMM. Computes
        the posteriors for one of the Gaussians. To get all the posteriors, you have
        to iterate over this function. This function is implemented for you because the
        numerical issues can be tricky. We use the logsumexp trick to make it work (see
        below).

        Arguments:
            features {np.ndarray} -- Numpy array containing data (n_samples, n_features).
            k {int} -- Index of which Gaussian to compute posteriors for.

        Returns:
            np.ndarray -- Posterior probabilities for the selected Gaussian k, of size
                (n_samples,).
        """
        num = self._log_likelihood(features, k)
        denom = np.array([
            self._log_likelihood(features, j)
            for j in range(self.n_clusters)
        ])

        # Below is a useful function for safely computing large exponentials. It's a common
        # machine learning trick called the logsumexp trick:
        #   https://www.xarg.org/2016/06/the-log-sum-exp-trick-in-machine-learning/

        max_value = denom.max(axis=0, keepdims=True)
        denom_sum = max_value + np.log(np.sum(np.exp(denom - max_value), axis=0))
        posteriors = np.exp(num - denom_sum)
        return posteriors
