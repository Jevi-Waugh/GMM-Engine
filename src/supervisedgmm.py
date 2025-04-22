from gmm import GuassianMixtureModel
import numpy as np
from scipy.stats import multivariate_normal

class SupervisedGMM(GuassianMixtureModel):
    
    def fit(self, X, Y) -> None:
        """
        Estimate the GMM parameters (means 𝝁, covariances 𝚺, priors 𝜋) for each class m using labeled data.

  
        Args:
            X :ndarray of shape (n_samples, n_features)
                Our data which is a matrix where each row is a data point and each
                column corresponds to features
                
            Y : ndarray of shape (n_samples,)
                The class label for each data point in 'X'. Each unique value 
                in 'Y' is treated as a seperate Guassian component
        
        Returns
        None
            This method sets the attributes (They are all lists of the same length):
            - self.means:  ndarray with the shape (n_components, n_features)
                 Mean vector for each class.
            - self.covariances:  ndarray with the shape (n_components, n_features, n_features)
                One covariance matrix per class.
            - self.priors:  ndarray with the shape (n_components,)
                Prior probability for each class.
        """
        # This is our theta θ
        self.means, self.covariances, self.priors = [],[],[]
        # Find all unique label in our Y 
        unique_labels = np.unique(Y)
        for unique_class in unique_labels:
            # Boolean indexing to select specific data based on their class
            X_group = X[Y == unique_class]
            mu_m = np.mean(X_group, axis=0)
            self.means.append(mu_m)
            # It's tempting to do -> self.covariances = np.cov(X_group, rowvar=False)
            # But lets's do it according to our mathematical definition of covariance matrix for each class
            n_m = len(X_group)
            # We reshape to get a column vector
            self.covariances.append(sum((x_i - mu_m).reshape(-1,1) @ (x_i - mu_m).reshape(-1,1).T for x_i in unique_class) / n_m)
            self.priors.append(len(X_group) / len(X))
            
            # Convert to numpy datatype for efficiency
            self.means = np.array(self.means)
            self.covariances = np.array(self.covariances)
            self.priors = np.array(self.priors)
    
    def predict(self, X) -> list:
        """
        Predict the class labels for a set of input data points using the trained GMM

        Args:
            X :ndarray of shape (n_samples, n_features)
                Our data which is a matrix where each row is a data point and each
                column corresponds to features

        Returns:
            y_pred : ndarray of shape (n_samples,)
                The predicted class label for each data point in `X`. Each label corresponds
                to the Gaussian component with the highest posterior probability:
        """
        # We create a likelihood matrix that contains the likelihood of eaxh x for every class m
        likelihood = np.zeros((len(X), self.n_components))
        for m in range(self.n_components):
            # Now we invoke our unormalised posterior probability
            likelihood[:, m] = self.priors[m] * multivariate_normal.pdf(X,self.means[m], self.covariances[m])
        
        # Now we retrieve the predicted class index per data point
        return np.argmax(likelihood, axis=1)
        