# Gaussian Mixture Models (GMM): Theory & Implementation

## Project Overview

This project will explore Gaussian Mixture Models (GMM) and their application in supervised and unsupervised learning. 
It includes a theoretical explanation, an implementation from scratch, and a comparison with scikit-learn’s GMM.

### Supervised Learning
### Goal

Estimate parameters $\theta = \{ \mu_m, \Sigma_m, \pi_m \}$ for each class $m$ using labeled data.

---

### Mean Vector

The mean of class $m$ is:

$$
\mu_m = \frac{1}{n_m} \sum_{i : y_i = m} x_i
$$

---

### Covariance Matrix

$$
\Sigma_m = \frac{1}{n_m} \sum_{i: y_i = m} (x_i - \mu_m)(x_i - \mu_m)^T
$$
sum((x_i - mu_m).reshape(-1,1) @ (x_i - mu_m).reshape(-1,1).T for x_i in unique_class) / n_m
---

### Prior Probability

$$
\pi_m = \frac{n_m}{n}
$$

---

### Prediction Rule

$$
\hat{y}_i = \arg\max_m \; \pi_m \cdot \mathcal{N}(x_i \mid \mu_m, \Sigma_m)
$$

<pre> ```python # The GMM prediction likelihoods = np.zeros((len(X), self.n_components)) for k in range(self.n_components): likelihoods[:, k] = self.priors[k] * multivariate_normal.pdf( X, mean=self.means[k], cov=self.covariances[k] ) y_pred = np.argmax(likelihoods, axis=1) ``` </pre>
