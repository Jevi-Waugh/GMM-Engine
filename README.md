# Gaussian Mixture Models (GMM): Theory & Implementation

## Project Overview

This project will explore Gaussian Mixture Models (GMM) and their application in supervised and unsupervised learning. 
It includes a theoretical explanation, an implementation from scratch, and a comparison with scikit-learn’s GMM.

### Project Update - 22 April 2024
I have starting working on supervised learning after working through the mathematical derivations and likelihood. The supervised classifier has been coded
and I just need to create some synthetic data and plot it with its contours.

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

### Prior Probability

$$
\pi_m = \frac{n_m}{n}
$$

---

### Covariance Matrix

$$
\Sigma_m = \frac{1}{n_m} \sum_{i: y_i = m} (x_i - \mu_m)(x_i - \mu_m)^T
$$
<pre> sum((x_i - mu_m).reshape(-1,1) @ (x_i - mu_m).reshape(-1,1).T for x_i in unique_class) / n_m</pre>
---

### Prediction Rule

$$
\hat{y}_i = \arg\max_m \ \pi_m \cdot \mathcal{N}(x_i \mid \mu_m, \Sigma_m)
$$

<pre>self.priors[k] * multivariate_normal.pdf( X, mean=self.means[k], cov=self.covariances[k] ) y_pred = np.argmax(likelihoods, axis=1)</pre>
