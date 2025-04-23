# Gaussian Mixture Models (GMM): Theory & Implementation

## Project Overview

This project will explore Gaussian Mixture Models (GMM) and their application in supervised and unsupervised learning. 
It includes a theoretical explanation, an implementation from scratch, and a comparison with scikit-learn’s GMM. I will initially approach this from a supervised learning perspective and then work towards unsupervised and semi-supervised learning.

### Project Update - 22 April 2024
I have starting working on supervised learning after working through the mathematical derivations and likelihood. The supervised classifier has been coded
and I just need to create some synthetic data and plot it with its contours.

### What is a Guassian Mixture Model?
A Gaussian Mixture Model (GMM) is called a mixture model because the overall distribution $p(x)$ is formed by combining multiple Gaussian components. Each component represents a potential class or cluster,  contributing one Gaussian to the mixture. We assume that each data point $x$ is generated from one of these Gaussian distributions, but we don’t  now which one. The model captures this uncertainty by assigning probabilities to each component, which effectively modeling the data as coming from a mixture of Gaussians.

The Gaussian mixture model makes use of the factorisation of the 
$$
\text{p(x,y) = p(x|y)p(y)}
$$

The second factor is the marginal distribution of y. Since y is categorical and takes values in the set $\{1,…,M\}$, this is given by a categorical distribution with $M$ parameters $\{\pi_m\}^{M}_{m=1}$

$$
\textit{Probability of a class} \\ p(y=1) = \pi_1 \\ \vdots \\ p(y=M) = \pi_M
$$

The GMM assumes that $p(x|y)$ has a Gaussian distribution for each y. $X$ modeled as Gaussian when conditioned on $y$. The output is learned with class-dependent mean vector $\mu_y$, and covariance $\Sigma_y$. Like any machine learning models, The GMM is leant from the training data. The unknown parameters to be learnt are $\theta = \{\mu_m,\Sigma_m,\pi_m\}_{m=1}^M$

According to Joint Probability:
$$
\text{p(x,y) = p(x|y)p(y) = N}(x|\mu_m, \Sigma_m) \ \cdot \pi_m
$$

### Supervised Learning of the Gaussian Mixture Model

This is our training data:

$$
\tau = \{X_i,Y_i\}_{i=1}^{n}
$$

Mathematically, we learn the GMM by maximising the log-likelihood of the training data and find the maximum $\theta$ which is the best parameters for a distribution:
$$
\hat{\theta} = \argmax_\theta \ ln \ p(\underbrace{\{x_i, y_i\}_{i=1}^n}_\tau | \theta)
$$

The unknown parameters to be learned are $\theta = \{\mu_m, \Sigma_m, \pi_m\}^M_{m=1}$. This means that we compute the log-likelihood of the data under all $M$ Gaussian components and sum them to obtain the total log-likelihood.

$$p(\underbrace{\{x_i, y_i\}_{i=1}^n}_\tau | \theta) \\ = \prod_{i=1}^n \underbrace{p(x_i,y_i|\theta)}_{joint \ likelihood} \\ \text{remember that} \ p(x_i,y_i|\theta) \equiv p(x,y) \\ =\prod_{i=1}^n p(x_i|y_i,\theta).p(y_i|\theta)$$

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
