class GuassianMixtureModel:
    def __init__(self, guassian_components: int):
        self.n_components = guassian_components
        self.means: list = None
        self.covariances: list = None
        self.priors: list = None
    
    def plot_contours(self):
        pass