import numpy as np


class PCA:

    def __init__(self, num_components):
        self.num_components = num_components

    def fit_transform(self, X):
        self.mean = np.mean(X, axis=0)
        X_meaned = X - self.mean
        
        cov_mat = np.cov(X_meaned , rowvar = False)
        
        eigen_values , eigen_vectors = np.linalg.eigh(cov_mat)
        
        sorted_index = np.argsort(eigen_values)[::-1]
        sorted_eigenvalue = eigen_values[sorted_index]
        sorted_eigenvectors = eigen_vectors[:,sorted_index]
        
        self.eigenvector_subset = sorted_eigenvectors[:,0:self.num_components]
        
        X_reduced = np.dot(self.eigenvector_subset.transpose() , X_meaned.transpose() ).transpose()
        
        return X_reduced
    
    def transform(self, X):
        X_meaned = X - self.mean
        eigenvector_subset = self.eigenvector_subset
        X_reduced = np.dot(eigenvector_subset.transpose() , X_meaned.transpose() ).transpose()
        return X_reduced
    
    def save_parameters(self, file_name):
            import json

            parameter_dict = {
                "mean": self.mean.tolist(),
                "eigenvector_subset": self.eigenvector_subset.tolist(),
            }

            with open(file_name, "w") as f:
                json.dump(parameter_dict, f)

    def load_parameters(self, file_name):
            import json

            with open(file_name, "r") as f:
                parameter_dict = json.load(f)

            self.mean = np.array(parameter_dict["mean"])
            self.eigenvector_subset = np.array(parameter_dict["eigenvector_subset"])

