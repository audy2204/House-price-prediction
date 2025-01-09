import numpy as np

class DecisionTree:
    def __init__(self, max_depth=7):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        self.tree = self.build_tree(X, y) # Changed _build_tree to build_tree

    def build_tree(self, X, y, depth=0): # Changed _build_tree to build_tree
        num_samples, num_features = X.shape
        if num_samples == 0 or (self.max_depth and depth >= self.max_depth):
            return np.mean(y)

        # Corrected method name to 'best_split'
        best_feature, best_threshold = self.best_split(X, y)
        if best_feature is None:
            return np.mean(y)

        left_indices = X[:, best_feature] <= best_threshold
        right_indices = X[:, best_feature] > best_threshold

        left_tree = self.build_tree(X[left_indices], y[left_indices], depth + 1) # Changed _build_tree to build_tree
        right_tree = self.build_tree(X[right_indices], y[right_indices], depth + 1) # Changed _build_tree to build_tree

        return (best_feature, best_threshold, left_tree, right_tree)

    def best_split(self, X, y):
        best_mse = float('inf')
        best_feature, best_threshold = None, None
        num_features = X.shape[1]

        for feature in range(num_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_indices = X[:, feature] <= threshold
                right_indices = X[:, feature] > threshold
                if len(y[left_indices]) == 0 or len(y[right_indices]) == 0:
                    continue

                # Corrected method name to 'calculate_mse'
                mse = self.calculate_mse(y[left_indices], y[right_indices])
                if mse < best_mse:
                    best_mse = mse
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def calculate_mse(self, left_y, right_y):
        total_samples = len(left_y) + len(right_y)
        if total_samples == 0:
            return 0
        left_mse = np.mean((left_y - np.mean(left_y)) ** 2) * len(left_y) / total_samples
        right_mse = np.mean((right_y - np.mean(right_y)) ** 2) * len(right_y) / total_samples
        return left_mse + right_mse

    def predict(self, X):
        return np.array([self.predict_sample(sample, self.tree) for sample in X])  # Changed '_predict_sample' to 'predict_sample'


    def predict_sample(self, sample, tree):
        if isinstance(tree, tuple):
            feature, threshold, left_tree, right_tree = tree
            if sample[feature] <= threshold:
                # Changed '_predict_sample' to 'predict_sample' for recursive calls
                return self.predict_sample(sample, left_tree)
            else:
                # Changed '_predict_sample' to 'predict_sample' for recursive calls
                return self.predict_sample(sample, right_tree)
        else:
            return tree
        
class RandomForestRegressor:
    def __init__(self, n_trees=10, max_depth=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y):
        for _ in range(self.n_trees):
            bootstrap_indices = np.random.choice(np.arange(len(X)), size=len(X), replace=True)
            X_bootstrap = X[bootstrap_indices]
            y_bootstrap = y[bootstrap_indices]
            tree = DecisionTree(max_depth=self.max_depth)
            tree.fit(X_bootstrap, y_bootstrap)
            self.trees.append(tree)

    def predict(self, X):
        tree_predictions = np.array([tree.predict(X) for tree in self.trees])
        return np.mean(tree_predictions, axis=0)
        