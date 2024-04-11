
import numpy as np
import pandas as pd

class Node:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class DecisionTree:
    def __init__(self, max_depth):
        self.max_depth = max_depth
        self.root = None

    def fit(self, X, Y):
        self.root = self._build_tree(X, Y)

    def fit_from_csv(self, csv_path, label_column):
        """Load data from a CSV file and fit the decision tree.

        Args:
            csv_path (str): The file path to the CSV file.
            label_column (str): The name of the column that contains the labels.
        """
        # Load the dataset
        data = pd.read_csv(csv_path)
        
        # Split the dataset into features and labels
        X = data.drop(label_column, axis=1).values
        Y = data[label_column].values
        
        # Fit the decision tree with the loaded data
        self.fit(X, Y)
    #end of function
    def predict(self, sample):
        """Predict the class label for a single sample by traversing the tree."""
        node = self.root
        while node.left or node.right:  # While we are not at a leaf node
            if sample[node.feature_index] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value

    def predict_all(self, X):
        """Predict the class labels for multiple samples."""
        return np.array([self.predict(sample) for sample in X])

    def _build_tree(self, X, Y, depth=0):
        if depth == self.max_depth or len(set(Y)) == 1:
            return Node(value=self._calculate_leaf_value(Y))

        best_split = self._find_best_split(X, Y)
        
        # Check if no valid split is found
        if not best_split:
            return Node(value=self._calculate_leaf_value(Y))

        left_subtree = self._build_tree(X[best_split["left_indices"]], Y[best_split["left_indices"]], depth+1)
        right_subtree = self._build_tree(X[best_split["right_indices"]], Y[best_split["right_indices"]], depth+1)

        return Node(feature_index=best_split["feature_index"], threshold=best_split["threshold"], left=left_subtree, right=right_subtree)


    def _calculate_leaf_value(self, Y):
        # Count occurrences of each class label in the subset Y
        class_labels, counts = np.unique(Y, return_counts=True)
        # Find and return the class label with the highest occurrence
        return class_labels[np.argmax(counts)]


    def _find_best_split(self, X, Y):
        best_split = {"gain": -np.inf, "feature_index": None, "threshold": None, "left_indices": None, "right_indices": None}
        n_samples, n_features = X.shape

        for feature_index in range(n_features):
            possible_thresholds = np.unique(X[:, feature_index])
            for threshold in possible_thresholds:
                left_indices = np.where(X[:, feature_index] <= threshold)[0]
                right_indices = np.where(X[:, feature_index] > threshold)[0]
                if len(left_indices) == 0 or len(right_indices) == 0:
                    continue
                
                gain = self._information_gain(Y, left_indices, right_indices)
                if gain > best_split["gain"]:
                    best_split["gain"] = gain
                    best_split["feature_index"] = feature_index
                    best_split["threshold"] = threshold
                    best_split["left_indices"] = left_indices
                    best_split["right_indices"] = right_indices
        
        # If no valid split is found, return an empty dictionary
        if best_split["feature_index"] is None:
            return {}

        return best_split

    def _entropy(self, Y):
        # np.unique with return_counts=True finds all unique class labels within Y
        # and counts how many times each unique label appears.
        _, counts = np.unique(Y, return_counts=True)
        
        # Calculates the probabilities of each class by dividing the count of each class
        # by the total number of instances.
        probabilities = counts / counts.sum()
        
        # Calculates the entropy of Y. Entropy is a measure of disorder or unpredictability.
        # High entropy means the data has high variance and it's hard to draw any conclusions from it.
        # It's calculated as the sum of the probabilities of each class times the log (base 2) of the
        # probability, then multiplied by -1 to make the sum positive.
        return -np.sum(probabilities * np.log2(probabilities))


    def _information_gain(self, Y, left_indices, right_indices):
        # Calculates the entropy of the entire dataset before the split.
        # This serves as the starting point for measuring improvement (gain).
        parent_entropy = self._entropy(Y)
        
        # n holds the total number of instances in Y.
        n = len(Y)
        
        # n_left and n_right hold the number of instances that would be in the left and right
        # child nodes if we split the dataset according to a particular feature's threshold.
        n_left, n_right = len(left_indices), len(right_indices)
        
        # child_entropy calculates the weighted average of the entropy of the child nodes
        # after a potential split. It combines the entropies of the left and right subsets,
        # weighted by their proportion of the total dataset.
        child_entropy = (n_left / n) * self._entropy(Y[left_indices]) + (n_right / n) * self._entropy(Y[right_indices])
        
        # The information gain is calculated as the difference between the entropy of the
        # parent node (before the split) and the combined entropy of the child nodes (after the split).
        # A higher gain means the split made the subsets more orderly (less entropy) compared
        # to the original dataset.
        gain = parent_entropy - child_entropy
        
        return gain

   # to find the find_best_split and calculate_leaf_value
   # find_best_split will find the best feature and threshold to split
   # calculate_leaf_value for classification is based on predicting which class a given input belongs to, the leaf node should represent the most common class among the 
   #training instances that end up in that leaf
 




     

    
