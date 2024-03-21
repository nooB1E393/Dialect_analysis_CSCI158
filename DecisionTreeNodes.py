import numpy as np

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

    def _build_tree(self, X, Y, depth=0):
        if depth == self.max_depth or len(set(Y)) == 1:
            return Node(value=self._calculate_leaf_value(Y))

        best_split = self._find_best_split(X, Y)
        if best_split["gain"] == 0:
            return Node(value=self._calculate_leaf_value(Y))

        left_subtree = self._build_tree(X[best_split["left_indices"]], Y[best_split["left_indices"]], depth+1)
        right_subtree = self._build_tree(X[best_split["right_indices"]], Y[best_split["right_indices"]], depth+1)

        return Node(feature_index=best_split["feature_index"], threshold=best_split["threshold"], left=left_subtree, right=right_subtree)

   # to find the find_best_split and calculate_leaf_value
 

    
