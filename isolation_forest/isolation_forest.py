import numpy as np


def c(n):
    """
    Average path length
    """
    if n > 2:
        return 2 * (np.log(n - 1) + 0.5772156649) - 2 * (n - 1) / n
    if n == 2:
        return 1
    return 0


class ExternalNode:
    def __init__(self, size, data):
        """
        External Node - same a leaf node
        """
        self.size = size
        self.data = data


class InternalNode:
    """
    Internal Node : Has two children (left and right) and test value (split index and split value).
    """
    def __init__(self, left, right, splitAtt, splitValue):
        self.left = left
        self.right = right
        self.splitAtt = splitAtt
        self.splitValue = splitValue


class IsolationTree:
    def __init__(self, height, height_limit):
        """
        Initialize an IsolationTree instance.

        Args:
            height: Current tree height
            height_limit: Height limit for tree
        """
        self.height = height
        self.height_limit = height_limit

    def fit(self, X):
        """
        Create an isolation tree.
        Set field self.root to the root of that tree and return it.
        """
        # If sub-sample is not divided, create an external node
        if self.height >= self.height_limit or X.shape[0] <= 2:
            self.root = ExternalNode(size=X.shape[0], data=X)
            return self.root

        # If sub-sample is divided, create an internal node
        Q = X.shape[1]  # Number of features in X
        splitAtt = np.random.randint(0, Q)  # Index of attribute to split on
        splitValue = np.random.uniform(min(X[:, splitAtt]), max(X[:, splitAtt]))  # Value to split on

        X_left = X[X[:, splitAtt] < splitValue]
        X_right = X[X[:, splitAtt] >= splitValue]

        left = IsolationTree(self.height + 1, self.height_limit)
        right = IsolationTree(self.height + 1, self.height_limit)
        left.fit(X_left)
        right.fit(X_right)
        self.root = InternalNode(left.root, right.root, splitAtt, splitValue)
        self.n_nodes = self.count_nodes(self.root)
        return self.root

    def count_nodes(self, root):
        """
        Traverse the tree and count the number of nodes.
        """
        count = 0
        stack = [root]
        while stack:
            node = stack.pop()
            count += 1
            if isinstance(node, InternalNode):
                stack.append(node.right)
                stack.append(node.left)
        return count


class IsolationForest:
    def __init__(self, n_trees, sub_sample_size):
        """
        Initialize an IsolationForest instance.

        Args:
            n_trees: The number of trees in the forest.
            sub_sample_size: Controls the training data size
        """
        self.sub_sample_size = sub_sample_size
        self.n_trees = n_trees

    def fit(self, X):
        """
        Create forest of IsolationTree
        """
        self.trees = []
        X = X.values  # Convert to numpy array
        height_limit = np.ceil(np.log2(self.sub_sample_size))
        for _ in range(self.n_trees):
            data_index = np.random.randint(0, X.shape[0], self.sub_sample_size)
            X_sub = X[data_index]
            tree = IsolationTree(0, height_limit)
            tree.fit(X_sub)
            self.trees.append(tree)
        return self

    def path_length(self, X):
        """
        Compute the path length for x_i in X using every isolation tree in forest,
        then compute the average for each x_i
        """
        paths = []
        for row in X:
            path = []
            for tree in self.trees:
                node = tree.root
                length = 0
                while isinstance(node, InternalNode):
                    if row[node.splitAtt] < node.splitValue:
                        node = node.left
                    else:
                        node = node.right
                    length += 1
                leaf_size = node.size
                pathLength = length + c(leaf_size)
                path.append(pathLength)
            paths.append(path)
        paths = np.array(paths)
        return np.mean(paths, axis=1)

    def anomaly_score(self, X):
        """
        Compute the anomaly score for each x_i in X
        """
        X = X.values
        avg_length = self.path_length(X)
        scores = np.array([np.power(2, -l / c(self.sub_sample_size)) for l in avg_length])
        return avg_length, scores

    def predict(self, scores):
        """
        Given an array of anomaly scores, return an array of predictions: 1 for normal and -1 for anomaly
        """
        prediction = np.array([1 if s < 0.5 else -1 for s in scores])
        return prediction
