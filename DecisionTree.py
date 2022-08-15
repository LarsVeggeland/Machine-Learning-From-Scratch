#---------- Imported libraries ----------

import numpy as np
from scipy import stats 
from matplotlib import pyplot as plt
from sklearn.metrics import precision_recall_fscore_support as metrics



#---------- Node class ----------

class Node:
    
    
    def __init__(self, values, labels, depth, precision, max_depth) -> None:
        self.impurity : float = None
        self.rule = None
        self.TYPE = None
        self.depth : int = depth
        self.max_depth : int = max_depth
        self.precision = precision
        self.l_child : Node = None
        self.r_child : Node = None
        self.build_node(values, labels)

        
    def set_rule(self, rule : tuple) -> None:
        self.rule = rule


    def set_l_child(self, node) -> None:
        self.l_child = node


    def set_r_child(self, node) -> None:
        self.r_child = node


    def set_TYPE(self, TYPE : int):
        self.TYPE = TYPE


    def decide(self, vector : list) -> object:
        if self.TYPE is None:
            return self.l_child.decide(vector) if vector[self.rule[0]] >= self.rule[1] else self.r_child.decide(vector)
        return self.TYPE


    def is_leaf(self) -> bool:
        return self.l_child is None


    def gini(self, values : list, labels) -> float:
        class_dist = {c : labels.count(c)  for c in set(labels)}
        #If there is only one class the impurity must be 0
        if len(class_dist) == 1: return 0
        
        # Find the distribution of the classes
        gini = 1 - sum([(dist/len(values))**2 for dist in class_dist.values()])

        # The gini impurity score of the provided valuesset
        return gini


    def split_node(self, values : list, labels : list, split_value : float, split_dimension : int) -> list:
        partition1 = [[], []]
        partition2 = [[], []]
        for i in range(len(values)):
            if values[i][split_dimension] >= split_value:
                partition1[0].append(values[i])
                partition1[1].append(labels[i])
            else:
                partition2[0].append(values[i])
                partition2[1].append(labels[i])
        
        return [partition1, partition2]


    def info_gain(self, values : list, labels : list, split_dimension : int, split_value : float) -> float:
        split = self.split_node(values, labels, split_value, split_dimension)
        partition1_gini = self.gini(split[0][0], split[0][1])
        partition2_gini = self.gini(split[1][0], split[1][1])

        # The new gini impurity score after dividing the node
        return (partition1_gini * len(split[0][0])/len(values) + partition2_gini * len(split[1][0])/len(values))


    def find_best_split(self, values : list, labels : list, precision = 10) -> tuple:
        optimal_impurity = 1
        optimal_split_value = None
        optimal_split_dimension = None
        dim_values = list(zip(np.array(values).T.tolist()))

        # Loop over a certain interval(every dimension)
        for dimension in range(len(values[0])):
            dim = list(dim_values[dimension])[0]

            # Get the interval within which all values for a given dimension resides
            min_value : float = min(dim)
            max_value : float = max(dim)
            delta : float = max_value - min_value

            # Iterates to find the best split in the interval for the given dimension, at the provided level of precision
            for val in range(precision): 
                split_value = min_value + delta * (1+val)/(precision)
                split_impurity = self.info_gain(values, labels, dimension, split_value)

                if split_impurity < optimal_impurity:
                    optimal_impurity = split_impurity
                    optimal_split_value = split_value
                    optimal_split_dimension = dimension
                    
        return ((optimal_split_dimension, optimal_split_value), optimal_impurity)


    def build_node(self, values, labels) -> None:
        try:
            self.impurity = self.gini(values, labels)
        except:
            print("Failed with values:", values)
            return
        best_split = self.find_best_split(values, labels)
        rule = best_split[0]

        if (best_split[1] < self.impurity) * (self.depth < self.max_depth):
            # This node will have child nodes and will therefore require a rule for deciding between them
            self.set_rule(rule)
            split = self.split_node(values, labels, split_dimension=rule[0], split_value=rule[1])
            left = Node(split[0][0], split[0][1], self.depth+1, precision=self.precision, max_depth=self.max_depth)
            right = Node(split[1][0], split[1][1], self.depth+1, precision=self.precision, max_depth=self.max_depth)
            self.set_l_child(left)
            self.set_r_child(right)

        else:
            # We are at a leaf node and must decide the type of object (class) we think we
            # are dealing with as defined by the path from the root node.
            # We select the type most frequently found in the provied labels
            self.set_TYPE(max(set(labels), key = labels.count))

        
    def __repr__(self) -> str:
        return f"""       Impurity: {self.impurity}
        Rule: {self.rule}
        Depth: {self.depth}
        TYPE: {self.TYPE}
        Is leaf: {self.l_child is None}"""



#---------- Decision Tree class ----------

class DecisionTree:


    def __init__(self, values, labels, precision : int = 100, max_depth = 10) -> None:
        self.values : list = values
        self.labels : list = labels
        self.max_depth = max_depth
        self.precision = precision
        self.root : Node = Node(self.values, self.labels ,depth=1, max_depth=self.max_depth, precision=self.precision)


    def predict(self, val : int) -> list:
        predictions = []
        for vector in val:
            predictions.append(self.root.decide(vector))
        return predictions



#---------- Util functions ----------

def get_training_data(filename : str) -> list:
    with open(filename) as file:
        values = []
        labels = []
        for line in file.readlines():
            line_values = [int(i) for i in line.split()]
            labels.append(line_values.pop())
            values.append(line_values)
    return values, labels


def get_test_data(filename : str) -> list:
    with open(filename) as file:
        values = []
        for line in file.readlines():
            values.append([int(i) for i in line.split()])
    return values


def plot_data(filename : str, with_labels : bool = True):
    if with_labels:
        values, labels = get_training_data(filename)
    else:
        values = get_test_data(filename)
    x = np.array([i[0] for i in values])
    y = np.array([i[1] for i in values])
    if with_labels:
        colors = []
        for i in labels:
            if i == 1: colors.append("blue")
            else: colors.append("green")
        plt.scatter(x, y, c=colors)
    else:
        colors = ['green', 'purple', 'brown', 'blue', 'orange']
        plt.scatter(x, y, c=colors)
    plt.axvline(x=10, color="red", linestyle="-")
    plt.show()


def plot_distributions(filename : str, dim : int):
    values, labels  =get_training_data(filename)
    type_1 = [values[i][dim] for i in range(len(labels)) if labels[i] == 1]
    type_2 = [values[i][dim] for i in range(len(labels)) if labels[i] == 2]

    mean_1 = sum(type_1)/len(type_1)
    mean_2 = sum(type_2)/len(type_2)

    std_1 = np.std(type_1)
    std_2 = np.std(type_2)
    # Borowed some code for to the actual plotting:
    # https://stackoverflow.com/questions/10138085/how-to-plot-normal-distribution
    x1 = np.linspace(mean_1 - 3*std_1, mean_1 + 3*std_1, 100)
    x2 = np.linspace(mean_2 - 3*std_2, mean_2 + 3*std_2, 100)
    plt.plot(x1, stats.norm.pdf(x1, mean_1, std_1))
    plt.plot(x2, stats.norm.pdf(x2, mean_2, std_2))
    plt.plot()
    plt.show()


def test_by_performance_eval_metrics(classifier):

    test_vals = [[2, 2], [10, 10], [15, 15], [10, 12]]
    test_labels = [1, 2, 2, 1]

    predictions = classifier.predict(test_vals)
    m = metrics(test_labels, predictions)
    
    print("\nRecall class 1:", m[0][0])
    print("Recall class 2:", m[0][1])

    print("\nPrecision class 1:", m[1][0])    
    print("Precision class 2:", m[1][1])

    print("\nF1 class 1:", m[2][0])
    print("\nF1 class 1:", m[2][1])
    

# Simply run the code to see the below results
values, labels = get_training_data("/Users/larsveggeland/Documents/Python/assignment1/assignment1_train.txt")
classifier = DecisionTree(values, labels)
print(f"\nDoes the classifier predict the training data perfectly? {classifier.predict(values) == labels}\n")
test_vals = get_test_data("/Users/larsveggeland/Documents/Python/assignment1/assignment1_test.txt")
print(f"classifier's prediction of test samples:\n{classifier.predict(test_vals)}")

test_by_performance_eval_metrics(classifier)