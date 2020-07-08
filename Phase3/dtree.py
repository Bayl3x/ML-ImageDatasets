import numpy as np
import pandas as pd
from pprint import pprint
import sys


class dtree:
    def __init__(self, max_depth=None, entropy_threshold=0.01):
        self.max_depth = max_depth
        self.entropy_threshold = entropy_threshold
        self.tree = None


    def fit(self, x_train, y_train):
        df = pd.DataFrame(x_train)
        df['label'] = y_train
        self.tree = self.decision_tree_algorithm(df)
        #pprint(self.tree)

    def predict(self, x_test):
        df = pd.DataFrame(x_test)
        df['classification'] = df.apply(self.classify_example, axis=1, args=(self.tree,))
        return df['classification'].values

    #takes numpy matrix and checks if all labels are equal
    #label should be the last column
    def check_purity(self, data):
        label_column = data[:, -1]
        unique_classes = np.unique(label_column)
        if len(unique_classes) == 1:
            return True
        else:
            return False

    #takes numpy matrix and assigns a classification based on majority vote
    def classify_data(self, data):
        label_column = data[:, -1]
        unique_classes, counts = np.unique(label_column, return_counts=True)
        index_of_largest_count = counts.argmax()
        classification = unique_classes[index_of_largest_count]

        return classification

    #creates many potential splits for each column
    def get_potential_splits(self, data):
        #dictionary {column_index: [split_value1, split_value2...]}
        potential_splits = {}
        _, n_columns = data.shape
        for column_index in range(n_columns-1):
            potential_splits[column_index] = []
            values = data[:, column_index]
            unique_values = np.unique(values)

            for index in range(len(unique_values)):
                if index != 0:
                    current_value = unique_values[index]
                    previous_value = unique_values[index-1]
                    potential_split = (current_value + previous_value) / 2

                    potential_splits[column_index].append(potential_split)
        
        return potential_splits

    #splits the data based on the given split column and split value
    def split_data(self, data, split_column, split_value):
        split_column_values = data[:, split_column]
        data_below = data[split_column_values <= split_value]
        data_above = data[split_column_values > split_value]

        return data_below, data_above

    #calulate entropy of a given numpy matrix
    def calculate_entropy(self, data):
        label_column = data[:, -1]
        _, counts = np.unique(label_column, return_counts=True)
        propabilities = counts / counts.sum()
        entropies = propabilities * -np.log2(propabilities)

        total_entropy = entropies.sum()

        return total_entropy

    #calculate entropy of split up data
    def calculate_overall_entropy(self, data_below, data_above):
        n_data_points = len(data_below) + len(data_above)
        p_data_below = len(data_below) / n_data_points
        p_data_above = len(data_above) / n_data_points

        overall_entropy = (p_data_below * self.calculate_entropy(data_below)) + (p_data_above * self.calculate_entropy(data_above))

        return overall_entropy

    def determine_best_split(self, data, potential_splits):
        #initialize overall_entropy to the first value we can find
        for potential_split_column in potential_splits:
            if len(potential_splits[potential_split_column]) > 0:
                data_below, data_above = self.split_data(data, potential_split_column, potential_splits[potential_split_column][0])
                overall_entropy = self.calculate_overall_entropy(data_below, data_above)
                best_split_column = potential_split_column
                best_split_value = potential_splits[potential_split_column][0]
                break


        #actually find the best split
        for column_index in potential_splits:
            for value in potential_splits[column_index]:
                data_below, data_above = self.split_data(data, split_column=column_index, split_value=value)
                current_overall_entropy = self.calculate_overall_entropy(data_below, data_above)

                if current_overall_entropy <= overall_entropy:
                    overall_entropy = current_overall_entropy
                    best_split_column = column_index
                    best_split_value = value

        if overall_entropy == 10:
            print(potential_splits)

        return best_split_column, best_split_value

    #{question: [yes_answer, no_answer]} where the answers are either 1) a class, or 2) another dictionary
    def decision_tree_algorithm(self, df, counter=0):
        #preprocessing
        if counter==0:
            data = df.values
        else:
            data = df

        #base case
        if self.check_purity(data):
            classification = self.classify_data(data)
            return classification

        #second base case
        if self.max_depth and counter >= self.max_depth:
            classification = self.classify_data(data)
            return classification

        counter += 1
        potential_splits = self.get_potential_splits(data)
        hasASplit = False
        for split in potential_splits:
            if len(potential_splits[split]) > 0:
                hasASplit = True
        if hasASplit == False:
            return self.classify_data(data)

        split_column, split_value = self.determine_best_split(data, potential_splits)
        data_below, data_above = self.split_data(data, split_column, split_value)

        current_entropy = self.calculate_entropy(data)
        entropy_after_split = self.calculate_overall_entropy(data_below, data_above)
        
        #third base case to not overfit
        if current_entropy - entropy_after_split < self.entropy_threshold:
            classification = self.classify_data(data)
            return classification


        #instantiate sub-tree
        question = "{} <= {}".format(split_column, split_value)
        sub_tree = {question: []}

        #get dictionary answers. Recursion here
        yes_answer = self.decision_tree_algorithm(data_below, counter)
        no_answer = self.decision_tree_algorithm(data_above, counter)

        sub_tree[question].append(yes_answer)
        sub_tree[question].append(no_answer)

        return sub_tree

    #classify a single example
    def classify_example(self, example, tree):
        question = list(tree.keys())[0]
        column_index, comparison_operator, value = question.split()
        
        if example[int(column_index)] <= float(value):
            answer = tree[question][0]
        else:
            answer = tree[question][1]

        #base case
        if not isinstance(answer, dict):
            return answer
        else:
            return self.classify_example(example, answer)

    #calculate accuracy of all entries in a given dataframe
    def calculate_accuracy(df, tree):
        df['classification'] = df.apply(classify_example, axis=1, args=(tree,))
        df['classification_correct'] = df.classification == df.label

        accuracy = df.classification_correct.mean()

        return accuracy