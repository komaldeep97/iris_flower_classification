# coding: utf-8

import numpy as np
import pandas as pd
import sys

import random


def split_tt_data(df, test_size):

    if isinstance(test_size, float):
        test_size = round(test_size * len(df))

    indices = df.index.tolist()
    test_indices = random.sample(population=indices, k=test_size)

    test_df = df.loc[test_indices]
    train_df = df.drop(test_indices)

    return train_df, test_df


def feature_type(df):

    feature_types = []
    n_unique_values_treshold = 15
    for feature in df.columns:
        if feature != "label":
            unique_values = df[feature].unique()
            example_value = unique_values[0]

            if (isinstance(example_value, str)) or (len(unique_values) <= n_unique_values_treshold):
                feature_types.append("categorical")
            else:
                feature_types.append("continuous")

    return feature_types


def accuracy_cal(predictions, labels):
    predictions_correct = predictions == labels
    accuracy = predictions_correct.mean()

    return accuracy


def check_purity(data):

    label_column = data[:, -1]
    unique_classes = np.unique(label_column)

    if len(unique_classes) == 1:
        return True
    else:
        return False


def data_classification(data):

    label_column = data[:, -1]
    unique_classes, counts_unique_classes = np.unique(
        label_column, return_counts=True)

    index = counts_unique_classes.argmax()
    classification = unique_classes[index]

    return classification


def potential_split(data, random_feature):

    splits = {}
    _, n_columns = data.shape
    # excluding the last column which is the label
    column_indices = list(range(n_columns - 1))

    if random_feature and random_feature <= len(column_indices):
        column_indices = random.sample(
            population=column_indices, k=random_feature)

    for column_index in column_indices:
        values = data[:, column_index]
        unique_values = np.unique(values)

        splits[column_index] = unique_values

    return splits


def cal_entropy(data):

    label_column = data[:, -1]
    _, counts = np.unique(label_column, return_counts=True)

    prob = counts / counts.sum()
    entropy = sum(prob * -np.log2(prob))

    return entropy


def cal_overall_entropy(data_below, data_above):
    n = len(data_below) + len(data_above)
    p_data_below = len(data_below) / n
    p_data_above = len(data_above) / n
    overall_entropy = (p_data_below * cal_entropy(data_below)
                       + p_data_above * cal_entropy(data_above))

    return overall_entropy


def best_split(data, splits):

    overall_entropy = 9999
    for column_index in splits:
        for value in splits[column_index]:
            data_below, data_above = split_data(
                data, split_column=column_index, split_value=value)
            current_overall_entropy = cal_overall_entropy(
                data_below, data_above)

            if current_overall_entropy <= overall_entropy:
                overall_entropy = current_overall_entropy
                best_split_column = column_index
                best_split_value = value

    return best_split_column, best_split_value


def split_data(data, split_column, split_value):

    split_column_values = data[:, split_column]

    type_of_feature = FEATURE_TYPES[split_column]
    if type_of_feature == "continuous":
        data_below = data[split_column_values <= split_value]
        data_above = data[split_column_values > split_value]

    else:
        data_below = data[split_column_values == split_value]
        data_above = data[split_column_values != split_value]

    return data_below, data_above


def decision_tree_algorithm(df, counter=0, min_samples=2, max_depth=5, random_feature=None):

    if counter == 0:
        global COLUMN_HEADERS, FEATURE_TYPES
        COLUMN_HEADERS = df.columns
        FEATURE_TYPES = feature_type(df)
        data = df.values
    else:
        data = df

    if (check_purity(data)) or (len(data) < min_samples) or (counter == max_depth):
        classification = data_classification(data)

        return classification

    else:
        counter += 1

        splits = potential_split(data, random_feature)
        split_column, split_value = best_split(
            data, splits)
        data_below, data_above = split_data(data, split_column, split_value)

        if len(data_below) == 0 or len(data_above) == 0:
            classification = data_classification(data)
            return classification

        feature_name = COLUMN_HEADERS[split_column]
        type_of_feature = FEATURE_TYPES[split_column]
        if type_of_feature == "continuous":
            question = "{} <= {}".format(feature_name, split_value)

        else:
            question = "{} = {}".format(feature_name, split_value)

        sub_tree = {question: []}

        yes_answer = decision_tree_algorithm(
            data_below, counter, min_samples, max_depth, random_feature)
        no_answer = decision_tree_algorithm(
            data_above, counter, min_samples, max_depth, random_feature)

        if yes_answer == no_answer:
            sub_tree = yes_answer
        else:
            sub_tree[question].append(yes_answer)
            sub_tree[question].append(no_answer)

        return sub_tree


def predict_example(example, tree):
    question = list(tree.keys())[0]
    feature_name, comparison_operator, value = question.split(" ")

    if comparison_operator == "<=":
        if example[feature_name] <= float(value):
            answer = tree[question][0]
        else:
            answer = tree[question][1]

    else:
        if str(example[feature_name]) == value:
            answer = tree[question][0]
        else:
            answer = tree[question][1]

    if not isinstance(answer, dict):
        return answer

    else:
        residual_tree = answer
        return predict_example(example, residual_tree)


def decision_tree_predictions(test_df, tree):
    predictions = test_df.apply(predict_example, args=(tree,), axis=1)
    return predictions


def bootstrapping(train_df, n_bootstrap):
    bootstrap_indices = np.random.randint(
        low=0, high=len(train_df), size=n_bootstrap)
    df_bootstrapped = train_df.iloc[bootstrap_indices]

    return df_bootstrapped


def random_forest_algorithm(train_df, n_trees, n_bootstrap, n_features, dt_max_depth):
    forest = []
    for i in range(n_trees):
        df_bootstrapped = bootstrapping(train_df, n_bootstrap)
        tree = decision_tree_algorithm(
            df_bootstrapped, max_depth=dt_max_depth, random_feature=n_features)
        forest.append(tree)

    return forest


def random_forest_predictions(test_df, forest):
    df_predictions = {}
    for i in range(len(forest)):
        column_name = "tree_{}".format(i)
        predictions = decision_tree_predictions(test_df, tree=forest[i])
        df_predictions[column_name] = predictions

    df_predictions = pd.DataFrame(df_predictions)
    random_forest_predictions = df_predictions.mode(axis=1)[0]

    return random_forest_predictions


df = pd.read_csv(sys.argv[1])
#converting the last coloumn to label
x = df.columns[-1]
df = df.rename(columns={x: 'label'})

column_names = []
for column in df.columns:
    name = column.replace(" ", "_")
    column_names.append(name)
df.columns = column_names

df.head()

random.seed(0)
train_df, test_df = split_tt_data(df, test_size=0.2)


forest = random_forest_algorithm(
    train_df, n_trees=4, n_bootstrap=800, n_features=2, dt_max_depth=4)
predictions = random_forest_predictions(test_df, forest)
accuracy = accuracy_cal(predictions, test_df.label)

print("Accuracy = {}".format(accuracy))
