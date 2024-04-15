from ucimlrepo import fetch_ucirepo
from scipy.stats import norm

# fetch dataset
breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17)

# data (as pandas dataframes)
data = breast_cancer_wisconsin_diagnostic.data.original

# shuffle data to ensure randomness and split 80% for training and 20% for testing
def shuffle_and_split_dataframe(df):
    df = df.sample(frac=1, random_state=1).reset_index(drop=True)
    split_index = int(0.8 * len(df))

    train_df = df[:split_index]
    test_df = df[split_index:]

    return train_df, test_df


# splitting dataframe to malignant and benign
def split_dataframe_to_targets(df):
    grouped = df.groupby("Diagnosis")
    group_m = grouped.get_group('M')
    group_b = grouped.get_group('B')
    return group_m, group_b


# training the model
def train_model(df):
    # Calculating prior values
    p_malignant = df.value_counts("Diagnosis")["M"] / len(df.index)
    p_benign = df.value_counts("Diagnosis")["B"] / len(df.index)

    # Splitting dataframe
    malignant_dataframe, bening_dataframe = split_dataframe_to_targets(df)

    # Calculating means and standard deviations for malignant and benign seperately
    malignant_mean_list = malignant_dataframe.describe().loc["mean"].to_list()[1:]
    malignant_std_list = malignant_dataframe.describe().loc["std"].to_list()[1:]
    bening_mean_list = bening_dataframe.describe().loc["mean"].to_list()[1:]
    bening_std_list = bening_dataframe.describe().loc["std"].to_list()[1:]

    return p_malignant, malignant_mean_list, malignant_std_list, p_benign, bening_mean_list, bening_std_list


# predicting the model
def predict_model(test_df, train_df):
    # Training the model
    p_malignant, malignant_mean_list, malignant_std_list, p_benign, bening_mean_list, bening_std_list = train_model(
        train_df)
    success = 0
    
    # calculating the probabilities of each row and comparing them to find accuracy
    for row in test_df.itertuples():
        malignant_prediction = p_malignant
        benign_prediction = p_benign
        for column in range(30):
            malignant_prediction *= norm.pdf(row[column + 2], malignant_mean_list[column],
                                             malignant_std_list[column])
            benign_prediction *= norm.pdf(row[column + 2], bening_mean_list[column], bening_std_list[column])
        if ((malignant_prediction > benign_prediction) and row[-1] == 'M') or (
                (malignant_prediction < benign_prediction) and row[-1] == 'B'):
            success += 1
    return success / len(test_df.index)


train_dataframe, test_dataframe = shuffle_and_split_dataframe(data)
print("Accuracy of test data: " + str(predict_model(test_dataframe, train_dataframe)))
print("Accuracy of training data: " + str(predict_model(train_dataframe, train_dataframe)))
