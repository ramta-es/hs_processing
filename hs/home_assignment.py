import pandas as pd
import pickle
from sklearn import metrics
import xgboost as xgb
from sklearn.model_selection import train_test_split
import argparse

parser = argparse.ArgumentParser(description='Arguments for home assignment selection')
parser.add_argument('--path_to_data', default='/Users/ramtahor/Downloads/d1.csv', help='Path to a csv file')
parser.add_argument('--speaker_id', default='Dana', help="Speaker's name for file name")
parser.add_argument('--directory', default='/Users/ramtahor/Desktop/home_ex/', help='Directory where the file would be saved')
parser.add_argument('--accuracy', default=0.9, help='Accuracy threshold for model saving')

# A function for saving the model
def save_model(model, out_dir):
    name = '_model'
    with open(out_dir + name, 'wb') as outfile:
        pickle.dump(model, outfile)
        print('file saved')


# This function organizes the data table in the right shape of data and label. it also gets a speaker_id and convert
# it to a number

def create_data_set(df, speaker_id):
    for i, row in df.iterrows():
        if df.at[i, 'speaker_id'] == speaker_id:
            df.at[i, 'speaker_id'] = 0
        else:
            df.at[i, 'speaker_id'] = 1

    X = df.loc[:, ['feature_1', 'feature_2', 'feature_3',
                   'feature_4', 'feature_5', 'feature_6', 'feature_7', 'feature_8',
                   'intonation_unit_id', 'speaker_id']]
    X = X.astype(float)
    y = df.loc[:, ['is_boundary_between_words']]
    return X, y

# main function
def main():
    args = parser.parse_args()
    df = pd.read_csv(args.path_to_data)

    X, y = create_data_set(df, args.speaker_id)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    clf = xgb.XGBRFClassifier(tree_method='exact', n_estimators=2, max_depth=4, use_label_encoder=False)
    model = clf.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = metrics.accuracy_score(y_true=y_test, y_pred=y_pred)

    if accuracy >= args.accuracy:
        save_model(model, args.directory)
    print(accuracy)


if __name__ == '__main__':
    main()
