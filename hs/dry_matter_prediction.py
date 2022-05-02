import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.svm
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from sklearn.model_selection import RepeatedKFold, cross_validate, train_test_split
from sklearn.utils import shuffle
import pywt.data


bk_ref_img_01 = np.load(
    r"C:\Users\g\Desktop\avocado_granot\20220330_124346__dark\20220330_124346__dark\stream01_dark_ref.npy")
bk_ref_img_00 = np.load(
    r"C:\Users\g\Desktop\avocado_granot\20220330_124346__dark\20220330_124346__dark\stream00_dark_ref.npy")

white_ref_01 = np.load(
    r"C:\Users\g\Desktop\avocado_granot\20220330_124214__white1\20220330_124214__white1\whit_ref_01.npy")
white_ref_00 = np.load(
    r"C:\Users\g\Desktop\avocado_granot\20220330_124108__white0\20220330_124108__white0\whit_ref_00.npy")

dm_folder = r"C:\Users\g\Desktop\avocado_granot\DM.csv"


# hs_image = U.reconstruct_image(U.ord_frame_list_all(im_folder))
# wb_image = U.reconstruct_image(U.ord_frame_list_all(wb_folder))
# bk_image = U.reconstruct_image(U.ord_frame_list_all(bk_folder))


# Clculate reflectance:

def find_reflectance(hs_img: np.ndarray, wb_img: np.ndarray, bk_img: np.ndarray) -> np.ndarray:
    wb_img = wb_img.reshape((224, 256, 1))
    bk_img = bk_img.reshape((224, 256, 1))
    return (hs_img - bk_img) / (wb_img + 1e-6 - bk_img)


# Get data:
# 1.
# draw contours with labelme
# on black and white images
# get masks
# get black ref
# get white ref
# calc reflectance
# ###done


# 2.


# # draw mask:
# mask = U.draw_conts(U.json_to_df(json_path), (hs_image.shape[1], hs_image.shape[2]))
#
# # mask image:
# hs_image = hs_image[mask != 0]
#
# # get median for each image
# hs_median = np.array([np.median(hs_image[i, :, :]) for i in range(hs_image.shape[0])])
# # add label to each median spectra of the DM value
# # split data to train-test sets

# # feed to the regressor


# Regressor:

# path_to_ts_04 = 'csv_files/test_data_dm_04.csv'
# path_to_tr_04 = 'csv_files/train_data_dm_04.csv'
#
# path_to_ts_03 = 'csv_files/test_data_dm_03.csv'
# path_to_tr_03 = 'csv_files/train_data_dm_03.csv'
#
#
# train_df = pd.read_csv(path_to_tr_03)
# test_df = pd.read_csv(path_to_ts_03)
#
# train_df_04 = pd.read_csv(path_to_tr_04)
# test_df_04 = pd.read_csv(path_to_ts_04)
#
# train_df = train_df.append(train_df_04, ignore_index=True)
# test_df = test_df.append(test_df_04, ignore_index=True)
#


ts = pd.read_csv('test_exp_2000')
tr = pd.read_csv('train_exp_2000')

# data = pd.read_csv(r"exp_2000_all_streams_df")
data = pd.read_csv(r"D:\avocado_granot_exp_2000\dm_with_median_normed_exp_2000.csv")
# print(data.columns)

#
# X_train, y_train = tr.loc[:, [f'{num}' for num in range(224)]].values, tr.loc[:, ['dm']].values
# for i in range(y_train.shape[0]):
#     y_train[i][0] = float(y_train[i])
#
# X_label = train_df.loc[:, ]
#
# X_test, y_test = ts.loc[:, [f'{num}' for num in range(224)]].values, ts.loc[:, ['dm']].values
# for i in range(y_test.shape[0]):
#     y_test[i][0] = float(y_test[i])
# f_i = [223, 158, 160, 162, 77, 76, 75, 167, 169, 172, 70, 174, 68, 67, 66, 65, 64, 63, 61, 60, 59, 58, 113, 56, 82, 55, 152, 85, 112, 118, 110, 120, 126, 139, 141, 142, 143, 99, 98, 97, 96, 95, 145, 93, 92, 146, 90, 89, 88, 87, 148, 151, 54, 57, 176, 28, 27, 194, 175, 24, 198, 200, 207, 212, 213, 214, 215, 14, 12, 11, 10, 7, 217, 218, 219, 220, 29, 30, 25, 31, 177, 178, 179, 180, 181, 182, 45, 183, 184, 185, 116, 186, 33, 188, 193, 192, 40, 37, 191, 121, 8, 9, 4, 44, 15, 123, 13, 32, 62, 168, 136, 144, 21, 166, 2, 0, 1, 36, 5, 16, 135, 34, 165, 19, 6, 17, 3, 53, 137, 105, 22, 23, 103, 128, 117, 196, 159, 111, 138, 199, 78, 170, 155, 202, 71, 132, 157, 156, 122, 51, 20, 147, 50, 163, 124, 46, 210, 47, 150, 115, 114, 43, 104, 206, 107, 125, 86, 204, 84, 69, 100, 119, 149, 18, 129, 130, 91, 41, 203, 109, 131, 102, 164, 73, 209, 35, 161, 171, 26, 94, 83, 42, 197, 74, 81, 38, 39, 187, 201, 173, 133, 153, 189, 106, 48, 127, 79, 134, 49, 221, 195, 52, 190, 154, 140, 72, 108, 80, 208, 101, 205, 216, 222, 211]
# mul = np.arange(223, -1, -1)

X, y = data.loc[:, [f'med_specstream03_can{num}' for num in range(224)]].values, data.loc[:, ['dm']].values
# X, y = data.loc[:, [f'{num}' for num in range(224)]].values, data.loc[:, ['dm']].values
# print(X[0])
# X = X[:, f_i]

# X = (X * mul)


for i in range(y.shape[0])
    y[i] = float(y[i][0].split('%')[0])
# x = np.arange(X.shape[1])

# X_grad = np.gradient(X, x, axis=1)
# X_grad_2 = np.gradient(X_grad, x, axis=1)
# X = np.concatenate((X, X_grad), axis=1)
ca = np.zeros((X.shape[0], 112))
for i in range(X.shape[0]):
    # X[i] = X[i] / np.max(X[i])
    ca[i], cd = pywt.dwt(X[i], 'db1')

# X = np.concatenate((X, ca), axis=1)
print(X.shape)
# print(X.shape, X_con.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3 )


def model_xgb(X_train, y_train, X_test, y_test):
    reg = xgb.XGBRFRegressor(tree_method='gpu_hist', n_estimators=6, max_depth=20)
    svm = sklearn.svm.SVR(kernel='rbf', degree=1, gamma='scale', coef0=0.0, tol=0.0001, C=1.0, epsilon=0.1, shrinking=True)
    score_df = pd.DataFrame(columns=['train_rmse', 'train_r2', 'val_r2', 'val_rmse', 'test_rmse'])
    X_train, y_train = shuffle(X_train, y_train)
    model = reg.fit(X_train, y_train.ravel())
    rkf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=2652124)
    scores = cross_validate(reg, X_train, y_train.ravel(), scoring=('neg_root_mean_squared_error', 'r2'),
                            return_estimator=True, return_train_score=True, cv=rkf, n_jobs=-1)
    model_retrained = scores['estimator'][np.argmax((scores['test_neg_root_mean_squared_error']))]
    model_retrained = model_retrained.fit(X_train, y_train)

    y_pred = model_retrained.predict(X_test)
    print('key', scores.keys())
    test_rmse = scores['test_neg_root_mean_squared_error']
    train_rmse = scores['train_neg_root_mean_squared_error']
    test_r2 = scores['test_r2']
    train_r2 = scores['train_r2']
    score_df['train_rmse'] = np.abs(train_rmse)
    score_df['train_r2'] = train_r2
    score_df['val_rmse'] = np.abs(test_rmse)
    score_df['val_r2'] = test_r2
    # print('feature importance = ', len(model.feature_importances_))

    rmse = mean_squared_error(y_test, y_pred, squared=True)
    score_df['test_rmse'] = rmse

    return model, rmse, y_test, y_pred, score_df#, model.feature_importances_

# C1

model, rmse, y_test, y_pred, score_df = model_xgb(X_train, y_train, X_test, y_test)
print(score_df)
score_df.to_csv('score_df')
# print(np.argsort(feature_importance))
# with open('feature_importance.txt', 'w') as m:
#     m.write(f'feature_importance: {list(np.argsort(feature_importance))}')

# rmse_0 = mean_squared_error(y_test, y_pred_0, squared=True)
# rmse_10 = mean_squared_error(y_test, y_pred_10, squared=True)
# rmse_19 = mean_squared_error(y_test, y_pred_19, squared=True)
# r2_score_0 = r2_score(y_test, y_pred_0)
# r2_score_10 = r2_score(y_test, y_pred_10)
# r2_score_19 = r2_score(y_test, y_pred_19)


# print('rmse 0', rmse_0)
# print('r2 score 0', r2_score_0)


# print(y_pred.reshape(20, 1).shape)
# z = np.polyfit(x.reshape(20), y_pred, 1)
# p = np.poly1d(z)

# plt.scatter(x=x, y=, label='real_dm', c='green')
# plt.scatter(x=x, y=y_pred_0, label='pred_dm_0', c='green')
# plt.scatter(x=x, y=y_pred_10, label='pred_dm_10', c='blue')
plt.scatter(x=y_test, y=y_pred, label='pred_dm', c='orange', s=4)
# plt.scatter(x=y_train, y=y_, label='pred_dm', c='green', s=2)
# y_test = y_test.astype(np.float32)
# print(type(y_test[0][0]), type(y_pred[0]))
# z = np.polyfit(y_test.reshape((40,)), y_pred, 4)
# print(z.shape)
# p = np.poly1d(z)
# plt.plot(y_test, p(y_pred), "r--")
# plt.plot(X_test, p(y_test), color="blue", linewidth=3)
# plt.scatter(x=y_train, y=y_pred, label='pred_dm', c='green')
# plt.plot(x=x, y=z, c='blue')
plt.legend()
plt.xlabel('measured dm')
plt.ylabel('predicted dm')
plt.grid()
plt.title('dry matter prediction with xgboostRF regressor and svm regressor')
plt.show()


### trendline ###


# prepare xgboost and SVM models that gets the same shape of data

# ####Cross validation####
# feed to regressor
# get RMSE on calibration, cross validation, prediction

# get and organize data form csv file

def save_params(param_1, param_2, param_3, param_4, param_5, param_6):
    with open('model_score.txt', 'w') as m:
        m.write(f'rmse train: {param_1}\n\n')
        m.write(f'rmse val: {param_2}\n\n')
        m.write(f'r2 train: {param_3}\n\n')
        m.write(f'r2 val: {param_4}\n\n')
        m.write(f'rmse test: {param_5}\n\n')
        m.write(f'r2 test : {param_6}\n\n')


def split_train_test(csv_path: str):
    dm = pd.read_csv(csv_path)
    # dm = dm.loc[:, ['box', 'side', 'dm']]
    ts = pd.DataFrame(columns=dm.columns)
    tr = pd.DataFrame(columns=dm.columns)
    for i, row in dm.iterrows():
        if ('6' in str(row['224']) and 'x' in row['side']) or ('9' in str(row['224']) and 'o' in row['side']):
            ts.loc[ts.shape[0]] = dm.loc[i]
            print(ts)
        else:
            tr.loc[tr.shape[0]] = dm.loc[i]
            # print(tr)
    return ts, tr


csv_path = 'exp_2000_all_streams_df'
