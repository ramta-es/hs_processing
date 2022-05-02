import pickle
import time
import numpy as np
from sklearn import metrics
import xgboost as xgb
import argparse

# ML: Cool that you used argparser, and cool that you added a description for every argument.
# ML: Looking at this, I don't immediately understand what it's for. You could add a help message that would explain
# the overall purpose of this script. Also, without this context I don't understand the arguments.

parser = argparse.ArgumentParser(description='Arguments for band selection')
parser.add_argument('--samp', default=2, help='Index jump in data sampling')
parser.add_argument('--start_train', default=0, help='Start pixel for image reshape')
parser.add_argument('--end_train', default=2000, help='end pixel for image reshape')
parser.add_argument('--start_test', default=2000, help='Start pixel for image reshape')
parser.add_argument('--end_test', default=3000, help='end pixel for image reshape')
parser.add_argument('--path_to_image', help='Path to full image')  # ML: it's very important to explain the necessary format. Otherwise no one else would be able to use this.
parser.add_argument('--path_to_mask', help='Path to mask')
parser.add_argument('--image_mode', default='resized', help='image mode for file name')
parser.add_argument('--fruit_name', default='Noname_file', help='Fruit name for file name')
parser.add_argument('--directory', help='Directory where the file would be saved')
parser.add_argument('--num_bands', default=20, help='Number of bands we want to save')


def img_reshape(img, mask, pix_b, pix_e, samp):
    shape = mask.shape
    img = img[:, ::samp * 2, pix_b:pix_e:samp]
    mask = mask[::samp * 2, pix_b:pix_e:samp]
    img_arr = np.concatenate((img, mask.reshape(-1, mask.shape[0], mask.shape[1]))).reshape(-1,
                                                                                            img.shape[1] * img.shape[
                                                                                                2]).T
    return img_arr, shape


'''Reshape non labeled data'''


def get_data_to_model(train_arr, test_arr, bands):
    X_train, y_train = train_arr[:, bands], train_arr[:, 224]
    X_test, y_test = test_arr[:, bands], test_arr[:, 224]
    return X_train, y_train, X_test, y_test


def model(X_train, y_train, X_test, y_test):
    # ML: this function does too many unrelated tasks.
    clf = xgb.XGBRFClassifier(tree_method='gpu_hist', n_estimators=10, max_depth=4)
    model = clf.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = metrics.accuracy_score(y_true=y_test, y_pred=y_pred)
    IOU = metrics.jaccard_score(y_true=y_test, y_pred=y_pred)  # ML: Variables should use snake case.
    return clf, accuracy, IOU


def save_model(model_params, fruit_name, img_mode, num_bands, dir):
    name = f'\{fruit_name}_model_params_{img_mode}_{num_bands}_bands'
    dir = dir
    with open(dir + name, 'wb') as outfile:
        pickle.dump(model_params, outfile)
        print('file saved')


# model, img_mode: str, des_bands, fruit_name
def assess_bands(**kwargs):
    # ML: "assess_bands" is not descriptive enough.
    # ML: why not list what's inside kwargs?
    list_check = []
    accuracy_list = []
    jacc_list = []
    list_bands = [i for i in range(224)]  # ML: another way: list(range(224))
    model_params = {'check_list': [], 'accuracy': [], 'IOU': [], 'time': []}
    start_iter = time.time()
    while len(list_check) < int(kwargs['dest_bands']):
        accuracy_in = 0.000
        iou_in = 0.000
        for j in range(len(list_bands)):
            print('j', j)
            tmp = list_check + [list_bands[j]]  # ML: "tmp" isn't a good name.
            print('tmp', tmp)
            X_train, y_train, X_test, y_test = get_data_to_model(kwargs['train_arr'], kwargs['test_arr'], bands=tmp)
            clf, accuracy_rf, iou_model = model(X_train, y_train, X_test, y_test)
            if iou_model > iou_in:
                accuracy_in = accuracy_rf
                iou_in = iou_model
                band = list_bands[j]
        if band not in list_check:  # ML: Why does PyCharm make the word 'band' yellow? This is usually a bad sign.
            list_check += [band]  #ML: see list.append()
            accuracy_list += [accuracy_in]
            jacc_list += [iou_in]
            list_bands.remove(band)
        '''saving to dict'''
        model_params = {'check_list': list_check, 'accuracy': accuracy_list, 'IOU': jacc_list, 'model': clf,
                        'time': ("--- %s seconds ---" % (time.time() - start_iter))}
        '''saving to file'''  # ML: comments should use the hash syntax (#)
    save_model(model_params, kwargs['fruit_name'], kwargs['img_mode'], len(list_check), kwargs['directory'])
# ML: the assess_bands implementation is very cumbersome. If you have a lot of params that you want to keep, consider
# using a dataclass for that.

'''When error for band variable referenced before assignment occures you must make sure that you have fruits in the data. Otherwise the IOU will stay on 0'''
# ML: not the place for a docstring.

def main():
    args = parser.parse_args()
    img = np.load(args.path_to_image)
    mask = np.load(args.path_to_mask)

    train_arr, train_shape = img_reshape(img=img, mask=mask, pix_b=int(args.start_train), pix_e=int(args.end_train), samp=args.samp)
    test_arr, test_shape = img_reshape(img=img, mask=mask, pix_b=int(args.start_test), pix_e=int(args.end_test), samp=args.samp)

    assess_bands(img_mode=args.image_mode, dest_bands=args.num_bands, train_arr=train_arr,
                 test_arr=test_arr, directory=args.directory, fruit_name=args.fruit_name)


if __name__ == '__main__':
    main()


# ML: Try to use more descriptive naming. It's hard to follow your code otherwise.
# Remember that "you don't write code fot computers, you write it for people" (I didn't invent this saying).
