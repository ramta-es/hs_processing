import pickle

import numpy as np


class HsFrame(object):  # ML: The name HsFrame doesn't describe the purpose of this class.
    def __init__(self, im_frame, model_path):
        # ML: In operational code, try to always (well, at least when it's used by someone else) add
        # type hints: `def __init__(self, im_frame: np.ndarray, model_path: str)`
        # ML: And a docstring. (what kind of model should be at that model_path?)
        self.bands, self.clf = self.get_model(model_path)
        self.hs_img = (np.load(im_frame))['x'][self.bands, :, :]
        self.shape = self.hs_img.shape
        self.mask = self.seg()

    def get_model(self, path):
        # ML: a private method (one that is used by this class only) should start with an underscore (_).
        # ML: PyCharm put a yellow underline beneath the method name. Why?
        with open(path, 'rb') as model:
            model = pickle.load(model)
            bands = model['check_list']  # ML: You assume some structure on the model. This should be documented.
            clf = model['model']
        return bands, clf

    def seg(self):
        # ML: We mentioned this, the c'tor (__init__ method) shouldn't get im_frame, rather this method should.
        self.mask = (self.clf.predict(self.hs_img.reshape(-1, self.hs_img.shape[1] * self.hs_img.shape[2]).T)).reshape(
            -1, self.shape[1], self.shape[2])
        return self.mask.reshape(self.shape[1], self.shape[2])

    '''Gets the path to the model file and returns the model parameters. for segmentation use the clf'''
    # ML: What does this docstring belong to? It's not the right place for a docstring.



    # ML: What are all these empty lines? :)  Use ctrl+alt+L to autoformat every file that you own.