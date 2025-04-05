import os
import gc
from functools import partial, reduce

import numpy as np
from tqdm import tqdm

from autovar.base import RegisteringChoiceType, register_var, VariableClass
from autovar.base.decorators import cache_outputs, requires


DEBUG = int(os.environ.get('DEBUG', 0))


class DatasetVarClass(VariableClass, metaclass=RegisteringChoiceType):
    """Defines the dataset to use"""
    var_name = 'dataset'

    @register_var(argument=r"fashionwo0", shown_name="fashion mnist without 0")
    @staticmethod
    def fashionwo0(auto_var, var_value, inter_var):
        from .mnist import fashion_leave_one_out
        return fashion_leave_one_out(0)

    @register_var(argument=r"fashionwo4", shown_name="mnist without 9")
    @staticmethod
    def fashionwo4(auto_var, var_value, inter_var):
        from .mnist import fashion_leave_one_out
        return fashion_leave_one_out(4)

    @register_var(argument=r"fashionwo9", shown_name="mnist without 9")
    @staticmethod
    def fashionwo9(auto_var, var_value, inter_var):
        from .mnist import fashion_leave_one_out
        return fashion_leave_one_out(9)

    @register_var(argument=r"mnistwo0", shown_name="mnist without 9")
    @staticmethod
    def mnistwo0(auto_var, var_value, inter_var):
        from .mnist import mnist_leave_one_out
        return mnist_leave_one_out(0)

    @register_var(argument=r"mnistwo1", shown_name="mnist without 9")
    @staticmethod
    def mnistwo1(auto_var, var_value, inter_var):
        from .mnist import mnist_leave_one_out
        return mnist_leave_one_out(1)

    @register_var(argument=r"mnistwo2", shown_name="mnist without 9")
    @staticmethod
    def mnistwo2(auto_var, var_value, inter_var):
        from .mnist import mnist_leave_one_out
        return mnist_leave_one_out(2)

    @register_var(argument=r"mnistwo3", shown_name="mnist without 9")
    @staticmethod
    def mnistwo3(auto_var, var_value, inter_var):
        from .mnist import mnist_leave_one_out
        return mnist_leave_one_out(3)

    @register_var(argument=r"mnistwo4", shown_name="mnist without 9")
    @staticmethod
    def mnistwo4(auto_var, var_value, inter_var):
        from .mnist import mnist_leave_one_out
        return mnist_leave_one_out(4)

    @register_var(argument=r"mnistwo5", shown_name="mnist without 9")
    @staticmethod
    def mnistwo5(auto_var, var_value, inter_var):
        from .mnist import mnist_leave_one_out
        return mnist_leave_one_out(5)

    @register_var(argument=r"mnistwo6", shown_name="mnist without 9")
    @staticmethod
    def mnistwo6(auto_var, var_value, inter_var):
        from .mnist import mnist_leave_one_out
        return mnist_leave_one_out(6)

    @register_var(argument=r"mnistwo7", shown_name="mnist without 9")
    @staticmethod
    def mnistwo7(auto_var, var_value, inter_var):
        from .mnist import mnist_leave_one_out
        return mnist_leave_one_out(7)

    @register_var(argument=r"mnistwo8", shown_name="mnist without 9")
    @staticmethod
    def mnistwo8(auto_var, var_value, inter_var):
        from .mnist import mnist_leave_one_out
        return mnist_leave_one_out(8)

    @register_var(argument=r"mnistwo9", shown_name="mnist without 9")
    @staticmethod
    def mnistwo9(auto_var, var_value, inter_var):
        from .mnist import mnistwo9
        return mnistwo9()

    @register_var(argument=r"calcedrepr-mnistwo9-(?P<result_path>[a-zA-Z0-9_\.\/-]+)", shown_name="mnist")
    @staticmethod
    def calcedrepr_mnistwo9(auto_var, var_value, inter_var, result_path):
        import joblib
        _, y_train, _, y_test, _ = DatasetVarClass.mnistwo9(None, None, None)
        result_path = os.path.join("./results/oos_repr/", result_path)
        res = joblib.load(result_path)
        x_train = res['trn_repr']
        x_test = res['tst_repr']
        res['trn_repr'].astype(np.float32, copy=False)
        res['tst_repr'].astype(np.float32, copy=False)
        res['oos_trn_repr'].astype(np.float32, copy=False)
        res['oos_tst_repr'].astype(np.float32, copy=False)
        rest = (res['oos_trn_repr'], res['oos_tst_repr'],
                9*np.ones(len(res['oos_trn_repr'])), 9*np.ones(len(res['oos_tst_repr'])))
        inter_var['is_img_data'] = False
        return x_train, y_train, x_test, y_test, rest

    @register_var(argument=r"calcedrepr-mnistwo(?P<unseen_no>[0-9])-(?P<result_path>[a-zA-Z0-9_\.\/-]+)", shown_name="mnist")
    @staticmethod
    def calcedrepr_mnist(auto_var, var_value, inter_var, unseen_no, result_path):
        from .mnist import mnist_leave_one_out
        import joblib
        unseen_no = int(unseen_no)
        _, y_train, _, y_test, _ = mnist_leave_one_out(unseen_no)
        result_path = os.path.join("./results/oos_repr/", result_path)
        res = joblib.load(result_path)
        x_train = res['trn_repr']
        x_test = res['tst_repr']
        res['trn_repr'].astype(np.float32, copy=False)
        res['tst_repr'].astype(np.float32, copy=False)
        res['oos_trn_repr'].astype(np.float32, copy=False)
        res['oos_tst_repr'].astype(np.float32, copy=False)
        rest = (res['oos_trn_repr'], res['oos_tst_repr'],
                unseen_no*np.ones(len(res['oos_trn_repr'])), unseen_no*np.ones(len(res['oos_tst_repr'])))
        inter_var['is_img_data'] = False
        return x_train, y_train, x_test, y_test, rest

    @register_var(argument=r"calcedreprold-mnistwo9-(?P<result_path>[a-zA-Z0-9_\.\/-]+)", shown_name="mnist")
    @staticmethod
    def calcedreprold_mnistwo9(auto_var, var_value, inter_var, result_path):
        import joblib
        _, y_train, _, y_test, _ = DatasetVarClass.mnistwo9(None, None, None)
        result_path = os.path.join("./results/oos_repr_bak/", result_path)
        res = joblib.load(result_path)
        x_train = res['trn_repr']
        x_test = res['tst_repr']
        res['trn_repr'].astype(np.float32, copy=False)
        res['tst_repr'].astype(np.float32, copy=False)
        res['oos_trn_repr'].astype(np.float32, copy=False)
        res['oos_tst_repr'].astype(np.float32, copy=False)
        rest = (res['oos_trn_repr'], res['oos_tst_repr'],
                9*np.ones(len(res['oos_trn_repr'])), 9*np.ones(len(res['oos_tst_repr'])))
        inter_var['is_img_data'] = False
        return x_train, y_train, x_test, y_test, rest

    @register_var(argument=r"calcedreprold-mnistwo(?P<unseen_no>[0-9])-(?P<result_path>[a-zA-Z0-9_\.\/-]+)", shown_name="mnist")
    @staticmethod
    def calcedreprold_mnist(auto_var, var_value, inter_var, unseen_no, result_path):
        from .mnist import mnist_leave_one_out
        import joblib
        unseen_no = int(unseen_no)
        _, y_train, _, y_test, _ = mnist_leave_one_out(unseen_no)
        result_path = os.path.join("./results/oos_repr_bak/", result_path)
        res = joblib.load(result_path)
        x_train = res['trn_repr']
        x_test = res['tst_repr']
        res['trn_repr'].astype(np.float32, copy=False)
        res['tst_repr'].astype(np.float32, copy=False)
        res['oos_trn_repr'].astype(np.float32, copy=False)
        res['oos_tst_repr'].astype(np.float32, copy=False)
        rest = (res['oos_trn_repr'], res['oos_tst_repr'],
                unseen_no*np.ones(len(res['oos_trn_repr'])), unseen_no*np.ones(len(res['oos_tst_repr'])))
        inter_var['is_img_data'] = False
        return x_train, y_train, x_test, y_test, rest

    @register_var(argument=r"mnist", shown_name="mnist")
    @staticmethod
    def mnist(auto_var, var_value, inter_var):
        from tensorflow.keras.datasets import mnist

        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train, x_test = x_train[:, :, :, np.newaxis], x_test[:, :, :, np.newaxis]
        x_train, x_test = x_train.astype(np.float32) / 255, x_test.astype(np.float32) / 255

        return x_train, y_train, x_test, y_test

    @register_var(argument=r"calcedrepr-cifar10wo9-(?P<result_path>[a-zA-Z0-9_\.\/-]+)", shown_name="cifar10wo0")
    @staticmethod
    def calcedrepr_cifar10wo9(auto_var, var_value, inter_var, result_path):
        import joblib
        _, y_train, _, y_test, oos = DatasetVarClass.cifar10wo9(None, None, None)
        result_path = os.path.join("./results/oos_repr/", result_path)
        res = joblib.load(result_path)
        res['trn_repr'].astype(np.float32, copy=False)
        res['tst_repr'].astype(np.float32, copy=False)
        res['oos_trn_repr'].astype(np.float32, copy=False)
        res['oos_tst_repr'].astype(np.float32, copy=False)
        x_train = res['trn_repr']
        x_test = res['tst_repr']
        rest = (res['oos_trn_repr'], res['oos_tst_repr'], oos[2], oos[3])
        inter_var['is_img_data'] = False
        return x_train, y_train, x_test, y_test, rest

    @register_var(argument=r"calcedrepr-cifar10wo8-(?P<result_path>[a-zA-Z0-9_\.\/-]+)", shown_name="cifar10wo0")
    @staticmethod
    def calcedrepr_cifar10wo8(auto_var, var_value, inter_var, result_path):
        import joblib
        _, y_train, _, y_test, oos = DatasetVarClass.cifar10wo8(None, None, None)
        result_path = os.path.join("./results/oos_repr/", result_path)
        res = joblib.load(result_path)
        res['trn_repr'].astype(np.float32, copy=False)
        res['tst_repr'].astype(np.float32, copy=False)
        res['oos_trn_repr'].astype(np.float32, copy=False)
        res['oos_tst_repr'].astype(np.float32, copy=False)
        x_train = res['trn_repr']
        x_test = res['tst_repr']
        rest = (res['oos_trn_repr'], res['oos_tst_repr'], oos[2], oos[3])
        inter_var['is_img_data'] = False
        return x_train, y_train, x_test, y_test, rest

    @register_var(argument=r"calcedrepr-cifar10wo7-(?P<result_path>[a-zA-Z0-9_\.\/-]+)", shown_name="cifar10wo0")
    @staticmethod
    def calcedrepr_cifar10wo7(auto_var, var_value, inter_var, result_path):
        import joblib
        _, y_train, _, y_test, oos = DatasetVarClass.cifar10wo7(None, None, None)
        result_path = os.path.join("./results/oos_repr/", result_path)
        res = joblib.load(result_path)
        res['trn_repr'].astype(np.float32, copy=False)
        res['tst_repr'].astype(np.float32, copy=False)
        res['oos_trn_repr'].astype(np.float32, copy=False)
        res['oos_tst_repr'].astype(np.float32, copy=False)
        x_train = res['trn_repr']
        x_test = res['tst_repr']
        rest = (res['oos_trn_repr'], res['oos_tst_repr'], oos[2], oos[3])
        inter_var['is_img_data'] = False
        return x_train, y_train, x_test, y_test, rest

    @register_var(argument=r"calcedrepr-cifar10wo6-(?P<result_path>[a-zA-Z0-9_\.\/-]+)", shown_name="cifar10wo0")
    @staticmethod
    def calcedrepr_cifar10wo6(auto_var, var_value, inter_var, result_path):
        import joblib
        _, y_train, _, y_test, oos = DatasetVarClass.cifar10wo6(None, None, None)
        result_path = os.path.join("./results/oos_repr/", result_path)
        res = joblib.load(result_path)
        res['trn_repr'].astype(np.float32, copy=False)
        res['tst_repr'].astype(np.float32, copy=False)
        res['oos_trn_repr'].astype(np.float32, copy=False)
        res['oos_tst_repr'].astype(np.float32, copy=False)
        x_train = res['trn_repr']
        x_test = res['tst_repr']
        rest = (res['oos_trn_repr'], res['oos_tst_repr'], oos[2], oos[3])
        inter_var['is_img_data'] = False
        return x_train, y_train, x_test, y_test, rest

    @register_var(argument=r"calcedrepr-cifar10wo5-(?P<result_path>[a-zA-Z0-9_\.\/-]+)", shown_name="cifar10wo0")
    @staticmethod
    def calcedrepr_cifar10wo5(auto_var, var_value, inter_var, result_path):
        import joblib
        _, y_train, _, y_test, oos = DatasetVarClass.cifar10wo5(None, None, None)
        result_path = os.path.join("./results/oos_repr/", result_path)
        res = joblib.load(result_path)
        res['trn_repr'].astype(np.float32, copy=False)
        res['tst_repr'].astype(np.float32, copy=False)
        res['oos_trn_repr'].astype(np.float32, copy=False)
        res['oos_tst_repr'].astype(np.float32, copy=False)
        x_train = res['trn_repr']
        x_test = res['tst_repr']
        rest = (res['oos_trn_repr'], res['oos_tst_repr'], oos[2], oos[3])
        inter_var['is_img_data'] = False
        return x_train, y_train, x_test, y_test, rest

    @register_var(argument=r"calcedrepr-cifar10wo4-(?P<result_path>[a-zA-Z0-9_\.\/-]+)", shown_name="cifar10wo0")
    @staticmethod
    def calcedrepr_cifar10wo4(auto_var, var_value, inter_var, result_path):
        import joblib
        _, y_train, _, y_test, oos = DatasetVarClass.cifar10wo4(None, None, None)
        result_path = os.path.join("./results/oos_repr/", result_path)
        res = joblib.load(result_path)
        res['trn_repr'].astype(np.float32, copy=False)
        res['tst_repr'].astype(np.float32, copy=False)
        res['oos_trn_repr'].astype(np.float32, copy=False)
        res['oos_tst_repr'].astype(np.float32, copy=False)
        x_train = res['trn_repr']
        x_test = res['tst_repr']
        rest = (res['oos_trn_repr'], res['oos_tst_repr'], oos[2], oos[3])
        inter_var['is_img_data'] = False
        return x_train, y_train, x_test, y_test, rest

    @register_var(argument=r"calcedrepr-cifar10wo3-(?P<result_path>[a-zA-Z0-9_\.\/-]+)", shown_name="cifar10wo0")
    @staticmethod
    def calcedrepr_cifar10wo3(auto_var, var_value, inter_var, result_path):
        import joblib
        _, y_train, _, y_test, oos = DatasetVarClass.cifar10wo3(None, None, None)
        result_path = os.path.join("./results/oos_repr/", result_path)
        res = joblib.load(result_path)
        res['trn_repr'].astype(np.float32, copy=False)
        res['tst_repr'].astype(np.float32, copy=False)
        res['oos_trn_repr'].astype(np.float32, copy=False)
        res['oos_tst_repr'].astype(np.float32, copy=False)
        x_train = res['trn_repr']
        x_test = res['tst_repr']
        rest = (res['oos_trn_repr'], res['oos_tst_repr'], oos[2], oos[3])
        inter_var['is_img_data'] = False
        return x_train, y_train, x_test, y_test, rest

    @register_var(argument=r"calcedrepr-cifar10wo2-(?P<result_path>[a-zA-Z0-9_\.\/-]+)", shown_name="cifar10wo0")
    @staticmethod
    def calcedrepr_cifar10wo2(auto_var, var_value, inter_var, result_path):
        import joblib
        _, y_train, _, y_test, oos = DatasetVarClass.cifar10wo2(None, None, None)
        result_path = os.path.join("./results/oos_repr/", result_path)
        res = joblib.load(result_path)
        res['trn_repr'].astype(np.float32, copy=False)
        res['tst_repr'].astype(np.float32, copy=False)
        res['oos_trn_repr'].astype(np.float32, copy=False)
        res['oos_tst_repr'].astype(np.float32, copy=False)
        x_train = res['trn_repr']
        x_test = res['tst_repr']
        rest = (res['oos_trn_repr'], res['oos_tst_repr'], oos[2], oos[3])
        inter_var['is_img_data'] = False
        return x_train, y_train, x_test, y_test, rest

    @register_var(argument=r"calcedrepr-cifar10wo1-(?P<result_path>[a-zA-Z0-9_\.\/-]+)", shown_name="cifar10wo0")
    @staticmethod
    def calcedrepr_cifar10wo1(auto_var, var_value, inter_var, result_path):
        import joblib
        _, y_train, _, y_test, oos = DatasetVarClass.cifar10wo1(None, None, None)
        result_path = os.path.join("./results/oos_repr/", result_path)
        res = joblib.load(result_path)
        res['trn_repr'].astype(np.float32, copy=False)
        res['tst_repr'].astype(np.float32, copy=False)
        res['oos_trn_repr'].astype(np.float32, copy=False)
        res['oos_tst_repr'].astype(np.float32, copy=False)
        x_train = res['trn_repr']
        x_test = res['tst_repr']
        rest = (res['oos_trn_repr'], res['oos_tst_repr'], oos[2], oos[3])
                #0*np.ones(len(res['oos_trn_repr'])), 0*np.ones(len(res['oos_tst_repr'])))
        inter_var['is_img_data'] = False
        return x_train, y_train, x_test, y_test, rest

    @register_var(argument=r"calcedrepr-cifar10wo0-(?P<result_path>[a-zA-Z0-9_\.\/-]+)", shown_name="cifar10wo0")
    @staticmethod
    def calcedrepr_cifar10wo0(auto_var, inter_var, result_path):
        import joblib
        _, y_train, _, y_test, oos = DatasetVarClass.cifar10wo0(None, None, None)
        result_path = os.path.join("./results/oos_repr/", result_path)
        res = joblib.load(result_path)
        res['trn_repr'].astype(np.float32, copy=False)
        res['tst_repr'].astype(np.float32, copy=False)
        res['oos_trn_repr'].astype(np.float32, copy=False)
        res['oos_tst_repr'].astype(np.float32, copy=False)
        x_train = res['trn_repr']
        x_test = res['tst_repr']
        rest = (res['oos_trn_repr'], res['oos_tst_repr'], oos[2], oos[3])
                #0*np.ones(len(res['oos_trn_repr'])), 0*np.ones(len(res['oos_tst_repr'])))
        inter_var['is_img_data'] = False
        return x_train, y_train, x_test, y_test, rest

    @register_var(argument=r"calcedrepr-cifar10-(?P<result_path>[a-zA-Z0-9_\.\/-]+)", shown_name="cifar10wo0")
    @staticmethod
    def calcedrepr_cifar10(auto_var, inter_var, result_path):
        import joblib
        _, y_train, _, y_test, _ = DatasetVarClass.cifar10(None, None, None)
        result_path = os.path.join("./results/oos_repr/", result_path)
        res = joblib.load(result_path)
        res['trn_repr'].astype(np.float32, copy=False)
        res['tst_repr'].astype(np.float32, copy=False)
        x_train = res['trn_repr']
        x_test = res['tst_repr']
        inter_var['is_img_data'] = False
        return x_train, y_train, x_test, y_test, None

    @register_var(argument=r"calcedrepr-cifar100coarse-(?P<result_path>[a-zA-Z0-9_\.\/-]+)", shown_name="cifar100")
    @staticmethod
    def calcedrepr_cifar100coarse(auto_var, inter_var, result_path):
        import joblib
        _, y_train, _, y_test, _ = DatasetVarClass.cifar100coarse(auto_var)
        result_path = os.path.join("./results/oos_repr/", result_path)
        res = joblib.load(result_path)
        res['trn_repr'].astype(np.float32, copy=False)
        res['tst_repr'].astype(np.float32, copy=False)
        x_train = res['trn_repr']
        x_test = res['tst_repr']
        inter_var['is_img_data'] = False
        return x_train, y_train, x_test, y_test, None

    @register_var(argument=r"calcedrepr-cifar100coarsewo(?P<unseen>\d)-(?P<result_path>[a-zA-Z0-9_\.\/-]+)", shown_name="cifar100")
    @staticmethod
    def calcedrepr_cifar100coarsewo(auto_var, inter_var, result_path, unseen):
        import joblib
        _, y_train, _, y_test, oos = DatasetVarClass.cifar100coarsewo(auto_var, unseen)
        result_path = os.path.join("./results/oos_repr/", result_path)
        res = joblib.load(result_path)
        res['trn_repr'].astype(np.float32, copy=False)
        res['tst_repr'].astype(np.float32, copy=False)
        res['oos_trn_repr'].astype(np.float32, copy=False)
        res['oos_tst_repr'].astype(np.float32, copy=False)
        x_train = res['trn_repr']
        x_test = res['tst_repr']
        rest = (res['oos_trn_repr'], res['oos_tst_repr'], oos[2], oos[3])
        inter_var['is_img_data'] = False
        return x_train, y_train, x_test, y_test, rest

    @register_var(argument=r"calcedrepr-cifar100coarsewo0-(?P<result_path>[a-zA-Z0-9_\.\/-]+)", shown_name="cifar10wo0")
    @staticmethod
    def calcedrepr_cifar100coarsewo0(auto_var, var_value, inter_var, result_path):
        import joblib
        _, y_train, _, y_test, oos = DatasetVarClass.cifar100coarsewo0(None, None, None)
        result_path = os.path.join("./results/oos_repr/", result_path)
        res = joblib.load(result_path)
        res['trn_repr'].astype(np.float32, copy=False)
        res['tst_repr'].astype(np.float32, copy=False)
        res['oos_trn_repr'].astype(np.float32, copy=False)
        res['oos_tst_repr'].astype(np.float32, copy=False)
        x_train = res['trn_repr']
        x_test = res['tst_repr']
        rest = (res['oos_trn_repr'], res['oos_tst_repr'], oos[2], oos[3])
                #0*np.ones(len(res['oos_trn_repr'])), 0*np.ones(len(res['oos_tst_repr'])))
        inter_var['is_img_data'] = False
        return x_train, y_train, x_test, y_test, rest

    @cache_outputs(cache_dir='./data/caches/dataset/')
    @register_var(argument=r"crepr-WRN_40_10-cifar10wo9-(?P<model_path>[a-zA-Z0-9_\.\/-]+)", shown_name="mnist")
    @staticmethod
    def crepr_WRN_40_10_cifar10wo9(auto_var, var_value, inter_var, model_path):
        import torch
        from ..models.torch_utils.archs import WRN_40_10
        model_path = os.path.join("./models/out_of_sample/", model_path)
        trnX, trny, tstX, tsty, rest = DatasetVarClass.cifar10wo9(None, None, None)
        model = WRN_40_10(n_classes=9, n_channels=3)
        model.load_state_dict(torch.load(model_path)['model_state_dict'])
        model.to("cuda")

        def extract_x(X):
            X = X.transpose(0, 3, 1, 2)
            dset = torch.utils.data.TensorDataset(torch.from_numpy(X))
            loader = torch.utils.data.DataLoader(
                dset, batch_size=64, shuffle=False, num_workers=12)
            ret = []
            for (x, ) in tqdm(loader, desc="[crepr]"):
                x = model.get_repr(x.to("cuda"), rtype="block2")
                ret.append(x.cpu().detach().numpy().astype(np.float32))
            del loader
            return np.concatenate(ret, axis=0)

        trnX = extract_x(trnX)
        tstX = extract_x(tstX)
        rest = list(rest)
        rest[0] = extract_x(rest[0])
        rest[1] = extract_x(rest[1])
        inter_var['is_img_data'] = False

        del model
        gc.collect()
        return trnX, trny, tstX, tsty, rest

    @cache_outputs(cache_dir='./data/caches/dataset/')
    @register_var(argument=r"crepr-WRN_40_10-cifar10wo8-(?P<model_path>[a-zA-Z0-9_\.\/-]+)", shown_name="mnist")
    @staticmethod
    def crepr_WRN_40_10_cifar10wo8(auto_var, var_value, inter_var, model_path):
        import torch
        from ..models.torch_utils.archs import WRN_40_10
        model_path = os.path.join("./models/out_of_sample/", model_path)
        trnX, trny, tstX, tsty, rest = DatasetVarClass.cifar10wo8(None, None, None)
        model = WRN_40_10(n_classes=9, n_channels=3)
        model.load_state_dict(torch.load(model_path)['model_state_dict'])
        model.to("cuda")

        def extract_x(X):
            X = X.transpose(0, 3, 1, 2)
            dset = torch.utils.data.TensorDataset(torch.from_numpy(X))
            loader = torch.utils.data.DataLoader(
                dset, batch_size=64, shuffle=False, num_workers=12)
            ret = []
            for (x, ) in tqdm(loader, desc="[crepr]"):
                x = model.get_repr(x.to("cuda"), rtype="block2")
                ret.append(x.cpu().detach().numpy().astype(np.float32))
            del loader
            return np.concatenate(ret, axis=0)

        trnX = extract_x(trnX)
        tstX = extract_x(tstX)
        rest = list(rest)
        rest[0] = extract_x(rest[0])
        rest[1] = extract_x(rest[1])
        inter_var['is_img_data'] = False

        del model
        gc.collect()
        return trnX, trny, tstX, tsty, rest

    @cache_outputs(cache_dir='./data/caches/dataset/')
    @register_var(argument=r"crepr-WRN_40_10-cifar10wo7-(?P<model_path>[a-zA-Z0-9_\.\/-]+)", shown_name="mnist")
    @staticmethod
    def crepr_WRN_40_10_cifar10wo7(auto_var, var_value, inter_var, model_path):
        import torch
        from ..models.torch_utils.archs import WRN_40_10
        model_path = os.path.join("./models/out_of_sample/", model_path)
        trnX, trny, tstX, tsty, rest = DatasetVarClass.cifar10wo7(None, None, None)
        model = WRN_40_10(n_classes=9, n_channels=3)
        model.load_state_dict(torch.load(model_path)['model_state_dict'])
        model.to("cuda")

        def extract_x(X):
            X = X.transpose(0, 3, 1, 2)
            dset = torch.utils.data.TensorDataset(torch.from_numpy(X))
            loader = torch.utils.data.DataLoader(
                dset, batch_size=64, shuffle=False, num_workers=12)
            ret = []
            for (x, ) in tqdm(loader, desc="[crepr]"):
                x = model.get_repr(x.to("cuda"), rtype="block2")
                ret.append(x.cpu().detach().numpy().astype(np.float32))
            del loader
            return np.concatenate(ret, axis=0)

        trnX = extract_x(trnX)
        tstX = extract_x(tstX)
        rest = list(rest)
        rest[0] = extract_x(rest[0])
        rest[1] = extract_x(rest[1])
        inter_var['is_img_data'] = False

        del model
        gc.collect()
        return trnX, trny, tstX, tsty, rest

    @cache_outputs(cache_dir='./data/caches/dataset/')
    @register_var(argument=r"crepr-WRN_40_10-cifar10wo6-(?P<model_path>[a-zA-Z0-9_\.\/-]+)", shown_name="mnist")
    @staticmethod
    def crepr_WRN_40_10_cifar10wo6(auto_var, var_value, inter_var, model_path):
        import torch
        from ..models.torch_utils.archs import WRN_40_10
        model_path = os.path.join("./models/out_of_sample/", model_path)
        trnX, trny, tstX, tsty, rest = DatasetVarClass.cifar10wo6(None, None, None)
        model = WRN_40_10(n_classes=9, n_channels=3)
        model.load_state_dict(torch.load(model_path)['model_state_dict'])
        model.to("cuda")

        def extract_x(X):
            X = X.transpose(0, 3, 1, 2)
            dset = torch.utils.data.TensorDataset(torch.from_numpy(X))
            loader = torch.utils.data.DataLoader(
                dset, batch_size=64, shuffle=False, num_workers=12)
            ret = []
            for (x, ) in tqdm(loader, desc="[crepr]"):
                x = model.get_repr(x.to("cuda"), rtype="block2")
                ret.append(x.cpu().detach().numpy().astype(np.float32))
            del loader
            return np.concatenate(ret, axis=0)

        trnX = extract_x(trnX)
        tstX = extract_x(tstX)
        rest = list(rest)
        rest[0] = extract_x(rest[0])
        rest[1] = extract_x(rest[1])
        inter_var['is_img_data'] = False

        del model
        gc.collect()
        return trnX, trny, tstX, tsty, rest

    @cache_outputs(cache_dir='./data/caches/dataset/')
    @register_var(argument=r"crepr-WRN_40_10-cifar10wo5-(?P<model_path>[a-zA-Z0-9_\.\/-]+)", shown_name="mnist")
    @staticmethod
    def crepr_WRN_40_10_cifar10wo5(auto_var, var_value, inter_var, model_path):
        import torch
        from ..models.torch_utils.archs import WRN_40_10
        model_path = os.path.join("./models/out_of_sample/", model_path)
        trnX, trny, tstX, tsty, rest = DatasetVarClass.cifar10wo5(None, None, None)
        model = WRN_40_10(n_classes=9, n_channels=3)
        model.load_state_dict(torch.load(model_path)['model_state_dict'])
        model.to("cuda")

        def extract_x(X):
            X = X.transpose(0, 3, 1, 2)
            dset = torch.utils.data.TensorDataset(torch.from_numpy(X))
            loader = torch.utils.data.DataLoader(
                dset, batch_size=64, shuffle=False, num_workers=12)
            ret = []
            for (x, ) in tqdm(loader, desc="[crepr]"):
                x = model.get_repr(x.to("cuda"), rtype="block2")
                ret.append(x.cpu().detach().numpy().astype(np.float32))
            del loader
            return np.concatenate(ret, axis=0)

        trnX = extract_x(trnX)
        tstX = extract_x(tstX)
        rest = list(rest)
        rest[0] = extract_x(rest[0])
        rest[1] = extract_x(rest[1])
        inter_var['is_img_data'] = False

        del model
        gc.collect()
        return trnX, trny, tstX, tsty, rest

    @cache_outputs(cache_dir='./data/caches/dataset/')
    @register_var(argument=r"crepr-WRN_40_10-cifar10wo4-(?P<model_path>[a-zA-Z0-9_\.\/-]+)", shown_name="")
    @staticmethod
    def crepr_WRN_40_10_cifar10wo4(auto_var, var_value, inter_var, model_path):
        import torch
        from ..models.torch_utils.archs import WRN_40_10
        model_path = os.path.join("./models/out_of_sample/", model_path)
        trnX, trny, tstX, tsty, rest = DatasetVarClass.cifar10wo4(None, None, None)
        model = WRN_40_10(n_classes=9, n_channels=3)
        model.load_state_dict(torch.load(model_path)['model_state_dict'])
        model.to("cuda")

        def extract_x(X):
            X = X.transpose(0, 3, 1, 2)
            dset = torch.utils.data.TensorDataset(torch.from_numpy(X))
            loader = torch.utils.data.DataLoader(
                dset, batch_size=64, shuffle=False, num_workers=12)
            ret = []
            for (x, ) in tqdm(loader, desc="[crepr]"):
                x = model.get_repr(x.to("cuda"), rtype="block2")
                ret.append(x.cpu().detach().numpy().astype(np.float32))
            del loader
            return np.concatenate(ret, axis=0)

        trnX = extract_x(trnX)
        tstX = extract_x(tstX)
        rest = list(rest)
        rest[0] = extract_x(rest[0])
        rest[1] = extract_x(rest[1])
        inter_var['is_img_data'] = False

        del model
        gc.collect()
        return trnX, trny, tstX, tsty, rest

    @cache_outputs(cache_dir='./data/caches/dataset/')
    @register_var(argument=r"crepr-WRN_40_10-cifar10wo3-(?P<model_path>[a-zA-Z0-9_\.\/-]+)", shown_name="")
    @staticmethod
    def crepr_WRN_40_10_cifar10wo3(auto_var, var_value, inter_var, model_path):
        import torch
        from ..models.torch_utils.archs import WRN_40_10
        model_path = os.path.join("./models/out_of_sample/", model_path)
        trnX, trny, tstX, tsty, rest = DatasetVarClass.cifar10wo3(None, None, None)
        model = WRN_40_10(n_classes=9, n_channels=3)
        model.load_state_dict(torch.load(model_path)['model_state_dict'])
        model.to("cuda")

        def extract_x(X):
            X = X.transpose(0, 3, 1, 2)
            dset = torch.utils.data.TensorDataset(torch.from_numpy(X))
            loader = torch.utils.data.DataLoader(
                dset, batch_size=64, shuffle=False, num_workers=12)
            ret = []
            for (x, ) in tqdm(loader, desc="[crepr]"):
                x = model.get_repr(x.to("cuda"), rtype="block2")
                ret.append(x.cpu().detach().numpy().astype(np.float32))
            del loader
            return np.concatenate(ret, axis=0)

        trnX = extract_x(trnX)
        tstX = extract_x(tstX)
        rest = list(rest)
        rest[0] = extract_x(rest[0])
        rest[1] = extract_x(rest[1])
        inter_var['is_img_data'] = False

        del model
        gc.collect()
        return trnX, trny, tstX, tsty, rest

    @cache_outputs(cache_dir='./data/caches/dataset/')
    @register_var(argument=r"crepr-WRN_40_10-cifar10wo2-(?P<model_path>[a-zA-Z0-9_\.\/-]+)", shown_name="mnist")
    @staticmethod
    def crepr_WRN_40_10_cifar10wo2(auto_var, var_value, inter_var, model_path):
        import torch
        from ..models.torch_utils.archs import WRN_40_10
        model_path = os.path.join("./models/out_of_sample/", model_path)
        trnX, trny, tstX, tsty, rest = DatasetVarClass.cifar10wo2(None, None, None)
        model = WRN_40_10(n_classes=9, n_channels=3)
        model.load_state_dict(torch.load(model_path)['model_state_dict'])
        model.to("cuda")

        def extract_x(X):
            X = X.transpose(0, 3, 1, 2)
            dset = torch.utils.data.TensorDataset(torch.from_numpy(X))
            loader = torch.utils.data.DataLoader(
                dset, batch_size=64, shuffle=False, num_workers=12)
            ret = []
            for (x, ) in tqdm(loader, desc="[crepr]"):
                x = model.get_repr(x.to("cuda"), rtype="block2")
                ret.append(x.cpu().detach().numpy().astype(np.float32))
            del loader
            return np.concatenate(ret, axis=0)

        trnX = extract_x(trnX)
        tstX = extract_x(tstX)
        rest = list(rest)
        rest[0] = extract_x(rest[0])
        rest[1] = extract_x(rest[1])
        inter_var['is_img_data'] = False

        del model
        gc.collect()
        return trnX, trny, tstX, tsty, rest

    @cache_outputs(cache_dir='./data/caches/dataset/')
    @register_var(argument=r"crepr-WRN_40_10-cifar10wo1-(?P<model_path>[a-zA-Z0-9_\.\/-]+)", shown_name="mnist")
    @staticmethod
    def crepr_WRN_40_10_cifar10wo1(auto_var, var_value, inter_var, model_path):
        import torch
        from ..models.torch_utils.archs import WRN_40_10
        model_path = os.path.join("./models/out_of_sample/", model_path)
        trnX, trny, tstX, tsty, rest = DatasetVarClass.cifar10wo1(None, None, None)
        model = WRN_40_10(n_classes=9, n_channels=3)
        model.load_state_dict(torch.load(model_path)['model_state_dict'])
        model.to("cuda")

        def extract_x(X):
            X = X.transpose(0, 3, 1, 2)
            dset = torch.utils.data.TensorDataset(torch.from_numpy(X))
            loader = torch.utils.data.DataLoader(
                dset, batch_size=64, shuffle=False, num_workers=12)
            ret = []
            for (x, ) in tqdm(loader, desc="[crepr]"):
                x = model.get_repr(x.to("cuda"), rtype="block2")
                ret.append(x.cpu().detach().numpy().astype(np.float32))
            del loader
            return np.concatenate(ret, axis=0)

        trnX = extract_x(trnX)
        tstX = extract_x(tstX)
        rest = list(rest)
        rest[0] = extract_x(rest[0])
        rest[1] = extract_x(rest[1])
        inter_var['is_img_data'] = False

        del model
        gc.collect()
        return trnX, trny, tstX, tsty, rest

    @cache_outputs(cache_dir='./data/caches/dataset/')
    @register_var(argument=r"crepr-WRN_40_10-cifar10wo0-(?P<model_path>[a-zA-Z0-9_\.\/-]+)", shown_name="mnist")
    @staticmethod
    def crepr_WRN_40_10_cifar10wo0(auto_var, var_value, inter_var, model_path):
        import torch
        from ..models.torch_utils.archs import WRN_40_10
        model_path = os.path.join("./models/out_of_sample/", model_path)
        trnX, trny, tstX, tsty, rest = DatasetVarClass.cifar10wo0(None, None, None)
        model = WRN_40_10(n_classes=9, n_channels=3)
        model.load_state_dict(torch.load(model_path)['model_state_dict'])
        model.to("cuda")

        def extract_x(X):
            X = X.transpose(0, 3, 1, 2)
            dset = torch.utils.data.TensorDataset(torch.from_numpy(X))
            loader = torch.utils.data.DataLoader(
                dset, batch_size=64, shuffle=False, num_workers=12)
            ret = []
            for (x, ) in tqdm(loader, desc="[crepr]"):
                x = model.get_repr(x.to("cuda"), rtype="block2")
                ret.append(x.cpu().detach().numpy().astype(np.float32))
            del loader
            return np.concatenate(ret, axis=0)

        trnX = extract_x(trnX)
        tstX = extract_x(tstX)
        rest = list(rest)
        rest[0] = extract_x(rest[0])
        rest[1] = extract_x(rest[1])
        inter_var['is_img_data'] = False

        del model
        gc.collect()
        return trnX, trny, tstX, tsty, rest

    @register_var(argument=r"cifar10wo0", shown_name="Cifar10 without 0 (airplane)")
    @staticmethod
    def cifar10wo0(auto_var, var_value, inter_var):
        """without the airplane class"""
        from tensorflow.keras.datasets import cifar10

        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        y_train, y_test = y_train.reshape(-1), y_test.reshape(-1)
        x_train, x_test = x_train.astype(np.float32) / 255, x_test.astype(np.float32) / 255
        rest = (x_train[y_train==0], x_test[y_test==0],
                0*np.ones((y_train==0).sum()), 0*np.ones((y_test==0).sum()))
        x_train, x_test = x_train[y_train != 0], x_test[y_test != 0]
        y_train, y_test = y_train[y_train != 0], y_test[y_test != 0]
        y_train[y_train == 9] = 0
        y_test[y_test == 9] = 0

        return x_train, y_train, x_test, y_test, rest

    @register_var(argument=r"cifar10wo9", shown_name="Cifar10 without 9 (truck)")
    @staticmethod
    def cifar10wo9(auto_var, var_value, inter_var):
        from tensorflow.keras.datasets import cifar10

        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        y_train, y_test = y_train.reshape(-1), y_test.reshape(-1)
        x_train, x_test = x_train.astype(np.float32) / 255, x_test.astype(np.float32) / 255
        rest = (x_train[y_train==9], x_test[y_test==9],
                9*np.ones((y_train==9).sum()), 9*np.ones((y_test==9).sum()))
        x_train, x_test = x_train[y_train != 9], x_test[y_test != 9]
        y_train, y_test = y_train[y_train != 9], y_test[y_test != 9]

        return x_train, y_train, x_test, y_test, rest

    @register_var(argument=r"cifar10wo1", shown_name="")
    @staticmethod
    def cifar10wo1(auto_var, var_value, inter_var):
        # automobile
        from .cifar import cifar10_leave_one_out
        return cifar10_leave_one_out(1)

    @register_var(argument=r"cifar10wo2", shown_name="")
    @staticmethod
    def cifar10wo2(auto_var, var_value, inter_var):
        # bird
        from .cifar import cifar10_leave_one_out
        return cifar10_leave_one_out(2)

    @register_var(argument=r"cifar10wo3", shown_name="")
    @staticmethod
    def cifar10wo3(auto_var, var_value, inter_var):
        # cat
        from .cifar import cifar10_leave_one_out
        return cifar10_leave_one_out(3)

    @register_var(argument=r"cifar10wo4", shown_name="")
    @staticmethod
    def cifar10wo4(auto_var, var_value, inter_var):
        # deer
        from .cifar import cifar10_leave_one_out
        return cifar10_leave_one_out(4)

    @register_var(argument=r"cifar10wo5", shown_name="")
    @staticmethod
    def cifar10wo5(auto_var, var_value, inter_var):
        # dog
        from .cifar import cifar10_leave_one_out
        return cifar10_leave_one_out(5)

    @register_var(argument=r"cifar10wo6", shown_name="")
    @staticmethod
    def cifar10wo6(auto_var, var_value, inter_var):
        # frog
        from .cifar import cifar10_leave_one_out
        return cifar10_leave_one_out(6)

    @register_var(argument=r"cifar10wo7", shown_name="")
    @staticmethod
    def cifar10wo7(auto_var, var_value, inter_var):
        # horse
        from .cifar import cifar10_leave_one_out
        return cifar10_leave_one_out(7)

    @register_var(argument=r"cifar10wo8", shown_name="")
    @staticmethod
    def cifar10wo8(auto_var, var_value, inter_var):
        # ship
        from .cifar import cifar10_leave_one_out
        return cifar10_leave_one_out(8)

    @register_var(argument=r"cifar10", shown_name="Cifar10")
    @staticmethod
    def cifar10(auto_var, var_value, inter_var):
        from tensorflow.keras.datasets import cifar10

        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        y_train, y_test = y_train.reshape(-1), y_test.reshape(-1)
        x_train, x_test = x_train.astype(np.float32) / 255, x_test.astype(np.float32) / 255

        return x_train, y_train, x_test, y_test, None

    @register_var(argument=r"cifar100coarse", shown_name="Cifar10")
    @staticmethod
    def cifar100coarse(auto_var):
        from tensorflow.keras.datasets import cifar100

        (x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='coarse')
        y_train, y_test = y_train.reshape(-1), y_test.reshape(-1)
        x_train, x_test = x_train.astype(np.float32) / 255, x_test.astype(np.float32) / 255

        return x_train, y_train, x_test, y_test, None

    @register_var(argument=r"cifar100wosp1", shown_name="Cifar10 without rand each superclass")
    @staticmethod
    def cifar100wosp1(auto_var, inter_var):
        from copy import deepcopy
        from tensorflow.keras.datasets import cifar100
        from .cifar import fine_labels, coarse_labels, superclass_mapping

        (trnX, trny), (tstX, tsty) = cifar100.load_data('fine')
        trny, tsty = trny.reshape(-1), tsty.reshape(-1)
        trnX, tstX = trnX.astype(np.float32) / 255, tstX.astype(np.float32) / 255

        random_state = np.random.RandomState(auto_var.get_var("random_seed"))
        unseen_idx = random_state.choice(np.arange(5), size=20, replace=True)

        unseen_categories = []
        for i, cat in enumerate(unseen_idx):
            unseen_categories.append(superclass_mapping[coarse_labels[i]][cat])

        oodtrX, oodtsX, oodtry, oodtsy = [], [], [], []
        now = 0
        lbl_mapping = {}
        for i in range(100):
            if fine_labels[i] in unseen_categories:
                oodtrX.append(trnX[trny == i])
                oodtry.append(trny[trny == i])
                oodtsX.append(tstX[tsty == i])
                oodtsy.append(tsty[tsty == i])
                trnX = trnX[trny != i]
                trny = trny[trny != i]
                tstX = tstX[tsty != i]
                tsty = tsty[tsty != i]
            else:
                if now != i:
                    trny[trny == i] = now
                    tsty[tsty == i] = now
                lbl_mapping[i] = now
                now += 1

        oodtrX, oodtry = np.concatenate(oodtrX, axis=0), np.concatenate(oodtry)
        oodtsX, oodtsy = np.concatenate(oodtsX, axis=0), np.concatenate(oodtsy)
        rest = (oodtrX, oodtsX, oodtry, oodtsy)
        inter_var['unseen_categories'] = unseen_categories
        inter_var['lbl_mapping'] = lbl_mapping

        return trnX, trny, tstX, tsty, rest

    #@register_var(argument=r"cifar100coarsewo0", shown_name="Cifar10 without 9 (truck)")
    #@staticmethod
    #def cifar100coarsewo0(auto_var, var_value, inter_var):
    #    # remove aquatic mammals
    #    from tensorflow.keras.datasets import cifar100
    #    (x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode="coarse")
    #    y_train, y_test = y_train.reshape(-1), y_test.reshape(-1)
    #    x_train, x_test = x_train.astype(np.float32) / 255, x_test.astype(np.float32) / 255
    #    rest = (x_train[y_train==0], x_test[y_test==0],
    #            0*np.ones((y_train==0).sum()), 0*np.ones((y_test==0).sum()))
    #    x_train, x_test = x_train[y_train != 0], x_test[y_test != 0]
    #    y_train, y_test = y_train[y_train != 0], y_test[y_test != 0]
    #    y_train[y_train == 19] = 0
    #    y_test[y_test == 19] = 0
    #    return x_train, y_train, x_test, y_test, rest

    @register_var(argument=r"cifar100coarsewo(?P<woc>\d)", shown_name="Cifar100")
    @staticmethod
    def cifar100coarsewo(auto_var, woc):
        # remove aquatic mammals
        from .cifar import cifar100coarse_leave_one_out
        woc = int(woc)
        return cifar100coarse_leave_one_out(woc)

    @cache_outputs(cache_dir='./data/caches/dataset/')
    @register_var(argument=r"crepr-WRN_40_10-cifar100wosp1-(?P<model_path>[a-zA-Z0-9_\.\/-]+)", shown_name="")
    @staticmethod
    def crepr_WRN_40_10_cifar100wosp1(auto_var, inter_var, model_path):
        import torch
        from ..models.torch_utils.archs import WRN_40_10
        model_path = os.path.join("./models/out_of_sample/", model_path)
        trnX, trny, tstX, tsty, rest = DatasetVarClass.cifar100wosp1(auto_var, inter_var)
        model = WRN_40_10(n_classes=80, n_channels=3)
        model.load_state_dict(torch.load(model_path)['model_state_dict'])
        model.to("cuda")

        def extract_x(X):
            X = X.transpose(0, 3, 1, 2)
            dset = torch.utils.data.TensorDataset(torch.from_numpy(X))
            loader = torch.utils.data.DataLoader(
                dset, batch_size=64, shuffle=False, num_workers=12)
            ret = []
            for (x, ) in tqdm(loader, desc="[crepr]"):
                x = model.get_repr(x.to("cuda"), rtype="block2")
                ret.append(x.cpu().detach().numpy().astype(np.float32))
            del loader
            return np.concatenate(ret, axis=0)

        trnX = extract_x(trnX)
        tstX = extract_x(tstX)
        rest = list(rest)
        rest[0] = extract_x(rest[0])
        rest[1] = extract_x(rest[1])
        inter_var['is_img_data'] = False

        del model
        gc.collect()
        return trnX, trny, tstX, tsty, rest

    @cache_outputs(cache_dir='./data/caches/dataset/')
    @register_var(argument=r"crepr-WRN_40_10-cifar100coarsewo(?P<unseen>\d)-(?P<model_path>[a-zA-Z0-9_\.\/-]+)", shown_name="")
    @staticmethod
    def crepr_WRN_40_10_cifar10coarsewo(auto_var, unseen, model_path):
        import torch
        from ..models.torch_utils.archs import WRN_40_10
        model_path = os.path.join("./models/out_of_sample/", model_path)
        trnX, trny, tstX, tsty, rest = DatasetVarClass.cifar100coarsewo(auto_var, unseen)
        model = WRN_40_10(n_classes=19, n_channels=3)
        model.load_state_dict(torch.load(model_path)['model_state_dict'])
        model.to("cuda")

        def extract_x(X):
            X = X.transpose(0, 3, 1, 2)
            dset = torch.utils.data.TensorDataset(torch.from_numpy(X))
            loader = torch.utils.data.DataLoader(
                dset, batch_size=64, shuffle=False, num_workers=12)
            ret = []
            for (x, ) in tqdm(loader, desc="[crepr]"):
                x = model.get_repr(x.to("cuda"), rtype="block2")
                ret.append(x.cpu().detach().numpy().astype(np.float32))
            del loader
            return np.concatenate(ret, axis=0)

        trnX = extract_x(trnX)
        tstX = extract_x(tstX)
        rest = list(rest)
        rest[0] = extract_x(rest[0])
        rest[1] = extract_x(rest[1])
        auto_var.inter_var['is_img_data'] = False

        del model
        gc.collect()
        return trnX, trny, tstX, tsty, rest

    @register_var(argument=r"calcedrepr-cifar100wosp1-(?P<result_path>[a-zA-Z0-9_\.\/-]+)", shown_name="cifar100coarsewo0")
    @staticmethod
    def calcedrepr_cifar100wosp1(auto_var, inter_var, result_path, data_dir="./data/"):
        import joblib
        _, y_train, _, y_test, oos = DatasetVarClass.cifar100wosp1(auto_var, inter_var)
        result_path = os.path.join("./results/oos_repr/", result_path)
        res = joblib.load(result_path)
        res['trn_repr'].astype(np.float32, copy=False)
        res['tst_repr'].astype(np.float32, copy=False)
        res['oos_trn_repr'].astype(np.float32, copy=False)
        res['oos_tst_repr'].astype(np.float32, copy=False)
        x_train = res['trn_repr']
        x_test = res['tst_repr']
        rest = (res['oos_trn_repr'], res['oos_tst_repr'], oos[2], oos[3])
        inter_var['is_img_data'] = False
        return x_train, y_train, x_test, y_test, rest

    @register_var(argument=r"fourq-(?P<n_samples>\d+)-(?P<noisy_level>\d+\.\d+|\d+)", shown_name="four quadrant")
    @staticmethod
    def fourq(auto_var, inter_var, n_samples, noisy_level):
        """
        """
        random_state = np.random.RandomState(auto_var.get_var("random_seed"))
        n_samples = int(n_samples)
        noisy_level = float(noisy_level)
        noisy_level = [[noisy_level, 0], [0, noisy_level]]

        def sample(train=False, n_samples=n_samples):
            q1 = random_state.multivariate_normal([1, 1], noisy_level, size=n_samples)
            q2 = random_state.multivariate_normal([1, -1], noisy_level, size=n_samples)
            q3 = random_state.multivariate_normal([-1, 1], noisy_level, size=n_samples)
            q4 = random_state.multivariate_normal([-1, -1], noisy_level, size=n_samples)

            if train:
                X = np.vstack((q1, q2, q3))
                y = np.concatenate((np.zeros(n_samples), np.ones(n_samples), 2*np.ones(n_samples)))
            else:
                X = np.vstack((q1, q2, q3, q4))
                y = np.concatenate((np.zeros(n_samples), np.ones(n_samples),
                                    2*np.ones(n_samples), 3*np.ones(n_samples)))

            return X, y.astype(np.int)

        inter_var['is_img_data'] = False
        trnX, trny = sample(True, n_samples)
        tstX, tsty = sample(True, 1000)
        return trnX, trny, tstX, tsty

    @register_var(argument=r"mulgaussv6-(?P<n_samples>\d+)-(?P<noisy_level>\d+\.\d+|\d+)", shown_name="threegauss")
    @staticmethod
    def mulgaussv6(auto_var, inter_var, n_samples, noisy_level):
        """Returns the two spirals dataset."""
        random_state = np.random.RandomState(auto_var.get_var("random_seed"))
        n_samples = int(n_samples)
        noisy_level = float(noisy_level)
        noisy_level = [[noisy_level, 0], [0, noisy_level]]

        def sample():
            X, y = [], []
            X.append(random_state.multivariate_normal([0, 0], noisy_level, size=n_samples))
            y.append(np.zeros(n_samples))
            X.append(random_state.multivariate_normal([3, 0], noisy_level, size=n_samples))
            y.append(np.ones(n_samples))
            X.append(random_state.multivariate_normal([5, 0], noisy_level, size=n_samples))
            y.append(np.zeros(n_samples))
            X.append(random_state.multivariate_normal([10, 0], noisy_level, size=n_samples))
            y.append(np.ones(n_samples))
            X = np.concatenate(X, axis=0)
            y = np.concatenate(y)

            return X, y.astype(np.int)

        inter_var['is_img_data'] = False
        trnX, trny = sample()
        trnX[:, 0] = (trnX[:, 0] - 5)
        tstX, tsty = sample()
        tstX[:, 0] = (tstX[:, 0] - 5)
        return trnX, trny, tstX, tsty

    @register_var(argument=r"mulgaussv5-(?P<n_samples>\d+)-(?P<noisy_level>\d+\.\d+|\d+)", shown_name="threegauss")
    @staticmethod
    def mulgaussv5(auto_var, inter_var, n_samples, noisy_level):
        """Returns the two spirals dataset."""
        random_state = np.random.RandomState(auto_var.get_var("random_seed"))
        n_samples = int(n_samples)
        noisy_level = float(noisy_level)
        noisy_level = [[noisy_level, 0], [0, noisy_level]]

        def sample():
            X, y = [], []
            X.append(random_state.multivariate_normal([0, 0], noisy_level, size=n_samples))
            y.append(np.zeros(n_samples))
            X.append(random_state.multivariate_normal([3, 0], noisy_level, size=n_samples))
            y.append(np.ones(n_samples))
            X.append(random_state.multivariate_normal([10, 0], noisy_level, size=n_samples))
            y.append(np.zeros(n_samples))
            X = np.concatenate(X, axis=0)
            y = np.concatenate(y)

            return X, y.astype(np.int)

        inter_var['is_img_data'] = False
        trnX, trny = sample()
        trnX[:, 0] = (trnX[:, 0] - 5)
        tstX, tsty = sample()
        tstX[:, 0] = (tstX[:, 0] - 5)
        return trnX, trny, tstX, tsty

    @register_var(argument=r"mulgaussv4-(?P<n_samples>\d+)-(?P<noisy_level>\d+\.\d+|\d+)", shown_name="threegauss")
    @staticmethod
    def mulgaussv4(auto_var, inter_var, n_samples, noisy_level):
        """Returns the two spirals dataset."""
        random_state = np.random.RandomState(auto_var.get_var("random_seed"))
        n_samples = int(n_samples)
        noisy_level = float(noisy_level)
        noisy_level = [[noisy_level, 0], [0, noisy_level]]

        def sample():
            X, y = [], []
            X.append(random_state.multivariate_normal([1, 0], noisy_level, size=n_samples))
            y.append(np.zeros(n_samples))
            X.append(random_state.multivariate_normal([2, 0], noisy_level, size=n_samples))
            y.append(np.ones(n_samples))
            X.append(random_state.multivariate_normal([5, 0], noisy_level, size=n_samples))
            y.append(np.zeros(n_samples))
            X.append(random_state.multivariate_normal([6, 0], noisy_level, size=n_samples))
            y.append(np.ones(n_samples))
            X.append(random_state.multivariate_normal([9, 0], noisy_level, size=n_samples))
            y.append(np.zeros(n_samples))
            X.append(random_state.multivariate_normal([10, 0], noisy_level, size=n_samples))
            y.append(np.ones(n_samples))
            X = np.concatenate(X, axis=0)
            y = np.concatenate(y)

            return X, y.astype(np.int)

        inter_var['is_img_data'] = False
        trnX, trny = sample()
        trnX[:, 0] = (trnX[:, 0] - 5)
        tstX, tsty = sample()
        tstX[:, 0] = (tstX[:, 0] - 5)
        return trnX, trny, tstX, tsty

    @register_var(argument=r"mulgaussv3-(?P<n_samples>\d+)-(?P<noisy_level>\d+\.\d+|\d+)", shown_name="threegauss")
    @staticmethod
    def mulgaussv3(auto_var, inter_var, n_samples, noisy_level):
        """Returns the two spirals dataset."""
        random_state = np.random.RandomState(auto_var.get_var("random_seed"))
        n_samples = int(n_samples)
        noisy_level = float(noisy_level)
        noisy_level = [[noisy_level, 0], [0, noisy_level]]

        def sample():
            X, y = [], []
            for i in range(1):
                X.append(random_state.multivariate_normal([i*2, 0], noisy_level, size=n_samples))
                X.append(random_state.multivariate_normal([i*2+1, 0], noisy_level, size=n_samples))
                y.append(np.zeros(n_samples))
                y.append(np.ones(n_samples))
            X = np.concatenate(X, axis=0)
            y = np.concatenate(y)

            return X, y.astype(np.int)

        inter_var['is_img_data'] = False
        trnX, trny = sample()
        trnX[:, 0] = (trnX[:, 0] - 0.5)
        tstX, tsty = sample()
        tstX[:, 0] = (tstX[:, 0] - 0.5)
        return trnX, trny, tstX, tsty

    @register_var(argument=r"mulgaussv2-(?P<n_samples>\d+)-(?P<noisy_level>\d+\.\d+|\d+)", shown_name="threegauss")
    @staticmethod
    def mulgaussv2(auto_var, inter_var, n_samples, noisy_level):
        """Returns the two spirals dataset."""
        random_state = np.random.RandomState(auto_var.get_var("random_seed"))
        n_samples = int(n_samples)
        noisy_level = float(noisy_level)
        noisy_level = [[noisy_level, 0], [0, noisy_level]]

        def sample():
            X, y = [], []
            for i in range(2):
                X.append(random_state.multivariate_normal([i*2, 0], noisy_level, size=n_samples))
                X.append(random_state.multivariate_normal([i*2+1, 0], noisy_level, size=n_samples))
                y.append(np.zeros(n_samples))
                y.append(np.ones(n_samples))
            X = np.concatenate(X, axis=0)
            y = np.concatenate(y)

            return X, y.astype(np.int)

        inter_var['is_img_data'] = False
        trnX, trny = sample()
        trnX[:, 0] = (trnX[:, 0] - 2) / 4
        tstX, tsty = sample()
        tstX[:, 0] = (tstX[:, 0] - 2) / 4
        return trnX, trny, tstX, tsty

    @register_var(argument=r"mulgaussv1-(?P<n_samples>\d+)-(?P<noisy_level>\d+\.\d+|\d+)", shown_name="threegauss")
    @staticmethod
    def mulgaussv1(auto_var, inter_var, n_samples, noisy_level):
        """Returns the two spirals dataset."""
        random_state = np.random.RandomState(auto_var.get_var("random_seed"))
        n_samples = int(n_samples)
        noisy_level = float(noisy_level)
        noisy_level = [[noisy_level, 0], [0, noisy_level]]

        def sample():
            X, y = [], []
            for i in range(3):
                X.append(random_state.multivariate_normal([i*2, 0], noisy_level, size=n_samples))
                X.append(random_state.multivariate_normal([i*2+1, 0], noisy_level, size=n_samples))
                y.append(np.zeros(n_samples))
                y.append(np.ones(n_samples))
            X = np.concatenate(X, axis=0)
            y = np.concatenate(y)

            return X, y.astype(np.int)

        inter_var['is_img_data'] = False
        trnX, trny = sample()
        trnX[:, 0] = (trnX[:, 0] - 3) / 6
        tstX, tsty = sample()
        tstX[:, 0] = (tstX[:, 0] - 3) / 6
        return trnX, trny, tstX, tsty

    @register_var(argument=r"threegaussv2-(?P<n_samples>\d+)-(?P<noisy_level>\d+\.\d+|\d+)", shown_name="threegauss")
    @staticmethod
    def threegaussv2(auto_var, inter_var, n_samples, noisy_level):
        """
        Returns the two spirals dataset.
        """
        random_state = np.random.RandomState(auto_var.get_var("random_seed"))
        n_samples = int(n_samples)
        noisy_level = float(noisy_level)
        noisy_level = [[noisy_level, 0], [0, noisy_level]]

        def sample():
            l1 = random_state.multivariate_normal([0, 4], noisy_level, size=n_samples)
            r1 = random_state.multivariate_normal([10, 0], noisy_level, size=n_samples)

            l2 = random_state.multivariate_normal([0, 0], noisy_level, size=n_samples*2)

            X = np.vstack((l1, r1, l2))
            y = np.concatenate((np.zeros(n_samples), np.zeros(n_samples), np.ones(n_samples*2)))

            return X, y.astype(np.int)

        inter_var['is_img_data'] = False
        trnX, trny = sample()
        tstX, tsty = sample()
        return trnX, trny, tstX, tsty

    @register_var(argument=r"threegauss-(?P<n_samples>\d+)-(?P<noisy_level>\d+\.\d+|\d+)", shown_name="threegauss")
    @staticmethod
    def threegauss(auto_var, inter_var, n_samples, noisy_level):
        """
        Returns the two spirals dataset.
        """
        random_state = np.random.RandomState(auto_var.get_var("random_seed"))
        n_samples = int(n_samples)
        noisy_level = float(noisy_level)
        noisy_level = [[noisy_level, 0], [0, noisy_level]]

        def sample():
            l1 = random_state.multivariate_normal([0, 0], noisy_level, size=n_samples)
            r1 = random_state.multivariate_normal([10, 0], noisy_level, size=n_samples)

            l2 = random_state.multivariate_normal([2, 0], noisy_level, size=n_samples)

            X = np.vstack((l1, r1, l2))
            y = np.concatenate((np.zeros(n_samples), np.zeros(n_samples), np.ones(n_samples)))

            return X, y.astype(np.int)

        inter_var['is_img_data'] = False
        trnX, trny = sample()
        tstX, tsty = sample()
        return trnX, trny, tstX, tsty

    @cache_outputs(cache_dir='./data/caches/dataset/')
    @requires(['random_seed'])
    @register_var(argument=r"imgnetsubset100resnext101", shown_name="")
    @staticmethod
    def imgnetsubset100resnext101(auto_var, data_dir="./data/"):
        from .imgnet import imagenet_subsample_100_feature
        trnX, trny, tstX, tsty = imagenet_subsample_100_feature(
                auto_var.get_var("random_seed"), "resnext101", data_dir=data_dir)
        gc.collect()
        return trnX, trny, tstX, tsty

    @register_var(argument=r"(?P<dataaug>[a-zA-Z0-9]+-)?imgnet100", shown_name="imgnet100")
    @staticmethod
    def imgnet100(auto_var, dataaug):
        from .imgnet100 import get_imgnet100
        from lolip.models.torch_utils import data_augs
        import torch
        torch.multiprocessing.set_sharing_strategy('file_system')
        if dataaug is None:
            trn_dset, tst_dset = get_imgnet100(None, None, None)
        else:
            trn_transform, tst_transform = getattr(data_augs, dataaug[:-1])()
            trn_dset, tst_dset = get_imgnet100(trn_transform, tst_transform, None)

        def load_data(dset):
            from torch.utils.data import DataLoader
            loader = DataLoader(dset, batch_size=32, shuffle=False, num_workers=24)
            X, y = zip(*[(x.numpy(), y.numpy()) for (x, y) in tqdm(loader)])
            return np.concatenate(X, axis=0).transpose(0, 2, 3, 1), np.concatenate(y)

        trnX, trny = load_data(trn_dset)
        tstX, tsty = load_data(tst_dset)
        return trnX, trny, tstX, tsty, None

    @register_var(argument=r"(?P<dataaug>[a-zA-Z0-9]+-)?imgnet100wo(?P<unseen>\d)", shown_name="imgnet100")
    @staticmethod
    def imgnet100wo(auto_var, dataaug, unseen):
        from .imgnet100 import get_imgnet100
        from lolip.models.torch_utils import data_augs
        import torch
        from tqdm import tqdm
        torch.multiprocessing.set_sharing_strategy('file_system')

        if dataaug is None:
            trn_dset, tst_dset, ood_dset_1, ood_dset_2 = get_imgnet100(None, None, [int(unseen)])
        else:
            trn_transform, tst_transform = getattr(data_augs, dataaug[:-1])()
            trn_dset, tst_dset, ood_dset_1, ood_dset_2 = get_imgnet100(trn_transform, tst_transform, [int(unseen)])

        def load_data(dset):
            from torch.utils.data import DataLoader
            loader = DataLoader(dset, batch_size=32, shuffle=False, num_workers=24)
            X, y = zip(*[(x.numpy(), y.numpy()) for (x, y) in tqdm(loader)])
            return np.concatenate(X, axis=0).transpose(0, 2, 3, 1), np.concatenate(y)

        trnX, trny = load_data(trn_dset)
        tstX, tsty = load_data(tst_dset)
        ood_1, _ = load_data(ood_dset_1)
        ood_2, _ = load_data(ood_dset_2)
        return trnX, trny, tstX, tsty, (ood_1, ood_2)

    @register_var(argument=r"calcedrepr-(?P<dataaug>[a-zA-Z0-9]+-)?imgnet100-(?P<result_path>[a-zA-Z0-9_\.\/-]+)", shown_name="cifar100")
    @staticmethod
    def calcedrepr_imgnet100(auto_var, inter_var, dataaug, result_path):
        import joblib
        _, y_train, _, y_test, _ = DatasetVarClass.imgnet100(auto_var, dataaug)
        result_path = os.path.join("./results/oos_repr/", result_path)
        res = joblib.load(result_path)
        res['trn_repr'].astype(np.float32, copy=False)
        res['tst_repr'].astype(np.float32, copy=False)
        x_train = res['trn_repr']
        x_test = res['tst_repr']
        inter_var['is_img_data'] = False
        return x_train, y_train, x_test, y_test, None


    @register_var(argument=r"calcedrepr-(?P<dataaug>[a-zA-Z0-9]+-)?imgnet100wo(?P<unseen>\d)-(?P<result_path>[a-zA-Z0-9_\.\/-]+)", shown_name="cifar100")
    @staticmethod
    def calcedrepr_imgnet100wo(auto_var, inter_var, dataaug, unseen, result_path):
        import joblib
        _, y_train, _, y_test, _ = DatasetVarClass.imgnet100wo(auto_var, dataaug, unseen)
        result_path = os.path.join("./results/oos_repr/", result_path)
        res = joblib.load(result_path)
        res['trn_repr'].astype(np.float32, copy=False)
        res['tst_repr'].astype(np.float32, copy=False)
        res['oos_trn_repr'].astype(np.float32, copy=False)
        res['oos_tst_repr'].astype(np.float32, copy=False)
        x_train = res['trn_repr']
        x_test = res['tst_repr']
        inter_var['is_img_data'] = False
        return x_train, y_train, x_test, y_test, (res['oos_trn_repr'], res['oos_tst_repr'])
