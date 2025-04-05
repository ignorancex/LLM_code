from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import pandas as pd
import numpy as np
from sklearn import preprocessing
import enum

class annototation_type(enum.Enum):
   asymmetry = 1
   border = 2
   color = 3

def get_train_validate_test_sets(ground_truth_file, seed, verbose, sanity_check):
    df = pd.read_csv(ground_truth_file)
    if sanity_check:
        df.loc[:, 'melanoma'] = np.ones(n)
        df.loc[:, 'seborrheic_keratosis'] = np.ones(n)
    class_label = df['melanoma'] + df['seborrheic_keratosis']
    class_id = df['image_id']

    X_train, X_test, y_train, y_test = train_test_split(
        class_id,
        class_label,
        test_size=0.125,
        random_state=seed,
        shuffle=True,
        stratify=class_label)
    X_train, X_validate, y_train, y_validate = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=seed,
        shuffle=True,
        stratify=y_train)

    class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
    class_weights = {i: class_weights[i] for i in range(2)}

    if verbose:
        print('in train set      = \n' + str(y_train.value_counts()))
        print('in validation set = \n' + str(y_validate.value_counts()))
        print('in test set       = \n' + str(y_test.value_counts()))

    return (X_train, y_train, X_validate, y_validate, X_test, y_test, class_weights)

def get_train_validate_test_sets_annotations(group_path, ground_truth_file, seed, verbose, sanity_check, type):
    df = pd.read_csv(ground_truth_file)
    class_label = df['melanoma'] + df['seborrheic_keratosis']
    class_id = df['image_id']
    group_path += '/'

    X_train, X_test, y_train, y_test = train_test_split(
        class_id,
        class_label,
        test_size=0.125,
        random_state=seed,
        shuffle=True,
        stratify=class_label)
    X_train, X_validate, y_train, y_validate = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=seed,
        shuffle=True,
        stratify=y_train)

    sample_weight_train = np.zeros(len(X_train))
    sample_weight_valid = np.zeros(len(X_validate))
    sample_weight_test = np.zeros(len(X_test))
    annotation_train = np.zeros(len(X_train))
    annotation_valid = np.zeros(len(X_validate))
    annotation_test = np.zeros(len(X_test))


    if type == annototation_type.asymmetry:
        annotation_id, annotation_label = get_asymm_score(group_path, sanity_check)
        if verbose: print('Asymmetry score is used')
    else:
        if type == annototation_type.border:
            if verbose: print('Border score is used')
            annotation_id, annotation_label = get_border_score(group_path)
        else:
            if type == annototation_type.color:
                if verbose: print('Color score is used')
                annotation_id, annotation_label = get_color_score(group_path)


    for i in range(len(X_train)):
        for j in range(len(annotation_id)):
            if X_train.iloc[i] == annotation_id[j]:
                sample_weight_train[i] = 1
                annotation_train[i] = annotation_label[j]
                break
        else:
            sample_weight_train[i] = 0
    if verbose:
        print('Annotations in train = ' + str(np.sum(sample_weight_train == 1)))

    for i in range(len(X_validate)):
        for j in range(len(annotation_id)):
            if X_validate.iloc[i] == annotation_id[j]:
                sample_weight_valid[i] = 1
                annotation_valid[i] = annotation_label[j]
                break
        else:
            sample_weight_valid[i] = 0
    if verbose:
        print('Annotations in validate = ' + str(np.sum(sample_weight_valid == 1)))

    for i in range(len(X_test)):
        for j in range(len(annotation_id)):
            if X_test.iloc[i] == annotation_id[j]:
                sample_weight_test[i] = 1
                annotation_test[i] = annotation_label[j]
                break
        else:
            sample_weight_test[i] = 0
    if verbose:
        print('Annotations in test = ' + str(np.sum(sample_weight_test == 1)))

    w = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
    class_weights = dict()
    class_weights[0] = w[0]
    class_weights[1] = w[1]

    if verbose:
        print('in train set = \n' + str(y_train.value_counts()))
        print('in validation set = \n' + str(y_validate.value_counts()))
        print('in test set = \n' + str(y_test.value_counts()))

    return (X_train, X_validate, X_test,
            y_train, y_validate, y_test,
            annotation_train, annotation_valid, annotation_test,
            sample_weight_train, sample_weight_valid, sample_weight_test, class_weights)

def get_color_score(group_path):
    df_1 = pd.read_excel(group_path + 'group01_C.xlsx') # 3
    df_2 = pd.read_excel(group_path + 'group02_C.xlsx') # 3
    df_4 = pd.read_excel(group_path + 'group04_C.xlsx') # 3
    df_51 = pd.read_excel(group_path + 'group05_C1.xlsx') # 3
    df_52 = pd.read_excel(group_path + 'group05_C2.xlsx') # 3
    df_6 = pd.read_excel(group_path + 'group06_C.xlsx') # 3
    df_7 = pd.read_excel(group_path + 'group07_C.xlsx') # 6
    df_8 = pd.read_excel(group_path + 'group08_C.xlsx') # 3

    a_1 = (preprocessing.scale(df_1['i']) + preprocessing.scale(df_1['ii']) + preprocessing.scale(
        df_1['iii'])) / 3.0
    a_2 = (preprocessing.scale(df_2['i']) + preprocessing.scale(df_2['ii']) + preprocessing.scale(
        df_2['iii'])) / 3.0
    a_4 = (preprocessing.scale(df_4['i']) + preprocessing.scale(df_4['ii']) + preprocessing.scale(
        df_4['iii'])) / 3.0
    a_51 = (preprocessing.scale(df_51['i']) + preprocessing.scale(df_51['ii']) + preprocessing.scale(
        df_51['iii'])) / 3.0
    a_52 = (preprocessing.scale(df_52['i']) + preprocessing.scale(df_52['ii']) + preprocessing.scale(
        df_52['iii'])) / 3.0
    a_6 = (preprocessing.scale(df_6['i']) + preprocessing.scale(df_6['ii']) + preprocessing.scale(
        df_6['iii'])) / 3.0
    a_7 = (preprocessing.scale(df_7['i']) + preprocessing.scale(df_7['ii']) + preprocessing.scale(df_7['iii']) +
           preprocessing.scale(df_7['iv']) + preprocessing.scale(df_7['v']) + preprocessing.scale(df_7['vi'])) / 6.0
    a_8 = (preprocessing.scale(df_8['i']) + preprocessing.scale(df_8['ii']) + preprocessing.scale(
        df_8['iii'])) / 3.0

    color_label = np.concatenate((a_1, a_2, a_4, a_51, a_52, a_6, a_7, a_8))
    color_id = np.concatenate(
        (df_1['id'], df_2['id'], df_4['id'], df_51['id'], df_52['id'], df_6['id'], df_7['id'], df_8['id']))

    return (color_id, color_label)


def get_border_score(group_path):
    df_1 = pd.read_excel(group_path + 'group01_B.xlsx') # 3
    df_2 = pd.read_excel(group_path + 'group02_B.xlsx') # 3
    df_3 = pd.read_excel(group_path + 'group03_B.xlsx') # 3
    df_4 = pd.read_excel(group_path + 'group04_B.xlsx') # 3
    df_51 = pd.read_excel(group_path + 'group05_B1.xlsx') # 2
    df_52 = pd.read_excel(group_path + 'group05_B2.xlsx') # 3
    df_6 = pd.read_excel(group_path + 'group06_B.xlsx') # 3
    df_7 = pd.read_excel(group_path + 'group07_B.xlsx') # 6
    df_8 = pd.read_excel(group_path + 'group08_B.xlsx') # 6

    a_1 = (preprocessing.scale(df_1['i']) + preprocessing.scale(df_1['ii']) + preprocessing.scale(
        df_1['iii'])) / 3.0
    a_2 = (preprocessing.scale(df_2['i']) + preprocessing.scale(df_2['ii']) + preprocessing.scale(
        df_2['iii'])) / 3.0
    a_3 = (preprocessing.scale(df_3['i']) + preprocessing.scale(df_3['ii']) + preprocessing.scale(
        df_3['iii'])) / 3.0
    a_4 = (preprocessing.scale(df_4['i']) + preprocessing.scale(df_4['ii']) + preprocessing.scale(
        df_4['iii'])) / 3.0
    a_51 = (preprocessing.scale(df_51['i']) + preprocessing.scale(df_51['ii'])) / 2.0
    a_52 = (preprocessing.scale(df_52['i']) + preprocessing.scale(df_52['ii']) + preprocessing.scale(
        df_52['iii'])) / 3.0
    a_6 = (preprocessing.scale(df_6['i']) + preprocessing.scale(df_6['ii']) + preprocessing.scale(
        df_6['iii'])) / 3.0
    a_7 = (preprocessing.scale(df_7['i']) + preprocessing.scale(df_7['ii']) + preprocessing.scale(df_7['iii']) +
           preprocessing.scale(df_7['iv']) + preprocessing.scale(df_7['v']) + preprocessing.scale(df_7['vi'])) / 6.0
    a_8 = (preprocessing.scale(df_8['i']) + preprocessing.scale(df_8['ii']) + preprocessing.scale(df_8['iii']) +
           preprocessing.scale(df_8['iv']) + preprocessing.scale(df_8['v']) + preprocessing.scale(df_8['vi'])) / 6.0

    border_label = np.concatenate((a_1, a_2, a_3, a_4, a_51, a_52, a_6, a_7, a_8))
    border_id = np.concatenate(
        (df_1['id'], df_2['id'], df_3['id'], df_4['id'], df_51['id'], df_52['id'], df_6['id'], df_7['id'], df_8['id']))

    return (border_id, border_label)

def get_asymm_score(group_path, sanity_check):
    # Label-Asymmetry Score
    df_1 = pd.read_excel(group_path + 'group01_A.xlsx')
    df_2 = pd.read_excel(group_path + 'group02_A.xlsx')
    df_3 = pd.read_excel(group_path + 'group03_A.xlsx')
    df_4 = pd.read_excel(group_path + 'group04_A.xlsx')
    df_5 = pd.read_excel(group_path + 'group05_A.xlsx')
    df_6 = pd.read_excel(group_path + 'group06_A.xlsx')
    df_7 = pd.read_excel(group_path + 'group07_A.xlsx')
    df_8 = pd.read_excel(group_path + 'group08_A.xlsx')


    a_1 = (preprocessing.scale(df_1['i']) + preprocessing.scale(df_1['ii']) + preprocessing.scale(
        df_1['iii'])) / 3.0
    a_2 = (preprocessing.scale(df_2['i']) + preprocessing.scale(df_2['ii']) + preprocessing.scale(
        df_2['iii'])) / 3.0
    a_3 = (preprocessing.scale(df_3['i']) + preprocessing.scale(df_3['ii']) + preprocessing.scale(
        df_3['iii'])) / 3.0
    a_4 = (preprocessing.scale(df_4['i']) + preprocessing.scale(df_4['ii']) + preprocessing.scale(
        df_4['iii'])) / 3.0
    a_5 = (preprocessing.scale(df_5['i']) + preprocessing.scale(df_5['ii']) + preprocessing.scale(
        df_5['iii'])) / 3.0
    a_6 = (preprocessing.scale(df_6['i']) + preprocessing.scale(df_6['ii']) + preprocessing.scale(
        df_6['iii'])) / 3.0
    a_7 = (preprocessing.scale(df_7['i']) + preprocessing.scale(df_7['ii']) + preprocessing.scale(df_7['iii']) +
           preprocessing.scale(df_7['iv']) + preprocessing.scale(df_7['v']) + preprocessing.scale(df_7['vi'])) / 6.0
    a_8 = (preprocessing.scale(df_8['i']) + preprocessing.scale(df_8['ii']) + preprocessing.scale(
        df_8['iii'])) / 3.0

    asymm_label = np.concatenate((a_1, a_2, a_3, a_4, a_5, a_6, a_7, a_8))
    asymm_id = np.concatenate(
        (df_1['ID'], df_2['ID'], df_3['ID'], df_4['ID'], df_5['ID'], df_6['ID'], df_7['ID'], df_8['ID']))

    if sanity_check:
        asymm_label = np.ones(len(asymm_label))

    return (asymm_id, asymm_label)