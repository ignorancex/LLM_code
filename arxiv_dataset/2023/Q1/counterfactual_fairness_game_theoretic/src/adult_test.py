import torch
import pandas as pd
import sys

from sklearn.linear_model import LogisticRegression
from model_arch.discriminator import DiscriminatorAdultAg
from dfencoder.autoencoder import AutoEncoder
from utils.helpers import preprocess_dataset
from utils.helpers import setup_logging
from utils.helpers import load_config
from utils.helpers import features_setting
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

def train_simple_classifier(Z_train,y_train, Z_test, clf):
    clf.fit(Z_train, y_train)
    return clf.predict(Z_test), clf.predict_proba(Z_test)[:,0].reshape(-1)

def get_predict(ae_model, generator, discriminator, df_train, df_test, dict_, l = ''):
    name = 'GD' + str(l)

    """Auto-encoder"""
    Z = ae_model.get_representation(df_train[dict_["full_features"]].copy()).cpu().detach().numpy()
    Z_test = ae_model.get_representation(df_test[dict_["full_features"]].copy()).cpu().detach().numpy()

    clf = LogisticRegression(solver='saga', max_iter=10)
    y_pred, y_pred_proba = train_simple_classifier(Z, df_train[dict_["target"]].values, Z_test, clf)
    df_test.loc[:,"AL"] = y_pred.reshape(-1)
    df_test.loc[:,"AL" + "_proba"] = clf.predict_proba(Z_test)[:,0].reshape(-1)

    clf = GradientBoostingClassifier()
    y_pred, y_pred_proba = train_simple_classifier(Z, df_train[dict_["target"]].values, Z_test, clf)
    df_test.loc[:,"AGbr"] = y_pred.reshape(-1)
    df_test.loc[:,"AGbr" + "_proba"] = clf.predict_proba(Z_test)[:,0].reshape(-1)

    """Generator"""
    Z = generator.custom_forward(df_train[dict_["normal_features"]].copy()).cpu().detach().numpy()
    Z_test = generator.custom_forward(df_test[dict_["normal_features"]].copy()).cpu().detach().numpy()

    clf = LogisticRegression(solver='saga', max_iter=10)
    y_pred, y_pred_proba = train_simple_classifier(Z, df_train[dict_["target"]].values, Z_test, clf)
    df_test.loc[:,"GL"] = y_pred.reshape(-1)
    df_test.loc[:,"GL" + "_proba"] = clf.predict_proba(Z_test)[:,0].reshape(-1)

    clf = GradientBoostingClassifier()
    y_pred, y_pred_proba = train_simple_classifier(Z, df_train[dict_["target"]].values, Z_test, clf)
    df_test.loc[:,"GGbr"] = y_pred.reshape(-1)
    df_test.loc[:,"GGbr" + "_proba"] = clf.predict_proba(Z_test)[:,0].reshape(-1)

    """Generator + Discriminator"""
    Z = generator.custom_forward(df_test[dict_["normal_features"]].copy())
    predictor_agnostic = discriminator(Z)
    y_pred = torch.argmax(predictor_agnostic, dim=1)
    y_pred = y_pred.reshape(-1).cpu().detach().numpy()
    df_test.loc[:,name] = y_pred
    df_test.loc[:,name + "_proba"] = predictor_agnostic.cpu().detach().numpy()[:,0]

    return df_test

if __name__ == "__main__":
    """Device"""
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    device = torch.device(dev)

    """Load configuration"""
    config_path = "/home/trduong/Data/counterfactual_fairness_game_theoric/configuration.yml"
    conf = load_config(config_path)

    """Set up logging"""
    logger = setup_logging(conf['log_train_adult'])

    """Load data"""
    data_path = conf['data_adult']
    df = pd.read_csv(data_path)

    """Setup features"""
    data_name = "adult"
    dict_ = features_setting(data_name)
    sensitive_features = dict_["sensitive_features"]
    normal_features = dict_["normal_features"]
    categorical_features = dict_["categorical_features"]
    continuous_features = dict_["continuous_features"]
    full_features = dict_["full_features"]
    target = dict_["target"]
    col_sensitive = ['race_0', 'race_1', 'sex_0', 'sex_1']

    """Preprocess data"""
    df = preprocess_dataset(df, [], categorical_features)
    df_generator = df[normal_features]
    df[target] = df[target].astype(float)

    df_train, df_test = train_test_split(df, test_size=0.2, random_state=0)
    # print(df_train[target].value_counts(), df_test[target].value_counts())
    # df_term = df.groupby(target).reset_index(drop=True)
    # print(df_term[target].value_counts())
    # df = df_test.copy()
    # sys.exit(1)
    """Load auto encoder"""
    df_autoencoder = df[full_features].copy()
    emb_size_gen = 128
    ae_model = AutoEncoder(
        input_shape=df[full_features].shape[1],
        encoder_layers=[512, 512, emb_size_gen],  # model architecture
        decoder_layers=[],  # decoder optional - you can create bottlenecks if you like
        activation='relu',
        swap_p=0.2,  # noise parameter
        lr=0.01,
        lr_decay=.99,
        batch_size=512,  # 512
        verbose=False,
        optimizer='sgd',
        scaler='gauss_rank',  # gauss rank scaling forces your numeric features into standard normal distributions
    )
    ae_model.to(device)
    ae_model.build_model(df[full_features].copy())
    ae_model.load_state_dict(torch.load(conf['adult_encoder']))
    ae_model.eval()

    """Load generator"""
    emb_size_gen = 256
    df_generator = df[normal_features]
    generator= AutoEncoder(
        input_shape = df_generator.shape[1],
        encoder_layers=[512, 512, emb_size_gen],  # model architecture
        decoder_layers=[],  # decoder optional - you can create bottlenecks if you like
        encoder_dropout = 0.5,
        decoder_dropout = 0.5,
        activation='tanh',
        swap_p=0.2,  # noise parameter
        lr=0.0001,
        lr_decay=.99,
        batch_size=512,  # 512
        verbose=False,
        optimizer='sgd',
        scaler='gauss_rank',  # gauss rank scaling forces your numeric features into standard normal distributions
    )
    generator.to(device)
    generator.build_model(df_generator)
    generator.load_state_dict(torch.load(conf['adult_generator']))
    generator.eval()


    """Load discriminator"""
    discriminator = DiscriminatorAdultAg(emb_size_gen)
    discriminator.to(device)
    discriminator.load_state_dict(torch.load(conf['adult_discriminator']))
    discriminator.eval()

    # df_generator = df_test[normal_features].copy()
    # df_autoencoder = df_test[full_features].copy()


    df_test = get_predict(ae_model, generator, discriminator, df_train, df_test, dict_)

    """Autoencoder + Linear regression"""
    # Z = ae_model.get_representation(df_autoencoder)
    # Z = Z.cpu().detach().numpy()
    # clf = LogisticRegression()
    # clf.fit(Z, df_test[target].values)
    # y_pred = clf.predict(Z)
    # df_test["AL_prediction"] = y_pred
    # df_test["AL_prediction_proba"] = clf.predict_proba(Z)[:,0]

    """Generator + Linear regression"""
    # Z = generator.custom_forward(df_generator)
    # Z = Z.cpu().detach().numpy()
    # clf = LogisticRegression()
    # clf.fit(Z, df_test[target].values)
    # y_pred = clf.predict(Z)
    # df_test["GL_prediction"] = y_pred
    # df_test["GL_prediction_proba"] = clf.predict_proba(Z)[:,0]

    """Generator + Discriminator"""
    # Z = generator.custom_forward(df_generator)
    # predictor_agnostic = discriminator_agnostic(Z)
    # y_pred = torch.argmax(predictor_agnostic, dim=1)
    # y_pred = y_pred.reshape(-1).cpu().detach().numpy()
    # df_test["GD_prediction"] = y_pred
    # df_test["GD_prediction_proba"] = predictor_agnostic.cpu().detach().numpy()[:,0]

    df_test.to_csv(conf["ivr_adult"], index = False)



