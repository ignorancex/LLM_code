import torch
import pandas as pd

from sklearn.linear_model import LogisticRegression
from model_arch.discriminator import DiscriminatorCompasAg
from dfencoder.autoencoder import AutoEncoder
from utils.helpers import preprocess_dataset
from utils.helpers import setup_logging
from utils.helpers import load_config
from utils.helpers import features_setting
from sklearn.model_selection import train_test_split

# def get_predict(ae_model, generator, discriminator, df, dict_, l = ''):
def get_predict(ae_model, generator, discriminator, df_train, df_test, dict_, l = ''):
    name = 'GD_prediction' + str(l)


    Z = ae_model.get_representation(df_train[dict_["full_features"]].copy())
    Z = Z.cpu().detach().numpy()
    reg = LogisticRegression(solver='saga', max_iter=10)
    reg.fit(Z, df_train[dict_["target"]].values)
    Z_test = ae_model.get_representation(df_test[dict_["full_features"]].copy()).cpu().detach().numpy()
    y_pred = reg.predict(Z_test)
    df_test.loc[:,"AL_prediction"] = y_pred.reshape(-1)
    df_test.loc[:,"AL_prediction" + "_proba"] = reg.predict_proba(Z_test)[:,0].reshape(-1)

    """Generator + Linear regression"""
    Z = generator.custom_forward(df_train[dict_["normal_features"]].copy())
    Z = Z.cpu().detach().numpy()
    reg = LogisticRegression(solver='saga', max_iter=10)
    reg.fit(Z, df_train[dict_["target"]].values)
    Z_test = generator.custom_forward(df_test[dict_["normal_features"]].copy()).cpu().detach().numpy()
    y_pred = reg.predict(Z_test)
    df_test.loc[:,"GL_prediction"] = y_pred.reshape(-1)
    df_test.loc[:,"GL_prediction" + "_proba"] = reg.predict_proba(Z_test)[:,0].reshape(-1)

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
    logger = setup_logging(conf['log_train_compas'])

    """Load data"""
    data_path = conf['data_compas']
    df = pd.read_csv(data_path)

    emb_size_gen = 256
    emb_size = 128

    """Setup features"""
    dict_ = features_setting('compas')
    sensitive_features = dict_["sensitive_features"]
    normal_features = dict_["normal_features"]
    categorical_features = dict_["categorical_features"]
    continuous_features = dict_["continuous_features"]
    discrete_features = dict_["discrete_features"]
    full_features = dict_["full_features"]
    target = dict_["target"]
    col_sensitive = ['race_0', 'race_1',
                     'sex_0', 'sex_1']
    standard_features = continuous_features + discrete_features

    """Preprocess data"""
    df = preprocess_dataset(df, [], categorical_features)
    df_generator = df[normal_features]
    df[target] = df[target].astype(float)

    df_train, df_test = train_test_split(df, test_size=0.2, random_state=0)
    # df = df_test.copy()

    """Load auto encoder"""
    df_autoencoder = df[full_features].copy()
    ae_model = AutoEncoder(
        input_shape=df[full_features].shape[1],
        encoder_layers=[512, 512, emb_size],  # model architecture
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
    ae_model.load_state_dict(torch.load(conf['compas_encoder']))
    ae_model.eval()

    """Load generator"""
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
    generator.load_state_dict(torch.load(conf['compas_generator']))
    generator.eval()

    """Load discriminator"""
    # emb_size = 128
    discriminator = DiscriminatorCompasAg(emb_size_gen)
    discriminator.to(device)
    discriminator.load_state_dict(torch.load(conf['compas_discriminator']))
    discriminator.eval()

    """Split dataset into train and test"""
    # df, df_test = train_test_split(df, test_size=0.1, random_state=0)
    # df = df_test.copy()

    # df_generator = df[normal_features]
    # df_autoencoder = df[full_features].copy()

    df_test = get_predict(ae_model, generator, discriminator, df_train, df_test, dict_)

    """Autoencoder + Linear regression"""
    # Z = ae_model.get_representation(df_autoencoder)
    # Z = Z.cpu().detach().numpy()
    # reg = LogisticRegression(solver='saga', max_iter=1000)
    # reg.fit(Z, df[target].values)
    # y_pred = reg.predict(Z)
    # df["AL_prediction"] = y_pred
    # df["AL_prediction_proba"] = reg.predict_proba(Z)[:,0]
    #
    # """Generator + Linear regression"""
    # Z = generator.custom_forward(df_generator)
    # Z = Z.cpu().detach().numpy()
    # reg = LogisticRegression(solver='saga', max_iter=1000)
    # reg.fit(Z, df[target].values)
    # y_pred = reg.predict(Z)
    # df["GL_prediction"] = y_pred
    # df["GL_prediction_proba"] = reg.predict_proba(Z)[:,0]
    #
    # """Generator + Discriminator"""
    # Z = generator.custom_forward(df_generator)
    # predictor_agnostic = discriminator_agnostic(Z)
    # y_pred = torch.argmax(predictor_agnostic, dim=1)
    # y_pred = y_pred.reshape(-1).cpu().detach().numpy()
    # df["GD_prediction"] = y_pred
    # df["GD_prediction_proba"] = predictor_agnostic.cpu().detach().numpy()[:,0]

    # print(df)
    df_test.to_csv(conf["ivr_compas"], index = False)



