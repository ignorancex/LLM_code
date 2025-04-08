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


    """Setup features"""
    data_name = "compas"
    dict_ = features_setting(data_name)
    sensitive_features = dict_["sensitive_features"]
    normal_features = dict_["normal_features"]
    categorical_features = dict_["categorical_features"]
    continuous_features = dict_["continuous_features"]
    full_features = dict_["full_features"]
    target = dict_["target"]
    col_sensitive = ['race_0', 'race_1', 'gender_0', 'gender_1']

    """Preprocess data"""
    df = preprocess_dataset(df, continuous_features, categorical_features)
    df_generator = df[normal_features]
    df[target] = df[target].astype(float)

    # df, df_test = train_test_split(df, test_size=0.1, random_state=0)
    # df = df_test.copy()

    """Load auto encoder"""
    df_autoencoder = df[full_features].copy()
    emb_size = 128
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
    emb_size = 128
    df_generator = df[normal_features]
    generator= AutoEncoder(
        input_shape = df_generator.shape[1],
        encoder_layers=[512, 512, emb_size],  # model architecture
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
    emb_size = 128
    discriminator_agnostic = DiscriminatorCompasAg(emb_size)
    discriminator_agnostic.to(device)
    discriminator_agnostic.load_state_dict(torch.load(conf['compas_discriminator']))
    discriminator_agnostic.eval()

    """Split dataset into train and test"""
    df, df_test = train_test_split(df, test_size=0.1, random_state=0)
    df = df_test.copy()

    df_generator = df[normal_features]
    df_autoencoder = df[full_features].copy()

    """Autoencoder + Linear regression"""
    Z = ae_model.get_representation(df_autoencoder)
    Z = Z.cpu().detach().numpy()
    reg = LogisticRegression(solver='saga', max_iter=1000)
    reg.fit(Z, df[target].values)
    y_pred = reg.predict(Z)
    df["AL_prediction"] = y_pred
    df["AL_prediction_proba"] = reg.predict_proba(Z)[:,0]

    """Generator + Linear regression"""
    Z = generator.custom_forward(df_generator)
    Z = Z.cpu().detach().numpy()
    reg = LogisticRegression(solver='saga', max_iter=1000)
    reg.fit(Z, df[target].values)
    y_pred = reg.predict(Z)
    df["GL_prediction"] = y_pred
    df["GL_prediction_proba"] = reg.predict_proba(Z)[:,0]

    """Generator + Discriminator"""
    Z = generator.custom_forward(df_generator)
    predictor_agnostic = discriminator_agnostic(Z)
    y_pred = torch.argmax(predictor_agnostic, dim=1)
    y_pred = y_pred.reshape(-1).cpu().detach().numpy()
    df["GD_prediction"] = y_pred
    df["GD_prediction_proba"] = predictor_agnostic.cpu().detach().numpy()[:,0]

    # print(df)
    df.to_csv(conf["result_ivr_compas"], index = False)



