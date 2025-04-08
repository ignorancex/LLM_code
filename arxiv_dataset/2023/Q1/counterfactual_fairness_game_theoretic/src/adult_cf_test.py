import torch
import torch.nn as nn
import pandas as pd
import yaml
import logging
import sys
import torch.nn.functional as F
import numpy as np
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression

from tqdm import tqdm
from model_arch.discriminator import Discriminator_Adult_Aw, Discriminator_Adult_Ag
from dfencoder.autoencoder import AutoEncoder
from dfencoder.dataframe import EncoderDataFrame
from utils.evaluate_func import evaluate_classifier, evaluate_distribution, evaluate_fairness
from utils.helpers import preprocess_dataset
from utils.helpers import setup_logging
from utils.helpers import load_config
from utils.helpers import features_setting
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from sklearn.metrics import classification_report

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
    data_path = conf['processed_data_adult']
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

    """Preprocess data"""
    df = preprocess_dataset(df, continuous_features, categorical_features)
    df_generator = df[normal_features]
    df[target] = df[target].astype(float)

    """Setup auto encoder"""
    df_autoencoder = df[full_features].copy()
    emb_size_ae = 128
    ae_model = AutoEncoder(
        input_shape=df[full_features].shape[1],
        encoder_layers=[512, 512, emb_size_ae],  # model architecture
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
    ae_model.build_model(df_autoencoder)
    ae_model.load_state_dict(torch.load(conf['adult_encoder']))
    ae_model.eval()

    """Logistic Regression"""
    Z = ae_model.get_representation(df_autoencoder)
    Y = np.array([float(i) for i in df[target].values])
    clf = LogisticRegression(solver='liblinear')
    clf.fit(Z.cpu().detach().numpy(), Y)
    y_pred = clf.predict(Z.cpu().detach().numpy())
    eval = evaluate_classifier(y_pred, df[target].values)
    logger.debug(eval)


    """Setup hyperparameter"""
    parameters = {}
    parameters['epochs'] = 1000
    parameters['learning_rate'] = 1e-4
    parameters['dataframe'] = df
    parameters['batch_size'] = 256
    parameters['problem'] = 'classification'
    lambda1, lambda2 = 0.5, 0.01

    """Hyperparameter"""
    learning_rate = parameters['learning_rate']
    epochs = parameters['epochs']
    dataframe = parameters['dataframe']
    batch_size = parameters['batch_size']
    problem = parameters['problem']

    """Setup generator and discriminator"""
    emb_size = 128
    generator= AutoEncoder(
        input_shape = df_generator.shape[1],
        encoder_layers=[512, 512, emb_size],  # model architecture
        decoder_layers=[],  # decoder optional - you can create bottlenecks if you like
        encoder_dropout = 0.15,
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
    generator.build_model(df_generator)
    generator.to(device)

    print(generator.categorical_fts)

    discriminator_agnostic = Discriminator_Adult_Ag(emb_size, problem)
    discriminator_awareness = Discriminator_Adult_Aw(emb_size + 4, problem)


    # discriminator_awareness = Discriminator_Adult_Aw(emb_size_ae, problem)
    # emb_size = 64
    # discriminator_awareness = Discriminator_Adult_Aw(emb_size_ae)
    # discriminator_awareness.to(device)
    # discriminator_awareness.load_state_dict(torch.load(conf['discriminator_awareness_adult']))
    # discriminator_awareness.eval()


    discriminator_agnostic.to(device)
    discriminator_awareness.to(device)

    optimizer1 = torch.optim.Adam(generator.parameters(), lr=1e-2)
    # optimizer2 = torch.optim.Adam(discriminator_agnostic.parameters(), lr=learning_rate)
    # optimizer3 = torch.optim.Adam(discriminator_awareness.parameters(), lr=learning_rate)


    optimizer2 = torch.optim.SGD(discriminator_agnostic.parameters(),
                                 lr=1e-2, momentum=0.9)
    optimizer3 = torch.optim.SGD(discriminator_awareness.parameters(),
                                 lr=1e-2, momentum=0.9)

    scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=150, gamma=0.1)
    scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=150, gamma=0.1)
    scheduler3 = torch.optim.lr_scheduler.StepLR(optimizer3, step_size=150, gamma=0.1)

    # scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optimizer1,
    #                                                   milestones=[30,80],
    #                                                   gamma=0.1)
    # scheduler2 = torch.optim.lr_scheduler.MultiStepLR(optimizer2,
    #                                                   milestones=[30,80],
    #                                                   gamma=0.1)
    # scheduler3 = torch.optim.lr_scheduler.MultiStepLR(optimizer3,
    #                                                   milestones=[30,80],
    #                                                   gamma=0.1)

    weights = [df[target].value_counts()[0], df[target].value_counts()[1]]
    normedWeights = [0.4, 0.6]
    normedWeights = torch.FloatTensor(normedWeights).to(device)
    loss_fn = nn.CrossEntropyLoss(normedWeights)

    step = 0
    for i in (range(epochs)):
        df_train = df.copy().sample(frac=1).reset_index(drop=True)
        losses = []
        losses1 = []
        losses2 = []
        """Split batch size"""
        skf = StratifiedKFold(n_splits=300, random_state=0, shuffle=True)
        for train_index, test_index in tqdm(skf.split(df_train[full_features], df_train[target])):

            batch_ae = df_train.iloc[test_index,:][full_features].copy()
            batch_Z = ae_model.get_representation(batch_ae)
            batch_generator = df_train.iloc[test_index, :][normal_features].copy()
            batch_generator = EncoderDataFrame(batch_generator)
            batch_generator_noise = batch_generator.swap(likelihood=0.1)

            Y = df_train.iloc[test_index,:][target].values
            Y = torch.Tensor(Y).to(device).reshape(-1,1).long()


            """Get only sensitive representation"""
            sex_feature = ae_model.categorical_fts['gender']
            cats = sex_feature['cats']
            emb = sex_feature['embedding']
            cat_index = batch_ae['gender'].values
            emb_cat_sex = []
            for c in cat_index:
                emb_cat_sex.append(emb.weight.data.cpu().numpy()[cats.index(c), :].tolist())

            race_feature = ae_model.categorical_fts['race']
            cats = race_feature['cats']
            emb = race_feature['embedding']
            cat_index = batch_ae['race'].values
            emb_cat_race = []
            for c in cat_index:
                emb_cat_race.append(emb.weight.data.cpu().numpy()[cats.index(c), :].tolist())

            emb_cat_race = torch.tensor(np.array(emb_cat_race).astype(np.float32)).to(device)
            emb_cat_sex = torch.tensor(np.array(emb_cat_sex).astype(np.float32)).to(device)
            emb = torch.cat((emb_cat_race, emb_cat_sex),1)

            """Concat generator and sensitive representation"""
            Z = generator.custom_forward(batch_generator)
            Z_noise = generator.custom_forward(batch_generator_noise)
            ZS = torch.cat((Z, emb), 1)
            # ZS = ae_model.get_representation(batch_ae)
            prediction_ag = discriminator_agnostic(Z)
            prediction_ag_noise = discriminator_agnostic(Z_noise)
            prediction_aw = discriminator_awareness(ZS)

            """Measure loss"""
            loss_agnostic = loss_fn(prediction_ag, Y.reshape(-1))
            loss_agnostic += loss_fn(prediction_ag_noise, Y.reshape(-1))
            loss_awareness = loss_fn(prediction_aw, Y.reshape(-1))
            diff_loss = F.leaky_relu(loss_agnostic - loss_awareness)
            gen_loss = 0.01 * diff_loss + loss_agnostic

            """Track loss"""
            losses.append(gen_loss.cpu().detach().numpy())
            losses1.append(loss_agnostic.cpu().detach().numpy())
            losses2.append(loss_awareness.cpu().detach().numpy())

            """Optimization progress"""
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            optimizer3.zero_grad()

            path = step % 10
            if path in [0, 1, 2]:
                gen_loss.backward()
                optimizer1.step()
            elif path in [3, 4, 5, 6]:
                loss_agnostic.backward()
                optimizer2.step()
            elif path in [7, 8, 9]:
                loss_awareness.backward()
                optimizer3.step()
            step += 1


        print("Parameters")
        print("Generator", generator.state_dict()['encoder_0.linear_layer.weight'][0][:10])
        print("Discriminator", discriminator_agnostic.state_dict()['hidden.weight'][0][:10])


        """Get the final prediction"""
        df_generator = df[normal_features].copy()
        Z = generator.get_representation(df_generator)
        y_pred = discriminator_agnostic(Z)
        y_pred = torch.argmax(y_pred, dim=1)
        y_pred = y_pred.reshape(-1).cpu().detach().numpy()
        y_true = df[target].values

        # unique_counts_1 = np.unique(y_true, return_counts=True)
        # unique_counts_2 = np.unique(y_pred, return_counts=True)
        # print(unique_counts_1)
        # print(unique_counts_2)

        # report = classification_report(y_true, y_pred)
        # print(report)

        """Evaluation"""
        eval = evaluate_classifier(y_pred, y_true)
        logger.debug("Epoch {}".format(i))
        logger.debug("Loss generator {:.5f}".format(sum(losses)/len(losses)))
        logger.debug("Loss discriminator 1 {:.5f}".format(sum(losses1)/len(losses1)))
        logger.debug("Loss discriminator 2 {:.5f}".format(sum(losses2)/len(losses2)))
        for key, value in eval.items():
            logger.debug("{} {:.4f}".format(key, value))
        logger.debug("-"*30)


    """Save model"""
    logger.debug("Saving model......")
    torch.save(generator.state_dict(), conf["adult_generator"])
    torch.save(discriminator_agnostic.state_dict(), conf["adult_discriminator"])

