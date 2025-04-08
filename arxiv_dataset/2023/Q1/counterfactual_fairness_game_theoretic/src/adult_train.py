import torch
import torch.nn as nn
import pandas as pd
import sys
import torch.nn.functional as F
import numpy as np
import visdom
import argparse

from model_arch.discriminator import DiscriminatorAdultAw, DiscriminatorAdultAg
from dfencoder.autoencoder import AutoEncoder
from dfencoder.dataframe import EncoderDataFrame
from utils.evaluate_func import evaluate_classifier, evaluate_distribution, evaluate_fairness
from utils.helpers import preprocess_dataset
from utils.helpers import setup_logging
from utils.helpers import load_config
from utils.helpers import features_setting
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

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

    """Parsing argument"""
    parser = argparse.ArgumentParser()

    """Parsing argument"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--lambda_weight', type=float, default=100)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--random_state', type=int, default=0)
    parser.add_argument('--epoch', type=int, default=50)
    args = parser.parse_args()

    lambda_weight = args.lambda_weight
    learning_rate = args.learning_rate
    epochs = args.epoch

    """Setup for dataset"""
    data_name = "adult"
    log_path = conf['log_train_adult']
    data_path = conf['data_adult']
    ae_path = conf['adult_encoder']
    generator_path = conf["adult_generator"]
    discriminator_path = conf["adult_discriminator"]


    """Set up logging"""
    logger = setup_logging(log_path)

    """Load data"""
    df = pd.read_csv(data_path)

    """Setup features"""
    dict_ = features_setting(data_name)
    sensitive_features = dict_["sensitive_features"]
    normal_features = dict_["normal_features"]
    categorical_features = dict_["categorical_features"]
    continuous_features = dict_["continuous_features"]
    full_features = dict_["full_features"]
    target = dict_["target"]

    """Preprocess data"""
    df = preprocess_dataset(df, [], categorical_features)
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
    ae_model.load_state_dict(torch.load(ae_path))
    ae_model.eval()

    """Setup hyperparameter"""
    parameters = {}
    # parameters['epochs'] = 200
    # parameters['learning_rate'] = learning_rate
    parameters['dataframe'] = df
    parameters['batch_size'] = 256
    parameters['problem'] = 'classification'
    lambda1, lambda2 = 0.5, 0.01

    """Hyperparameter"""
    # learning_rate = parameters['learning_rate']
    # epochs = parameters['epochs']
    dataframe = parameters['dataframe']
    batch_size = parameters['batch_size']
    problem = parameters['problem']

    emb_size_gen = 256

    """Setup generator and discriminator"""
    generator= AutoEncoder(
        input_shape = df_generator.shape[1],
        encoder_layers=[512, 512, emb_size_gen],  # model architecture
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


    discriminator_agnostic = DiscriminatorAdultAg(emb_size_gen, problem)
    discriminator_awareness = DiscriminatorAdultAw(emb_size_gen + emb_size_ae, problem)
    discriminator_agnostic.to(device)
    discriminator_awareness.to(device)

    optimizer1 = torch.optim.Adam(generator.parameters(), lr=1e-2)
    optimizer2 = torch.optim.SGD(discriminator_agnostic.parameters(),lr=1e-2, momentum=0.9)
    optimizer3 = torch.optim.SGD(discriminator_awareness.parameters(),lr=1e-2, momentum=0.9)

    scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=50, gamma=0.1)
    scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=50, gamma=0.1)
    scheduler3 = torch.optim.lr_scheduler.StepLR(optimizer3, step_size=50, gamma=0.1)

    weights = [df[target].value_counts()[0], df[target].value_counts()[1]]
    normedWeights = [0.3, 0.7]
    normedWeights = torch.FloatTensor(normedWeights).to(device)
    loss_fn = nn.CrossEntropyLoss(normedWeights)

    losses = []
    losses_aware = []
    losses_gen = []
    learning_rates = []
    step = 0
    for i in (range(epochs)):
        sum_loss = []
        sum_loss_aware = []
        sum_loss_gen = []

        df_train = df.copy().sample(frac=1).reset_index(drop=True)
        """Split batch size"""
        skf = StratifiedKFold(n_splits=120, random_state=0, shuffle=True)
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
            # Z_noise = generator.custom_forward(batch_generator_noise)
            ZS = torch.cat((Z, batch_Z), 1)

            prediction_ag = discriminator_agnostic(Z)
            # prediction_ag_noise = discriminator_agnostic(Z_noise)
            prediction_aw = discriminator_awareness(ZS)

            """Measure loss"""
            loss_agnostic = loss_fn(prediction_ag, Y.reshape(-1))
            # loss_agnostic += loss_fn(prediction_ag_noise, Y.reshape(-1))
            loss_awareness = loss_fn(prediction_aw, Y.reshape(-1))
            diff_loss = F.rrelu(loss_agnostic - loss_awareness)
            gen_loss = lambda_weight * diff_loss + loss_agnostic

            """Track loss"""
            sum_loss.append(loss_agnostic.cpu().detach().numpy())
            sum_loss_aware.append(loss_awareness.cpu().detach().numpy())
            sum_loss_gen.append(gen_loss.cpu().detach().numpy())

            """Optimization progress"""
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            optimizer3.zero_grad()

            path = step % 7
            if path in [0]:
                gen_loss.backward()
                optimizer1.step()
            elif path in [1, 2, 3]:
                loss_agnostic.backward()
                optimizer2.step()
            elif path in [4, 5, 6]:
                loss_awareness.backward()
                optimizer3.step()
            step += 1

        """Update learning rate"""
        current_lr = scheduler1.get_last_lr()[0]
        scheduler1.step()
        scheduler2.step()
        scheduler3.step()
        learning_rates.append(current_lr)
        if current_lr <= 1e-9:
            optimizer1.param_groups[0]['lr'] = learning_rate
            optimizer2.param_groups[0]['lr'] = learning_rate
            optimizer3.param_groups[0]['lr'] = learning_rate

        """Get the final prediction"""
        df_generator = df[normal_features].copy()
        Z = generator.custom_forward(df_generator)
        y_pred = discriminator_agnostic(Z)
        y_pred = torch.argmax(y_pred, dim=1)
        y_pred = y_pred.reshape(-1).cpu().detach().numpy()
        y_true = df[target].values

        l1 = sum(sum_loss) / len(sum_loss)
        l2 = sum(sum_loss_aware) / len(sum_loss)
        l3 = sum(sum_loss_gen) / len(sum_loss)
        losses.append(l1)
        losses_aware.append(l2)
        losses_gen.append(l3)

        """Evaluation"""
        eval = evaluate_classifier(y_pred, y_true)
        logger.debug("Epoch {}".format(i))
        # logger.debug('Loss Agnostic {:.4f}'.format(l1))
        # logger.debug('Loss Awareness {:.4f}'.format(l2))
        # logger.debug('Generator loss {:.4f}'.format(l3))


        display_loss = '->'.join([str(round(x,3)) for x in [losses[0], losses[-1]]])
        display_aware = '->'.join([str(round(x,3)) for x in [losses_aware[0], losses_aware[-1]]])
        display_gen = '->'.join([str(round(x,3)) for x in [losses_gen[0], losses_gen[-1]]])

        logger.debug("Adult Loss Agnostic {}".format(display_loss))
        logger.debug("Adult Loss Awareness {}".format(display_aware))
        logger.debug("Adult Generator Agnostic {}".format(display_gen))
        logger.debug("Precision {:.3f}, Recall {:.3f}, F-measure {:.3f}".format(eval['Precision'], eval['Recall'], eval['F1 Score']))

        logger.debug("-"*30)

    viz = visdom.Visdom()

    viz.line(
        Y=np.array(losses),
        X=np.array(range(epochs)),
        opts=dict(title='Agnostic loss', webgl=True)
    )

    viz.line(
        Y=np.array(losses_aware),
        X=np.array(range(epochs)),
        opts=dict(title='Awareness loss', webgl=True)
    )

    viz.line(
        Y=np.array(losses_gen),
        X=np.array(range(epochs)),
        opts=dict(title='Generator loss', webgl=True)
    )

    """Save model"""
    logger.debug("Saving model......")
    torch.save(generator.state_dict(), generator_path)
    torch.save(discriminator_agnostic.state_dict(), discriminator_path)
