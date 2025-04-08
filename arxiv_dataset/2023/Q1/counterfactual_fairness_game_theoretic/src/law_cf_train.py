import torch
import pandas as pd
import sys 
import numpy as np

from tqdm import tqdm 
from model_arch.discriminator import Discriminator_Law
from dfencoder.autoencoder import AutoEncoder
from dfencoder.dataframe import EncoderDataFrame
from utils.evaluate_func import evaluate_pred, evaluate_distribution, evaluate_fairness
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
    logger = setup_logging(conf['log_train_law'])

    """Load data"""
    data_path = conf['data_law']
    df = pd.read_csv(data_path)

    df, df_test = train_test_split(df, test_size=0.1, random_state=0)

    """Setup features"""
    data_name = "law"
    dict_ = features_setting("law")
    sensitive_features = dict_["sensitive_features"]
    normal_features = dict_["normal_features"]
    categorical_features = dict_["categorical_features"]
    continuous_features = dict_["continuous_features"]
    full_features = dict_["full_features"]
    target = dict_["target"]

    selected_race = ['White', 'Black']
    df = df[df['race'].isin(selected_race)]
    df = df.reset_index(drop = True)

    """Preprocess data"""
    df = preprocess_dataset(df, continuous_features, categorical_features)
    df['ZFYA'] = (df['ZFYA']-df['ZFYA'].mean())/df['ZFYA'].std()
    df = df[['LSAT', 'UGPA', 'sex', 'race', 'ZFYA']]

    """Setup auto encoder"""
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
    ae_model.load_state_dict(torch.load(conf['law_encoder']))
    ae_model.eval()

    """Setup hyperparameter"""    
    logger.debug('Setup hyperparameter')
    parameters = {}
    parameters['epochs'] = 100
    parameters['learning_rate'] = 1e-9
    parameters['dataframe'] = df
    parameters['batch_size'] = 64
    parameters['problem'] = 'regression'

    """Hyperparameter"""
    learning_rate = parameters['learning_rate']
    epochs = parameters['epochs']
    dataframe = parameters['dataframe']
    batch_size = parameters['batch_size']
    problem = parameters['problem']
    
    """Setup generator and discriminator"""
    emb_size = 64
    discriminator_agnostic = Discriminator_Law(emb_size, problem)
    discriminator_awareness = Discriminator_Law(emb_size + 4, problem)
    discriminator_agnostic.to(device)
    discriminator_awareness.to(device)

    """Setup generator"""
    df_generator = df[normal_features]
    generator= AutoEncoder(
        input_shape = df_generator.shape[1],
        encoder_layers=[256, 256, emb_size],  # model architecture
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

    generator.build_model(df_generator)

    """Optimizer"""
    optimizer1 = torch.optim.Adam(
        generator.parameters(), lr=learning_rate
    )
    optimizer2 = torch.optim.SGD(discriminator_agnostic.parameters(),
                                 lr=learning_rate, momentum=0.9)
    optimizer3 = torch.optim.SGD(discriminator_awareness.parameters(),
                                 lr=learning_rate, momentum=0.9)

    scheduler1 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer1, 'min')
    scheduler2 = torch.optim.lr_scheduler.CyclicLR(optimizer2, base_lr=learning_rate, max_lr=0.001)
    scheduler3 = torch.optim.lr_scheduler.CyclicLR(optimizer3, base_lr=learning_rate, max_lr=0.001)

    """Training"""
    n_updates = len(df)// batch_size
    logger.debug('Training')
    logger.debug('Number of updates {}'.format(n_updates))
    logger.debug('Dataframe length {}'.format(len(df)))
    logger.debug('Batchsize {}'.format((batch_size)))

    loss_function = torch.nn.MSELoss()
    loss_function = torch.nn.SmoothL1Loss()

    step = 0
    for i in (range(epochs)):
        df_train = df.copy().sample(frac=1).reset_index(drop=True)

        sum_loss = []
        sum_loss_aware = []
        sum_loss_gen = []
        for j in tqdm(range(n_updates)):
            path = step % 10

            df_term = df_train.loc[batch_size*j:batch_size*(j+1)].reset_index(drop=True)
            df_term_generator = df_term[normal_features].copy()
            df_term_generator = EncoderDataFrame(df_term_generator)
            df_term_generator_noise = df_term_generator.swap(likelihood=0.1)
            df_term_autoencoder = df_term[full_features].copy()

            """Label"""
            Y = torch.Tensor(df_term[target].values).to(device).reshape(-1,1)

            """Feed forward"""
            Z = generator.custom_forward(df_term_generator)
            Z_noise = generator.custom_forward(df_term_generator_noise)

            """Get the representation from autoencoder model"""
            S = ae_model.get_representation(
                df_term_autoencoder[full_features]
            )

            """Get only sensitive representation"""
            sex_feature = ae_model.categorical_fts['sex']
            cats = sex_feature['cats']
            emb = sex_feature['embedding']
            cat_index = df_term_autoencoder['sex'].values
            emb_cat_sex = []
            for c in cat_index:
                emb_cat_sex.append(emb.weight.data.cpu().numpy()[cats.index(c), :].tolist())

            race_feature = ae_model.categorical_fts['race']
            cats = race_feature['cats']
            emb = race_feature['embedding']
            cat_index = df_term_autoencoder['race'].values
            emb_cat_race = []
            for c in cat_index:
                emb_cat_race.append(emb.weight.data.cpu().numpy()[cats.index(c), :].tolist())

            emb_cat_race = torch.tensor(np.array(emb_cat_race).astype(np.float32)).to(device)
            emb_cat_sex = torch.tensor(np.array(emb_cat_sex).astype(np.float32)).to(device)
            emb = torch.cat((emb_cat_race, emb_cat_sex),1)

            ZS = torch.cat((emb, Z), 1)

            """Prediction and calculate loss"""
            predictor_awareness = discriminator_awareness(ZS)
            predictor_agnostic = discriminator_agnostic(Z)
            predictor_agnostic_noise = discriminator_agnostic(Z_noise)

            """Discriminator loss"""
            loss_agnostic = loss_function(predictor_agnostic, Y)
            loss_agnostic += loss_function(predictor_agnostic_noise, Y)
            loss_awareness = loss_function(predictor_awareness, Y)
            diff_loss = torch.max(torch.tensor(0).to(device), loss_agnostic - loss_awareness)

            "Generator loss"
            gen_loss = 0.1 * diff_loss + loss_agnostic

            """Track loss"""
            sum_loss.append(loss_agnostic)
            sum_loss_aware.append(loss_awareness)
            sum_loss_gen.append(gen_loss)

            """Optimizing progress"""
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            optimizer3.zero_grad()

            for p in discriminator_awareness.parameters():
                if p.grad is not None:  # In general, C is a NN, with requires_grad=False for some layers
                    p.grad.data.mul_(-1)  # Update of grad.data not tracked in computation graph

            if path in [0, 1, 2, 3, 4]:
                gen_loss.backward()
                optimizer1.step()
                scheduler1.step(gen_loss)
            elif path in [5, 6]:
                loss_agnostic.backward()
                optimizer2.step()
                scheduler2.step()
            elif path in [7, 8, 9]:
                loss_awareness.backward()
                optimizer3.step()
                scheduler3.step()
            else:
                raise ValueError("Invalid path number. ")

            step += 1

            del df_term_generator
            del df_term_generator_noise
            del emb_cat_race
            del emb_cat_sex
            del emb
            del ZS


        df_train = df.copy()
        df_generator = df_train[normal_features].copy()

        """Get the final prediction"""
        Z = generator.get_representation(df_generator)
        predictor_agnostic = discriminator_agnostic(Z)
        y_pred = predictor_agnostic.cpu().detach().numpy().reshape(-1)
        y_true = df_train[target].values

        """Evaluation"""
        df_train['inv_prediction'] = y_pred

        eval = evaluate_pred(y_pred, y_true)
        eval_fairness = evaluate_fairness(sensitive_features, df_train, 'inv_prediction')

        """Log to file"""
        logger.debug("Epoch {}".format(i))
        logger.debug('Loss Agnostic {:.4f}'.format(sum(sum_loss)/len(sum_loss)))
        logger.debug('Loss Awareness {:.4f}'.format(sum(sum_loss_aware)/len(sum_loss)))
        logger.debug('Generator loss {:.4f}'.format(sum(sum_loss_gen)/len(sum_loss)))
        logger.debug("RMSE {:.4f}".format(eval['RMSE']))
        logger.debug("Fairness {:.7f}".format(eval_fairness['sinkhorn']))

        del df_train
        del df_generator
        del eval
        del eval_fairness
        del Z
        del predictor_agnostic
        del y_pred
        del y_true


    """Save model"""
    logger.debug("Saving model......")
    torch.save(generator.state_dict(), conf["law_generator"])
    torch.save(discriminator_agnostic.state_dict(), conf["law_discriminator"])

    """Output to file"""
    sys.modules[__name__].__dict__.clear()


    
    