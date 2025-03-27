import json
import argparse
import numpy as np
# from hp_optim import best_model, load, reset_seeds
from hp_optim_cwc import best_model, load, reset_seeds
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from pim import PredictionInterval

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='naval')
    parser.add_argument('--n_epochs', type=int, default=3000)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--project', type=str, default='models')
    args = parser.parse_args()
    n_hidden = 100 if args.dataset in ['protein', 'song-year'] else 50
    batch_size = args.batch_size if args.dataset != 'song-year' else 1000
    if args.dataset == 'song-year':
        n_ens = 1
    elif args.dataset == 'protein':
        n_ens = 5
    else:
        n_ens = 20
    # Load data
    x_al, y_al = load(args.dataset)
    # Load best hp configuration
    with open(f'{args.project}/best_hp_{args.dataset}.json', 'r') as jsonfile:
        best_hp = json.load(jsonfile)
    # Training callbacks
    ckpt_path = f'{args.project}/weights_{args.dataset}.h5'
    checkpoint = ModelCheckpoint(
        ckpt_path,
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=True,
        verbose=0, 
        mode='min'
    )
    es_pred = EarlyStopping(monitor='val_loss', mode='min', patience=100)
    es_pim = EarlyStopping(monitor='loss', mode='min', patience=100)
    # Reserve space for results
    mpiw_va = np.empty((n_ens, 2))
    picp_va = np.empty((n_ens, 2))
    picp_te = np.empty((n_ens, 2)) 
    # Run experiments
    for seed in range(n_ens):
        reset_seeds(seed)
        print(f'Running experiment {seed+1}/{n_ens}')
        # Randomly choose train and test set
        x_tr, x_te, y_tr, y_te = train_test_split(x_al, y_al, 
                                test_size=0.1, random_state=seed+1)
        # Randomly choose validation set from 20% of previous training set
        x_tr, x_va, y_tr, y_va = train_test_split(x_tr, y_tr, 
                                test_size=0.2, random_state=seed+1)
        # Standardize the data
        s_tr_x = StandardScaler().fit(x_tr)
        s_tr_y = StandardScaler().fit(y_tr)
        x_tr = s_tr_x.transform(x_tr)
        x_va = s_tr_x.transform(x_va)
        x_te = s_tr_x.transform(x_te)
        y_tr = s_tr_y.transform(y_tr)
        y_va = s_tr_y.transform(y_va)
        y_te = s_tr_y.transform(y_te)
        # Train best model
        model = best_model(best_hp, n_hidden)
        _ = model.fit(x_tr, y_tr, 
                epochs = args.n_epochs, 
                batch_size = batch_size,
                validation_data = (x_va, y_va),
                callbacks=[es_pred, checkpoint],
                verbose=0
            )
        model.load_weights(ckpt_path)
        y_pred_va = model.predict(x_va)
        y_pred_te = model.predict(x_te)
        err_va = y_va - y_pred_va 
        err_te = y_te - y_pred_te
        
        umodel = PredictionInterval(0.95, err='asym')
        umodel.compile(loss=umodel.pim.loss, optimizer=Adam(lr=0.005))
        _ = umodel.fit(err_va, err_va, 
                        epochs=5000, 
                        batch_size=x_va.shape[0], 
                        callbacks=[es_pim], 
                        verbose=0)
        mpiw_va_1 = sum(umodel.pim.radius)
        picp_va_1 = np.mean(umodel.pim.picp(err_va).numpy())
        picp_te_1 = np.mean(umodel.pim.picp(err_te).numpy())
        print(f' PI ends: mpiw_va = {mpiw_va_1:.2f}, picp_va = {picp_va_1:.2f}, picp_te = {picp_te_1:.2f}')
        ### as if calibrated
        umodel = PredictionInterval(0.95)
        umodel.compile(loss=umodel.pim.loss, optimizer=Adam(lr=0.005))
        _ = umodel.fit(err_va, err_va, 
                        epochs=5000, 
                        batch_size=x_va.shape[0], 
                        callbacks=[es_pim], 
                        verbose=0)
        mpiw_va_2 = 2*umodel.pim.radius[0]
        picp_va_2 = umodel.pim.picp(err_va).numpy()[0]
        picp_te_2 = umodel.pim.picp(err_te).numpy()[0]
        print(f' PI radius: mpiw_va = {mpiw_va_2:.2f}, picp_va = {picp_va_2:.2f}, picp_te = {picp_te_2:.2f}')
        # Save results
        mpiw_va[seed,0] = mpiw_va_1
        mpiw_va[seed,1] = mpiw_va_2
        picp_va[seed,0] = picp_va_1
        picp_va[seed,1] = picp_va_2
        picp_te[seed,0] = picp_te_1
        picp_te[seed,1] = picp_te_2
        np.save(f'{args.project}/mpiw_va_{args.dataset}.npy', mpiw_va)
        np.save(f'{args.project}/picp_va_{args.dataset}.npy', picp_va)
        np.save(f'{args.project}/picp_te_{args.dataset}.npy', picp_te)