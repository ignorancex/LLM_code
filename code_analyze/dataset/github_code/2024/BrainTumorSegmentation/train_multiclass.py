from argparse import ArgumentParser

from keras.callbacks import ModelCheckpoint

from utils.dataset import MRIDataset
from utils.loss import log_cosh_dice_loss
from utils.metric import dice_coef_multilabel, peritumoral_edema, enhancing_tumor, core_tumor
from utils.models import attention_multiclass_model
from utils.optimizers import LH_Adam


def train(dataset_dir: str, weights_path: str):
    dataset = MRIDataset(cropped_dataset_path=dataset_dir)

    batch_size = 6
    train_img_datagen = dataset.cropped_train_datagen(batch_size)
    val_img_datagen = dataset.cropped_val_datagen(batch_size)

    steps_per_epoch, val_steps_per_epoch = dataset.cropped_steps_per_epoch(batch_size)

    model = attention_multiclass_model(48, 48, 128, 4, 4)

    learning_rate = 0.0003
    optimizer = LH_Adam(learning_rate)

    model.compile(optimizer=optimizer, loss=log_cosh_dice_loss,
                  metrics=[dice_coef_multilabel, peritumoral_edema, core_tumor, enhancing_tumor])
    # model.load_weights(weights_path)

    checkpoint_callback = ModelCheckpoint(filepath=weights_path, save_weights_only=True, save_best_only=True)
    model.fit(x=train_img_datagen,
              steps_per_epoch=steps_per_epoch,
              epochs=300,
              verbose=1,
              validation_data=val_img_datagen,
              validation_steps=val_steps_per_epoch,
              callbacks=[checkpoint_callback])


if __name__ == '__main__':
    arg_parser = ArgumentParser()

    arg_parser.add_argument("-d", "--dataset_dir",
                            help="Directory for the training dataset. Should contain the 'train' and 'val' "
                                 "directories.",
                            required=True)

    arg_parser.add_argument("-w", "--weights",
                            help="Path to the model's weights. This is the path of where to save the weights",
                            required=True)

    args = arg_parser.parse_args()
    train(args.dataset_dir, args.weights)
