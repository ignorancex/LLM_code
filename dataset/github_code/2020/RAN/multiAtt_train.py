from kenn.AttCaptionsModel import AttCaptionsModel
from kenn.AttCaptionsSolver import AttCaptionsSolver
from kenn.utils import load_captions_data


def main():
    # load train dataset
    data = load_captions_data(path='opportunity\data\gestures\sequential_label', split='train/')
    word_to_idx = data['word_to_idx']

    model = AttCaptionsModel(word_to_idx, dim_data=[600, 113], dim_feature=[67, 128], dim_embed=34,
                             dim_hidden=128, n_time_step=9, prev2out=False,
                             ctx2out=True, alpha_c=1, selector=False, dropout=True)

    solver = AttCaptionsSolver(model, data, n_epochs=1000, batch_size=50, update_rule='adam',
                               learning_rate=0.00025, print_every=50, bool_save_model=True,
                               pretrained_model=None, model_path='model/oppo/',
                               log_path='log/', bool_val=True, generated_caption_len=9)

    solver.train()


if __name__ == "__main__":
    main()
