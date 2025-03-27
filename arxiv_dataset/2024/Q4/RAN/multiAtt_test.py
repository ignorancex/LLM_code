from kenn.AttCaptionsModel import AttCaptionsModel
from kenn.AttCaptionsSolver import AttCaptionsSolver
from kenn.utils import load_captions_data


def main():
    # load train dataset
    data = load_captions_data(path='opportunity\data\gestures\sequential_label', split='test/')
    word_to_idx = data['word_to_idx']

    model = AttCaptionsModel(word_to_idx, dim_data=[600, 113], dim_feature=[67, 128], dim_embed=34,
                             dim_hidden=128, n_time_step=9, prev2out=False,
                             ctx2out=True, alpha_c=1, selector=False, dropout=True)

    solver = AttCaptionsSolver(model, data,
                               pretrained_model=None, test_model='model/oppo/model-1000',
                               log_path='log/oppo/', bool_val=True, bool_selector=False,
                               generated_caption_len=9)

    solver.test()


if __name__ == "__main__":
    main()
