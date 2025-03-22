import torch
from torch.nn import functional as F

from models.utils.continual_model import ContinualModel
from utils.args import ArgumentParser, add_experiment_args, add_management_args

from collections import OrderedDict
from torch.optim import SGD


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via SLCA.')
    add_management_args(parser)
    add_experiment_args(parser)

    # Specific parameters of the method.
    parser.add_argument("--weights_path", default="/homes/mmosconi/continual_action/work_dir/ntu60/xview_joint/xview_joint_pre-train_spatio-temporal_0.5_0.5.pt", help="Path in which weights are put.")
    parser.add_argument('--lr_rps', type=float, default=0.0001, help='Learning rate representation layer.')
    parser.add_argument('--lr_cls', type=float, default=0.01, help='Learning rate classification layer.')

    return parser


# Be aware that this method has only the SL part of SLCA.
class SLCA(ContinualModel):
    NAME = 'slca'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(SLCA, self).__init__(backbone, loss, args, transform)

        # pre-trained weights loading
        weights = torch.load(self.args.weights_path)

        weights = OrderedDict([[k.split('module.')[-1], v.cuda()] for k, v in weights.items()])

        model_dict = self.net.state_dict()

        # Filter out unnecessary keys
        weights = {k: v for k, v in weights.items() if k in model_dict}
        # Overwrite entries in the existing state dict
        model_dict.update(weights)

        try:
            self.net.load_state_dict(weights, strict=False)
        except BaseException:
            state = self.net.state_dict()
            diff = list(set(state.keys()).difference(set(weights.keys())))
            print('Can not find these weights:')
            for d in diff:
                print('  ' + d)
            state.update(weights)
            self.net.load_state_dict(state)

        # SL
        classifier = []
        classifier.extend(self.net.classifier.parameters())

        representations = [p for p in self.net.parameters() if p not in set(classifier)]

        self.opt = SGD([{"params": representations, "lr": self.args.lr_rps},
                        {"params": classifier, "lr": self.args.lr_cls}], lr=self.args.lr)  # , nesterov=self.args.optim_nesterov, momentum=self.args.optim_mom, weight_decay=self.args.optim_wd)

    def observe(self, inputs, labels, not_aug_inputs):

        self.opt.zero_grad()

        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)

        loss.backward()
        self.opt.step()

        return loss.item()
