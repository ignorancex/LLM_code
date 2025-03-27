# slot = [NUM, ]
import argparse
import warnings

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--log_level', type=str, choices={'INFO', 'DEBUG', 'ERROR'},
                    default='DEBUG', help='logging level INFO,DEBUG,ERROR default=INFO')
parser.add_argument('--algo', choices={'rwb', 'a2c', 'ppo', 'acer'}, default='rwb')
parser.add_argument('--pooling', choices={'attention', 'mean', 'max', 'slot_gate', 'slot_gate_full_attn'},
                     default='slot_gate')
parser.add_argument('--slot_num', type=int, default=1, help='slot number')
parser.add_argument('--max_turn', type=int, default=24,
                    help='the maximum count of user simulator response')
parser.add_argument('--train_steps', type=int, default=200)
parser.add_argument('--description', type=str, default="")
parser.add_argument('--USE_VISDOM', action='store_true', default=False)

# manager settings
parser.add_argument('--manager_emb_dim', type=int, default=100)
parser.add_argument('--manager_char_dim', type=int, default=64)
parser.add_argument('--manager_hidden_dim', type=int, default=64)
parser.add_argument('--manager_state_dim', type=int, default=64)
parser.add_argument('--manager_hidden_one_dim', type=int, default=32)
parser.add_argument('--manager_hidden_two_dim', type=int, default=32)
parser.add_argument('--learning_rate', type=float, default=1e-3)

# sl settings
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--seed', type=int, default=10)
parser.add_argument('--docker', action='store_true', default=False)
parser.add_argument('--fix_nlu', action='store_true', default=False)
parser.add_argument('--patience','--patience', help='', required=False, default=6, type=int)
parser.add_argument('--device', choices={'cpu', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'}, default='cpu')

MAX_STATE_LEN = 20
MAX_STATE_BLOCK_NUM = 5
MAX_UTTER_LEN = 30

PAD_TOKEN = 0
UNK_TOKEN = 1
SEP_TOKEN = 2
BOS_TOKEN = 3
EOS_TOKEN = 4

RegularExpressions = ['^1\d{10}$', '^1\d{1,}']


intent_tag_id2name = ['offer', 'inform', 'affirm', 'deny', 'update', 'ack', 'finish', 'ask_state', 'bye', 'restart', 'ask_repeat',
                      'doubt_identity', 'how_signal', 'bad_signal', 'good_signal', 'wait', 'other']
intent_tag_id2name = ['<pad>'] + intent_tag_id2name
intent_tag_names = {w: i for i, w in enumerate(intent_tag_id2name)}

sys_action_id2name = ['request', 'continue', 'req_more', 'implicit_confirm', 'explicit_confirm', 'ack', 'req_correct',
                      'compare', 'ask_restart', 'bye', 'good_signal', 'how_signal', 'bad_signal', 'repeat', 'robot', 'other']
sys_action_id2name = ['<pad>'] + sys_action_id2name
sys_action_names = {w: i for i, w in enumerate(sys_action_id2name)}