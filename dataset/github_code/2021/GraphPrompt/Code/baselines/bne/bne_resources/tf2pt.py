import argparse
import math
import json
from encoder import *
import os

class BNE:
    def __init__(self, session, params):
        self.session = session
        self.encoder = NameEncoder(params, session)
        print(self.encoder)
        # Load pretrained model parameters.
        print('Load pretrained model from {}'.format(arg_model + '/best_model'))

        session.run(tf.global_variables_initializer())
        variables = [_ for _ in tf.global_variables()
                     if _ not in self.encoder.ignored_vars]

        print(variables)
        self.saver = tf.train.Saver(variables, max_to_keep=1)

        self.saver.restore(session, arg_model + '/best_model')

        # Load pretrained word embeddings.
        emb = self.encoder.we_core.eval(session)

        
        emb_holder = tf.placeholder(tf.float32, [len(emb), params['we_dim']])
    
        init_op = self.encoder.we_core.assign(emb_holder)
        
        session.run(init_op, feed_dict={emb_holder: emb})


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--model", type=str, default = "./models/BNE_SGsc", help='Path to the pretrained encoder.')

    args = parser.parse_args()
    arg_model = args.model

    config = tf.ConfigProto(allow_soft_placement=True)

    with tf.Session(config=config) as session:
        # Load the pretrained model's configuration.

        with open(arg_model + '/params.json', encoding='utf8') as f:
            params = json.load(f)
        with open('w2id.json') as g:
            params['w2id'] = json.load(g)
            
        # Declare and load pretrained model.
        model = BNE(session, params)
        encoder = model.encoder


