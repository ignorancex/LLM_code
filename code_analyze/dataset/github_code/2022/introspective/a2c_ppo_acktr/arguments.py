import argparse

import torch


def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument(
        '--algo', default='ppo', help='algorithm to use: a2c | ppo | acktr')
    parser.add_argument(
        '--lr', type=float, default=7e-4, help='learning rate (default: 7e-4)')
    parser.add_argument(
        '--eps',
        type=float,
        default=1e-5,
        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.99,
        help='RMSprop optimizer apha (default: 0.99)')
    parser.add_argument(
        '--use-gae',
        action='store_true',
        default=False,
        help='use generalized advantage estimation')
    parser.add_argument(
        '--gae-lambda',
        type=float,
        default=0.95,
        help='gae lambda parameter (default: 0.95)')
    parser.add_argument(
        '--entropy-coef',
        type=float,
        default=0.01,
        help='entropy term coefficient (default: 0.01)')
    parser.add_argument(
        '--value-loss-coef',
        type=float,
        default=0.5,
        help='value loss coefficient (default: 0.5)')
    parser.add_argument(
        '--max-grad-norm',
        type=float,
        default=0.5,
        help='max norm of gradients (default: 0.5)')
    parser.add_argument(
        '--seed', type=int, default=3, help='random seed (default: 3)')
    parser.add_argument(
        '--cuda-deterministic',
        action='store_true',
        default=False,
        help="sets flags for determinism when using CUDA (potentially slow!)")
    parser.add_argument(
        '--num-processes',
        type=int,
        default=64,
        help='how many training CPU processes to use (default: 64)')
    parser.add_argument(
        '--num-steps',
        type=int,
        default=50,
        help='number of forward steps in A2C (default: 5)')
    parser.add_argument(
        '--ppo-epoch',
        type=int,
        default=1,
        help='number of ppo epochs (default: 1)')
    parser.add_argument(
        '--num-mini-batch',
        type=int,
        default=1,
        help='number of batches for ppo (default: 1)')
    parser.add_argument(
        '--clip-param',
        type=float,
        default=0.2,
        help='ppo clip parameter (default: 0.2)')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=10,
        help='log interval, one log per n updates (default: 10)')
    parser.add_argument(
        '--save-interval',
        type=int,
        default=1000,
        help='save interval, one save per n updates (default: 1000)')
    parser.add_argument(
        '--no-cuda',
        action='store_true',
        default=False,
        help='disables CUDA training')
    parser.add_argument(
        '--video',
        action='store_true',
        default=False,
        help='video')
    parser.add_argument(
        '--planning', default='high', help='planning to use: low | mid | high')
    parser.add_argument(
        '--speed', default='average', help='speed to use: veryslow | slow | average | fast | veryfast')
    parser.add_argument(
        '--vision', default='long', help='vision to use: short | medium | long')
    parser.add_argument(
        '--prey-weights', default='', help='weights for prey for eval')
    parser.add_argument(
        '--predator-weights', default='', help='weights for predator for eval')

    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    assert args.planning in ['low', 'mid', 'high']
    assert args.speed in ['veryslow', 'slow', 'average', 'fast', 'veryfast']
    assert args.vision in ['short', 'medium', 'long']

    return args
