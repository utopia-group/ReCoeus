import argparse
import logging
from pathlib import Path
import torch
import sys

from remote import open_connection
from model import Policy
from policy_gradient import Reinforce, Parameters

LOG = logging.getLogger(__name__)


def train(verifier, args):
    num_actions = verifier.get_num_actions()
    num_features = verifier.get_num_features()
    num_training = verifier.get_num_training()
    if num_actions <= 0:
        LOG.error(f'Illegal action count: {num_actions}')
        sys.exit(1)
    if num_features <= 0:
        LOG.error(f'Illegal feature count: {num_features}')
        sys.exit(1)
    if num_training <= 0:
        LOG.error(f'Illegal training example count: {num_training}')
        sys.exit(1)

    LOG.warning(
        'Verifier connected '
        f'(actions count = {num_actions}, '
        f'features count = {num_features}, '
        f'training example count = {num_training})'
    )

    policy = Policy(num_features, num_actions)
    params = Parameters(
        seed=args.seed if args.seed is not None else torch.initial_seed() & ((1 << 63) - 1),
        num_training=num_training,
        num_episodes=args.num_episodes,
        batch_size=args.batch_size,
        restart_count=args.restart_count,
        discount_factor=args.gamma,
        learning_rate=args.learning_rate,
        tracking_window=args.tracking_window,
        save_interval=args.save_interval,
    )
    trainer = Reinforce(verifier, policy, params, args.output)
    trainer.train()


def main(args):
    if args.output is not None:
        LOG.info(f'Setting output path to {args.output}')
    if args.num_episodes <= 0:
        LOG.error(f'Episode count must be positive: {args.num_episodes}')
        sys.exit(1)
    if args.save_interval is not None and args.save_interval <= 0:
        LOG.error(f'Save interval must be positive: {args.save_interval}')
        sys.exit(1)
    if args.batch_size <= 0:
        LOG.error(f'Batch size must be positive: {args.batch_size}')
        sys.exit(1)
    if args.restart_count <= 0:
        LOG.error(f'Restart count must be positive: {args.restart_count}')
        sys.exit(1)
    if args.tracking_window <= 0:
        LOG.error(f'Tracking window must be positive: {args.tracking_window}')
        sys.exit(1)
    if args.gamma > 1.0 or args.gamma <= 0.0:
        LOG.error(f'Discount factor must be in (0, 1]: {args.batch_size}')
        sys.exit(1)

    LOG.info(f'Connecting to verifier at remote host {args.addr}:{args.port}...')
    try:
        with open_connection((args.addr, args.port)) as verifier:
            train(verifier, args)
    except FileExistsError:
        LOG.error('Model output directory already exists.')
        LOG.error(
            'To prevent accidental overwrites, please remove or rename the existing file first.'
        )
        sys.exit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Coeus learning engine')
    parser.add_argument(
        '-a',
        '--addr',
        metavar='HOST',
        type=str,
        default='localhost',
        help='Host name of the server')
    parser.add_argument(
        '-p',
        '--port',
        metavar='PORT',
        type=int,
        default=12345,
        help='Remote port of the server')
    parser.add_argument(
        '-n',
        '--num-episodes',
        type=int,
        default=1000,
        metavar='N',
        help='Max number of training episodes (default: 1000)')
    parser.add_argument(
        '-o',
        '--output',
        type=Path,
        metavar='PATH',
        help='Directory where output files (trained models, event logs) are stored. '
             'Note that nothing will be saved if this argument is absent')
    parser.add_argument(
        '-g',
        '--gamma',
        type=float,
        default=0.99,
        metavar='G',
        help='Discount factor (default: 0.99)')
    parser.add_argument(
        '-l',
        '--learning-rate',
        type=float,
        default=1e-3,
        metavar='L',
        help='Learning rate (default: 0.001)')
    parser.add_argument(
        '-b',
        '--batch-size',
        type=int,
        default=32,
        metavar='B',
        help='Batch size (default: 1)'
    )
    parser.add_argument(
        '-r',
        '--restart-count',
        type=int,
        default=1,
        metavar='R',
        help='Number of rollouts on the same benchmark. '
             'Setting r>1 may benefit from conflict analysis if it is enabled '
             'on the server side. (default: 1)'
    )
    parser.add_argument(
        '-s',
        '--seed',
        type=int,
        metavar='SEED',
        help='Random seed (default: auto chosen)')
    parser.add_argument(
        '-w',
        '--tracking-window',
        type=int,
        default=250,
        metavar='W',
        help='How many episodes are considered when tracking training statistics (default: 250)')
    parser.add_argument(
        '-i',
        '--save-interval',
        type=int,
        metavar='I',
        help='Interval between saving trained model. '
             'By default models are only saved after the training is done')
    parser.add_argument(
        '-v',
        '--verbose',
        dest='verbose_count',
        action="count",
        default=0,
        help="increases log verbosity for each occurence.")
    args = parser.parse_args()

    if args.verbose_count == 0:
        log_level = logging.WARNING
    elif args.verbose_count == 1:
        log_level = logging.INFO
    else:
        log_level = logging.DEBUG
    logging.basicConfig(
        level=log_level,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    main(args)
