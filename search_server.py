import argparse
import functools
import logging
from pathlib import Path
import re
import sys

from model import Policy
from remote import establish_simple_server


LOG = logging.getLogger(__name__)


def handler(policy, state):
    features = state['features']
    available_actions = state['available_actions']
    prob_tensor = policy.predict_masked_normalized(features, available_actions)
    return prob_tensor.tolist()


def find_policy(args):
    if args.iteration is not None:
        return args.model / f'policy_{str(args.iteration)}.pt'

    final_policy_path = args.model / 'policy_final.pt'
    if final_policy_path.is_file():
        return final_policy_path

    # We need to enumerate the model files and find the last one
    max_iteration = 0
    for file_path in args.model.iterdir():
        if file_path.suffix != '.pt':
            continue
        match = re.match(r'policy_(\d+)', file_path.name)
        if match is not None:
            iteration = int(match.group(1))
            if iteration > max_iteration:
                max_iteration = iteration
                final_policy_path = file_path
    return final_policy_path


def main(args):
    policy_path = find_policy(args)
    if not policy_path.is_file():
        LOG.error(f'Cannot find model at: {policy_path}')
        sys.exit(1)

    policy = Policy.load_for_eval(policy_path)
    LOG.info(f'Policy loaded from {policy_path}')

    LOG.info(f'Server starting at {args.addr}:{args.port}')
    establish_simple_server(args.addr, args.port, functools.partial(handler, policy))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Demo server that can talk to Coeus search client')
    parser.add_argument(
        'model',
        type=Path,
        help='Directory that holds the model file')
    parser.add_argument(
        '-i',
        '--iteration',
        metavar='I',
        type=int,
        help='Specify the which version of the model to use (indexed by iteration number). '
             'By default, use the one that gets trained the longest')
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
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    main(args)
