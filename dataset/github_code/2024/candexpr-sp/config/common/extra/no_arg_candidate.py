import argparse
from functools import cache

def update_grammar(grammar, action_names_for_arg_candidate=None):
    # grammar = configuration._make_grammar()
    action_name = _get_action_name(action_names_for_arg_candidate)
    if action_name in ['nothing', 'all']:
        return grammar
    else:
        action = grammar.name_to_action(action_name)
        action.arg_candidate = None
        return grammar


using_arg_candidate = True


@cache
def _get_cmd_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-arg-candidate-for', dest='action_name_without_arg_candidate', help='arg-candidate is disabled for the specified action')
    args, unknown = parser.parse_known_args()

    return args


def _get_action_name(action_names_for_arg_candidate=None):
    args = _get_cmd_args()
    if args.action_name_without_arg_candidate == 'nothing':
        return 'nothing'
    elif args.action_name_without_arg_candidate == 'all':
        # global using_arg_candidate
        # using_arg_candidate = False
        return 'all'
    else:
        if action_names_for_arg_candidate is not None:
            # if not is_action_with_arg_candidate(
            #         args.action_name_without_arg_candidate,
            #         action_names_for_arg_candidate):
            #     breakpoint()
            assert is_action_with_arg_candidate(
                args.action_name_without_arg_candidate,
                action_names_for_arg_candidate)
        return args.action_name_without_arg_candidate


def get_using_arg_candidate():
    args = _get_cmd_args()
    if args.action_name_without_arg_candidate == 'all':
        return False
    else:
        return True


def is_action_with_arg_candidate(action_name, action_names_for_arg_candidate):
    return action_name in action_names_for_arg_candidate
