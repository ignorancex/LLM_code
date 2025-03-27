from src import TaskHost
import os, argparse, importlib

# Hope we can get rid of absolute path in training scripts.
root_path = os.path.dirname(os.path.abspath(__file__))

main_procedure_translator = {
    'TPP_train': 'TPP',
    'TPP_plot': 'TPP'
}

sub_procedure_translator = {
    'TPP_train': 'Trainer',
    'TPP_plot': 'Plotter'
}

if __name__ == '__main__':
    # Train
    parser = argparse.ArgumentParser()

    # Enumerate subparsers from procedure_names
    # we need main_procedure_translator and sub_procedure_translator to translate
    # procedure names into the correct procedure and argument class.
    procedure_names = [
        'TPP_train',
        'TPP_plot'
        ]

    subparsers = parser.add_subparsers(help = 'Define the procedure name.')
    for procedure_name in procedure_names:
        '''
        Get the argument list
        '''
        main_procedure = main_procedure_translator[procedure_name]
        sub_procedure_argument_prefix = main_procedure + sub_procedure_translator[procedure_name]

        tmp_parser_hook = subparsers.add_parser(procedure_name, help = f'We use {procedure_name}.')
        procedure = importlib.import_module('src.' + main_procedure)
        argument_class_name = sub_procedure_argument_prefix + 'Arguments'
        getattr(procedure, argument_class_name)(tmp_parser_hook, root_path)

    agent = TaskHost(parser = parser, root_path = root_path)
    agent.start()