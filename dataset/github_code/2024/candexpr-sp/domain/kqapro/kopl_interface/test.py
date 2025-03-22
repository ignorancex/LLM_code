
from tqdm import tqdm

from dhnamlib.pylib.filesys import json_load, python_pretty_save

from .execution import postprocess_prediction
from .original import execute_kopl_program
from ..execution import KoPLCompiler


def test_dataset():
    from configuration import config

    dataset = json_load('./dataset/kqapro/train.json')

    correct_count = 0
    incorrect_example_info = []

    for example_idx, example in tqdm(enumerate(dataset)):
        denotation_by_kopl = execute_kopl_program(config.context, example['program'])
        answer_by_kopl = postprocess_prediction(denotation_by_kopl)
        if answer_by_kopl == example['answer']:
            correct_count += 1
        else:
            incorrect_example_info.append((example_idx, (example['answer'], answer_by_kopl)))

    print(f'#correct: {correct_count}')
    print(f'#incorrect: {len(dataset) - correct_count}')
    print(f'acc: {correct_count / len(dataset)}')

    incorrect_example_info_file_path = './_tmp_incorrect_example_info.txt'
    python_pretty_save(incorrect_example_info, incorrect_example_info_file_path)
    print(f'incorrect_example_info is saved as {incorrect_example_info_file_path}')


def test_dataset_size():
    dataset = json_load('./dataset/kqapro/train.json')
    print(len(dataset))


def test_dataset_answers():
    dataset = json_load('./dataset/kqapro/train.json')
    for example_idx, example in tqdm(enumerate(dataset)):
        if example['answer'] not in example['choices']:
            breakpoint
            print(example)
    print('done')


# context = KoPLEngine(json.load(open(os.path.join(args.input_dir, 'kb.json'))))


def test_grammar():
    from tqdm import tqdm

    from splogic.base.grammar import read_grammar
    from dhnamlib.pylib.filesys import json_load
    from dhnamlib.pylib.time import TimeMeasure
    from .execution import postprocess_prediction
    from .execution import postprocess_denotation
    from .kopl_original import execute_kopl_program
    from .grammar import KQAProGrammar
    from . import kopl_transfer
    from configuration import config

    grammar = read_grammar('./domain/kqapro/grammar.lissp', grammar_cls=KQAProGrammar)
    compiler = KoPLCompiler()
    dataset = json_load('./dataset/kqapro/train.json')

    # action_seq = kopl_transfer.kopl_to_action_seq(grammar, dataset[0]['program'])

    tm = TimeMeasure()
    kopl_to_action_seq_cumtime = 0
    get_last_state_cumtime = 0
    compile_tree_cumtime = 0
    program_cumtime = 0
    postprocess_prediction_cumtime = 0

    for example_idx, example in tqdm(enumerate(dataset)):
        labeled_kopl_program = example['program']
        answer = example['answer']

        try:
            with tm:
                action_seq = kopl_transfer.kopl_to_action_seq(grammar, config.context, labeled_kopl_program)
            kopl_to_action_seq_cumtime += tm.interval

            with tm:
                last_state = grammar.search_state_cls.get_last_state(action_seq, verifying=True)
            get_last_state_cumtime += tm.interval

            with tm:
                program = compiler.compile_tree(last_state.tree)
            compile_tree_cumtime += tm.interval

            with tm:
                denotation = program(config.context)
            program_cumtime += tm.interval

            with tm:
                prediction = postprocess_prediction(denotation)
            postprocess_prediction_cumtime += tm.interval

            denotation_by_kopl = execute_kopl_program(config.context, labeled_kopl_program)
            prediction_by_kopl = postprocess_prediction(denotation_by_kopl)

            if answer != prediction:
                if denotation == denotation_by_kopl:
                    assert prediction == prediction_by_kopl
                    print(f'The labeled answer of id {example_idx} is incorrect. Expected {prediction_by_kopl} but got {answer}')
                else:
                    breakpoint()
                    print(f'Incorrect prediction for an example of index {example_idx}')
                    # raise Exception('incorrect prediction')
        except Exception as e:
            if len(e.args) > 0 and e.args[0] == '_map_action_seq':
                breakpoint()
                print(f'error in _map_action_seq. opened: {e.args[1]} . children: {e.args[2]}, action: {e.args[3]}')
                print(f'skip {example_idx}')
            else:
                raise e

    cum_time_dict = dict(
        kopl_to_action_seq = kopl_to_action_seq_cumtime,
        get_last_state     = get_last_state_cumtime,
        compile_tree       = compile_tree_cumtime,
        program            = program_cumtime,
        postprocess_prediction = postprocess_prediction_cumtime)  # cumulative time
    avg_time_dict = dict([k, v / len(dataset)] for k, v in cum_time_dict.items())

    print()
    print('=== Cumulative time ===')
    for k, v in avg_time_dict.items():
        print(f'- {k}: {v}')

    print('=== Average time ===')
    for k, v in avg_time_dict.items():
        print(f'- {k}: {v}')


# def _test():
#     from splogic.base.grammar import read_grammar
#     from dhnamlib.pylib.filesys import json_load
#     from dhnamlib.pylib.time import TimeMeasure
#     # from dhnamlib.pylib.cProfiling import run_context
#     # import cProfile

#     grammar = read_grammar('./language/kopl/grammar.lissp', grammar_cls=KQAProGrammar)
#     compiler = grammar.compiler_cls()
#     # kb = json_load('./_tmp_data-indented/indented-kb.json')
#     dataset = json_load('./_tmp_data-indented/indented-train.json')

#     action_seq = kopl_transfer.kopl_to_action_seq(grammar, dataset[0]['program'])

#     def test():
#         tm = TimeMeasure()

#         tm.check()
#         last_state = grammar.search_state_cls.get_last_state(action_seq, verifying=True)
#         print(f'actions to the last state: {tm.elapse()}')

#         print(f'logical form: {last_state.tree.get_expr_str("visual")}')

#         tm.check()
#         program = compiler.compile_tree(last_state.tree)
#         print(f'compiling a tree: {tm.elapse()}')

#         tm.check
#         denotation = program(config.context)
#         print(f'executing a program: {tm.elapse()}')

#         print(f'denotation: {denotation}')

#     # with TimeMeasure() as tm:
#     #     test()
#     #     # run_context('test()', sort='cumtime')
#     # print(f'Time: {tm.interval} seconds')

#     test()

#     # breakpoint()
#     pass
