
def execute_kopl_program(engine, kopl_program, ignore_error=False, show_details=False):
    functions = list(fun_call['function'] for fun_call in kopl_program)
    inputs = list(fun_call['inputs'] for fun_call in kopl_program)
    
    return engine.forward(functions, inputs, ignore_error=ignore_error, show_details=show_details)
