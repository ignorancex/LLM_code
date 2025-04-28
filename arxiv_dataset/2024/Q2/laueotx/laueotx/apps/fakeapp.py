import click


# parser.add_argument('-v', '--verbosity', type=str, default='info', choices=('critical', 'error', 'warning', 'info', 'debug'), 
#                     help='logging level')
# parser.add_argument('--conf', type=str, required=True, 
#                     help='configuration yaml file')
# parser.add_argument('--dir_out', type=str, required=True, 
#                     help='output dir')
# parser.add_argument('--params_ot', type=str, default=None,
#                     help='parameters for the OT, string will be parsed, format: --params_ot=key1=value1,key2=value2')
# parser.add_argument('--test', action='store_true',
#                     help='test mode')
# parser.add_argument('--n_grid', type=int, default=2000,
#                     help='number of grid points from which to initialize coordinate descent')
# parser.add_argument('--calibrate_coniga', action='store_true',
#                     help='if to calibrate the coniga sample')
# parser.add_argument('--calibrate_fenimn', action='store_true',
#                     help='if to calibrate the fenimn sample')

# click.option('--params-ot', multiple=True,type=str, type=(str, int), help='Parameters for the OT, string will be parsed. Use it multiple times to add multiple key/value pairs, e.g.: --params-ot key1 value1 --params-ot key2 value2')

import functools
def common_options(f):
    @click.option('--conf', "-c", required=True, type=click.File(), show_default=True,help='Configuration yaml file')
    @click.option('output_dir','--output-dir', "-o", required=True, type=click.Path(), show_default=True,help='Directory to store the produced files')
    @click.option('--n-grid', default=2000, show_default=True,help='number of grid points from which to initialize coordinate descent')
    @click.option("--calibrate-coniga/--no-calibrate-coninga",default=False, help="Calibrate the coniga sample")
    @click.option("--calibrate-fenimn/--no-calibrate-fenimn", default=False, help="Calibrate the fenimn sample")
    @functools.wraps(f)
    def wrapper_common_options(*args, **kwargs):
        return f(*args, **kwargs)

    return wrapper_common_options


@click.group("fakeapp")
def main():
    pass

@main.command()
@common_options
@click.argument("tasks", nargs=-1)
def singlegrain(conf, output_dir, n_grid, calibrate_coniga, calibrate_fenimn,tasks):
    for x in range(count):
        click.echo(f"Hello {name}!")

@main.command()
@common_options
@click.argument("tasks", nargs=-1)
def multigrain(conf, output_dir, n_grid, calibrate_coniga, calibrate_fenimn,tasks):
    click.echo("tasks")
