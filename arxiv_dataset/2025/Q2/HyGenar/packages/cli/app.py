import typer
from packages.cli.data import data
from packages.cli.bnfs import bnfs

#TODO:
# 1. [ ] evaluate various LLMs for bnf generation (syntax correctness)
# 2. [ ] use reflexion to optimize bnf generation and evaluate again ( syntax correctness): feed back non-terminal not defined, terminal wrong escape and more
# 3. [ ] evaluate generated bnf for validity (accept all positive examples and rejects all negative examples)
# 4. [ ] write a new algorithm to optimize bnf generation for enhancing validity performance


app = typer.Typer()
app.add_typer(data, name="data") # for data generation
app.add_typer(bnfs, name="bnfs") # for bnfs generation