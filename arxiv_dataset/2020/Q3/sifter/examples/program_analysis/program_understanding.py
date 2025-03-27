"""Comparative program understanding between shell implementation snippets."""
import random
import os
from timeit import default_timer as timer
from tactic_utils import Fix, ApplyRulesMatching
from ui import serve
from lazy_structure import LazyStructure
from analyzelib import LoadDocument, AnalyzeCodelets
from analogy_utils import Analogy

def Main():
    """Runs the analogy-maker and displays the output."""
    start = timer()
    print("Setting up the structure...")
    random.seed(24)
    demo_path = os.environ.get("BUILD_WORKSPACE_DIRECTORY", ".")
    demo_path += "/examples/program_analysis/paper_demos"

    extra_special = [
        "static_shell_builtins", "shell_builtins", "builtin_address_internal",
        "builtin_datas", "builtin_lookup", "_builtin", "builtin_"]
    bash = LoadDocument(f"{demo_path}/bash.txt", extra_special=extra_special)
    fish = LoadDocument(f"{demo_path}/fish.txt", extra_special=extra_special)

    # These could be provided, eg., by an AST, or we could even have Sifter
    # rewrite rules which identify them. We can also make each the membor of
    # its sub-struct then make the sub-struct a member of the parent struct.
    bash.AnnotateChunks(dict({
        bash.FindChunks("shell_builtins")[0]: "/:Semantics:Collection",
        bash.FindChunks("cd")[2]: "/:Semantics:CollectionMember",
    }))
    fish.AnnotateChunks(dict({
        fish.FindChunks("builtin_datas")[0]: "/:Semantics:Collection",
        fish.FindChunks("cd")[2]: "/:Semantics:CollectionMember",
    }))
    bash.AnnotateChunks(dict({
        bash.FindChunks("shell_builtins")[-1]: "/:Semantics:FunctionBody",
        bash.FindChunks("builtin_address_internal")[-1]: "/:Semantics:Function",
    }))
    fish.AnnotateChunks(dict({
        fish.FindChunks("builtin_datas")[-1]: "/:Semantics:FunctionBody",
        fish.FindChunks("builtin_lookup")[-1]: "/:Semantics:Function",
    }))

    structure = LazyStructure([bash, fish], AnalyzeCodelets)
    for document in structure.documents:
        for chunk in document.chunks:
            structure.ChunkToNode(document, chunk)

    print("Identifying word pairs...")
    for word in ["cd", "shell_builtins", "builtin_datas"]:
        ApplyRulesMatching(structure.rt, "SameWord", dict({
            "/:Rules:SameWord:MustMap:Word": structure.dictionary[word].full_name,
        }))

    print("Finding the analogy...")
    analogy = Analogy.Begin(structure.rt, dict({
        "/:Mapper:NoSlipRules:A":
            structure.NodeOfChunk(bash, bash.FindChunk("cd")),
        "/:Mapper:NoSlipRules:B":
            structure.NodeOfChunk(fish, fish.FindChunk("cd")),
    }), extend_here=False)

    Fix(analogy.ExtendMap,
        ["/:Semantics:CollectionMember", "/:Semantics:Function",
         "/:Semantics:FunctionBody", "/:SameWord"])

    print(timer() - start)
    serve.start_server(structure)

if __name__ == "__main__":
    Main()
