"""Learning to generalize a program optimization."""
import random
import os
from timeit import default_timer as timer
from ui import serve
from lazy_structure import LazyStructure
from analyzelib import LoadDocument, AnalyzeCodelets, CompleteAnalogyTactic

def Main():
    """Runs the analogy-maker and displays the output."""
    start = timer()
    print("Setting up the structure...")
    random.seed(24)
    demo_path = os.environ.get("BUILD_WORKSPACE_DIRECTORY", ".")
    demo_path += "/examples/program_analysis/paper_demos"
    sources = []
    for i in range(1, 4):
        with open(f"{demo_path}/gemm{i}.before.txt", "r") as peek:
            n_lines = len(peek.readlines())
        chunks = [(0, n_lines - 1)]
        before = LoadDocument(f"{demo_path}/gemm{i}.before.txt", chunks)
        after = LoadDocument(f"{demo_path}/gemm{i}.after.txt", chunks)
        sources.append((before, after))

    def annotateGTHalf(doc, outer_name, inner_name):
        outer_chunks = list(doc.FindChunks(outer_name))[-2:]
        inner_chunk = list(doc.FindChunks(inner_name))[-1]
        for outer_chunk in outer_chunks:
            doc.AnnotateChunks(dict({
                outer_chunk: "/:Semantics:Greater",
                inner_chunk: "/:Semantics:LTHalf",
            }))

    # This could be provided, eg., by an abstract interpreter.
    annotateGTHalf(sources[0][0], "outer", "inner")
    annotateGTHalf(sources[1][0], "outer", "A_cols")
    annotateGTHalf(sources[2][0], "AB_rowcol", "inner")

    structure = LazyStructure(
        [source for sourcelist in sources for source in sourcelist],
        AnalyzeCodelets)

    for document in structure.documents:
        for chunk in document.chunks:
            structure.ChunkToNode(document, chunk)

    for i in range(0, 6, 2):
        structure.AnnotateDocuments(dict({
            i: "/:TransformPair:Before",
            (i + 1): "/:TransformPair:After",
        }))

    structure.MarkDocumentGenerated(5)
    CompleteAnalogyTactic(structure, sources)
    structure.GetGeneratedDocument(5)

    print(timer() - start)
    serve.start_server(structure)

if __name__ == "__main__":
    Main()
