"""Learning to generalize an API migration."""
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
    chunks = [(0, 7), (9, 5)]
    sources = []
    for i in range(1, 4):
        before = LoadDocument(f"{demo_path}/api{i}.before.txt", chunks)
        after = LoadDocument(f"{demo_path}/api{i}.after.txt",
                             chunks if i != 3 else [])
        sources.append((before, after))

    chunks = []
    docs_before = LoadDocument(f"{demo_path}/docs.before.txt")
    docs_after = LoadDocument(f"{demo_path}/docs.after.txt")

    structure = LazyStructure(
        [source for sourcelist in sources for source in sourcelist]
        + [docs_before, docs_after], AnalyzeCodelets)

    for document in structure.documents[:-2]:
        for chunk in document.chunks:
            structure.ChunkToNode(document, chunk)

    def insertChunks(doc, chunk_texts):
        for chunk in doc.chunks:
            if doc.ChunkWord(chunk) in chunk_texts:
                structure.ChunkToNode(doc, chunk)

    doc_words = set({
        "cam_record_video", "cam_record_audio", "cam_record_frame",
        "On", "error", "failure", "returns",
        "-1", "-2", "-3", "-4", "-5", "-6",
    })
    insertChunks(docs_before, doc_words)
    insertChunks(docs_after, doc_words)

    for i in range(0, 8, 2):
        structure.AnnotateDocuments(dict({
            i: "/:TransformPair:Before",
            (i + 1): "/:TransformPair:After",
        }))
        if i != 6:
            structure.AnnotateDocuments(dict({
                i: "/:Documentation:Code",
                6: "/:Documentation:Docs",
            }))
            structure.AnnotateDocuments(dict({
                (i + 1): "/:Documentation:Code",
                7: "/:Documentation:Docs",
            }))

    structure.MarkDocumentGenerated(5)
    CompleteAnalogyTactic(structure, sources)
    structure.GetGeneratedDocument(5)

    print(timer() - start)
    serve.start_server(structure)

if __name__ == "__main__":
    Main()
