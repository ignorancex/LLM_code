"""Parse a LazyStructure to extract abstractions for the UI."""
from collections import defaultdict
from lazy_structure import LazyGeneratedTextDocument

def parse_lazy_structure(structure):
    """Returns a dict representation of a LazyStructure.

    Returns a dictionary with
    {
        "documents": [
            {
                "text": str,
                "chunks": [(global_id, start, length)],
            }
        ],
        "maps": [
            [chunk_1_gid, chunk_2_gid, ...]
        ],
    }

    maps[i][j] lists all chunks corresponding to the jth node in abstraction i.
    """
    ts = structure.ts
    parsed = dict({"documents": [], "maps": []})
    # (1) Add the documents.
    all_chunk_nodes = set()
    for document in structure.documents:
        parsed_doc = dict({
            "text": document.text,
            "chunks": [],
            "generated": isinstance(document, LazyGeneratedTextDocument),
        })
        for chunk in document.chunks:
            chunk_node = structure.NodeOfChunk(document, chunk)
            if ts.has_node(chunk_node):
                start, length = chunk
                parsed_doc["chunks"].append((chunk_node, start, length))
                all_chunk_nodes.add(chunk_node)
        # Add fake chunks for the rest of the document.
        parsed_doc["chunks"] = pad_chunks(parsed_doc["chunks"], document)
        parsed["documents"].append(parsed_doc)
    # (2) Add the maps.
    abstract_nodes = defaultdict(set)
    for fact in ts.lookup(None, None, "/:Mapper:Abstraction"):
        for other_fact in ts.lookup(fact[1], None, None):
            if other_fact[1] in all_chunk_nodes:
                abstract_nodes[other_fact[2]].add(other_fact[1])
    parsed["maps"] = sorted(map(sorted, abstract_nodes.values()))
    return parsed

def pad_chunks(chunks, document):
    """Fills in gaps in @chunks.

    For example, @chunks may only contain chunks of the document which have
    actually been added to the structure. This iterates over @chunks, and
    anywhere a gap is found it inserts a new chunk with the node name as False.

    @chunks should be a list of triples (node_name, start, length). The return
    value is of the same format.
    """
    padded = [(False, 0, 0)]
    chunks = chunks + [(False, len(document.text), 0)]
    for (global_id, start, length) in chunks:
        last_end = padded[-1][1] + padded[-1][2]
        if last_end < start:
            padded.append((False, last_end, start - last_end))
        padded.append((global_id, start, length))
    return padded[1:-1]
