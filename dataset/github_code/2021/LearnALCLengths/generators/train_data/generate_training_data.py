import sys, os
import argparse

this_file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(this_file_path.split('train_data')[0])
sys.path.append(this_file_path.split('generators')[0])

if __name__=="__main__":
    from data_generator import KBToData
    from helper_classes.embedding_triples import RDFTriples
    
    def map_kb_name_to_path(kb_name):
        if not "family-benchmark" in kb_name:
            return this_file_path.split('generators')[0]+"Datasets/"+kb_name+"/"+kb_name+".owl"
        return this_file_path.split('generators')[0]+"Datasets/"+kb_name+"/"+kb_name+"_rich_background.owl"
        
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--kb', nargs='+', type=str, default=["all"], help="List of knowledge bases for which to extract triples and generate training data class expressions")
    parser.add_argument('--refinement_path_length', type=int, default=5, help="The depth of a sequence of refinements from a given root node, higher ==> more compute power")
    parser.add_argument('--max_child_length', type=int, default=15, help="The maximum child length in the refinement operator")
    parser.add_argument('--refinement_expressivity', type=float, default=0.5, help="The expressivity of the refinement operator, note that we use ExpressRefinement")
    parser.add_argument('--num_rand_samples', type=int, default=300, help="The number of nodes to samples from the refinements of the top concept (T)")
    parser.add_argument('--min_num_pos_examples', type=int, default=1, help="The minimum number of positive examples a class expression should have to be included in training data")
    
    args = parser.parse_args()
    if not "all" in args.kb:
        kbs = list(map(lambda x: map_kb_name_to_path(x), args.kb))
    else:
        dirs = [dr for dr in os.listdir(this_file_path.split('generators')[0]+"Datasets/") if not dr.startswith(".")]
        kbs = list(map(lambda x: map_kb_name_to_path(x), dirs))
        
    for kb in kbs:
        triples = RDFTriples(source_kg_path=kb)
        triples.export_triples()

        kb_to_data = KBToData(path=kb, path_length=args.refinement_path_length, min_child_length=1, max_child_length = args.max_child_length, refinement_expressivity=args.refinement_expressivity, num_rand_samples=args.num_rand_samples, min_num_pos_examples=args.min_num_pos_examples)

        kb_to_data.generate_concepts().save_train_data()