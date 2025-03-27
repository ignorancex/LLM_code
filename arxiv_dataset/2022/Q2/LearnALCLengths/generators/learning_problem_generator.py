import random, json
from collections import defaultdict, Counter
from ontolearn.refinement_operators import ExpressRefinement
from ontolearn import KnowledgeBase
from owlapy.render import DLSyntaxObjectRenderer
import os

class LearningProblemGenerator:
    """
    Learning problem generator.
    """
    
    def __init__(self, kb_path=None, depth=2, num_rand_samples=150, max_num_probs_per_length=50, max_ref_child_length=6, refinement_expressivity=0.1, min_num_pos_examples=100, max_num_pos_examples=1000):
        assert kb_path is not None, "Provide a path for the knowledge base"
        self.kb = KnowledgeBase(path=kb_path)
        self.rho = ExpressRefinement(self.kb, max_child_length=max_ref_child_length, expressivity=refinement_expressivity)
        self.depth = depth
        self.num_rand_samples = num_rand_samples
        self.max_num_probs_per_length = max_num_probs_per_length
        self.min_num_pos_examples = min_num_pos_examples
        self.max_num_pos_examples = max_num_pos_examples
        self.dl_syntax_renderer = DLSyntaxObjectRenderer()
        self.path = kb_path
        
        atomic_concepts = frozenset(self.kb.ontology().classes_in_signature())
        self.atomic_concept_names: Final = frozenset([self.dl_syntax_renderer.render(a) for a in atomic_concepts])
        
                
    def apply_rho(self, concept):
        refinements = {ref for ref in self.rho.refine(concept)}
        if refinements:
            return list(refinements)
        
    def generate(self):
        print("#"*70)
        print("Started generating learning problems on %s" % self.path.split("/")[-1].split(".")[0]+" knowledge graph")
        print("#"*70)
        roots = self.apply_rho(self.kb.thing)
        print ("|Thing refinements|: ", len(roots))
        Refinements = set()
        Refinements.update(roots)
        for root in random.sample(roots, k=self.num_rand_samples):
            current_state = root
            for _ in range(self.depth):
#                 try:
                refts = self.apply_rho(current_state)
                current_state = random.choice(refts) if refts else None
                if current_state is None:
                    break
                Refinements.update(refts)
#                 except AttributeError:
#                     continue
        return Refinements

    def Filter(self, max_num_concept_per_length=200):
        self.learning_problems = defaultdict(lambda : defaultdict(list))
        length_tracker = Counter()
        All_individuals = set(self.kb.individuals())
        print("Number of individuals in the knowledge graph: {} \n".format(len(All_individuals)))
        self.train_concepts = dict()
        generated_concept_descriptions = sorted(self.generate(), key=lambda c: self.kb.cl(c))
        cardinality = len(generated_concept_descriptions)
        print('\n Number of concept descriptions generated: ', cardinality, "\n")
        count = 0
        for concept in generated_concept_descriptions:
            temp_set = self.kb.individuals_set(concept)
            count += 1
            if length_tracker[self.kb.cl(concept)] >= self.max_num_probs_per_length or not self.min_num_pos_examples <= len(temp_set) <= self.max_num_pos_examples:
                continue
            if count % 100 == 0:
                print('Progress: ', 100 * float(count)/cardinality, "%")
            valid_neg = [ind.get_iri().as_str().split("/")[-1] for ind in All_individuals if not self.kb.individuals_set(ind).issubset(self.kb.individuals_set(concept))]
            valid_pos = [ind.get_iri().as_str().split("/")[-1] for ind in self.kb.individuals(concept)]
            if not temp_set in self.train_concepts and (length_tracker[self.kb.cl(concept)] < self.max_num_probs_per_length):
                self.train_concepts[temp_set] = self.kb.cl(concept)
                length_tracker.update([self.kb.cl(concept)])
            else:
                continue
            self.learning_problems[self.dl_syntax_renderer.render(concept)]['positive examples'].extend(valid_pos)
            self.learning_problems[self.dl_syntax_renderer.render(concept)]['negative examples'].extend(valid_neg)
        return self
            
    def save_learning_problems(self):
        data = defaultdict(lambda: dict())
        for concept_name in self.learning_problems:
            data[concept_name].update({"positive examples":self.learning_problems[concept_name]['positive examples'], "negative examples":self.learning_problems[concept_name]['negative examples']})
        if not os.path.exists(self.path[:self.path.rfind("/")]+"/Learning_problems/"):
            os.mkdir(self.path[:self.path.rfind("/")]+"/Learning_problems/")
        with open(self.path[:self.path.rfind("/")]+"/Learning_problems/learning_problems.json", "w") as file:
            json.dump(data, file, ensure_ascii=False, indent=3)
        print("Learning problems saved at %s"% self.path[:self.path.rfind("/")]+"/Learning_problems/")
            