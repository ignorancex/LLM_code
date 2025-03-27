import sys, os

sys.path.append(os.path.dirname(os.path.realpath(__file__)).split('generators')[0])

from generators.concept_description import ConceptDescriptionGenerator
from ontolearn import KnowledgeBase
from ontolearn.refinement_operators import ExpressRefinement
from helper_classes.decompose import Decompose
from collections import defaultdict
import random, os, copy, json
from typing import Final
from owlapy.render import DLSyntaxObjectRenderer

class KBToData:
    """
    This class takes an owl file, loads it using ontolearn.base.KnowledgeBase resulting in a knowledge base.
    A refinement operator is used to generate new concepts of various lengths.
    Next, from the large number of concept descriptions generated, we remove all longer redundant descriptions. Finally, a classifier is to use the pair (instances of target concept, length of target concept) for training. Hence, the lengths and the respective positive and negative examples of each 'target concept' are stored in dedicated dictionaries.
    """

    def __init__(self, path='', path_length=12, min_child_length=2, max_child_length=15, refinement_expressivity=0.3, downsample_refinements=True, num_rand_samples=150, min_num_pos_examples=100):
        self.path = path
        self.dl_syntax_renderer = DLSyntaxObjectRenderer()
        self.min_num_pos_examples = min_num_pos_examples
        self.kb = KnowledgeBase(path=path)
        max_num_pos_examples = min(2000, self.kb.individuals_count()-1)
        self.max_num_pos_examples = max_num_pos_examples
        relation_names: Final = frozenset([rel.get_iri().get_remainder() for rel in self.kb.ontology().object_properties_in_signature()])
        self.Decompose = Decompose(relation_names)
        self.num_ex = min(1000, self.kb.individuals_count()//2)
        atomic_concepts = frozenset(self.kb.ontology().classes_in_signature())
        self.atomic_concept_names: Final = frozenset([self.dl_syntax_renderer.render(a) for a in atomic_concepts])
        individual_types = dict()
        print("Number of atomic concepts: ", len(self.atomic_concept_names))

        rho = ExpressRefinement(knowledge_base=self.kb, max_child_length=max_child_length, downsample = downsample_refinements, expressivity=refinement_expressivity)
        self.lp_gen = ConceptDescriptionGenerator(knowledge_base=self.kb, refinement_operator=rho, depth=path_length, num_rand_samples=num_rand_samples)
        
    def generate_concepts(self):
        random.seed(1)
        print()
        print("#"*80)
        print("Started generating data on the "+self.path.split("/")[-1].split(".")[0]+" knowledge base")
        print("#"*80)
        print()
        self.concept_pos_neg = defaultdict(lambda: defaultdict(list))
        All_individuals = set(self.kb.individuals())
        self.train_concepts = []
        print("Number of individuals in the knowledge graph: {} \n".format(len(All_individuals)))
        Concepts = sorted(self.lp_gen.generate(), key=lambda c: self.kb.cl(c))
        print("Concepts generation done!\n")
        print("Number of atomic concepts: ", len(self.atomic_concept_names))
        print("Longest concept length: ", self.kb.cl(Concepts[-1]), "\n")
        print("Total number of concept descriptions generated: ", len(Concepts), "\n")
        print("Now computing similarities...")
        # Find the best valid macthing concept for each target concept
        matching_map = dict()
        Targets = dict()
        count = 0
        num_concepts = len(Concepts)
        for concept in Concepts:
            count += 1
            if count % 100 == 0: 
                print("Filtering Progress: ", 100*(count/num_concepts), "%")
            temp_set = self.kb.individuals_set(concept)
            if temp_set in Targets or not self.min_num_pos_examples <= len(temp_set) <= self.max_num_pos_examples:
                continue
            Targets[temp_set] = concept #Getting concepts with distinct instance sets
        Targets = list(Targets.values())
        num_targets = float(len(Targets))
        counter = 0
        print("Done filtering")
        print("Remaining: ", len(Targets))     
        for i, target_concept in enumerate(Targets):
            neg = list({ind.get_iri().as_str().split("/")[-1] for ind in All_individuals if not self.kb.individuals_set(ind).issubset(self.kb.individuals_set(target_concept))})
            pos = [ind.get_iri().as_str().split("/")[-1] for ind in self.kb.individuals(target_concept)]
            if (i+1)%500 == 0:
                print("Progression: {}%".format(round(100.*(i+1)/num_targets, ndigits=2)))
            if min(len(neg),len(pos)) >= self.num_ex//2:
                if len(pos) > len(neg):
                    num_neg_ex = self.num_ex//2
                    num_pos_ex = self.num_ex-num_neg_ex
                else:
                    num_pos_ex = self.num_ex//2
                    num_neg_ex = self.num_ex-num_pos_ex
            elif len(pos) > len(neg):
                num_neg_ex = len(neg)
                num_pos_ex = self.num_ex-num_neg_ex
            elif len(pos) < len(neg):
                num_pos_ex = len(pos)
                num_neg_ex = self.num_ex-num_pos_ex
            else:
                continue
            positive = random.sample(pos, num_pos_ex)
            negative = random.sample(neg, num_neg_ex)
            self.concept_pos_neg[target_concept]["positive"] = positive
            self.concept_pos_neg[target_concept]["negative"] = negative
        print("Data generation completed")
        return self


    def save_train_data(self):
        data = defaultdict(lambda: dict())
        for concept in self.concept_pos_neg:
            data[self.dl_syntax_renderer.render(concept)].update({"positive examples":self.concept_pos_neg[concept]["positive"], "negative examples":self.concept_pos_neg[concept]["negative"], "target concept length":self.kb.cl(concept)})
        if not os.path.exists(self.path[:self.path.rfind("/")]+"/Train_data/"):
            os.mkdir(self.path[:self.path.rfind("/")]+"/Train_data/")
        with open(self.path[:self.path.rfind("/")]+"/Train_data/Data.json", "w") as file:
            json.dump(data, file, ensure_ascii=False, indent=3)
        print("Data saved at %s"% "/"+("/").join(self.path.split("/")[1:-1]))
        
