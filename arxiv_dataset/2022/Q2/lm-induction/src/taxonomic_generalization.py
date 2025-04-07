import argparse
import csv
import os
import random
import torch

from collections import defaultdict
from minicons import cwe
from multiverse.vsm_utils import cosine
from multiverse import vsm
from ordered_set import OrderedSet
from reasoner import Reasoner
from tqdm import tqdm
from world import World

random.seed(42)
torch.manual_seed(42)

parser = argparse.ArgumentParser("Taxonomic Generalization Experiments")

parser.add_argument(
    "--model",
    "-m",
    default = "axxl-property",
    type = str
)

parser.add_argument(
    "--device",
    "-d",
    default = "cuda:0",
    type = str
)

parser.add_argument(
    "--balanced",
    "-b",
    action="store_true"
)

parser.add_argument(
    "--testrun",
    "-t",
    action="store_true"
)

args = parser.parse_args()

# will be argparsed
MODEL = args.model
BALANCED = False
if args.balanced:
    BALANCED = True
DEVICE = args.device

model_lr = {
    'rl-property': 2.5074850299401197e-06,
    'bl-property': 2.508483033932136e-06,
    'axxl-property': 3.0054890219560877e-06,
}

negative_sampler = {
    'axxl-property': '../../induction/checkpoints/finetuned_models/axxl-property',
    'bl-property': '../../induction/checkpoints/finetuned_models/bl-property',
    'rl-property': '../../induction/checkpoints/finetuned_models/rl-property',
}

world = World(concept_path = "../data/concept_senses.csv", 
             feature_path = '../data/experimental splits/train_1ns.csv', 
             matrix_path = "../data/train_1ns_matrix.txt")
world.create()

base_categories = ['bird.n.01', 'mammal.n.01', 'reptile.n.01', 'fish.n.01', 'insect.n.01', 'mollusk.n.01']

PROPERTIES = ['can dax', 'can fep', 'is vorpal', 'is mimsy', 'has blickets', 'has feps', 'is a wug', 'is a tove']

negative_space = defaultdict(list)
for category in base_categories:
    negative_space[category] = [c for c in list(OrderedSet(world.taxonomy['animal.n.01'].descendants()) - OrderedSet(world.taxonomy[category].descendants())) if c not in ['guinea_pig', 'stick_insect']]

negative_space.default_factory = None

vector_queries = [[c, c] for c in world.concepts]
    
sampler = vsm.VectorSpaceModel(MODEL)
embedder = cwe.CWE(negative_sampler[MODEL])
concept_vectors = embedder.extract_representation(vector_queries, layer="static")
sampler.load_vectors_from_tensor(concept_vectors, world.concepts)

trial = 1
stimuli = []

for i, category in enumerate(base_categories):
    node = world.taxonomy[category]
    num_descendants = len(node.descendants())
    # Repeat experiment 10 times
    for rep in range(10):
        
        # avoid multi word entities
        space = [c for c in node.descendants() if c not in ['guinea_pig', 'stick_insect']]
        
        # sample concepts from descendants of the category
        samples = random.sample(space, 5)
        
        # False categories to prevent biasing "can dax" to always be true.
        false_cats = []
        for k, v in world.taxonomy.items():
            if k not in world.taxonomy['animal.n.01'].path() and 'animal.n.01' not in v.ancestors() and len(v.descendants()) >= 5:
                false_cats.append(k)
                
        false_cat = random.sample(false_cats, 1)[0]
        false_space = world.taxonomy[false_cat].descendants()
        false_samples = random.sample(false_space, len(samples))
        
        for j in range(0, len(samples)):
            stim = samples[0:j+1]
            false_stim = false_samples[0:j+1]
            
            space = negative_space[category]
            within = [c for c in list(set(world.taxonomy[category].descendants()) - set(stim)) if c not in ['guinea_pig', 'stick_insect']]
        
            # similarity (cosine) based sampling scheme - model dependent
            topk_similar = cosine(sampler(stim).mean(0).unsqueeze(0), sampler(space)).topk(len(within))
            within_topk_similar = cosine(sampler(stim).mean(0).unsqueeze(0), sampler(within)).topk(len(within))

            sim = topk_similar.values.mean().item()
            within_sim = within_topk_similar.values.mean().item()

            outside_similar = [space[i] for i in topk_similar.indices.squeeze().tolist()]
            
            # random sampling scheme - globally consistent
            outside_random = random.sample(space, len(within))
            random_sim = cosine(sampler(stim).mean(0).unsqueeze(0), sampler(outside_random)).mean().item()
            
            # compute feature overlaps (jaccard between feature vectors)
            overlap_within = world.similarity(stim, within).mean().item()
            overlap_similar = world.similarity(stim, outside_similar).mean().item()
            overlap_random = world.similarity(stim, outside_random).mean().item()
            
            stimuli.append([trial, category, stim, false_stim, len(stim), within, outside_similar, outside_random, within_sim, sim, random_sim, overlap_within, overlap_similar, overlap_random])
            
            trial += 1

results = []

for prop in PROPERTIES:
    print(f"-------- Property: {prop} --------")
    for k, stimulus in enumerate(tqdm(stimuli)):
        
        reasoner = Reasoner(f'../../induction/checkpoints/finetuned_models/{MODEL}',
                            learning_rate = model_lr[MODEL], 
                            lexicon = world.lexicon,
                            device = DEVICE)
        
        trial, category, stim, false_stim, n, within, outside_similar, outside_random, within_sim, similar_sim, random_sim, overlap_within, overlap_similar, overlap_random = stimulus
        
        if BALANCED:
            labels = torch.tensor([1] * len(stim) + [0] * len(false_stim))
            premise = reasoner.prepare_stimuli(stim+false_stim, prop)
        else:
            labels = torch.tensor([1] * len(stim))
            premise = reasoner.prepare_stimuli(stim, prop)
            
        # prepare stimuli -- adds articles to concepts and pairs them with the property.
        conclusion_within = reasoner.prepare_stimuli(within, prop)
        conclusion_similar = reasoner.prepare_stimuli(outside_similar, prop)
        conclusion_random = reasoner.prepare_stimuli(outside_random, prop)
        
        reasoner.adapt(premise, labels, 20)
        
        logprob_within = reasoner.generalize(conclusion_within)[:, 1]
        logprob_similar = reasoner.generalize(conclusion_similar)[:, 1]
        logprob_random = reasoner.generalize(conclusion_random)[:, 1]
        reasoner.model.to('cpu')
        
        result = {
            'within': logprob_within.mean().item(), 
            'outside_similar': logprob_similar.mean().item(), 
            'outside_random': logprob_random.mean().item()
        }

        results.append([trial, MODEL, prop, category, n, within_sim, similar_sim, random_sim, overlap_within, overlap_similar, overlap_random, reasoner.stopping_epoch, result['within'], result['outside_similar'], result['outside_random']])

with open(f'../data/results/{MODEL}-induction-tg.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['trial', 'model', 'property', 'category', 'n', 'similarity_within', 'similarity_similar', 'similarity_random', 'overlap_within', 'overlap_similar', 'overlap_random', 'stopping_epoch', 'logprob_within', 'logprob_similar', 'logprob_random'])
    writer.writerows(results)
