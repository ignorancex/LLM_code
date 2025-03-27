"""
For licensing see accompanying LICENSE file.
Copyright (C) 2024 Apple Inc. All Rights Reserved.

Test functions.
"""

from helpers import load_models
from sae import SAE

def test_top_activations(sae, sentence):
    print('TEST 1: get_top_activations')
    print('getting top features for sentence:', sentence)
    top_activations, similar_features = sae.get_top_activations_from_sentence(
        sentence, 0)
    top_k = 10
    print(top_activations[:top_k])
    print(similar_features)
    return top_activations


def test_add_features_and_invert(sae, top_features, sentence):
    print('TEST 2: add_features_and_invert')
    print('adding features and inverting for sentence:', sentence)
    k = 3  # top k features
    # feature_ids = top_features['feature'][:k].tolist()
    new_sentence = sae.add_features_and_invert(
        sentence, 0, top_features)
    return new_sentence


def test_generate_new_points(sae, sentence, top_features, gen_num):
    print('TEST 3: generate_new_points')
    print('generating new points for sentence:', sentence)
    k = 3  # top k features
    # feature_ids = top_features['feature'][:k].tolist()
    new_points = sae.generate_new_points(sentence, -1, top_features, gen_num)
    print(new_points)
    return new_points


def test_get_top_neighbors(sae, sentence_id):
    print('TEST 4: get_top_neighbors')
    print('getting top neighbors for sentence:', sentence_id)
    top_neighbors = sae.get_top_neighbors(sentence_id)
    return top_neighbors


def test_generate_points_llm(sae, sentence, gen_num, prompt):
    print('TEST 5: generate_points_llm')
    print('generating new points for sentence:', sentence)
    new_points = sae.generate_new_points_llm(sentence, gen_num, prompt)
    print(new_points)
    return new_points


def test_interpolate_points(sae, sent1, id1, sent2, id2, gen_num):
    print('TEST 6: interpolate_points')
    print('interpolating between sentences:', sent1, 'and', sent2)
    new_points = sae.interpolate_between_points(
        sent1, id1, sent2, id2, gen_num)
    print(new_points)
    return new_points


def test_generate_prompts(sae, sentence):
    print('TEST 7: generate_prompts')
    print('generating prompts for sentence:', sentence)
    prompts = sae.get_prompt_ideas(sentence)
    return prompts


def main():
    model_dict = load_models()
    dataset = 'redteam'
    sae = SAE(model_dict, dataset=dataset)
    # sentence = "The quick brown fox jumps over the lazy dog."
    sentence = "A time traveller interviews major historical figures at three points in their lives : Their 16th birthday , the day after they made their most important decision , and the day before they die ."
    # sentence2 = "An elephant is walking in the forest."
    sentence2 = "A toy boat floats out to sea and has an adventure ."

    # top_features = test_top_activations(sae, sentence)
    # print('-----------------------------------')

    # top_features = [{"id": 0, "weight": 1}, {
    #     "id": 1, "weight": 2}, {"id": 2, "weight": 1}]
    # top_features = [{"id": 0, "weight": -0.5}, {
    #     "id": 4623, "weight": 0.5}] # wiki
    top_features = [{"id": 6752, "weight": -0.5}, {
        "id": 6666, "weight": 0.5}]  # redteam
    # test_add_features_and_invert(sae, top_features, sentence)
    # print('-----------------------------------')

    gen_num = 5
    test_generate_new_points(sae, sentence, top_features, gen_num)
    print('-----------------------------------')

    # test_get_top_neighbors(sae, 0)
    # print('-----------------------------------')

    # prompt = "make this sentence more formal"
    prompt = "Explore different cultural contexts by changing the historical figures to representatives from diverse civilizations throughout history."
    test_generate_points_llm(sae, sentence, gen_num, prompt)
    print('-----------------------------------')

    # test_interpolate_points(sae, sentence, -1, sentence2, -1, gen_num)
    # print('-----------------------------------')

    # test_generate_prompts(sae, sentence)


if __name__ == '__main__':
    main()
