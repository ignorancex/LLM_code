"""
For licensing see accompanying LICENSE file.
Copyright (C) 2024 Apple Inc. All Rights Reserved.
"""

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import os
from os.path import dirname, abspath, join
import time
import json
from datetime import datetime
import numpy as np

from helpers import convert_points_to_serializable, load_models
from sae import SAE

# get current date
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# outputs folder
outputs_folder = '../outputs'

app = Flask(__name__)
CORS(app)
static_folder = '../frontend/build'

rootDir = dirname(abspath(''))
print('root dir:', rootDir)
print('-----------------------------------')
model_dict = load_models()
python_version = model_dict['python_version']
print('-----------------------------------')

""" UPDATE HERE IF YOU ADD A NEW DATASET """
# load SAEs
sae_wiki = SAE(model_dict, dataset='wiki')
print('-----------------------------------')
sae_chi = SAE(model_dict, dataset='chi')
print('-----------------------------------')

SAE_DICT = {
    'wiki': sae_wiki,
    'chi': sae_chi
}

""" END UPDATE """

# Path to read data


@app.route("/data/<dataset>", methods=['GET'])
def read_data(dataset):
    time_start = time.time()

    dataset_name = f"{dataset}_data.json"
    d = json.load(open(join(rootDir, 'data', dataset, dataset_name), 'r'))
    print("New dataset:", dataset)
    time_end = time.time()
    print('Embeddings shape:', SAE_DICT[dataset].embeddings.shape)

    print("Data loaded! Time elapsed: {} seconds".format(time_end - time_start))
    print('-----------------------------------')
    return d

# Path to get top activations from sentence


@app.route("/top_activations_and_neighbors", methods=['GET'])
def get_top_activations():
    dataset = request.args.get('dataset')
    sentence = request.args.get('sentence')
    id = request.args.get('id')

    if not dataset or not sentence or not id:
        return jsonify({'error': 'Dataset, sentence, and id are required'}), 400

    id = int(id)

    print('Getting top features for sentence:', sentence)
    sae = SAE_DICT[dataset]
    top_features, similar_features = sae.get_top_activations_from_sentence(
        sentence, id)
    print(f'Found {len(similar_features)} similar features')

    # this is a df so we need to convert it to json
    top_features = top_features.to_dict(orient='records')

    print('Getting neighbors for feature:', id)
    neighbors = sae.get_top_neighbors(id)

    # Convert neighbors to a list if it's a NumPy array
    if isinstance(neighbors, np.ndarray):
        neighbors = neighbors.tolist()

    result = {'top_features': top_features, 'neighbors': neighbors,
              'similar_features': similar_features}
    print('-----------------------------------')
    return jsonify(result)


# path to generate new points from sentence and set of features
@app.route("/generate_points", methods=['GET'])
def generate_points():
    dataset = request.args.get('dataset')
    sentence = request.args.get('sentence')
    id = request.args.get('sent_id')
    features = request.args.get('feature_ids')
    gen_num = request.args.get('gen_num')

    if not dataset or not sentence or not id or not gen_num or not features:
        return jsonify({'error': 'Dataset, sentence, id, gen_num, and feature_ids are required'}), 400

    gen_num = int(gen_num)
    print(f'[SAE] Generating {gen_num} new points for sentence:', sentence)
    # features is a list of {id: int, weight: float}
    features = json.loads(features)
    # make sure the ids are integers and weights are floats
    features = [{'id': int(f['id']), 'weight': float(f['weight'])}
                for f in features]
    print('With features:', features)

    start_time = time.time()
    sae = SAE_DICT[dataset]
    # generated_points = sae.generate_new_points(sentence, int(id), features)
    generated_points = sae.generate_new_points(
        sentence, int(id), features, gen_num)
    end_time = time.time()
    total_time = end_time - start_time

    # generated points is an array of {'sentence': str, 'umap_x': int, 'umap_y': int}
    # so we need to convert it to json

    serializable_points = convert_points_to_serializable(generated_points)
    print('Done! Time elapsed:', total_time)
    print('-----------------------------------')
    return serializable_points

# path to generate new points from sentence and prompt


@app.route("/generate_points_llm", methods=['GET'])
def generate_points_llm():
    dataset = request.args.get('dataset')
    sentence = request.args.get('sentence')
    prompt = request.args.get('prompt')
    gen_num = request.args.get('gen_num')

    if not dataset or not sentence or not prompt or not gen_num:
        return jsonify({'error': 'Dataset, sentence, prompt, and gen_num are required'}), 400

    gen_num = int(gen_num)
    print(f'[LLM] Generating {gen_num} new points for sentence:', sentence)
    print('With prompt:', prompt)

    start_time = time.time()
    sae = SAE_DICT[dataset]
    generated_points = sae.generate_new_points_llm(sentence, gen_num, prompt)
    end_time = time.time()
    total_time = end_time - start_time

    # generated points is an array of {'sentence': str, 'umap_x': int, 'umap_y': int}
    # so we need to convert it to json
    serializable_points = convert_points_to_serializable(generated_points)
    print('Done! Time elapsed:', total_time)
    print('-----------------------------------')
    return serializable_points

# path to interpolate between two points


@app.route("/interpolate_points", methods=['GET'])
def interpolate_points():
    dataset = request.args.get('dataset')
    sent1 = request.args.get('sent1')
    id1 = request.args.get('id1')
    sent2 = request.args.get('sent2')
    id2 = request.args.get('id2')
    gen_num = request.args.get('gen_num')

    if not dataset or not sent1 or not id1 or not sent2 or not id2 or not gen_num:
        return jsonify({'error': 'Dataset, sent1, id1, sent2, id2, and gen_num are required'}), 400

    gen_num = int(gen_num)
    print(
        f'[INT] Interpolating {gen_num} new points between sentences:', sent1, 'and', sent2)

    start_time = time.time()
    sae = SAE_DICT[dataset]
    interpolated_points = sae.interpolate_between_points(
        sent1, int(id1), sent2, int(id2), gen_num)
    end_time = time.time()
    total_time = end_time - start_time

    # generated points is an array of {'sentence': str, 'umap_x': int, 'umap_y': int, 'weight': float}
    # so we need to convert it to json
    serializable_points = convert_points_to_serializable(interpolated_points)
    print('Done! Time elapsed:', total_time)
    print('-----------------------------------')
    return serializable_points

# path to add a sentence manually to dataset


@app.route("/add_sentence_manual", methods=['POST'])
def add_sentence_manual():
    data = request.json
    if not data:
        return jsonify({'error': 'No data received'}), 400

    dataset = data.get('dataset')
    sentence = data.get('sentence')
    print('Adding sentence to dataset:', sentence)

    start_time = time.time()
    sae = SAE_DICT[dataset]
    new_points = sae.add_new_sentence(sentence)
    serializable_points = convert_points_to_serializable(new_points)
    end_time = time.time()
    total_time = end_time - start_time
    print('Done! Time elapsed:', total_time)
    print('-----------------------------------')
    return jsonify({'success': True, 'data': serializable_points})


# path to remove point from dataset


@app.route("/remove_sentence", methods=['DELETE'])
def remove_sentence():
    dataset = request.args.get('dataset')
    id = request.args.get('id')
    if not dataset or not id:
        return jsonify({'error': 'Dataset and ID are required'}), 400
    try:
        id = int(id)
        print(f'Removing point: {id}')

        sae = SAE_DICT[dataset]
        sae.remove_embedding(id)
        print('-----------------------------------')
        return jsonify({'success': True, 'message': 'Point removed successfully'})
    except ValueError:
        return jsonify({'error': 'Invalid ID format'}), 400
    except Exception as e:
        print(f"Error removing embedding: {str(e)}")
        return jsonify({'error': 'Failed to remove point'}), 500

# path to edit sentence in dataset


@app.route("/edit_sentence", methods=['PUT'])
def edit_sentence():
    # Handle both query parameters and JSON body
    data = request.json
    if not data:
        return jsonify({'error': 'No data received'}), 400

    dataset = data.get('dataset')
    id = data.get('id')
    new_sentence = data.get('new_sentence')
    if not dataset or not id or not new_sentence:
        return jsonify({'error': 'Dataset, ID, and new_sentence are required'}), 400

    try:
        id = int(id)
        print(f'Editing sentence {id} in dataset:', dataset)
        print('New sentence:', new_sentence)

        sae = SAE_DICT[dataset]
        new_points = sae.edit_sentence(id, new_sentence)
        serializable_points = convert_points_to_serializable(new_points)
        print('-----------------------------------')
        return jsonify({'success': True, 'data': serializable_points})
    except ValueError:
        return jsonify({'error': 'Invalid ID format'}), 400
    except Exception as e:
        print(f"Error editing embedding: {str(e)}")
        return jsonify({'error': 'Failed to edit point'}), 500

# path to add multiple embeddings to dataset


@app.route("/add_sentences", methods=['POST'])
def add_sentences():
    data = request.json
    if not data:
        return jsonify({'error': 'No data received'}), 400
    dataset = data.get('dataset')
    sentences = data.get('sentences')
    total_sentences = data.get('total_sentences')

    if not dataset or not sentences or not total_sentences:
        error_msg = f"Dataset, sentences, and total_sentences are required. Received: dataset={dataset}, sentences={sentences}, total_sentences={total_sentences}"
        print(error_msg)
        return jsonify({'error': error_msg}), 400

    try:
        sae = SAE_DICT[dataset]
        existing_emb_length = len(sae.embeddings)
        if existing_emb_length >= total_sentences:
            return jsonify({'success': True, 'message': 'No new points need to be added'})
        print(f'Adding {len(sentences)} points to dataset')
        sae.add_sentence_embeddings(sentences)
        print('-----------------------------------')
        return jsonify({'success': True, 'message': 'Points added successfully'})
    except Exception as e:
        print(f"Error adding embeddings: {str(e)}")
        return jsonify({'error': 'Failed to add points'}), 500

# path to generate prompt ideas for a sentence


@app.route("/get_prompt_ideas", methods=['GET'])
def get_prompt_ideas():
    dataset = request.args.get('dataset')
    sentence = request.args.get('sentence')
    if not dataset or not sentence:
        return jsonify({'error': 'Dataset and sentence are required'}), 400
    print('Getting prompts for sentence:', sentence)
    sae = SAE_DICT[dataset]
    prompt_ideas = sae.get_prompt_ideas(sentence)
    result = {'prompt_ideas': prompt_ideas}
    print('-----------------------------------')
    return jsonify(result)

# path to re-embed all sentences in the dataset


@app.route("/reembed_sentences", methods=['POST'])
def reembed_sentences():
    data = request.json
    if not data:
        return jsonify({'error': 'No data received'}), 400

    dataset = data.get('dataset')
    sae = SAE_DICT[dataset]
    new_points = sae.reembed_all_sentences()
    serializable_points = convert_points_to_serializable(new_points)
    print('-----------------------------------')
    return serializable_points

# path to download data


@app.route("/download_data", methods=['POST'])
def download_data():
    data = request.json
    if not data:
        return jsonify({'error': 'No data received'}), 400

    filename = data.get('filename')
    data_content = data.get('data')

    if not filename or not data_content:
        return jsonify({'error': 'Missing filename or data'}), 400

    # replace all slashes with dashes
    filename = filename.replace('/', '-')

    # Ensure the outputs folder exists
    os.makedirs(outputs_folder, exist_ok=True)

    # write data to file
    file_path = os.path.join(
        outputs_folder, f"{filename}")
    with open(file_path, 'w') as f:
        json.dump(data_content, f, indent=2)

    print(f'File saved at: {file_path}')
    print('-----------------------------------')

    return jsonify({'success': True, 'message': 'Data downloaded successfully'})

# Path for main Svelte page


@app.route("/")
def base():
    return send_from_directory(static_folder, 'index.html')

# Path for all static files


@app.route("/<path:path>")
def home(path):
    return send_from_directory(static_folder, path)


if __name__ == "__main__":
    print('Starting Flask server...')
    print('-----------------------------------')
    app.run(debug=True, port=5000)
