from openai import OpenAI
import os
from music_cnn_core import GENRES
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description='Generate textual descriptions per genre that serve as prompts for '
                                             'MusicGen. Note: Requires an OpenAI API key to be set as'
                                             'environment variable OPENAI_API_KEY.')
parser.add_argument('--descriptions-folder', type=str, required=True, help='Path where descriptions will be stored')
args = parser.parse_args()
output_folder = args.descriptions_folder

client = OpenAI()

N_RUNS = 10  # Each run generates 10 descriptions
if not os.path.isdir(output_folder):
    os.mkdir(output_folder)

for genre in GENRES:
    if genre == 'HipHop':
        genre_str = 'Hip Hop'
    else:
        genre_str = genre

    genre_folder = os.path.join(output_folder, genre)
    if not os.path.isdir(genre_folder):
        os.mkdir(genre_folder)
    for run in range(N_RUNS):
        print(genre, run + 1, ' / ', N_RUNS)
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            temperature=0.5,
            n=10,
            messages=[
                {"role": "system", "content": "You are a music expert writing short textual descriptions for songs."},
                {"role": "user", "content": f"Write ten descriptions for an instrumental {genre_str} track. "
                                            f"Each description contains a single sentence. "
                                            f"It mentions that it is an instrumental {genre_str} track and give details on "
                                            f"tempo and instruments."}
            ]
          )
        for c, choice in enumerate(completion.choices):
            responses = choice.message.content.split('\n')
            responses = [x.split('. ')[-1] for x in responses]
            trg_path = os.path.join(genre_folder, genre + '_run_' + str(run) + '_choice_' + str(c) + '.txt')
            with open(trg_path, 'w') as fp:
                for r in responses:
                    fp.write(r)
                    fp.write('\n')


