from torch_geometric.datasets import Planetoid, Amazon, Twitch, Coauthor
import os
from dotenv import load_dotenv


def main():
    load_dotenv('.env')

    inp_name = input("Enter dataset to be downloaded: ")
    cora_path = os.getenv('Cora')
    pubmed_path = os.getenv('Pubmed')
    citeseer_path = os.getenv('CiteSeer')
    computers_path = os.getenv('Computers')
    photos_path = os.getenv('Photo')
    twitchen_path = os.getenv('Twitch_EN')
    twitchru_path = os.getenv('Twitch_RU')
    physics_path = os.getenv('Physics')
    cs_path = os.getenv('CS')

    if inp_name == 'cora':
        cora = Planetoid(root=cora_path, name='Cora')
    elif inp_name == 'pubmed':
        pubmed = Planetoid(root=pubmed_path, name='PubMed')
    elif inp_name == 'citeseer':
        citeseer = Planetoid(root=citeseer_path, name='CiteSeer')
    elif inp_name == 'computers':
        computers = Amazon(root=computers_path, name='Computers')
    elif inp_name == 'photos':
        photos = Amazon(root=photos_path, name='Photo')
    elif inp_name == 'twitch-en':
        twitch_en = Twitch(root=twitchen_path, name='EN')
    elif inp_name == 'twitch-ru':
        twitch_ru = Twitch(root=twitchru_path, name='RU')
    elif inp_name == 'physics':
        physics = Coauthor(root=physics_path, name='Physics')
    elif inp_name == 'cs':
        cs = Coauthor(root=cs_path, name='CS')


if __name__ == '__main__':
    main()
