import json
import argparse
import os
import re

def book_to_structure(book_text):
    structure = {}
    chapters = book_text.split('\chapter{')[1:]
    for c in chapters:
        chapter_name = c.split('}')[0]
        cur = {}
        for s in c.split('\section*{')[1:]:
            section_name = s.split('}')[0]
            section_text = '\section*{' + s
            cur[section_name] = section_text
        structure[chapter_name] = cur
    return structure

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', type=str, required=True)
    args = parser.parse_args()
    
    book_path = "../../books/"

    if args.domain == 'causality':
        book_text = open(os.path.join(book_path, 'Introduction_to_Causal_Inference.tex'), 'r').read()
        structure = book_to_structure(book_text)
            
    elif args.domain == 'physics':
        structure = {}
        for i in range(3):
            book_text = open(os.path.join(book_path, 'UniversityPhysicsVol'+str(i+1)+'-WEB.tex'), 'r').read()
            vol_structure = book_to_structure(book_text)
            structure = {**structure, **vol_structure}
                
    elif args.domain == 'chemistry':
        book_text = open(os.path.join(book_path, 'Atkins_Physical_Chemistry.tex'), 'r').read()
        structure = book_to_structure(book_text)
 
    print(len(structure))
    json.dump(structure, open(f'outputs/structure_{args.domain}.json', 'w'), indent=2)