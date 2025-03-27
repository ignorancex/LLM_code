
class Decompose:
    def __init__(self, relation_names):
        self.relation_names = relation_names
    

    def decompose(self, concept_name: str):
        pieces = []
        i = 0
        while i < len(concept_name):
            concept = ''
            while i < len(concept_name) and not concept_name[i] in ['(', ')', '⊔', '⊓', '∃', '∀', '¬', '.', ' ']:
                concept += concept_name[i]
                i += 1
            if concept:
                pieces.append(concept)
            i += 1
        relation_free_pieces = set(pieces)-self.relation_names
        main_pieces = set(pieces)
        return main_pieces, relation_free_pieces
    
        
 