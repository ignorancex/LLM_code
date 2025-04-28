

class MoleculeConstructError(Exception):
    def __init__(self, *args: object) -> None:
        super(MoleculeConstructError, self).__init__(*args)
        
class ProteinConstructError(Exception):
    def __init__(self, *args: object) -> None:
        super(ProteinConstructError).__init__(*args)