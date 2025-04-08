from rdkit import Chem

def get_vdw_type(m):
    types = []
    for i in range(m.GetNumAtoms()):
        types.append('0')
    
    for i in range(m.GetNumAtoms()):
        atom = m.GetAtomWithIdx(i)
        batoms = atom.GetNeighbors()
        symbol =  atom.GetSymbol()
        charge = atom.GetFormalCharge()
        if symbol == 'C':
            if atom.GetIsAromatic():
                if types[i] == '0':#c-O-
                    types[i] = '12' #aromatic
                for batom in batoms:
                    if batom.GetSymbol() == 'H':
                        types[batom.GetIdx()] = '1'
            else:
                for batom in batoms:
                    if batom.GetSymbol() == 'H':
                        types[batom.GetIdx()] = '6'
                if types[i] == '0':#C-O-, C=O
                    hyb = str(atom.GetHybridization())
                    if hyb == 'SP3':
                        types[i] = '11'
                    elif hyb == 'SP2':
                        types[i] = '14'
                    elif hyb == 'SP':
                        types[i] = '15'
        
        elif symbol == 'O':
            if atom.GetIsAromatic():
                types[i] = '24' #aromatic
            else:
                if charge == 0:
                    hyb = str(atom.GetHybridization())
                    if hyb == 'SP3':
                        types[i] = '21'
                        for batom in batoms:
                            if batom.GetSymbol() == 'H':
                                types[batom.GetIdx()] = '2'#hydroxyl
                    elif hyb == 'SP2':
                        types[i] = '24'
                        if len(batoms) == 1:
                            for batom in batoms:
                                if batom.GetSymbol() == 'C':
                                    types[batom.GetIdx()] = '13'#carbonyl
                        elif len(batoms) == 2:#phenol
                            for batom in batoms:
                                if batom.GetSymbol() == 'H':
                                    types[batom.GetIdx()] = '2'#hydroxyl
                if charge == -1:
                    types[i] = '26'
                    for batom in batoms:
                        if batom.GetSymbol() == 'C':
                            if batom.GetIsAromatic():
                                types[batom.GetIdx()] = '18'
                            else:
                                types[batom.GetIdx()] = '17'
            
        elif symbol == 'N':
            if atom.GetIsAromatic():
                if charge == 0:
                    types[i] = '31' #aromatic
                    for batom in batoms:
                        if batom.GetSymbol() == 'H':
                            types[batom.GetIdx()] = '1'
                elif charge == 1:
                    types[i] = '39'
                    for batom in batoms:
                        if batom.GetSymbol() == 'H':
                            types[batom.GetIdx()] = '9'
            else:
                if charge == 0:
                    for batom in batoms:
                        if batom.GetSymbol() == 'H':
                            types[batom.GetIdx()] = '4'
                        
                    hyb = str(atom.GetHybridization())
                    if hyb == 'SP3':
                        types[i] = '34'
                    elif hyb == 'SP2':
                        types[i] = '36'
                    elif hyb == 'SP':
                        types[i] = '37'

                elif charge == 1:
                    types[i] = '38'
                    for batom in batoms:
                        if batom.GetSymbol() == 'H':
                            types[batom.GetIdx()] = '8'#carbonyl
                
                elif charge == -1:
                    types[i] = '38'
        
        elif symbol == 'P':
            types[i] = '41'
            for batom in batoms:
                if batom.GetSymbol() == 'H':
                    types[batom.GetIdx()] = '6'

        elif symbol == 'S':
            types[i] = '42'
            for batom in batoms:
                if batom.GetSymbol() == 'H':
                    types[batom.GetIdx()] = '7'

        elif symbol == 'F':
            types[i] = '51'

        elif symbol == 'Cl':
            types[i] = '52'

        elif symbol == 'Br':
            types[i] = '53'

        elif symbol == 'I':
            types[i] = '54'

    patt = Chem.MolFromSmarts('[CX3](=[O])[NX3]') #amide
    if m.HasSubstructMatch(patt):
        matches = m.GetSubstructMatches(patt)
        for match in matches:
            types[match[1]] = '23'
            types[match[2]] = '33'
            for batom in m.GetAtomWithIdx(match[2]).GetNeighbors():
                if batom.GetSymbol() == 'H':
                    types[batom.GetIdx()] = '5'
    
    patt = Chem.MolFromSmarts('[CX3](=O)[O-]')#carboxlate
    if m.HasSubstructMatch(patt):
        matches = m.GetSubstructMatches(patt)
        for match in matches:
            types[match[0]] = '16'
            types[match[1]] = '25'
            types[match[2]] = '25'
    
    patt = Chem.MolFromSmarts('[CX3](=O)[OX2H1]')#carboxlic acid
    if m.HasSubstructMatch(patt):
        matches = m.GetSubstructMatches(patt)
        for match in matches:
            types[match[0]] = '16'
            types[match[1]] = '23'
            types[match[2]] = '23'
            for batom in m.GetAtomWithIdx(match[2]).GetNeighbors():
                if batom.GetSymbol() == 'H':
                    types[batom.GetIdx()] = '3'
    
    patt = Chem.MolFromSmarts('[NX3+](=O)[O-]') #nitro
    if m.HasSubstructMatch(patt):
        matches = m.GetSubstructMatches(patt)
        for match in matches:
            types[match[0]] = '32'
            types[match[1]] = '22'
            types[match[2]] = '22'
    return types
