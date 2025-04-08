from rdkit import Chem

class Atom:
    def __init__(self):
        #self.xyz = None
        self.symbol = None
        self.type = None
        self.index = None

class Molecule:
    def __init__(self, mol):
        self.atomlist = []
        self.mol = mol

    def get_atom_type(self):
        for i in range(self.mol.GetNumAtoms()):
            atom = Atom()
            symbol = self.mol.GetAtomWithIdx(i).GetSymbol()
            atom.symbol = symbol
            atom.index = i
            if symbol == 'H':
                atom.type = 22
            else:
                atom.type = -1
            self.atomlist.append(atom)

        #Csp3
        patt = Chem.MolFromSmarts('[CX4]')
        if self.mol.HasSubstructMatch(patt):
            matches = self.mol.GetSubstructMatches(patt)
            for match in matches:
                self.atomlist[match[0]].type = 1
        
        #Osp3
        patt = Chem.MolFromSmarts('[OX2]')
        if self.mol.HasSubstructMatch(patt):
            matches = self.mol.GetSubstructMatches(patt)
            for match in matches:
                atom = self.mol.GetAtomWithIdx(match[0])
                self.atomlist[match[0]].type = 43

                H = False; ring = False; aro = False
                for batom in atom.GetNeighbors():
                    if batom.GetSymbol() == 'H':
                        H = batom.GetIdx()
                    elif batom.GetIsAromatic():
                        aro = True
                    elif batom.IsInRing():
                        ring = True
                if H:
                    if aro:
                        self.atomlist[H].type = 25
                    elif ring:
                        self.atomlist[H].type = 24
                    else:
                        self.atomlist[H].type = 23
        
        #Nsp3
        patt = Chem.MolFromSmarts('[NX3]')
        if self.mol.HasSubstructMatch(patt):
            matches = self.mol.GetSubstructMatches(patt)
            for match in matches:
                atom = self.mol.GetAtomWithIdx(match[0])
                if atom.GetFormalCharge() == 0:
                    self.atomlist[match[0]].type = 60
                    Hs = []; aro = False
                    for batom in atom.GetNeighbors():
                        if batom.GetSymbol() == 'H':
                            Hs.append(batom.GetIdx())
                        else:
                            if batom.GetIsAromatic():
                                aro = True
                                
                    if aro:
                        for i in Hs:
                            self.atomlist[i].type = 29
                    elif Hs:
                        if len(Hs) == 2:
                            for i in Hs:
                                self.atomlist[i].type = 26
                        else:
                            self.atomlist[Hs[0]].type = 27
                #else:
                    #guanidium
                    #print ('NX3 but charge=',atom.GetFormalCharge())
        
        #Ssp3
        patt = Chem.MolFromSmarts('[SX2]')
        if self.mol.HasSubstructMatch(patt):
            matches = self.mol.GetSubstructMatches(patt)
            for match in matches:
                atom = self.mol.GetAtomWithIdx(match[0])
                self.atomlist[match[0]].type = 81

                H = False; aro = False
                for batom in atom.GetNeighbors():
                    if batom.GetSymbol() == 'H':
                        H = batom.GetIdx()
                    elif batom.GetIsAromatic():
                        aro = True
                    
                if H:
                    if aro:
                        self.atomlist[H].type = 35
                    else:
                        self.atomlist[H].type = 34
        
        #Csp2
        patt = Chem.MolFromSmarts('[CX3]')
        if self.mol.HasSubstructMatch(patt):
            matches = self.mol.GetSubstructMatches(patt)
            for match in matches:
                    self.atomlist[match[0]].type = 2

        #Nsp2
        patt = Chem.MolFromSmarts('[NX2]')
        if self.mol.HasSubstructMatch(patt):
            matches = self.mol.GetSubstructMatches(patt)
            for match in matches:
                atom = self.mol.GetAtomWithIdx(match[0])
                if atom.GetFormalCharge() == 0:
                    self.atomlist[match[0]].type = 61
                    for batom in atom.GetNeighbors():
                        if batom.GetSymbol() == 'H':
                            self.atomlist[batom.GetIdx()].type = 28

        #Osp243
        patt = Chem.MolFromSmarts('O=*')
        if self.mol.HasSubstructMatch(patt):
            matches = self.mol.GetSubstructMatches(patt)
            for match in matches:
                self.atomlist[match[0]].type = 44
                batom = self.mol.GetAtomWithIdx(match[1])
                if batom.GetSymbol() == 'C':
                    self.atomlist[match[1]].type = 5

        #Ssp2
        patt = Chem.MolFromSmarts('[SX1]=*')
        if self.mol.HasSubstructMatch(patt):
            matches = self.mol.GetSubstructMatches(patt)
            #print (matches)
            for match in matches:
                self.atomlist[match[0]].type = 82
                batom = self.mol.GetAtomWithIdx(match[1])
                if batom.GetSymbol() == 'C':
                    self.atomlist[match[1]].type = 5
        
        
        
        #Csp
        patt = Chem.MolFromSmarts('[$(C#*),$(C(=*)=*)]')
        if self.mol.HasSubstructMatch(patt):
            matches = self.mol.GetSubstructMatches(patt)
            for match in matches:
                atom = self.mol.GetAtomWithIdx(match[0])
                self.atomlist[match[0]].type = 3
                for batom in atom.GetNeighbors():
                    if batom.GetSymbol() == 'H':
                        self.atomlist[batom.GetIdx()].type = 38

        #Nsp
        patt = Chem.MolFromSmarts('[$(N#*),$(N(=*)=*),$([N-]=*)]')
        if self.mol.HasSubstructMatch(patt):
            matches = self.mol.GetSubstructMatches(patt)
            for match in matches:
                atom = self.mol.GetAtomWithIdx(match[0])
                
                if self.mol.GetAtomWithIdx(match[0]).GetFormalCharge() == 1:
                    self.atomlist[match[0]].type = 71
                elif self.mol.GetAtomWithIdx(match[0]).GetFormalCharge() == 0:
                    self.atomlist[match[0]].type = 62
                elif self.mol.GetAtomWithIdx(match[0]).GetFormalCharge() == -1:
                    self.atomlist[match[0]].type = 72
    
        #ar
        patt = Chem.MolFromSmarts('[c,o,s]')
        if self.mol.HasSubstructMatch(patt):
            matches = self.mol.GetSubstructMatches(patt)
            for match in matches:
                atom = self.mol.GetAtomWithIdx(match[0])
                symbol = atom.GetSymbol()
                if symbol == 'C':
                    if self.atomlist[match[0]].type == -1:
                        self.atomlist[match[0]].type = 4
                        for batom in atom.GetNeighbors():
                            if batom.GetSymbol() == 'H':
                                self.atomlist[batom.GetIdx()].type = 36
                                break
                    '''
                    #CHEMBL1922320
                    else:
                        print (self.atomlist[match[0]].type)
                        xx
                    '''
                elif symbol == 'O':
                    self.atomlist[match[0]].type = 45
                else:
                    self.atomlist[match[0]].type = 83
        
        #n
        patt = Chem.MolFromSmarts('n')
        if self.mol.HasSubstructMatch(patt):
            matches = self.mol.GetSubstructMatches(patt)
            for match in matches:
                n = self.mol.GetAtomWithIdx(match[0])
                charge = n.GetFormalCharge()
                H = False; Cs = []
                for batom in n.GetNeighbors():
                    if batom.GetSymbol() == 'H':
                        H = batom.GetIdx()
                    elif batom.GetSymbol() == 'C':
                        Cs.append(batom.GetIdx())
                
                if charge == 0:
                    if H:
                        self.atomlist[match[0]].type = 64
                        self.atomlist[H].type = 37
                    else:
                        self.atomlist[match[0]].type = 63

                elif charge == 1:
                    if H:
                        self.atomlist[match[0]].type = 74
                        self.atomlist[H].type = 40
                    else:
                        self.atomlist[match[0]].type = 73

                    for C in Cs:
                        if self.mol.GetAtomWithIdx(C).GetIsAromatic():
                            self.atomlist[C].type = 17
                        else:
                            self.atomlist[C].type = 18
        
        #Nsp3+
        patt = Chem.MolFromSmarts('[NX4+]')
        if self.mol.HasSubstructMatch(patt):
            matches = self.mol.GetSubstructMatches(patt)
            for match in matches:
                #print (match)
                Hs = []; Cs = []; cs = []
                for batom in self.mol.GetAtomWithIdx(match[0]).GetNeighbors():
                    if batom.GetSymbol() == 'H':
                        Hs.append(batom.GetIdx())
                    elif batom.GetSymbol() == 'C':
                        if batom.GetIsAromatic():
                            cs.append(batom.GetIdx())#not exist
                        else:
                            Cs.append(batom.GetIdx())
                    
                self.atomlist[match[0]].type = 69
                for H in Hs:
                    self.atomlist[H].type = 39
                for C in Cs:
                    if len(self.mol.GetAtomWithIdx(C).GetNeighbors()) == 4: #Csp3
                        self.atomlist[C].type = 11
                    elif len(self.mol.GetAtomWithIdx(C).GetNeighbors()) == 3:
                        self.atomlist[C].type = 12
                    else:
                        print ('N+ - Csp?')
                        xx
                for c in cs:#N+(C3)c
                    self.atomlist[c].type = 17
                    '''
                    print ('N+ - aromatic?')
                    print (self.mol.GetProp('Smiles'))
                    writer = Chem.SDWriter('error.sdf')
                    writer.write(self.mol)
                    exit()
                    '''

        #Nsp2+
        patt = Chem.MolFromSmarts('[NX3+]')
        if self.mol.HasSubstructMatch(patt):
            matches = self.mol.GetSubstructMatches(patt)
            for match in matches:
                Hs = []; Cs = []; cs = []
                for batom in self.mol.GetAtomWithIdx(match[0]).GetNeighbors():
                    if batom.GetSymbol() == 'H':
                        Hs.append(batom.GetIdx())
                    elif batom.GetSymbol() == 'C':
                        if batom.GetIsAromatic():
                            cs.append(batom.GetIdx())#not exist
                        else:
                            Cs.append(batom.GetIdx())
                    
                self.atomlist[match[0]].type = 70
                for H in Hs:
                    self.atomlist[H].type = 39
                for C in Cs:
                    if len(self.mol.GetAtomWithIdx(C).GetNeighbors()) == 4: #Csp3
                        self.atomlist[C].type = 11
                    elif len(self.mol.GetAtomWithIdx(C).GetNeighbors()) == 3:
                        self.atomlist[C].type = 13
                    else:
                        print ('N+ - Csp?')
                for c in cs:#nitro
                    1
                    '''
                    print ('N+ - aromatic?')
                    print (self.mol.GetProp('Smiles'))
                    writer = Chem.SDWriter('error.sdf')
                    writer.write(self.mol)
                    exit()
                    '''

        #amide
        patt = Chem.MolFromSmarts('[O,S]=C-N')
        if self.mol.HasSubstructMatch(patt):
            matches = self.mol.GetSubstructMatches(patt)
            S = False
            for match in matches:
                if self.mol.GetAtomWithIdx(match[0]).GetSymbol() == 'O':
                    self.atomlist[match[0]].type = 49
                else:
                    S = True
                    self.atomlist[match[0]].type = 87
                self.atomlist[match[1]].type = 6
                self.atomlist[match[2]].type = 68
                Hs = []
                for batom in self.mol.GetAtomWithIdx(match[2]).GetNeighbors():
                    if batom.GetSymbol() == 'H':
                        Hs.append(batom.GetIdx())
                if len(Hs) == 2:
                    for H in Hs:
                        if S:
                            self.atomlist[H].type = 32
                        else:
                            self.atomlist[H].type = 30
                elif Hs:
                    if S:
                        self.atomlist[Hs[0]].type = 33
                    else:
                        self.atomlist[Hs[0]].type = 31
        
        #carboxylate
        #patt = Chem.MolFromSmarts('[#6]C(=O)[O-]')
        patt = Chem.MolFromSmarts('[#6,#7]C(=O)[O-]')#including carbamic acid
        if self.mol.HasSubstructMatch(patt):
            matches = self.mol.GetSubstructMatches(patt)
            for match in matches:
                atom = self.mol.GetAtomWithIdx(match[0])
                if atom.GetSymbol() == 'C':
                    self.atomlist[match[0]].type = 20
                else:
                    self.atomlist[match[0]].type = 75

                for batom in atom.GetNeighbors():
                    if batom.GetSymbol() == 'H':
                        self.atomlist[batom.GetIdx()].type = 41
                self.atomlist[match[1]].type = 14
                self.atomlist[match[2]].type = 53
                self.atomlist[match[3]].type = 53
        
        #N-oxide or sulfoxide
        patt = Chem.MolFromSmarts('[O-][#7+,#16+]')
        if self.mol.HasSubstructMatch(patt):
            matches = self.mol.GetSubstructMatches(patt)
            for match in matches:
                self.atomlist[match[0]].type = 59
                if self.mol.GetAtomWithIdx(match[1]).GetSymbol() == 'N':
                    self.atomlist[match[1]].type = 79
                else:
                    self.atomlist[match[1]].type = 94

        #nitro
        patt = Chem.MolFromSmarts('[N+](=O)[O-]')
        if self.mol.HasSubstructMatch(patt):
            matches = self.mol.GetSubstructMatches(patt)
            for match in matches:
                self.atomlist[match[0]].type = 78
                self.atomlist[match[1]].type = 58
                self.atomlist[match[2]].type = 58
        
        #Guanidinium
        patt = Chem.MolFromSmarts('C(N)=[+N]')
        if self.mol.HasSubstructMatch(patt):
            matches = self.mol.GetSubstructMatches(patt)
            for match in matches:
                
                self.atomlist[match[1]].type = 80
                self.atomlist[match[2]].type = 80
                for i in match[1:]:
                    for batom in self.mol.GetAtomWithIdx(i).GetNeighbors():
                        if batom.GetSymbol() == 'H':
                            self.atomlist[batom.GetIdx()].type = 42
                        elif batom.GetSymbol() == 'C':
                            if batom.GetIsAromatic():
                                self.atomlist[batom.GetIdx()].type = 17
                            else:
                                self.atomlist[batom.GetIdx()].type = 11 #use Csp3-Nsp3, may be Csp3-Nsp2
                self.atomlist[match[0]].type = 19
        '''
        #sulfonium
        #S3+
        patt = Chem.MolFromSmarts('[SX3+]')
        if self.mol.HasSubstructMatch(patt):
            matches = self.mol.GetSubstructMatches(patt)
            for match in matches:
                self.atomlist[match[0]].type = 84
        '''

        #S4
        patt = Chem.MolFromSmarts('[SX3](=O)(*)*')
        if self.mol.HasSubstructMatch(patt):
            matches = self.mol.GetSubstructMatches(patt)
            for match in matches:
                self.atomlist[match[0]].type = 89
                self.atomlist[match[1]].type = 52
                for i in match[2:]:
                    symbol = self.mol.GetAtomWithIdx(i).GetSymbol()
                    if symbol == 'O':
                        self.atomlist[i].type = 48
                    elif symbol == 'N':
                        self.atomlist[i].type = 67
                    elif symbol == 'C':
                        self.atomlist[i].type = 7
        
        #S6
        #S6=[N,S] were not considered
        patt = Chem.MolFromSmarts('S(=O)(=O)(*)*')
        if self.mol.HasSubstructMatch(patt):
            matches = self.mol.GetSubstructMatches(patt)
            for match in matches:
                O = []; O_ = []; C = []; N = []; S = []
                for i in match[3:]:
                    atom = self.mol.GetAtomWithIdx(i)
                    if atom.GetSymbol() == 'O':
                        if atom.GetFormalCharge() == -1:
                            O_.append(i)
                        else:
                            O.append(i)
                    elif atom.GetSymbol() == 'C':
                        C.append(i)
                    elif atom.GetSymbol() == 'N':
                        N.append(i)
                    elif atom.GetSymbol() == 'S':
                        S.append(i)
                if O_:
                    self.atomlist[match[0]].type = 93
                    self.atomlist[match[1]].type = 55
                    self.atomlist[match[2]].type = 55
                    for i in O_:
                        self.atomlist[i].type = 55
                    for i in O:
                        self.atomlist[i].type = 57
                    for i in N:
                        self.atomlist[i].type = 77
                    for i in C:
                        self.atomlist[i].type = 16
                    for i in S:
                        self.atomlist[i].type = 92
                else:
                    self.atomlist[match[0]].type = 88
                    self.atomlist[match[1]].type = 51
                    self.atomlist[match[2]].type = 51
                    for i in O:
                        self.atomlist[i].type = 47
                    for i in N:
                        self.atomlist[i].type = 65
                    for i in C:
                        self.atomlist[i].type = 9
                    for i in S:
                        self.atomlist[i].type = 85
                
        #P3
        patt = Chem.MolFromSmarts('[PX3]')
        if self.mol.HasSubstructMatch(patt):
            matches = self.mol.GetSubstructMatches(patt)
            for match in matches:
                self.atomlist[match[0]].type = 95
        
        #P3+
        patt = Chem.MolFromSmarts('[PX4+]')
        if self.mol.HasSubstructMatch(patt):
            matches = self.mol.GetSubstructMatches(patt)
            for match in matches:
                self.atomlist[match[0]].type = 98
                for batom in self.mol.GetAtomWithIdx(match[0]).GetNeighbors():
                    if batom.GetSymbol() == 'C':
                        self.atomlist[batom.GetIdx()].type = 21
                    else:
                        print (batom.GetSymbol(), 'bonded to PX4+', self.mol.GetProp('_Name'))

        #P5
        patt = Chem.MolFromSmarts('P(=[O,S])(*)(*)*')
        if self.mol.HasSubstructMatch(patt):
            matches = self.mol.GetSubstructMatches(patt)
            for match in matches:
                O = []; O_ = []; C = []; N = []; S = []
                for i in match[2:]:
                    atom = self.mol.GetAtomWithIdx(i)
                    if atom.GetSymbol() == 'O':
                        if atom.GetFormalCharge() == -1:
                            O_.append(atom.GetIdx())
                        else:
                            O.append(atom.GetIdx())
                    elif atom.GetSymbol() == 'C':
                        C.append(atom.GetIdx())
                    elif atom.GetSymbol() == 'N':
                        N.append(atom.GetIdx())
                    elif atom.GetSymbol() == 'S':
                        S.append(atom.GetIdx())
                #print (O_,O,C,N)
                if O_:
                    self.atomlist[match[0]].type = 97
                    if self.mol.GetAtomWithIdx(match[1]).GetSymbol() == 'O':
                        self.atomlist[match[1]].type = 54
                    elif self.mol.GetAtomWithIdx(match[1]).GetSymbol() == 'S':
                        self.atomlist[match[1]].type = 91
                    for i in O_:
                        self.atomlist[i].type = 54
                    for i in O:
                        self.atomlist[i].type = 56
                    for i in N:
                        self.atomlist[i].type = 76
                    for i in C:
                        self.atomlist[i].type = 15
                    for i in S:
                        self.atomlist[i].type = 90
                else:
                    self.atomlist[match[0]].type = 96
                    if self.mol.GetAtomWithIdx(match[1]).GetSymbol() == 'O':
                        self.atomlist[match[1]].type = 50
                    elif self.mol.GetAtomWithIdx(match[1]).GetSymbol() == 'S':
                        self.atomlist[match[1]].type = 86
                    for i in O:
                        self.atomlist[i].type = 46
                    for i in N:
                        self.atomlist[i].type = 66
                    for i in C:
                        self.atomlist[i].type = 8
                    for i in S:
                        self.atomlist[i].type = 84
        
        patt = Chem.MolFromSmarts('[F,Cl,Br,I]')
        if self.mol.HasSubstructMatch(patt):
            matches = self.mol.GetSubstructMatches(patt)
            for match in matches:
                atom = self.mol.GetAtomWithIdx(match[0])
                symbol = atom.GetSymbol()
                if symbol == 'F':
                    self.atomlist[match[0]].type = 99
                elif symbol == 'Cl':
                    self.atomlist[match[0]].type = 100
                elif symbol == 'Br':
                    self.atomlist[match[0]].type = 101
                elif symbol == 'I':
                    self.atomlist[match[0]].type = 102
                batom = atom.GetNeighbors()[0]
                if batom.GetSymbol() == 'C' and len(batom.GetNeighbors()) == 4:
                    self.atomlist[batom.GetIdx()].type = 10
        
        types = []
        for atom in self.atomlist:
            if atom.type == -1:
                return None
            types.append(atom.type)

        return types
        '''
        for i, j in enumerate(self.atomlist):
            print (i+1,j)
        print ('\n')
        '''
        
def CheckMol(mol):

    if mol.GetNumAtoms() < 6:
        return 0
    
    symbols = ['H','C','O','N','S','P','F','Cl','Br','I']
    for i in range(mol.GetNumAtoms()):
        if mol.GetAtomWithIdx(i).GetSymbol() not in symbols:
            return 0
    return 1


if __name__ == "__main__":
    #print (Chem.MolFromSmiles('CN=[N+]=[N-]').GetSubstructMatches(Chem.MolFromSmarts('[C,N](=*)=*')))
    smiles = 'C[S+](C)(C)[O-]'
    print (smiles)
    m = Chem.MolFromSmiles(smiles)
    
    m = Chem.AddHs(m)
    print (Chem.MolToSmiles(m))
    mol = Molecule(m)
    types = mol.GetAtomTypes()
    for i, t in enumerate(types):
        print (i,t)
    exit()
    supplier = Chem.SDMolSupplier('test.sd', True, False)
    for m in supplier:
        mol = Molecule(m)
        mol.GetAtomTypes()
        print (m.GetProp('_Name'))
        for i in range(len(mol.atomlist)):
            print (i+1,mol.atomlist[i].type)
        print ('\n')
    xx

    type2n = {}
    for i in range(77):
        type2n[i] = 0

    for sdf in ['ibs2019mar_nc_out','ibs2019mar_sc1_out','ibs2019mar_sc2_out','ibs2019mar_sc3_out']:
        #supplier = Chem.SDMolSupplier('Z:\\lab\\Project\\Database\\%s.sdf'%sdf, True, False)
        supplier = Chem.SDMolSupplier('test.sd', True, False)
        
        for m in supplier:
            #if m.GetProp('_Name') == 'STOCK1N-00105':
                if CheckMol(m):
                    print ('%s\n'%m.GetProp('_Name'))
                    mol = Molecule(m)
                    mol.GetAtomTypes()
                    for atom in mol.atomlist:
                        print (atom.index+1, atom.type)
                        '''
                        if atom.type == 46:
                            print (atom.index+1, m.GetProp('_Name'))
                            exit()
                            '''
                        if atom.type == -1:
                            print (atom.index+1, atom.symbol, m.GetProp('_Name'))
                            #exit()
                        else:
                            type2n[atom.type] += 1
    #for key, value in type2n.items():
        #print (key,value)
    '''
    files = fileL('D:\\Research\\Deep learning\\PDB\\SDF\\single_prep','sdf')
    with open('TypeCount.txt','w') as streem:
        type2count = {}
        for f in files:
            supplier = Chem.SDMolSupplier(f, True, False)
            self.mol = supplier[0]
            if CheckMol(self.mol):
                
                self.atomlist = GetAtomTypes(self.mol)
                for t in self.atomlist:
                    if t == 0:
                        print (os.path.basename(f))
                        print (self.atomlist)
                        #exit()
                    if t in type2count:
                        type2count[t] += 1
                    else:
                        type2count[t] = 1
        for t, c in type2count.items():
            streem.write('%s\t%s\n'%(t,c))
            '''