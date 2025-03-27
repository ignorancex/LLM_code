import sieves.AffixationSieve
import sieves.CompoundPhraseSieve
import sieves.DiseaseModifierSynonymsSieve
import sieves.HyphenationSieve
import sieves.PartialMatchNCBISieve
import sieves.PartialMatchSieve
import sieves.PrepositionalTransformSieve
import sieves.Sieve
import sieves.SimpleNameSieve
import sieves.StemmingSieve
import sieves.SymbolReplacementSieve
import util.Concept
import util.Terminology


class MultiPassSieveNormalizer():
    def __init__(self):
        pass
     
    maxSieveLevel =  10 
    def pass_(self, concept, currentSieveLevel):
        if not concept.getCui()=="":
            concept.setAlternateCuis(Sieve.getAlternateCuis(concept.getCui()))
            concept.setNormalizingSieveLevel(currentSieveLevel-1)
            return False
        
        if currentSieveLevel > maxSieveLevel:
            return False
        
        return True
        
    
    def applyMultiPassSieve(self, concept):
        currentSieveLevel = 1     
        concept.setCui(Sieve.exactMatchSieve(concept.getName()));      
        if not self.pass_(concept, ++currentSieveLevel):
            return
        
        concept.setCui(Sieve.exactMatchSieve(concept.getNameExpansion()))
        if not self.pass_(concept, ++currentSieveLevel):
            return

        concept.setCui(PrepositionalTransformSieve.apply(concept))
        if not self.pass_(concept, ++currentSieveLevel):
            return
        
        # Sieve 4
        concept.setCui(SymbolReplacementSieve.apply(concept))
        if not self.pass_(concept, ++currentSieveLevel):
            return
        
        # Sieve 5
        concept.setCui(HyphenationSieve.apply(concept))
        if not self.pass_(concept, ++currentSieveLevel):         
            return
        
        # Sieve 6
        concept.setCui(AffixationSieve.apply(concept));
        if not self.pass_(concept, ++currentSieveLevel):
            return       
        
        # Sieve 7
        concept.setCui(DiseaseModifierSynonymsSieve.apply(concept))
        if not self.pass_ss(concept, ++currentSieveLevel):           
            return
        
        # Sieve 8
        concept.setCui(StemmingSieve.apply(concept))
        if not self.pass_(concept, ++currentSieveLevel):
            return      
        
        # Sieve 9
        concept.setCui(CompoundPhraseSieve.applyNCBI(concept.getName()) if Main.test_data_dir.toString().contains("ncbi") else CompoundPhraseSieve.apply(concept.getName()))
        if not self.pass_(concept, ++currentSieveLevel):            
            return
                
        # Sieve 10
        concept.setCui(SimpleNameSieve.apply(concept))
        self.pass_(concept, ++currentSieveLevel) 
        --currentSieveLevel
        if not concept.getCui()=="":
            return                
        # Sieve 10
        concept.setCui(PartialMatchNCBISieve.apply(concept) if Main.test_data_dir.toString().contains("ncbi") else PartialMatchSieve.apply(concept))
        self.pass_(concept, ++currentSieveLevel)       
