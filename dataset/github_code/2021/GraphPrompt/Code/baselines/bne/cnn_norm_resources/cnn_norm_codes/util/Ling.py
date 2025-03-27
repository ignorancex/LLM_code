import os

class Ling():
    def __init__(self) -> None:
        stopwords = []
        with open(os.path.join("../text_resources/stopwords.txt"), 'r',encoding="ISO-8859-1") as stopwords_file:
            for line in stopwords_file:
                line = str(line).replace("\n","")
                stopwords.append(line)
        self.suffixMap  = {}
        self.set_suffix_map()
        self.prefixMap = {}
        self.setPrefixMap()
        self.AFFIX = "ganglioma|cancer"
        self.affixMap = {}
        self.setAffixMap()
        self.SINGULAR_DISORDER_SYNONYMS = ["disease", "disorder", "condition", "syndrome", "symptom",
            "abnormality", "NOS", "event", "episode", "issue", "impairment"]
            
        self.PLURAL_DISORDER_SYNONYMS = ["diseases", "disorders", "conditions", "syndromes", "symptoms", "abnormalities", "events", "episodes", "issues", "impairments"]
    
    def set_suffix_map(self, file_path = os.path.join("../text_resources/suffix.txt")):
        f = open(file_path,"r")
        for line in f:
            s = str(line).strip()
            tokens = s.split("\\|\\|")
            try:
                self.suffixMap[tokens[0]] = tokens[1]
            except:
                continue
    
    def getSuffixMap(self):
        return self.suffixMap

    def getSuffix(self, str_, len_):
        if (len(str_) < len_):
            return ""
        return str_[0:len(str_) - len_]


    def getSuffixStr(self, str_):
        return self.getSuffix(str_, 10) if self.suffixMap.__contains__(self.getSuffix(str_, 10)) else (
                self.getSuffix(str_, 7) if self.suffixMap.__contains__(self.getSuffix(str_, 7)) else (
                self.getSuffix(str_, 6) if self.suffixMap.__contains__(self.getSuffix(str_, 6)) else ( 
                self.getSuffix(str_, 5) if self.suffixMap.__contains__(self.getSuffix(str_, 5)) else ( 
                self.getSuffix(str_, 4) if self.suffixMap.__contains__(self.getSuffix(str_, 4)) else ( 
                self.getSuffix(str_, 3) if self.suffixMap.__contains__(self.getSuffix(str_, 3)) else (self.getSuffix(str_, 2) if self.suffixMap.__contains__(self.getSuffix(str_, 2)) else ""))))))
    
    def setPrefixMap(self, file_path = os.path.join("../text_resources/prefix.txt")):
        f = open(file_path,"r")
        for line in f:
            line = str(line).strip()
            tokens = line.split("||")
            value = "" if len(tokens)==1 else tokens[1]
            self.prefixMap[tokens[0]] =  value

    def getPrefixMap(self):
        return self.prefixMap

    def getPrefix(self, str_, len_):
        if len(str_) < len_:
            return ""
        return str_[0:len_]
     
    def getPrefixStr(self, str_):
        return  self.getPrefix(str_, 5) if self.prefixMap.__contains__(self.getPrefix(str_, 5)) else (
                self.getPrefix(str_, 4) if self.prefixMap.__contains__(self.getPrefix(str_, 4)) else 
                (self.getPrefix(str_, 3) if self.prefixMap.__contains__(self.getPrefix(str_, 3)) else ""))  
    

    def setAffixMap(self,file_name = os.path.join("../text_resources/affix.txt")):
        f = open(file_name,"r")
        for line_ in f:
            line_ = str(line_).strip()
            tokens = line_.split("||")
            value  = "" if len(tokens) == 1 else tokens[1]
            self.affixMap[tokens[0]] = value


    def getAffixMap(self):
        return self.affixMap

    """
    def exactTokenMatch(String phrase1, String phrase2) {
    List<String> tokens = new ArrayList<>(Arrays.asList(phrase1.split("\\s+")));
    tokens.removeAll(new ArrayList<>(Arrays.asList(phrase2.split("\\s+"))));
    return tokens.isEmpty() && phrase1.split("\\s+").length == phrase2.split("\\s+").length ? true : false;
}
    
    """



"""
     if self.suffix_map.has_key(self.get_suffix(str_,2)):
            x1 = self.get_suffix_len(str_,2)
        else:
            x1  = ""

        if self.suffix_map.has_key(self.get_suffix(str_,3)):
            x2 = self.get_suffix_len(str_,3)
        else:
            x2 = x1

        if self.suffix_map.has_key(self.get_suffix(str_,4)):
            x3 = self.get_suffix_len(str_,4)
        else:
            x3 = x2

        if self.suffix_map.has_key(self.get_suffix(str_,5)):
            x4 = self.get_suffix_len(str_,5)
        else:
            x4 = x3

        if self.suffix_map.has_key(self.get_suffix(str_,6)):
            x5 = self.get_suffix_len(str_,6)
        else:
            x5 = x4

        if self.suffix_map.has_key(self.get_suffix(str_,7)):
            x6 = self.get_suffix_len(str_,7)
        else:
            x6 = x5
        return x6
"""