import json
import io
import logging 
import datetime
import os
import sys
config = None

# A bit chaos in this file

def __convert2Bool(v):
    falsev= v in ["false","False","None","Null","none","null","0",0,False,0.0]
    if falsev:
        return False
    truev= v in ["true","True",True]
    if not truev:
        try:
            return float(v)!=0
        except:
            pass
        try:
            return int(v)>0
        except:
            pass
        raise BaseException("Cannot convert to bool for "+str(v))
    else:
        return True
    
def _assignDicValue(fullDic,b,name=""):
    # is a value, store to fullDic
    fullDic["{%s}"%name]=str(b)
    try:
        intb = int(b)
        fullDic["{%s$int}"%name]=intb
    except:
        pass
    try:
        floatb = float(b)
        fullDic["{%s$float}"%name]=floatb
    except:
        pass
    try:
        boolb = __convert2Bool(b)
        fullDic["{%s$bool}"%name]=boolb
        #print("Bool %s = %s > %s"%(name,str(b),str(boolb)))
    except:
        pass

def _replace(v,replaceDic,prefix=""):
    if replaceDic is None:
        return v
    if isinstance(v,tuple):
        values=[]
        for i in v:
            values.append(_replace(i,replaceDic))
        return tuple(values)
    elif isinstance(v,str):
        newv =  _replaceInner(v,replaceDic)
        _assignDicValue(replaceDic,newv,prefix)
        return newv
    return v

def _replaceWithType(s,k,v):
    if not isinstance(s,str):
        return s
    s = s.replace(k,str(v))
    if "$int}" in k:
        try:
            #print("%s | %s | %s"%(s,k,str(v)))
            return int(s.strip())
        except:
            pass
    elif "$float}" in k:
        try:
            return float(s.strip())
        except:
            pass
    elif "$bool}" in k:
        try:
            return __convert2Bool(s.strip())
        except:
            pass
    return s

def _replaceInner(v,replaceDic):
    i = 0
    while True:
        left = v.find("{",i)
        right = v.find("}",i)
        if left<0 or right<0 or right<left:
            break
        key = v[left:(right+1)]
        if key not in replaceDic.keys():
            i=right
            if "$" in key:
                logging.warning("Cannot replace dic %s, please indicate its value in the command line"%key)
            else:
                logging.warning("Cannot replace dic %s, please check sequence of definition"%key)
            continue
        value = replaceDic[key]
        forwardLen = len(v)
        v = _replaceWithType(v,key,value)
        if not isinstance(v,str):
            break
        curLen=len(v)
        if forwardLen<=curLen:
            i=right
    return v

'''
    Json -> Obj
    replaceDic replace string place_holders with specific values

    Undefined keys can be visited, which will return False in default.
    Each ConfigObj is also callable, the parameter indicates its default value, if it does not exists in the father's dictionary
'''
class ConfigObj(object):
    def __init__(self, d, replaceDic=None,father=None,key=None):
        self.__father=father
        self.__key=key
        for a, b in d.items():
            name = str(self)[1:]+"."+a
            if name[0]==".":
                name=name[1:]
            if isinstance(b, (list, tuple)):
                setattr(self, a, [ConfigObj(x, replaceDic,self,"%s[%d]"%(a,i)) if isinstance(x, dict) else _replace(x, replaceDic,name) for i,x in enumerate(b)])
            else:
                setattr(self, a, ConfigObj(b, replaceDic,self,a) if isinstance(b, dict) else _replace(b, replaceDic,name))

    # if attribute do not exists, return False
    def __getattr__(self, item):
        if item.startswith("__"):
            return super(ConfigObj, self).__getattr__(item)
        logging.warning("Undefined Config attribute %s.<%s>"%(str(self),item))
        obj=ConfigObj({},{},self,item)
        setattr(self,item,obj)
        return obj

    def __str__(self):
        if self.__father is not None:
            if self.__key is not None:
                return "%s.%s"%(str(self.__father),self.__key)
            else:
                return "<Unknown>"
        else:
            return ""

    def print(self,head=1):
        logging.info(">"*(head-1)+" "+str(self))
        for k,v in self.__dict__.items():
            if not k.startswith("_"):
                logging.info(" "*head + " %s = %s"%(k,str(v)))
                if isinstance(v,ConfigObj):
                    v.print(head+1)
                elif isinstance(v,list):
                    head+=1
                    for i,t in enumerate(v):
                        logging.info(" "*head+"[%d] = %s"%(i,str(t)))
                        if isinstance(t,ConfigObj):
                            t.print(head+1)

    @staticmethod
    def default(config,name,defaultValue):
        vars = name.strip().split(".")
        obj=config
        for var in vars[0:-1]:
            obj = getattr(obj,var)
        if vars[-1] not in obj.__dict__.keys():
            logging.info("Config %s.%s = %s"%(str(obj),vars[-1],str(defaultValue)))
            setattr(obj,vars[-1],defaultValue)


'''
    Generate Dictionary for json:
        output:
        key:
            {name1}
            {name1.name2.name3}
            {name1.name2.name3.name4....}
        value:
            a string value
'''
def _genDic(d,fullDic,prefix=""):
    for a, b in d.items():
        if isinstance(b, (list, tuple)):
            for x in b:
                if isinstance(x, dict):
                    _genDic(x,fullDic,prefix+a+".")
                else:
                    # is tuple
                    pass
        else:
            if isinstance(b, dict):
                 _genDic(b,fullDic,prefix+a+".") 
            else:
                # is a value, store to fullDic
                _assignDicValue(fullDic,b,prefix+a)

'''
    transform ConfigObj to Dictionary
'''
def obj2dic(obj):
    if isinstance(obj,(str,int,float,bool)):
        return obj
    elif isinstance(obj,(tuple,list,set,frozenset)):
        fv=[]
        for x in obj:
            fv.append(obj2dic(x))
        return fv
    elif isinstance(obj,dict):
        nd={}
        for k,v in obj.items():
            if not k.startswith("_"):
                nd[obj2dic(k)]=obj2dic(v)
        return nd
    elif isinstance(obj,ConfigObj):
        nd2={}
        for k,v in obj.__dict__.items():
            if not k.startswith("_"):
                nd2[obj2dic(k)]=obj2dic(v)
        return nd2
    pr = {}
    for name in dir(obj):
        value = getattr(obj, name)
        try:
            if not name.startswith('_') and not callable(value):
                pr[name] = obj2dic(value)
        except:
            pass
    return pr

def anaInputParams():
    vs = [v for v in sys.argv[1:] if len(v)>0]

    i=0
    dic={}

    while i<len(vs):
        v=vs[i]
        if v.startswith("--"):
            if v=="--config_file":
                i+=2
                continue
            key = v[2:]
            i+=1
            value = vs[i]
            dic["{$%s}"%key]=value
            try:
                numv = int(value)
                dic["{$%s$int}"%key]=numv
            except:
                pass
        else:
            logging.warning("Unknown value %s at %d"%(v,i))
        i+=1
    for k,v in dic.items():
        print("Input Param %s = %s \t(%s)"%(k,str(v),str(type(v))))
    return dic


def toConfigObj(obj,dic={}):
    dt = datetime.datetime.now()
    dic["{$time}"]=dt.strftime("%Y_%m_%d__%H_%M_%S")
    _genDic(obj,dic)
    print("Generated Dictionary (%d)"%len(dic))
    dic = __processDic(dic)
    print("Processed Dictionary (%d)"%len(dic))
    finalObj=ConfigObj(obj,dic)
    print("Final Dictionary (%d)"%len(dic))
    return finalObj

def __loadJson(v):
    v=v.strip()
    #logging.info("Load Config File %s"%v)
    print("Load Config File %s"%v)
    try:
        f = open(v,"r")
        obj = json.load(f)
        f.close()
        return obj
    except BaseException as e:
        print("Cannot load config file %s -> %s"%(v,str(e)))
    return None

def __replaceWithDic(s,dic):
    
    if not isinstance(s,str):
        return s

    for k,v in dic.items():
        if k in s:
            s = _replaceWithType(s,k,v)
            if not isinstance(s,str):
                break
    return s

def __processDic(dic):
    flag=True
    while flag:
        flag=False
        newDic={}
        for k,v in dic.items():
            nv=__replaceWithDic(v,dic)
            if nv!=v:
                flag=True
            newDic[k]=nv
        dic=newDic
    return dic

def __preprocess(obj,path,dic={}):
    folder,_ = os.path.split(path)
    if isinstance(obj,dict):
        newObj={}
        for k,v in obj.items():
            if k=="{$include}":
                realPath = __replaceWithDic(os.path.join(folder,v.lstrip()),dic)
                objt = __preprocess(__loadJson(realPath),realPath,dic)
                newObj={**newObj,**objt}
                pass
            elif isinstance(v,str) and v.startswith("{$include}"):
                realPath = __replaceWithDic(os.path.join(folder,v.replace("{$include}","").lstrip()),dic)
                newObj[k]=__preprocess(__loadJson(realPath),realPath,dic)
            else:
                newObj[k]=__preprocess(v,path,dic)
        for k,v in newObj.items():
            obj[k]=v
    elif isinstance(obj,list):
        newList=[]
        for i,a in enumerate(obj):
            if isinstance(a,str) and a.startswith("{$include}"):
                realPath = __replaceWithDic(os.path.join(folder,v.replace("{$include}","").lstrip()),dic)
                objt=__preprocess(__loadJson(realPath),realPath,dic)
                newList.append((i,objt))
            else:
                newList.append((i,__preprocess(a,path,dic)))
        for i,a in newList:
            obj[i]=a
    return obj

def loadConfig(path):
    print("Load Config From %s"%path)
    dic = anaInputParams()
    obj = __preprocess(__loadJson(path),path,dic)
    print("Load complete")
    return toConfigObj(obj,dic)


# a.b.c.d = value
def setConfig(config, name, value):
    if "$" in name:
        logging.warning("Cannot set config %s = %s"%(name,str(value)))
        return
    names = name.split(".")
    if len(names)==0:
        logging.warning("[SetConfig] name is empty")
        return
    curc = config
    for i in range(len(names)-1):
        curc = getattr(curc,names[i])
    setattr(curc,names[-1],value)

