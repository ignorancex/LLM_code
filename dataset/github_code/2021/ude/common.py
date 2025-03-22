import os

col2anno = {'officehome_train':'concepts65.txt','officehome_test':'concepts65.txt', 'domainnet_train':'concepts345.txt', 'domainnet_test':'concepts345.txt'}

ROOT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),"tomm2021ude-data")

def get_anno_path(collection, domain, annotation_name, concept, rootpath=ROOT_PATH):
    anno_path = os.path.join(rootpath, collection, 'Annotations', 'Image', annotation_name, domain, '%s.txt' % concept)
    return anno_path

def get_pred_path(test_collection, domain, model_name, rootpath=ROOT_PATH):
    annotation_name = col2anno[test_collection]
    pred_path = os.path.join(rootpath, test_collection, 'Predictions', annotation_name, domain, model_name, 'id.concept.score.txt')
    return pred_path

def read_imset(collection, domain, rootpath=ROOT_PATH):
    imset_file = os.path.join(rootpath, collection, 'ImageSets', '%s.txt' % domain)
    imset = [x.strip() for x in open(imset_file) if x.strip()]
    assert(len(set(imset)) == len(imset))
    return imset


def read_domain_list(collection, rootpath=ROOT_PATH):
    domain_file = os.path.join(rootpath, collection, 'Annotations', 'domains.txt')
    domain_list = [x.strip() for x in open(domain_file) if x.strip()]
    assert(len(set(domain_list)) == len(domain_list))
    return domain_list


def read_concept_list(collection, annotation_name, rootpath=ROOT_PATH):
    concept_file = os.path.join(rootpath, collection, 'Annotations', annotation_name)
    concepts = [x.strip() for x in open(concept_file).readlines() if x.strip()]
    assert(len(set(concepts)) == len(concepts))
    return concepts


def read_anno(anno_path):
    pos_img_list = []
    for line in open(anno_path):
        imgid, label = line.strip().split()
        assert(int(label) == 1)
        pos_img_list.append(imgid)
    assert(len(set(pos_img_list)) == len(pos_img_list))
    return pos_img_list


def read_full_anno(collection, domain, annotation_name, rootpath=ROOT_PATH):
    concepts = read_concept_list(collection, annotation_name, rootpath)
    gt = {}

    for concept in concepts:
        anno_path = get_anno_path(collection, domain, annotation_name, concept, rootpath)
        pos_img_list = read_anno(anno_path)
        for im in pos_img_list:
            assert (im not in gt) 
            gt[im] = concept
    return gt

def read_pred(pred_path):
    lines = open(pred_path).readlines()
    pred = {}

    for line in lines:
        imgid, concept, score = line.strip().split()
        pred[imgid] = concept
    return pred

def compute_accuracy(preds, gts):
    assert(len(preds) == len(gts))
    count=0.0
    for imgid,pred_y in preds.items():
        if pred_y == gts[imgid]:
            count = count + 1
    accuracy = count / len(preds)
    return accuracy


if __name__ == '__main__':
    gts = {'a':1, 'b':0}
    preds = {'a':0, 'b':0}
    print (compute_accuracy(preds, gts))
    
    collection = 'domainnet_test'
    annotation_name = col2anno[collection]
    concept = 'knife'
    domain = 'real'
    model_name = 'ResNet50_%s' % domain
    model_name = 'KDDE_CDAN_ResNet50_%s_clipart' % domain
    anno_path = get_anno_path(collection, domain, annotation_name, concept)
    pos_img_list = read_anno(anno_path)
    print (anno_path, len(pos_img_list))
    
    pred_path = get_pred_path(collection, domain, model_name)
    preds = read_pred(pred_path)
    print (pred_path, len(preds))
    print (len(read_imset(collection, domain)))
    print (read_domain_list(collection))
    print (read_concept_list(collection, annotation_name))
    
    gts = read_full_anno(collection, domain, annotation_name)
    print (len(gts))
    print ('accuracy', compute_accuracy(preds, gts))

