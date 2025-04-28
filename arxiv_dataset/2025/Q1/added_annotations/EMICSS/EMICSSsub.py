#!/usr/bin/env python

#
# Generated Thu Aug  1 15:07:24 2024 by generateDS.py version 2.38.6.
# Python 3.7.6 (default, Dec 30 2019, 19:38:28)  [Clang 11.0.0 (clang-1100.0.33.16)]
#
# Command line options:
#   ('-o', 'EMICSS/EMICSS.py')
#   ('-s', 'EMICSS/EMICSSsub.py')
#
# Command line arguments:
#   ../../emicss-schema/current/emdb_emicss.xsd
#
# Command line:
#   /usr/local/bin/generateDS.py -o "EMICSS/EMICSS.py" -s "EMICSS/EMICSSsub.py" ../../emicss-schema/current/emdb_emicss.xsd
#
# Current working directory (os.getcwd()):
#   added_annotations
#

import os
import sys
from lxml import etree as etree_

import ??? as supermod

def parsexml_(infile, parser=None, **kwargs):
    if parser is None:
        # Use the lxml ElementTree compatible parser so that, e.g.,
        #   we ignore comments.
        parser = etree_.ETCompatXMLParser()
    try:
        if isinstance(infile, os.PathLike):
            infile = os.path.join(infile)
    except AttributeError:
        pass
    doc = etree_.parse(infile, parser=parser, **kwargs)
    return doc

def parsexmlstring_(instring, parser=None, **kwargs):
    if parser is None:
        # Use the lxml ElementTree compatible parser so that, e.g.,
        #   we ignore comments.
        try:
            parser = etree_.ETCompatXMLParser()
        except AttributeError:
            # fallback to xml.etree
            parser = etree_.XMLParser()
    element = etree_.fromstring(instring, parser=parser, **kwargs)
    return element

#
# Globals
#

ExternalEncoding = ''
SaveElementTreeNode = True

#
# Data representation classes
#


class emicssSub(supermod.emicss):
    def __init__(self, emdb_id=None, version='0.9.5', schema_location=None, dbs=None, entry_ref_dbs=None, primary_citation=None, weights=None, sample=None, **kwargs_):
        super(emicssSub, self).__init__(emdb_id, version, schema_location, dbs, entry_ref_dbs, primary_citation, weights, sample,  **kwargs_)
supermod.emicss.subclass = emicssSub
# end class emicssSub


class cross_ref_dbSub(supermod.cross_ref_db):
    def __init__(self, name=None, source=None, accession_id=None, uniprot_start=None, uniprot_end=None, type_=None, provenance=None, score=None, **kwargs_):
        super(cross_ref_dbSub, self).__init__(name, source, accession_id, uniprot_start, uniprot_end, type_, provenance, score,  **kwargs_)
supermod.cross_ref_db.subclass = cross_ref_dbSub
# end class cross_ref_dbSub


class dbsTypeSub(supermod.dbsType):
    def __init__(self, collection_date=None, db=None, **kwargs_):
        super(dbsTypeSub, self).__init__(collection_date, db,  **kwargs_)
supermod.dbsType.subclass = dbsTypeSub
# end class dbsTypeSub


class dbTypeSub(supermod.dbType):
    def __init__(self, source=None, version=None, **kwargs_):
        super(dbTypeSub, self).__init__(source, version,  **kwargs_)
supermod.dbType.subclass = dbTypeSub
# end class dbTypeSub


class entry_ref_dbsTypeSub(supermod.entry_ref_dbsType):
    def __init__(self, entry_ref_db=None, **kwargs_):
        super(entry_ref_dbsTypeSub, self).__init__(entry_ref_db,  **kwargs_)
supermod.entry_ref_dbsType.subclass = entry_ref_dbsTypeSub
# end class entry_ref_dbsTypeSub


class entry_ref_dbTypeSub(supermod.entry_ref_dbType):
    def __init__(self, source=None, accession_id=None, provenance=None, **kwargs_):
        super(entry_ref_dbTypeSub, self).__init__(source, accession_id, provenance,  **kwargs_)
supermod.entry_ref_dbType.subclass = entry_ref_dbTypeSub
# end class entry_ref_dbTypeSub


class primary_citationTypeSub(supermod.primary_citationType):
    def __init__(self, doi=None, provenance=None, ref_citation=None, authors=None, **kwargs_):
        super(primary_citationTypeSub, self).__init__(doi, provenance, ref_citation, authors,  **kwargs_)
supermod.primary_citationType.subclass = primary_citationTypeSub
# end class primary_citationTypeSub


class ref_citationTypeSub(supermod.ref_citationType):
    def __init__(self, source=None, accession_id=None, provenance=None, **kwargs_):
        super(ref_citationTypeSub, self).__init__(source, accession_id, provenance,  **kwargs_)
supermod.ref_citationType.subclass = ref_citationTypeSub
# end class ref_citationTypeSub


class authorsTypeSub(supermod.authorsType):
    def __init__(self, author=None, **kwargs_):
        super(authorsTypeSub, self).__init__(author,  **kwargs_)
supermod.authorsType.subclass = authorsTypeSub
# end class authorsTypeSub


class authorTypeSub(supermod.authorType):
    def __init__(self, name=None, orcid_id=None, order=None, provenance=None, **kwargs_):
        super(authorTypeSub, self).__init__(name, orcid_id, order, provenance,  **kwargs_)
supermod.authorType.subclass = authorTypeSub
# end class authorTypeSub


class weightsTypeSub(supermod.weightsType):
    def __init__(self, weight_info=None, **kwargs_):
        super(weightsTypeSub, self).__init__(weight_info,  **kwargs_)
supermod.weightsType.subclass = weightsTypeSub
# end class weightsTypeSub


class weight_infoTypeSub(supermod.weight_infoType):
    def __init__(self, pdb_id=None, assemblies=None, weight=None, unit=None, provenance=None, **kwargs_):
        super(weight_infoTypeSub, self).__init__(pdb_id, assemblies, weight, unit, provenance,  **kwargs_)
supermod.weight_infoType.subclass = weight_infoTypeSub
# end class weight_infoTypeSub


class sampleTypeSub(supermod.sampleType):
    def __init__(self, name=None, cross_ref_dbs=None, supramolecules=None, macromolecules=None, **kwargs_):
        super(sampleTypeSub, self).__init__(name, cross_ref_dbs, supramolecules, macromolecules,  **kwargs_)
supermod.sampleType.subclass = sampleTypeSub
# end class sampleTypeSub


class cross_ref_dbsTypeSub(supermod.cross_ref_dbsType):
    def __init__(self, cross_ref_db=None, **kwargs_):
        super(cross_ref_dbsTypeSub, self).__init__(cross_ref_db,  **kwargs_)
supermod.cross_ref_dbsType.subclass = cross_ref_dbsTypeSub
# end class cross_ref_dbsTypeSub


class supramoleculesTypeSub(supermod.supramoleculesType):
    def __init__(self, supramolecule=None, **kwargs_):
        super(supramoleculesTypeSub, self).__init__(supramolecule,  **kwargs_)
supermod.supramoleculesType.subclass = supramoleculesTypeSub
# end class supramoleculesTypeSub


class supramoleculeTypeSub(supermod.supramoleculeType):
    def __init__(self, type_=None, id=None, copies=None, provenance=None, name=None, cross_ref_dbs=None, **kwargs_):
        super(supramoleculeTypeSub, self).__init__(type_, id, copies, provenance, name, cross_ref_dbs,  **kwargs_)
supermod.supramoleculeType.subclass = supramoleculeTypeSub
# end class supramoleculeTypeSub


class cross_ref_dbsType1Sub(supermod.cross_ref_dbsType1):
    def __init__(self, cross_ref_db=None, **kwargs_):
        super(cross_ref_dbsType1Sub, self).__init__(cross_ref_db,  **kwargs_)
supermod.cross_ref_dbsType1.subclass = cross_ref_dbsType1Sub
# end class cross_ref_dbsType1Sub


class macromoleculesTypeSub(supermod.macromoleculesType):
    def __init__(self, macromolecule=None, **kwargs_):
        super(macromoleculesTypeSub, self).__init__(macromolecule,  **kwargs_)
supermod.macromoleculesType.subclass = macromoleculesTypeSub
# end class macromoleculesTypeSub


class macromoleculeTypeSub(supermod.macromoleculeType):
    def __init__(self, type_=None, id=None, copies=None, provenance=None, name=None, ccd_id=None, cross_ref_dbs=None, **kwargs_):
        super(macromoleculeTypeSub, self).__init__(type_, id, copies, provenance, name, ccd_id, cross_ref_dbs,  **kwargs_)
supermod.macromoleculeType.subclass = macromoleculeTypeSub
# end class macromoleculeTypeSub


class cross_ref_dbsType2Sub(supermod.cross_ref_dbsType2):
    def __init__(self, cross_ref_db=None, **kwargs_):
        super(cross_ref_dbsType2Sub, self).__init__(cross_ref_db,  **kwargs_)
supermod.cross_ref_dbsType2.subclass = cross_ref_dbsType2Sub
# end class cross_ref_dbsType2Sub


def get_root_tag(node):
    tag = supermod.Tag_pattern_.match(node.tag).groups()[-1]
    rootClass = None
    rootClass = supermod.GDSClassesMapping.get(tag)
    if rootClass is None and hasattr(supermod, tag):
        rootClass = getattr(supermod, tag)
    return tag, rootClass


def parse(inFilename, silence=False):
    parser = None
    doc = parsexml_(inFilename, parser)
    rootNode = doc.getroot()
    rootTag, rootClass = get_root_tag(rootNode)
    if rootClass is None:
        rootTag = 'emicss'
        rootClass = supermod.emicss
    rootObj = rootClass.factory()
    rootObj.build(rootNode)
    # Enable Python to collect the space used by the DOM.
    if not SaveElementTreeNode:
        doc = None
        rootNode = None
    if not silence:
        sys.stdout.write('<?xml version="1.0" ?>\n')
        rootObj.export(
            sys.stdout, 0, name_=rootTag,
            namespacedef_='',
            pretty_print=True)
    return rootObj


def parseEtree(inFilename, silence=False):
    parser = None
    doc = parsexml_(inFilename, parser)
    rootNode = doc.getroot()
    rootTag, rootClass = get_root_tag(rootNode)
    if rootClass is None:
        rootTag = 'emicss'
        rootClass = supermod.emicss
    rootObj = rootClass.factory()
    rootObj.build(rootNode)
    mapping = {}
    rootElement = rootObj.to_etree(None, name_=rootTag, mapping_=mapping)
    reverse_mapping = rootObj.gds_reverse_node_mapping(mapping)
    # Enable Python to collect the space used by the DOM.
    if not SaveElementTreeNode:
        doc = None
        rootNode = None
    if not silence:
        content = etree_.tostring(
            rootElement, pretty_print=True,
            xml_declaration=True, encoding="utf-8")
        sys.stdout.write(content)
        sys.stdout.write('\n')
    return rootObj, rootElement, mapping, reverse_mapping


def parseString(inString, silence=False):
    if sys.version_info.major == 2:
        from StringIO import StringIO
    else:
        from io import BytesIO as StringIO
    parser = None
    rootNode= parsexmlstring_(inString, parser)
    rootTag, rootClass = get_root_tag(rootNode)
    if rootClass is None:
        rootTag = 'emicss'
        rootClass = supermod.emicss
    rootObj = rootClass.factory()
    rootObj.build(rootNode)
    # Enable Python to collect the space used by the DOM.
    if not SaveElementTreeNode:
        rootNode = None
    if not silence:
        sys.stdout.write('<?xml version="1.0" ?>\n')
        rootObj.export(
            sys.stdout, 0, name_=rootTag,
            namespacedef_='')
    return rootObj


def parseLiteral(inFilename, silence=False):
    parser = None
    doc = parsexml_(inFilename, parser)
    rootNode = doc.getroot()
    rootTag, rootClass = get_root_tag(rootNode)
    if rootClass is None:
        rootTag = 'emicss'
        rootClass = supermod.emicss
    rootObj = rootClass.factory()
    rootObj.build(rootNode)
    # Enable Python to collect the space used by the DOM.
    if not SaveElementTreeNode:
        doc = None
        rootNode = None
    if not silence:
        sys.stdout.write('#from ??? import *\n\n')
        sys.stdout.write('import ??? as model_\n\n')
        sys.stdout.write('rootObj = model_.rootClass(\n')
        rootObj.exportLiteral(sys.stdout, 0, name_=rootTag)
        sys.stdout.write(')\n')
    return rootObj


USAGE_TEXT = """
Usage: python ???.py <infilename>
"""


def usage():
    print(USAGE_TEXT)
    sys.exit(1)


def main():
    args = sys.argv[1:]
    if len(args) != 1:
        usage()
    infilename = args[0]
    parse(infilename)


if __name__ == '__main__':
    #import pdb; pdb.set_trace()
    main()
