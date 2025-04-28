#!/usr/bin/env python
# -*- coding: utf-8 -*-

#
# Generated Thu Aug  1 15:07:22 2024 by generateDS.py version 2.38.6.
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

import sys
try:
    ModulenotfoundExp_ = ModuleNotFoundError
except NameError:
    ModulenotfoundExp_ = ImportError
from six.moves import zip_longest
import os
import re as re_
import base64
import datetime as datetime_
import decimal as decimal_
try:
    from lxml import etree as etree_
except ModulenotfoundExp_ :
    from xml.etree import ElementTree as etree_


Validate_simpletypes_ = True
SaveElementTreeNode = True
if sys.version_info.major == 2:
    BaseStrType_ = basestring
else:
    BaseStrType_ = str


def parsexml_(infile, parser=None, **kwargs):
    if parser is None:
        # Use the lxml ElementTree compatible parser so that, e.g.,
        #   we ignore comments.
        try:
            parser = etree_.ETCompatXMLParser()
        except AttributeError:
            # fallback to xml.etree
            parser = etree_.XMLParser()
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
# Namespace prefix definition table (and other attributes, too)
#
# The module generatedsnamespaces, if it is importable, must contain
# a dictionary named GeneratedsNamespaceDefs.  This Python dictionary
# should map element type names (strings) to XML schema namespace prefix
# definitions.  The export method for any class for which there is
# a namespace prefix definition, will export that definition in the
# XML representation of that element.  See the export method of
# any generated element type class for an example of the use of this
# table.
# A sample table is:
#
#     # File: generatedsnamespaces.py
#
#     GenerateDSNamespaceDefs = {
#         "ElementtypeA": "http://www.xxx.com/namespaceA",
#         "ElementtypeB": "http://www.xxx.com/namespaceB",
#     }
#
# Additionally, the generatedsnamespaces module can contain a python
# dictionary named GenerateDSNamespaceTypePrefixes that associates element
# types with the namespace prefixes that are to be added to the
# "xsi:type" attribute value.  See the exportAttributes method of
# any generated element type and the generation of "xsi:type" for an
# example of the use of this table.
# An example table:
#
#     # File: generatedsnamespaces.py
#
#     GenerateDSNamespaceTypePrefixes = {
#         "ElementtypeC": "aaa:",
#         "ElementtypeD": "bbb:",
#     }
#

try:
    from generatedsnamespaces import GenerateDSNamespaceDefs as GenerateDSNamespaceDefs_
except ModulenotfoundExp_ :
    GenerateDSNamespaceDefs_ = {}
try:
    from generatedsnamespaces import GenerateDSNamespaceTypePrefixes as GenerateDSNamespaceTypePrefixes_
except ModulenotfoundExp_ :
    GenerateDSNamespaceTypePrefixes_ = {}

#
# You can replace the following class definition by defining an
# importable module named "generatedscollector" containing a class
# named "GdsCollector".  See the default class definition below for
# clues about the possible content of that class.
#
try:
    from generatedscollector import GdsCollector as GdsCollector_
except ModulenotfoundExp_ :

    class GdsCollector_(object):

        def __init__(self, messages=None):
            if messages is None:
                self.messages = []
            else:
                self.messages = messages

        def add_message(self, msg):
            self.messages.append(msg)

        def get_messages(self):
            return self.messages

        def clear_messages(self):
            self.messages = []

        def print_messages(self):
            for msg in self.messages:
                print("Warning: {}".format(msg))

        def write_messages(self, outstream):
            for msg in self.messages:
                outstream.write("Warning: {}\n".format(msg))


#
# The super-class for enum types
#

try:
    from enum import Enum
except ModulenotfoundExp_ :
    Enum = object

#
# The root super-class for element type classes
#
# Calls to the methods in these classes are generated by generateDS.py.
# You can replace these methods by re-implementing the following class
#   in a module named generatedssuper.py.

try:
    from generatedssuper import GeneratedsSuper
except ModulenotfoundExp_ as exp:
    
    class GeneratedsSuper(object):
        __hash__ = object.__hash__
        tzoff_pattern = re_.compile(r'(\+|-)((0\d|1[0-3]):[0-5]\d|14:00)$')
        class _FixedOffsetTZ(datetime_.tzinfo):
            def __init__(self, offset, name):
                self.__offset = datetime_.timedelta(minutes=offset)
                self.__name = name
            def utcoffset(self, dt):
                return self.__offset
            def tzname(self, dt):
                return self.__name
            def dst(self, dt):
                return None
        def gds_format_string(self, input_data, input_name=''):
            return input_data
        def gds_parse_string(self, input_data, node=None, input_name=''):
            return input_data
        def gds_validate_string(self, input_data, node=None, input_name=''):
            if not input_data:
                return ''
            else:
                return input_data
        def gds_format_base64(self, input_data, input_name=''):
            return base64.b64encode(input_data)
        def gds_validate_base64(self, input_data, node=None, input_name=''):
            return input_data
        def gds_format_integer(self, input_data, input_name=''):
            return '%d' % input_data
        def gds_parse_integer(self, input_data, node=None, input_name=''):
            try:
                ival = int(input_data)
            except (TypeError, ValueError) as exp:
                raise_parse_error(node, 'Requires integer value: %s' % exp)
            return ival
        def gds_validate_integer(self, input_data, node=None, input_name=''):
            try:
                value = int(input_data)
            except (TypeError, ValueError):
                raise_parse_error(node, 'Requires integer value')
            return value
        def gds_format_integer_list(self, input_data, input_name=''):
            if len(input_data) > 0 and not isinstance(input_data[0], BaseStrType_):
                input_data = [str(s) for s in input_data]
            return '%s' % ' '.join(input_data)
        def gds_validate_integer_list(
                self, input_data, node=None, input_name=''):
            values = input_data.split()
            for value in values:
                try:
                    int(value)
                except (TypeError, ValueError):
                    raise_parse_error(node, 'Requires sequence of integer values')
            return values
        def gds_format_float(self, input_data, input_name=''):
            return ('%.15f' % input_data).rstrip('0')
        def gds_parse_float(self, input_data, node=None, input_name=''):
            try:
                fval_ = float(input_data)
            except (TypeError, ValueError) as exp:
                raise_parse_error(node, 'Requires float or double value: %s' % exp)
            return fval_
        def gds_validate_float(self, input_data, node=None, input_name=''):
            try:
                value = float(input_data)
            except (TypeError, ValueError):
                raise_parse_error(node, 'Requires float value')
            return value
        def gds_format_float_list(self, input_data, input_name=''):
            if len(input_data) > 0 and not isinstance(input_data[0], BaseStrType_):
                input_data = [str(s) for s in input_data]
            return '%s' % ' '.join(input_data)
        def gds_validate_float_list(
                self, input_data, node=None, input_name=''):
            values = input_data.split()
            for value in values:
                try:
                    float(value)
                except (TypeError, ValueError):
                    raise_parse_error(node, 'Requires sequence of float values')
            return values
        def gds_format_decimal(self, input_data, input_name=''):
            return_value = '%s' % input_data
            if '.' in return_value:
                return_value = return_value.rstrip('0')
                if return_value.endswith('.'):
                    return_value = return_value.rstrip('.')
            return return_value
        def gds_parse_decimal(self, input_data, node=None, input_name=''):
            try:
                decimal_value = decimal_.Decimal(input_data)
            except (TypeError, ValueError):
                raise_parse_error(node, 'Requires decimal value')
            return decimal_value
        def gds_validate_decimal(self, input_data, node=None, input_name=''):
            try:
                value = decimal_.Decimal(input_data)
            except (TypeError, ValueError):
                raise_parse_error(node, 'Requires decimal value')
            return value
        def gds_format_decimal_list(self, input_data, input_name=''):
            if len(input_data) > 0 and not isinstance(input_data[0], BaseStrType_):
                input_data = [str(s) for s in input_data]
            return ' '.join([self.gds_format_decimal(item) for item in input_data])
        def gds_validate_decimal_list(
                self, input_data, node=None, input_name=''):
            values = input_data.split()
            for value in values:
                try:
                    decimal_.Decimal(value)
                except (TypeError, ValueError):
                    raise_parse_error(node, 'Requires sequence of decimal values')
            return values
        def gds_format_double(self, input_data, input_name=''):
            return '%s' % input_data
        def gds_parse_double(self, input_data, node=None, input_name=''):
            try:
                fval_ = float(input_data)
            except (TypeError, ValueError) as exp:
                raise_parse_error(node, 'Requires double or float value: %s' % exp)
            return fval_
        def gds_validate_double(self, input_data, node=None, input_name=''):
            try:
                value = float(input_data)
            except (TypeError, ValueError):
                raise_parse_error(node, 'Requires double or float value')
            return value
        def gds_format_double_list(self, input_data, input_name=''):
            if len(input_data) > 0 and not isinstance(input_data[0], BaseStrType_):
                input_data = [str(s) for s in input_data]
            return '%s' % ' '.join(input_data)
        def gds_validate_double_list(
                self, input_data, node=None, input_name=''):
            values = input_data.split()
            for value in values:
                try:
                    float(value)
                except (TypeError, ValueError):
                    raise_parse_error(
                        node, 'Requires sequence of double or float values')
            return values
        def gds_format_boolean(self, input_data, input_name=''):
            return ('%s' % input_data).lower()
        def gds_parse_boolean(self, input_data, node=None, input_name=''):
            if input_data in ('true', '1'):
                bval = True
            elif input_data in ('false', '0'):
                bval = False
            else:
                raise_parse_error(node, 'Requires boolean value')
            return bval
        def gds_validate_boolean(self, input_data, node=None, input_name=''):
            if input_data not in (True, 1, False, 0, ):
                raise_parse_error(
                    node,
                    'Requires boolean value '
                    '(one of True, 1, False, 0)')
            return input_data
        def gds_format_boolean_list(self, input_data, input_name=''):
            if len(input_data) > 0 and not isinstance(input_data[0], BaseStrType_):
                input_data = [str(s) for s in input_data]
            return '%s' % ' '.join(input_data)
        def gds_validate_boolean_list(
                self, input_data, node=None, input_name=''):
            values = input_data.split()
            for value in values:
                value = self.gds_parse_boolean(value, node, input_name)
                if value not in (True, 1, False, 0, ):
                    raise_parse_error(
                        node,
                        'Requires sequence of boolean values '
                        '(one of True, 1, False, 0)')
            return values
        def gds_validate_datetime(self, input_data, node=None, input_name=''):
            return input_data
        def gds_format_datetime(self, input_data, input_name=''):
            if input_data.microsecond == 0:
                _svalue = '%04d-%02d-%02dT%02d:%02d:%02d' % (
                    input_data.year,
                    input_data.month,
                    input_data.day,
                    input_data.hour,
                    input_data.minute,
                    input_data.second,
                )
            else:
                _svalue = '%04d-%02d-%02dT%02d:%02d:%02d.%s' % (
                    input_data.year,
                    input_data.month,
                    input_data.day,
                    input_data.hour,
                    input_data.minute,
                    input_data.second,
                    ('%f' % (float(input_data.microsecond) / 1000000))[2:],
                )
            if input_data.tzinfo is not None:
                tzoff = input_data.tzinfo.utcoffset(input_data)
                if tzoff is not None:
                    total_seconds = tzoff.seconds + (86400 * tzoff.days)
                    if total_seconds == 0:
                        _svalue += 'Z'
                    else:
                        if total_seconds < 0:
                            _svalue += '-'
                            total_seconds *= -1
                        else:
                            _svalue += '+'
                        hours = total_seconds // 3600
                        minutes = (total_seconds - (hours * 3600)) // 60
                        _svalue += '{0:02d}:{1:02d}'.format(hours, minutes)
            return _svalue
        @classmethod
        def gds_parse_datetime(cls, input_data):
            tz = None
            if input_data[-1] == 'Z':
                tz = GeneratedsSuper._FixedOffsetTZ(0, 'UTC')
                input_data = input_data[:-1]
            else:
                results = GeneratedsSuper.tzoff_pattern.search(input_data)
                if results is not None:
                    tzoff_parts = results.group(2).split(':')
                    tzoff = int(tzoff_parts[0]) * 60 + int(tzoff_parts[1])
                    if results.group(1) == '-':
                        tzoff *= -1
                    tz = GeneratedsSuper._FixedOffsetTZ(
                        tzoff, results.group(0))
                    input_data = input_data[:-6]
            time_parts = input_data.split('.')
            if len(time_parts) > 1:
                micro_seconds = int(float('0.' + time_parts[1]) * 1000000)
                input_data = '%s.%s' % (
                    time_parts[0], "{}".format(micro_seconds).rjust(6, "0"), )
                dt = datetime_.datetime.strptime(
                    input_data, '%Y-%m-%dT%H:%M:%S.%f')
            else:
                dt = datetime_.datetime.strptime(
                    input_data, '%Y-%m-%dT%H:%M:%S')
            dt = dt.replace(tzinfo=tz)
            return dt
        def gds_validate_date(self, input_data, node=None, input_name=''):
            return input_data
        def gds_format_date(self, input_data, input_name=''):
            _svalue = '%04d-%02d-%02d' % (
                input_data.year,
                input_data.month,
                input_data.day,
            )
            try:
                if input_data.tzinfo is not None:
                    tzoff = input_data.tzinfo.utcoffset(input_data)
                    if tzoff is not None:
                        total_seconds = tzoff.seconds + (86400 * tzoff.days)
                        if total_seconds == 0:
                            _svalue += 'Z'
                        else:
                            if total_seconds < 0:
                                _svalue += '-'
                                total_seconds *= -1
                            else:
                                _svalue += '+'
                            hours = total_seconds // 3600
                            minutes = (total_seconds - (hours * 3600)) // 60
                            _svalue += '{0:02d}:{1:02d}'.format(
                                hours, minutes)
            except AttributeError:
                pass
            return _svalue
        @classmethod
        def gds_parse_date(cls, input_data):
            tz = None
            if input_data[-1] == 'Z':
                tz = GeneratedsSuper._FixedOffsetTZ(0, 'UTC')
                input_data = input_data[:-1]
            else:
                results = GeneratedsSuper.tzoff_pattern.search(input_data)
                if results is not None:
                    tzoff_parts = results.group(2).split(':')
                    tzoff = int(tzoff_parts[0]) * 60 + int(tzoff_parts[1])
                    if results.group(1) == '-':
                        tzoff *= -1
                    tz = GeneratedsSuper._FixedOffsetTZ(
                        tzoff, results.group(0))
                    input_data = input_data[:-6]
            dt = datetime_.datetime.strptime(input_data, '%Y-%m-%d')
            dt = dt.replace(tzinfo=tz)
            return dt.date()
        def gds_validate_time(self, input_data, node=None, input_name=''):
            return input_data
        def gds_format_time(self, input_data, input_name=''):
            if input_data.microsecond == 0:
                _svalue = '%02d:%02d:%02d' % (
                    input_data.hour,
                    input_data.minute,
                    input_data.second,
                )
            else:
                _svalue = '%02d:%02d:%02d.%s' % (
                    input_data.hour,
                    input_data.minute,
                    input_data.second,
                    ('%f' % (float(input_data.microsecond) / 1000000))[2:],
                )
            if input_data.tzinfo is not None:
                tzoff = input_data.tzinfo.utcoffset(input_data)
                if tzoff is not None:
                    total_seconds = tzoff.seconds + (86400 * tzoff.days)
                    if total_seconds == 0:
                        _svalue += 'Z'
                    else:
                        if total_seconds < 0:
                            _svalue += '-'
                            total_seconds *= -1
                        else:
                            _svalue += '+'
                        hours = total_seconds // 3600
                        minutes = (total_seconds - (hours * 3600)) // 60
                        _svalue += '{0:02d}:{1:02d}'.format(hours, minutes)
            return _svalue
        def gds_validate_simple_patterns(self, patterns, target):
            # pat is a list of lists of strings/patterns.
            # The target value must match at least one of the patterns
            # in order for the test to succeed.
            found1 = True
            for patterns1 in patterns:
                found2 = False
                for patterns2 in patterns1:
                    mo = re_.search(patterns2, target)
                    if mo is not None and len(mo.group(0)) == len(target):
                        found2 = True
                        break
                if not found2:
                    found1 = False
                    break
            return found1
        @classmethod
        def gds_parse_time(cls, input_data):
            tz = None
            if input_data[-1] == 'Z':
                tz = GeneratedsSuper._FixedOffsetTZ(0, 'UTC')
                input_data = input_data[:-1]
            else:
                results = GeneratedsSuper.tzoff_pattern.search(input_data)
                if results is not None:
                    tzoff_parts = results.group(2).split(':')
                    tzoff = int(tzoff_parts[0]) * 60 + int(tzoff_parts[1])
                    if results.group(1) == '-':
                        tzoff *= -1
                    tz = GeneratedsSuper._FixedOffsetTZ(
                        tzoff, results.group(0))
                    input_data = input_data[:-6]
            if len(input_data.split('.')) > 1:
                dt = datetime_.datetime.strptime(input_data, '%H:%M:%S.%f')
            else:
                dt = datetime_.datetime.strptime(input_data, '%H:%M:%S')
            dt = dt.replace(tzinfo=tz)
            return dt.time()
        def gds_check_cardinality_(
                self, value, input_name,
                min_occurs=0, max_occurs=1, required=None):
            if value is None:
                length = 0
            elif isinstance(value, list):
                length = len(value)
            else:
                length = 1
            if required is not None :
                if required and length < 1:
                    self.gds_collector_.add_message(
                        "Required value {}{} is missing".format(
                            input_name, self.gds_get_node_lineno_()))
            if length < min_occurs:
                self.gds_collector_.add_message(
                    "Number of values for {}{} is below "
                    "the minimum allowed, "
                    "expected at least {}, found {}".format(
                        input_name, self.gds_get_node_lineno_(),
                        min_occurs, length))
            elif length > max_occurs:
                self.gds_collector_.add_message(
                    "Number of values for {}{} is above "
                    "the maximum allowed, "
                    "expected at most {}, found {}".format(
                        input_name, self.gds_get_node_lineno_(),
                        max_occurs, length))
        def gds_validate_builtin_ST_(
                self, validator, value, input_name,
                min_occurs=None, max_occurs=None, required=None):
            if value is not None:
                try:
                    validator(value, input_name=input_name)
                except GDSParseError as parse_error:
                    self.gds_collector_.add_message(str(parse_error))
        def gds_validate_defined_ST_(
                self, validator, value, input_name,
                min_occurs=None, max_occurs=None, required=None):
            if value is not None:
                try:
                    validator(value)
                except GDSParseError as parse_error:
                    self.gds_collector_.add_message(str(parse_error))
        def gds_str_lower(self, instring):
            return instring.lower()
        def get_path_(self, node):
            path_list = []
            self.get_path_list_(node, path_list)
            path_list.reverse()
            path = '/'.join(path_list)
            return path
        Tag_strip_pattern_ = re_.compile(r'\{.*\}')
        def get_path_list_(self, node, path_list):
            if node is None:
                return
            tag = GeneratedsSuper.Tag_strip_pattern_.sub('', node.tag)
            if tag:
                path_list.append(tag)
            self.get_path_list_(node.getparent(), path_list)
        def get_class_obj_(self, node, default_class=None):
            class_obj1 = default_class
            if 'xsi' in node.nsmap:
                classname = node.get('{%s}type' % node.nsmap['xsi'])
                if classname is not None:
                    names = classname.split(':')
                    if len(names) == 2:
                        classname = names[1]
                    class_obj2 = globals().get(classname)
                    if class_obj2 is not None:
                        class_obj1 = class_obj2
            return class_obj1
        def gds_build_any(self, node, type_name=None):
            # provide default value in case option --disable-xml is used.
            content = ""
            content = etree_.tostring(node, encoding="unicode")
            return content
        @classmethod
        def gds_reverse_node_mapping(cls, mapping):
            return dict(((v, k) for k, v in mapping.items()))
        @staticmethod
        def gds_encode(instring):
            if sys.version_info.major == 2:
                if ExternalEncoding:
                    encoding = ExternalEncoding
                else:
                    encoding = 'utf-8'
                return instring.encode(encoding)
            else:
                return instring
        @staticmethod
        def convert_unicode(instring):
            if isinstance(instring, str):
                result = quote_xml(instring)
            elif sys.version_info.major == 2 and isinstance(instring, unicode):
                result = quote_xml(instring).encode('utf8')
            else:
                result = GeneratedsSuper.gds_encode(str(instring))
            return result
        def __eq__(self, other):
            def excl_select_objs_(obj):
                return (obj[0] != 'parent_object_' and
                        obj[0] != 'gds_collector_')
            if type(self) != type(other):
                return False
            return all(x == y for x, y in zip_longest(
                filter(excl_select_objs_, self.__dict__.items()),
                filter(excl_select_objs_, other.__dict__.items())))
        def __ne__(self, other):
            return not self.__eq__(other)
        # Django ETL transform hooks.
        def gds_djo_etl_transform(self):
            pass
        def gds_djo_etl_transform_db_obj(self, dbobj):
            pass
        # SQLAlchemy ETL transform hooks.
        def gds_sqa_etl_transform(self):
            return 0, None
        def gds_sqa_etl_transform_db_obj(self, dbobj):
            pass
        def gds_get_node_lineno_(self):
            if (hasattr(self, "gds_elementtree_node_") and
                    self.gds_elementtree_node_ is not None):
                return ' near line {}'.format(
                    self.gds_elementtree_node_.sourceline)
            else:
                return ""
    
    
    def getSubclassFromModule_(module, class_):
        '''Get the subclass of a class from a specific module.'''
        name = class_.__name__ + 'Sub'
        if hasattr(module, name):
            return getattr(module, name)
        else:
            return None


#
# If you have installed IPython you can uncomment and use the following.
# IPython is available from http://ipython.scipy.org/.
#

## from IPython.Shell import IPShellEmbed
## args = ''
## ipshell = IPShellEmbed(args,
##     banner = 'Dropping into IPython',
##     exit_msg = 'Leaving Interpreter, back to program.')

# Then use the following line where and when you want to drop into the
# IPython shell:
#    ipshell('<some message> -- Entering ipshell.\nHit Ctrl-D to exit')

#
# Globals
#

ExternalEncoding = ''
# Set this to false in order to deactivate during export, the use of
# name space prefixes captured from the input document.
UseCapturedNS_ = True
CapturedNsmap_ = {}
Tag_pattern_ = re_.compile(r'({.*})?(.*)')
String_cleanup_pat_ = re_.compile(r"[\n\r\s]+")
Namespace_extract_pat_ = re_.compile(r'{(.*)}(.*)')
CDATA_pattern_ = re_.compile(r"<!\[CDATA\[.*?\]\]>", re_.DOTALL)

# Change this to redirect the generated superclass module to use a
# specific subclass module.
CurrentSubclassModule_ = None

#
# Support/utility functions.
#


def showIndent(outfile, level, pretty_print=True):
    if pretty_print:
        for idx in range(level):
            outfile.write('    ')


def quote_xml(inStr):
    "Escape markup chars, but do not modify CDATA sections."
    if not inStr:
        return ''
    s1 = (isinstance(inStr, BaseStrType_) and inStr or '%s' % inStr)
    s2 = ''
    pos = 0
    matchobjects = CDATA_pattern_.finditer(s1)
    for mo in matchobjects:
        s3 = s1[pos:mo.start()]
        s2 += quote_xml_aux(s3)
        s2 += s1[mo.start():mo.end()]
        pos = mo.end()
    s3 = s1[pos:]
    s2 += quote_xml_aux(s3)
    return s2


def quote_xml_aux(inStr):
    s1 = inStr.replace('&', '&amp;')
    s1 = s1.replace('<', '&lt;')
    s1 = s1.replace('>', '&gt;')
    return s1


def quote_attrib(inStr):
    s1 = (isinstance(inStr, BaseStrType_) and inStr or '%s' % inStr)
    s1 = s1.replace('&', '&amp;')
    s1 = s1.replace('<', '&lt;')
    s1 = s1.replace('>', '&gt;')
    if '"' in s1:
        if "'" in s1:
            s1 = '"%s"' % s1.replace('"', "&quot;")
        else:
            s1 = "'%s'" % s1
    else:
        s1 = '"%s"' % s1
    return s1


def quote_python(inStr):
    s1 = inStr
    if s1.find("'") == -1:
        if s1.find('\n') == -1:
            return "'%s'" % s1
        else:
            return "'''%s'''" % s1
    else:
        if s1.find('"') != -1:
            s1 = s1.replace('"', '\\"')
        if s1.find('\n') == -1:
            return '"%s"' % s1
        else:
            return '"""%s"""' % s1


def get_all_text_(node):
    if node.text is not None:
        text = node.text
    else:
        text = ''
    for child in node:
        if child.tail is not None:
            text += child.tail
    return text


def find_attr_value_(attr_name, node):
    attrs = node.attrib
    attr_parts = attr_name.split(':')
    value = None
    if len(attr_parts) == 1:
        value = attrs.get(attr_name)
    elif len(attr_parts) == 2:
        prefix, name = attr_parts
        if prefix == 'xml':
            namespace = 'http://www.w3.org/XML/1998/namespace'
        else:
            namespace = node.nsmap.get(prefix)
        if namespace is not None:
            value = attrs.get('{%s}%s' % (namespace, name, ))
    return value


def encode_str_2_3(instr):
    return instr


class GDSParseError(Exception):
    pass


def raise_parse_error(node, msg):
    if node is not None:
        msg = '%s (element %s/line %d)' % (msg, node.tag, node.sourceline, )
    raise GDSParseError(msg)


class MixedContainer:
    # Constants for category:
    CategoryNone = 0
    CategoryText = 1
    CategorySimple = 2
    CategoryComplex = 3
    # Constants for content_type:
    TypeNone = 0
    TypeText = 1
    TypeString = 2
    TypeInteger = 3
    TypeFloat = 4
    TypeDecimal = 5
    TypeDouble = 6
    TypeBoolean = 7
    TypeBase64 = 8
    def __init__(self, category, content_type, name, value):
        self.category = category
        self.content_type = content_type
        self.name = name
        self.value = value
    def getCategory(self):
        return self.category
    def getContenttype(self, content_type):
        return self.content_type
    def getValue(self):
        return self.value
    def getName(self):
        return self.name
    def export(self, outfile, level, name, namespace,
               pretty_print=True):
        if self.category == MixedContainer.CategoryText:
            # Prevent exporting empty content as empty lines.
            if self.value.strip():
                outfile.write(self.value)
        elif self.category == MixedContainer.CategorySimple:
            self.exportSimple(outfile, level, name)
        else:    # category == MixedContainer.CategoryComplex
            self.value.export(
                outfile, level, namespace, name_=name,
                pretty_print=pretty_print)
    def exportSimple(self, outfile, level, name):
        if self.content_type == MixedContainer.TypeString:
            outfile.write('<%s>%s</%s>' % (
                self.name, self.value, self.name))
        elif self.content_type == MixedContainer.TypeInteger or \
                self.content_type == MixedContainer.TypeBoolean:
            outfile.write('<%s>%d</%s>' % (
                self.name, self.value, self.name))
        elif self.content_type == MixedContainer.TypeFloat or \
                self.content_type == MixedContainer.TypeDecimal:
            outfile.write('<%s>%f</%s>' % (
                self.name, self.value, self.name))
        elif self.content_type == MixedContainer.TypeDouble:
            outfile.write('<%s>%g</%s>' % (
                self.name, self.value, self.name))
        elif self.content_type == MixedContainer.TypeBase64:
            outfile.write('<%s>%s</%s>' % (
                self.name,
                base64.b64encode(self.value),
                self.name))
    def to_etree(self, element, mapping_=None, nsmap_=None):
        if self.category == MixedContainer.CategoryText:
            # Prevent exporting empty content as empty lines.
            if self.value.strip():
                if len(element) > 0:
                    if element[-1].tail is None:
                        element[-1].tail = self.value
                    else:
                        element[-1].tail += self.value
                else:
                    if element.text is None:
                        element.text = self.value
                    else:
                        element.text += self.value
        elif self.category == MixedContainer.CategorySimple:
            subelement = etree_.SubElement(
                element, '%s' % self.name)
            subelement.text = self.to_etree_simple()
        else:    # category == MixedContainer.CategoryComplex
            self.value.to_etree(element)
    def to_etree_simple(self, mapping_=None, nsmap_=None):
        if self.content_type == MixedContainer.TypeString:
            text = self.value
        elif (self.content_type == MixedContainer.TypeInteger or
                self.content_type == MixedContainer.TypeBoolean):
            text = '%d' % self.value
        elif (self.content_type == MixedContainer.TypeFloat or
                self.content_type == MixedContainer.TypeDecimal):
            text = '%f' % self.value
        elif self.content_type == MixedContainer.TypeDouble:
            text = '%g' % self.value
        elif self.content_type == MixedContainer.TypeBase64:
            text = '%s' % base64.b64encode(self.value)
        return text
    def exportLiteral(self, outfile, level, name):
        if self.category == MixedContainer.CategoryText:
            showIndent(outfile, level)
            outfile.write(
                'model_.MixedContainer(%d, %d, "%s", "%s"),\n' % (
                    self.category, self.content_type,
                    self.name, self.value))
        elif self.category == MixedContainer.CategorySimple:
            showIndent(outfile, level)
            outfile.write(
                'model_.MixedContainer(%d, %d, "%s", "%s"),\n' % (
                    self.category, self.content_type,
                    self.name, self.value))
        else:    # category == MixedContainer.CategoryComplex
            showIndent(outfile, level)
            outfile.write(
                'model_.MixedContainer(%d, %d, "%s",\n' % (
                    self.category, self.content_type, self.name,))
            self.value.exportLiteral(outfile, level + 1)
            showIndent(outfile, level)
            outfile.write(')\n')


class MemberSpec_(object):
    def __init__(self, name='', data_type='', container=0,
            optional=0, child_attrs=None, choice=None):
        self.name = name
        self.data_type = data_type
        self.container = container
        self.child_attrs = child_attrs
        self.choice = choice
        self.optional = optional
    def set_name(self, name): self.name = name
    def get_name(self): return self.name
    def set_data_type(self, data_type): self.data_type = data_type
    def get_data_type_chain(self): return self.data_type
    def get_data_type(self):
        if isinstance(self.data_type, list):
            if len(self.data_type) > 0:
                return self.data_type[-1]
            else:
                return 'xs:string'
        else:
            return self.data_type
    def set_container(self, container): self.container = container
    def get_container(self): return self.container
    def set_child_attrs(self, child_attrs): self.child_attrs = child_attrs
    def get_child_attrs(self): return self.child_attrs
    def set_choice(self, choice): self.choice = choice
    def get_choice(self): return self.choice
    def set_optional(self, optional): self.optional = optional
    def get_optional(self): return self.optional


def _cast(typ, value):
    if typ is None or value is None:
        return value
    return typ(value)

#
# Data representation classes.
#


class provenance_type(str, Enum):
    """Annotations done from the respective database. Could be more than one
    database"""
    EMDB='EMDB'
    UNI_PROT='UniProt'
    UNI_PROT_KB='UniProtKB'
    PD_BE='PDBe'
    PD_BEKB='PDBe-KB'
    ALPHA_FOLDDB='AlphaFold DB'
    EMPIAR='EMPIAR'
    EUROPE_PMC='EuropePMC'
    COMPLEX_PORTAL='Complex Portal'
    CH_EMBL='ChEMBL'
    CH_EBI='ChEBI'
    DRUG_BANK='DrugBank'
    PD_BECCD='PDBe-CCD'
    PUB_MED='PubMed'
    PUB_MED_CENTRAL='PubMed Central'
    ISSN='ISSN'
    DOI='DOI'
    GO='GO'
    INTER_PRO='InterPro'
    PFAM='Pfam'
    CATH='CATH'
    SCOP='SCOP'
    SCOP_2='SCOP2'
    SCOP_2_B='SCOP2B'
    RFAM='Rfam'


class sample_kind(str, Enum):
    COMPLEX='complex'
    PROTEIN='protein'
    LIGAND='ligand'
    RNA='rna'


class emicss(GeneratedsSuper):
    """EMDB entry idEMICSS schema version"""
    __hash__ = GeneratedsSuper.__hash__
    subclass = None
    superclass = None
    def __init__(self, emdb_id=None, version='0.9.5', schema_location=None, dbs=None, entry_ref_dbs=None, primary_citation=None, weights=None, sample=None, gds_collector_=None, **kwargs_):
        self.gds_collector_ = gds_collector_
        self.gds_elementtree_node_ = None
        self.original_tagname_ = None
        self.parent_object_ = kwargs_.get('parent_object_')
        self.ns_prefix_ = None
        self.emdb_id = _cast(None, emdb_id)
        self.emdb_id_nsprefix_ = None
        self.version = _cast(None, version)
        self.version_nsprefix_ = None
        self.schema_location = _cast(None, schema_location)
        self.schema_location_nsprefix_ = None
        self.dbs = dbs
        self.dbs_nsprefix_ = None
        self.entry_ref_dbs = entry_ref_dbs
        self.entry_ref_dbs_nsprefix_ = None
        self.primary_citation = primary_citation
        self.primary_citation_nsprefix_ = None
        self.weights = weights
        self.weights_nsprefix_ = None
        self.sample = sample
        self.sample_nsprefix_ = None
    def factory(*args_, **kwargs_):
        if CurrentSubclassModule_ is not None:
            subclass = getSubclassFromModule_(
                CurrentSubclassModule_, emicss)
            if subclass is not None:
                return subclass(*args_, **kwargs_)
        if emicss.subclass:
            return emicss.subclass(*args_, **kwargs_)
        else:
            return emicss(*args_, **kwargs_)
    factory = staticmethod(factory)
    def get_ns_prefix_(self):
        return self.ns_prefix_
    def set_ns_prefix_(self, ns_prefix):
        self.ns_prefix_ = ns_prefix
    def get_dbs(self):
        return self.dbs
    def set_dbs(self, dbs):
        self.dbs = dbs
    def get_entry_ref_dbs(self):
        return self.entry_ref_dbs
    def set_entry_ref_dbs(self, entry_ref_dbs):
        self.entry_ref_dbs = entry_ref_dbs
    def get_primary_citation(self):
        return self.primary_citation
    def set_primary_citation(self, primary_citation):
        self.primary_citation = primary_citation
    def get_weights(self):
        return self.weights
    def set_weights(self, weights):
        self.weights = weights
    def get_sample(self):
        return self.sample
    def set_sample(self, sample):
        self.sample = sample
    def get_emdb_id(self):
        return self.emdb_id
    def set_emdb_id(self, emdb_id):
        self.emdb_id = emdb_id
    def get_version(self):
        return self.version
    def set_version(self, version):
        self.version = version
    def get_schema_location(self):
        return self.schema_location
    def set_schema_location(self, schema_location):
        self.schema_location = schema_location
    def validate_emdb_id_type(self, value):
        # Validate type emdb_id_type, a restriction on xsd:token.
        if value is not None and Validate_simpletypes_ and self.gds_collector_ is not None:
            if not isinstance(value, str):
                lineno = self.gds_get_node_lineno_()
                self.gds_collector_.add_message('Value "%(value)s"%(lineno)s is not of the correct base simple type (str)' % {"value": value, "lineno": lineno, })
                return False
            if not self.gds_validate_simple_patterns(
                    self.validate_emdb_id_type_patterns_, value):
                self.gds_collector_.add_message('Value "%s" does not match xsd pattern restrictions: %s' % (encode_str_2_3(value), self.validate_emdb_id_type_patterns_, ))
    validate_emdb_id_type_patterns_ = [['^(EMD-\\d{4,})$']]
    def hasContent_(self):
        if (
            self.dbs is not None or
            self.entry_ref_dbs is not None or
            self.primary_citation is not None or
            self.weights is not None or
            self.sample is not None
        ):
            return True
        else:
            return False
    def export(self, outfile, level, namespaceprefix_='', namespacedef_='', name_='emicss', pretty_print=True):
        imported_ns_def_ = GenerateDSNamespaceDefs_.get('emicss')
        if imported_ns_def_ is not None:
            namespacedef_ = imported_ns_def_
        if pretty_print:
            eol_ = '\n'
        else:
            eol_ = ''
        if self.original_tagname_ is not None and name_ == 'emicss':
            name_ = self.original_tagname_
        if UseCapturedNS_ and self.ns_prefix_:
            namespaceprefix_ = self.ns_prefix_ + ':'
        showIndent(outfile, level, pretty_print)
        outfile.write('<%s%s%s' % (namespaceprefix_, name_, namespacedef_ and ' ' + namespacedef_ or '', ))
        already_processed = set()
        self.exportAttributes(outfile, level, already_processed, namespaceprefix_, name_='emicss')
        if self.hasContent_():
            outfile.write('>%s' % (eol_, ))
            self.exportChildren(outfile, level + 1, namespaceprefix_, namespacedef_, name_='emicss', pretty_print=pretty_print)
            showIndent(outfile, level, pretty_print)
            outfile.write('</%s%s>%s' % (namespaceprefix_, name_, eol_))
        else:
            outfile.write('/>%s' % (eol_, ))
    def exportAttributes(self, outfile, level, already_processed, namespaceprefix_='', name_='emicss'):
        if self.emdb_id is not None and 'emdb_id' not in already_processed:
            already_processed.add('emdb_id')
            outfile.write(' emdb_id=%s' % (self.gds_encode(self.gds_format_string(quote_attrib(self.emdb_id), input_name='emdb_id')), ))
        if self.version != "0.9.5" and 'version' not in already_processed:
            already_processed.add('version')
            outfile.write(' version=%s' % (self.gds_encode(self.gds_format_string(quote_attrib(self.version), input_name='version')), ))
        if self.schema_location is not None and 'schema_location' not in already_processed:
            already_processed.add('schema_location')
            outfile.write(' schema_location=%s' % (self.gds_encode(self.gds_format_string(quote_attrib(self.schema_location), input_name='schema_location')), ))
    def exportChildren(self, outfile, level, namespaceprefix_='', namespacedef_='', name_='emicss', fromsubclass_=False, pretty_print=True):
        if pretty_print:
            eol_ = '\n'
        else:
            eol_ = ''
        if self.dbs is not None:
            namespaceprefix_ = self.dbs_nsprefix_ + ':' if (UseCapturedNS_ and self.dbs_nsprefix_) else ''
            self.dbs.export(outfile, level, namespaceprefix_, namespacedef_='', name_='dbs', pretty_print=pretty_print)
        if self.entry_ref_dbs is not None:
            namespaceprefix_ = self.entry_ref_dbs_nsprefix_ + ':' if (UseCapturedNS_ and self.entry_ref_dbs_nsprefix_) else ''
            self.entry_ref_dbs.export(outfile, level, namespaceprefix_, namespacedef_='', name_='entry_ref_dbs', pretty_print=pretty_print)
        if self.primary_citation is not None:
            namespaceprefix_ = self.primary_citation_nsprefix_ + ':' if (UseCapturedNS_ and self.primary_citation_nsprefix_) else ''
            self.primary_citation.export(outfile, level, namespaceprefix_, namespacedef_='', name_='primary_citation', pretty_print=pretty_print)
        if self.weights is not None:
            namespaceprefix_ = self.weights_nsprefix_ + ':' if (UseCapturedNS_ and self.weights_nsprefix_) else ''
            self.weights.export(outfile, level, namespaceprefix_, namespacedef_='', name_='weights', pretty_print=pretty_print)
        if self.sample is not None:
            namespaceprefix_ = self.sample_nsprefix_ + ':' if (UseCapturedNS_ and self.sample_nsprefix_) else ''
            self.sample.export(outfile, level, namespaceprefix_, namespacedef_='', name_='sample', pretty_print=pretty_print)
    def build(self, node, gds_collector_=None):
        self.gds_collector_ = gds_collector_
        if SaveElementTreeNode:
            self.gds_elementtree_node_ = node
        already_processed = set()
        self.ns_prefix_ = node.prefix
        self.buildAttributes(node, node.attrib, already_processed)
        for child in node:
            nodeName_ = Tag_pattern_.match(child.tag).groups()[-1]
            self.buildChildren(child, node, nodeName_, gds_collector_=gds_collector_)
        return self
    def buildAttributes(self, node, attrs, already_processed):
        value = find_attr_value_('emdb_id', node)
        if value is not None and 'emdb_id' not in already_processed:
            already_processed.add('emdb_id')
            self.emdb_id = value
            self.emdb_id = ' '.join(self.emdb_id.split())
            self.validate_emdb_id_type(self.emdb_id)    # validate type emdb_id_type
        value = find_attr_value_('version', node)
        if value is not None and 'version' not in already_processed:
            already_processed.add('version')
            self.version = value
            self.version = ' '.join(self.version.split())
        value = find_attr_value_('schema_location', node)
        if value is not None and 'schema_location' not in already_processed:
            already_processed.add('schema_location')
            self.schema_location = value
    def buildChildren(self, child_, node, nodeName_, fromsubclass_=False, gds_collector_=None):
        if nodeName_ == 'dbs':
            obj_ = dbsType.factory(parent_object_=self)
            obj_.build(child_, gds_collector_=gds_collector_)
            self.dbs = obj_
            obj_.original_tagname_ = 'dbs'
        elif nodeName_ == 'entry_ref_dbs':
            obj_ = entry_ref_dbsType.factory(parent_object_=self)
            obj_.build(child_, gds_collector_=gds_collector_)
            self.entry_ref_dbs = obj_
            obj_.original_tagname_ = 'entry_ref_dbs'
        elif nodeName_ == 'primary_citation':
            obj_ = primary_citationType.factory(parent_object_=self)
            obj_.build(child_, gds_collector_=gds_collector_)
            self.primary_citation = obj_
            obj_.original_tagname_ = 'primary_citation'
        elif nodeName_ == 'weights':
            obj_ = weightsType.factory(parent_object_=self)
            obj_.build(child_, gds_collector_=gds_collector_)
            self.weights = obj_
            obj_.original_tagname_ = 'weights'
        elif nodeName_ == 'sample':
            obj_ = sampleType.factory(parent_object_=self)
            obj_.build(child_, gds_collector_=gds_collector_)
            self.sample = obj_
            obj_.original_tagname_ = 'sample'
# end class emicss


class cross_ref_db(GeneratedsSuper):
    """Name as in the cross reference databaseDatabase/resource name Unique ID
    provided by the databaseStarting residue number as in UniProtEnding
    residue number as in UniProtThree main aspects Gene Ontology is
    categorised onResource the mapping is fetched fromIf the complex in the
    sample mathces exactly to the complex in the Complex Protal database
    the score is 1.0. If anything between score 0.9 and 0.5, the complex
    matches only partially with the complex in the Complex Portal. Score
    below 0.5 is omitted."""
    __hash__ = GeneratedsSuper.__hash__
    subclass = None
    superclass = None
    def __init__(self, name=None, source=None, accession_id=None, uniprot_start=None, uniprot_end=None, type_=None, provenance=None, score=None, gds_collector_=None, **kwargs_):
        self.gds_collector_ = gds_collector_
        self.gds_elementtree_node_ = None
        self.original_tagname_ = None
        self.parent_object_ = kwargs_.get('parent_object_')
        self.ns_prefix_ = None
        self.name = _cast(None, name)
        self.name_nsprefix_ = None
        self.source = _cast(None, source)
        self.source_nsprefix_ = None
        self.accession_id = _cast(None, accession_id)
        self.accession_id_nsprefix_ = None
        self.uniprot_start = _cast(int, uniprot_start)
        self.uniprot_start_nsprefix_ = None
        self.uniprot_end = _cast(int, uniprot_end)
        self.uniprot_end_nsprefix_ = None
        self.type_ = _cast(None, type_)
        self.type__nsprefix_ = None
        self.provenance = _cast(None, provenance)
        self.provenance_nsprefix_ = None
        self.score = _cast(float, score)
        self.score_nsprefix_ = None
    def factory(*args_, **kwargs_):
        if CurrentSubclassModule_ is not None:
            subclass = getSubclassFromModule_(
                CurrentSubclassModule_, cross_ref_db)
            if subclass is not None:
                return subclass(*args_, **kwargs_)
        if cross_ref_db.subclass:
            return cross_ref_db.subclass(*args_, **kwargs_)
        else:
            return cross_ref_db(*args_, **kwargs_)
    factory = staticmethod(factory)
    def get_ns_prefix_(self):
        return self.ns_prefix_
    def set_ns_prefix_(self, ns_prefix):
        self.ns_prefix_ = ns_prefix
    def get_name(self):
        return self.name
    def set_name(self, name):
        self.name = name
    def get_source(self):
        return self.source
    def set_source(self, source):
        self.source = source
    def get_accession_id(self):
        return self.accession_id
    def set_accession_id(self, accession_id):
        self.accession_id = accession_id
    def get_uniprot_start(self):
        return self.uniprot_start
    def set_uniprot_start(self, uniprot_start):
        self.uniprot_start = uniprot_start
    def get_uniprot_end(self):
        return self.uniprot_end
    def set_uniprot_end(self, uniprot_end):
        self.uniprot_end = uniprot_end
    def get_type(self):
        return self.type_
    def set_type(self, type_):
        self.type_ = type_
    def get_provenance(self):
        return self.provenance
    def set_provenance(self, provenance):
        self.provenance = provenance
    def get_score(self):
        return self.score
    def set_score(self, score):
        self.score = score
    def validate_provenance_type(self, value):
        # Validate type provenance_type, a restriction on xsd:token.
        if value is not None and Validate_simpletypes_ and self.gds_collector_ is not None:
            if not isinstance(value, str):
                lineno = self.gds_get_node_lineno_()
                self.gds_collector_.add_message('Value "%(value)s"%(lineno)s is not of the correct base simple type (str)' % {"value": value, "lineno": lineno, })
                return False
            value = value
            enumerations = ['EMDB', 'UniProt', 'UniProtKB', 'PDBe', 'PDBe-KB', 'AlphaFold DB', 'EMPIAR', 'EuropePMC', 'Complex Portal', 'ChEMBL', 'ChEBI', 'DrugBank', 'PDBe-CCD', 'PubMed', 'PubMed Central', 'ISSN', 'DOI', 'GO', 'InterPro', 'Pfam', 'CATH', 'SCOP', 'SCOP2', 'SCOP2B', 'Rfam']
            if value not in enumerations:
                lineno = self.gds_get_node_lineno_()
                self.gds_collector_.add_message('Value "%(value)s"%(lineno)s does not match xsd enumeration restriction on provenance_type' % {"value" : encode_str_2_3(value), "lineno": lineno} )
                result = False
    def hasContent_(self):
        if (

        ):
            return True
        else:
            return False
    def export(self, outfile, level, namespaceprefix_='', namespacedef_='', name_='cross_ref_db', pretty_print=True):
        imported_ns_def_ = GenerateDSNamespaceDefs_.get('cross_ref_db')
        if imported_ns_def_ is not None:
            namespacedef_ = imported_ns_def_
        if pretty_print:
            eol_ = '\n'
        else:
            eol_ = ''
        if self.original_tagname_ is not None and name_ == 'cross_ref_db':
            name_ = self.original_tagname_
        if UseCapturedNS_ and self.ns_prefix_:
            namespaceprefix_ = self.ns_prefix_ + ':'
        showIndent(outfile, level, pretty_print)
        outfile.write('<%s%s%s' % (namespaceprefix_, name_, namespacedef_ and ' ' + namespacedef_ or '', ))
        already_processed = set()
        self.exportAttributes(outfile, level, already_processed, namespaceprefix_, name_='cross_ref_db')
        if self.hasContent_():
            outfile.write('>%s' % (eol_, ))
            self.exportChildren(outfile, level + 1, namespaceprefix_, namespacedef_, name_='cross_ref_db', pretty_print=pretty_print)
            outfile.write('</%s%s>%s' % (namespaceprefix_, name_, eol_))
        else:
            outfile.write('/>%s' % (eol_, ))
    def exportAttributes(self, outfile, level, already_processed, namespaceprefix_='', name_='cross_ref_db'):
        if self.name is not None and 'name' not in already_processed:
            already_processed.add('name')
            outfile.write(' name=%s' % (self.gds_encode(self.gds_format_string(quote_attrib(self.name), input_name='name')), ))
        if self.source is not None and 'source' not in already_processed:
            already_processed.add('source')
            outfile.write(' source=%s' % (self.gds_encode(self.gds_format_string(quote_attrib(self.source), input_name='source')), ))
        if self.accession_id is not None and 'accession_id' not in already_processed:
            already_processed.add('accession_id')
            outfile.write(' accession_id=%s' % (self.gds_encode(self.gds_format_string(quote_attrib(self.accession_id), input_name='accession_id')), ))
        if self.uniprot_start is not None and 'uniprot_start' not in already_processed:
            already_processed.add('uniprot_start')
            outfile.write(' uniprot_start="%s"' % self.gds_format_integer(self.uniprot_start, input_name='uniprot_start'))
        if self.uniprot_end is not None and 'uniprot_end' not in already_processed:
            already_processed.add('uniprot_end')
            outfile.write(' uniprot_end="%s"' % self.gds_format_integer(self.uniprot_end, input_name='uniprot_end'))
        if self.type_ is not None and 'type_' not in already_processed:
            already_processed.add('type_')
            outfile.write(' type=%s' % (self.gds_encode(self.gds_format_string(quote_attrib(self.type_), input_name='type')), ))
        if self.provenance is not None and 'provenance' not in already_processed:
            already_processed.add('provenance')
            outfile.write(' provenance=%s' % (self.gds_encode(self.gds_format_string(quote_attrib(self.provenance), input_name='provenance')), ))
        if self.score is not None and 'score' not in already_processed:
            already_processed.add('score')
            outfile.write(' score="%s"' % self.gds_format_float(self.score, input_name='score'))
    def exportChildren(self, outfile, level, namespaceprefix_='', namespacedef_='', name_='cross_ref_db', fromsubclass_=False, pretty_print=True):
        pass
    def build(self, node, gds_collector_=None):
        self.gds_collector_ = gds_collector_
        if SaveElementTreeNode:
            self.gds_elementtree_node_ = node
        already_processed = set()
        self.ns_prefix_ = node.prefix
        self.buildAttributes(node, node.attrib, already_processed)
        for child in node:
            nodeName_ = Tag_pattern_.match(child.tag).groups()[-1]
            self.buildChildren(child, node, nodeName_, gds_collector_=gds_collector_)
        return self
    def buildAttributes(self, node, attrs, already_processed):
        value = find_attr_value_('name', node)
        if value is not None and 'name' not in already_processed:
            already_processed.add('name')
            self.name = value
        value = find_attr_value_('source', node)
        if value is not None and 'source' not in already_processed:
            already_processed.add('source')
            self.source = value
            self.source = ' '.join(self.source.split())
            self.validate_provenance_type(self.source)    # validate type provenance_type
        value = find_attr_value_('accession_id', node)
        if value is not None and 'accession_id' not in already_processed:
            already_processed.add('accession_id')
            self.accession_id = value
        value = find_attr_value_('uniprot_start', node)
        if value is not None and 'uniprot_start' not in already_processed:
            already_processed.add('uniprot_start')
            self.uniprot_start = self.gds_parse_integer(value, node, 'uniprot_start')
            if self.uniprot_start < 0:
                raise_parse_error(node, 'Invalid NonNegativeInteger')
        value = find_attr_value_('uniprot_end', node)
        if value is not None and 'uniprot_end' not in already_processed:
            already_processed.add('uniprot_end')
            self.uniprot_end = self.gds_parse_integer(value, node, 'uniprot_end')
            if self.uniprot_end < 0:
                raise_parse_error(node, 'Invalid NonNegativeInteger')
        value = find_attr_value_('type', node)
        if value is not None and 'type' not in already_processed:
            already_processed.add('type')
            self.type_ = value
        value = find_attr_value_('provenance', node)
        if value is not None and 'provenance' not in already_processed:
            already_processed.add('provenance')
            self.provenance = value
            self.provenance = ' '.join(self.provenance.split())
            self.validate_provenance_type(self.provenance)    # validate type provenance_type
        value = find_attr_value_('score', node)
        if value is not None and 'score' not in already_processed:
            already_processed.add('score')
            value = self.gds_parse_float(value, node, 'score')
            self.score = value
    def buildChildren(self, child_, node, nodeName_, fromsubclass_=False, gds_collector_=None):
        pass
# end class cross_ref_db


class dbsType(GeneratedsSuper):
    """List of databases annotated for this EMDB entry"""
    __hash__ = GeneratedsSuper.__hash__
    subclass = None
    superclass = None
    def __init__(self, collection_date=None, db=None, gds_collector_=None, **kwargs_):
        self.gds_collector_ = gds_collector_
        self.gds_elementtree_node_ = None
        self.original_tagname_ = None
        self.parent_object_ = kwargs_.get('parent_object_')
        self.ns_prefix_ = None
        if isinstance(collection_date, BaseStrType_):
            initvalue_ = datetime_.datetime.strptime(collection_date, '%Y-%m-%d').date()
        else:
            initvalue_ = collection_date
        self.collection_date = initvalue_
        if db is None:
            self.db = []
        else:
            self.db = db
        self.db_nsprefix_ = None
    def factory(*args_, **kwargs_):
        if CurrentSubclassModule_ is not None:
            subclass = getSubclassFromModule_(
                CurrentSubclassModule_, dbsType)
            if subclass is not None:
                return subclass(*args_, **kwargs_)
        if dbsType.subclass:
            return dbsType.subclass(*args_, **kwargs_)
        else:
            return dbsType(*args_, **kwargs_)
    factory = staticmethod(factory)
    def get_ns_prefix_(self):
        return self.ns_prefix_
    def set_ns_prefix_(self, ns_prefix):
        self.ns_prefix_ = ns_prefix
    def get_db(self):
        return self.db
    def set_db(self, db):
        self.db = db
    def add_db(self, value):
        self.db.append(value)
    def insert_db_at(self, index, value):
        self.db.insert(index, value)
    def replace_db_at(self, index, value):
        self.db[index] = value
    def get_collection_date(self):
        return self.collection_date
    def set_collection_date(self, collection_date):
        self.collection_date = collection_date
    def hasContent_(self):
        if (
            self.db
        ):
            return True
        else:
            return False
    def export(self, outfile, level, namespaceprefix_='', namespacedef_='', name_='dbsType', pretty_print=True):
        imported_ns_def_ = GenerateDSNamespaceDefs_.get('dbsType')
        if imported_ns_def_ is not None:
            namespacedef_ = imported_ns_def_
        if pretty_print:
            eol_ = '\n'
        else:
            eol_ = ''
        if self.original_tagname_ is not None and name_ == 'dbsType':
            name_ = self.original_tagname_
        if UseCapturedNS_ and self.ns_prefix_:
            namespaceprefix_ = self.ns_prefix_ + ':'
        showIndent(outfile, level, pretty_print)
        outfile.write('<%s%s%s' % (namespaceprefix_, name_, namespacedef_ and ' ' + namespacedef_ or '', ))
        already_processed = set()
        self.exportAttributes(outfile, level, already_processed, namespaceprefix_, name_='dbsType')
        if self.hasContent_():
            outfile.write('>%s' % (eol_, ))
            self.exportChildren(outfile, level + 1, namespaceprefix_, namespacedef_, name_='dbsType', pretty_print=pretty_print)
            showIndent(outfile, level, pretty_print)
            outfile.write('</%s%s>%s' % (namespaceprefix_, name_, eol_))
        else:
            outfile.write('/>%s' % (eol_, ))
    def exportAttributes(self, outfile, level, already_processed, namespaceprefix_='', name_='dbsType'):
        if self.collection_date is not None and 'collection_date' not in already_processed:
            already_processed.add('collection_date')
            outfile.write(' collection_date="%s"' % self.gds_format_date(self.collection_date, input_name='collection_date'))
    def exportChildren(self, outfile, level, namespaceprefix_='', namespacedef_='', name_='dbsType', fromsubclass_=False, pretty_print=True):
        if pretty_print:
            eol_ = '\n'
        else:
            eol_ = ''
        for db_ in self.db:
            namespaceprefix_ = self.db_nsprefix_ + ':' if (UseCapturedNS_ and self.db_nsprefix_) else ''
            db_.export(outfile, level, namespaceprefix_, namespacedef_='', name_='db', pretty_print=pretty_print)
    def build(self, node, gds_collector_=None):
        self.gds_collector_ = gds_collector_
        if SaveElementTreeNode:
            self.gds_elementtree_node_ = node
        already_processed = set()
        self.ns_prefix_ = node.prefix
        self.buildAttributes(node, node.attrib, already_processed)
        for child in node:
            nodeName_ = Tag_pattern_.match(child.tag).groups()[-1]
            self.buildChildren(child, node, nodeName_, gds_collector_=gds_collector_)
        return self
    def buildAttributes(self, node, attrs, already_processed):
        value = find_attr_value_('collection_date', node)
        if value is not None and 'collection_date' not in already_processed:
            already_processed.add('collection_date')
            try:
                self.collection_date = self.gds_parse_date(value)
            except ValueError as exp:
                raise ValueError('Bad date attribute (collection_date): %s' % exp)
    def buildChildren(self, child_, node, nodeName_, fromsubclass_=False, gds_collector_=None):
        if nodeName_ == 'db':
            obj_ = dbType.factory(parent_object_=self)
            obj_.build(child_, gds_collector_=gds_collector_)
            self.db.append(obj_)
            obj_.original_tagname_ = 'db'
# end class dbsType


class dbType(GeneratedsSuper):
    """Name of the databaseVersion of the database if available"""
    __hash__ = GeneratedsSuper.__hash__
    subclass = None
    superclass = None
    def __init__(self, source=None, version=None, gds_collector_=None, **kwargs_):
        self.gds_collector_ = gds_collector_
        self.gds_elementtree_node_ = None
        self.original_tagname_ = None
        self.parent_object_ = kwargs_.get('parent_object_')
        self.ns_prefix_ = None
        self.source = _cast(None, source)
        self.source_nsprefix_ = None
        self.version = _cast(None, version)
        self.version_nsprefix_ = None
    def factory(*args_, **kwargs_):
        if CurrentSubclassModule_ is not None:
            subclass = getSubclassFromModule_(
                CurrentSubclassModule_, dbType)
            if subclass is not None:
                return subclass(*args_, **kwargs_)
        if dbType.subclass:
            return dbType.subclass(*args_, **kwargs_)
        else:
            return dbType(*args_, **kwargs_)
    factory = staticmethod(factory)
    def get_ns_prefix_(self):
        return self.ns_prefix_
    def set_ns_prefix_(self, ns_prefix):
        self.ns_prefix_ = ns_prefix
    def get_source(self):
        return self.source
    def set_source(self, source):
        self.source = source
    def get_version(self):
        return self.version
    def set_version(self, version):
        self.version = version
    def validate_provenance_type(self, value):
        # Validate type provenance_type, a restriction on xsd:token.
        if value is not None and Validate_simpletypes_ and self.gds_collector_ is not None:
            if not isinstance(value, str):
                lineno = self.gds_get_node_lineno_()
                self.gds_collector_.add_message('Value "%(value)s"%(lineno)s is not of the correct base simple type (str)' % {"value": value, "lineno": lineno, })
                return False
            value = value
            enumerations = ['EMDB', 'UniProt', 'UniProtKB', 'PDBe', 'PDBe-KB', 'AlphaFold DB', 'EMPIAR', 'EuropePMC', 'Complex Portal', 'ChEMBL', 'ChEBI', 'DrugBank', 'PDBe-CCD', 'PubMed', 'PubMed Central', 'ISSN', 'DOI', 'GO', 'InterPro', 'Pfam', 'CATH', 'SCOP', 'SCOP2', 'SCOP2B', 'Rfam']
            if value not in enumerations:
                lineno = self.gds_get_node_lineno_()
                self.gds_collector_.add_message('Value "%(value)s"%(lineno)s does not match xsd enumeration restriction on provenance_type' % {"value" : encode_str_2_3(value), "lineno": lineno} )
                result = False
    def hasContent_(self):
        if (

        ):
            return True
        else:
            return False
    def export(self, outfile, level, namespaceprefix_='', namespacedef_='', name_='dbType', pretty_print=True):
        imported_ns_def_ = GenerateDSNamespaceDefs_.get('dbType')
        if imported_ns_def_ is not None:
            namespacedef_ = imported_ns_def_
        if pretty_print:
            eol_ = '\n'
        else:
            eol_ = ''
        if self.original_tagname_ is not None and name_ == 'dbType':
            name_ = self.original_tagname_
        if UseCapturedNS_ and self.ns_prefix_:
            namespaceprefix_ = self.ns_prefix_ + ':'
        showIndent(outfile, level, pretty_print)
        outfile.write('<%s%s%s' % (namespaceprefix_, name_, namespacedef_ and ' ' + namespacedef_ or '', ))
        already_processed = set()
        self.exportAttributes(outfile, level, already_processed, namespaceprefix_, name_='dbType')
        if self.hasContent_():
            outfile.write('>%s' % (eol_, ))
            self.exportChildren(outfile, level + 1, namespaceprefix_, namespacedef_, name_='dbType', pretty_print=pretty_print)
            outfile.write('</%s%s>%s' % (namespaceprefix_, name_, eol_))
        else:
            outfile.write('/>%s' % (eol_, ))
    def exportAttributes(self, outfile, level, already_processed, namespaceprefix_='', name_='dbType'):
        if self.source is not None and 'source' not in already_processed:
            already_processed.add('source')
            outfile.write(' source=%s' % (self.gds_encode(self.gds_format_string(quote_attrib(self.source), input_name='source')), ))
        if self.version is not None and 'version' not in already_processed:
            already_processed.add('version')
            outfile.write(' version=%s' % (self.gds_encode(self.gds_format_string(quote_attrib(self.version), input_name='version')), ))
    def exportChildren(self, outfile, level, namespaceprefix_='', namespacedef_='', name_='dbType', fromsubclass_=False, pretty_print=True):
        pass
    def build(self, node, gds_collector_=None):
        self.gds_collector_ = gds_collector_
        if SaveElementTreeNode:
            self.gds_elementtree_node_ = node
        already_processed = set()
        self.ns_prefix_ = node.prefix
        self.buildAttributes(node, node.attrib, already_processed)
        for child in node:
            nodeName_ = Tag_pattern_.match(child.tag).groups()[-1]
            self.buildChildren(child, node, nodeName_, gds_collector_=gds_collector_)
        return self
    def buildAttributes(self, node, attrs, already_processed):
        value = find_attr_value_('source', node)
        if value is not None and 'source' not in already_processed:
            already_processed.add('source')
            self.source = value
            self.source = ' '.join(self.source.split())
            self.validate_provenance_type(self.source)    # validate type provenance_type
        value = find_attr_value_('version', node)
        if value is not None and 'version' not in already_processed:
            already_processed.add('version')
            self.version = value
            self.version = ' '.join(self.version.split())
    def buildChildren(self, child_, node, nodeName_, fromsubclass_=False, gds_collector_=None):
        pass
# end class dbType


class entry_ref_dbsType(GeneratedsSuper):
    """Annotations on EMDB entry level"""
    __hash__ = GeneratedsSuper.__hash__
    subclass = None
    superclass = None
    def __init__(self, entry_ref_db=None, gds_collector_=None, **kwargs_):
        self.gds_collector_ = gds_collector_
        self.gds_elementtree_node_ = None
        self.original_tagname_ = None
        self.parent_object_ = kwargs_.get('parent_object_')
        self.ns_prefix_ = None
        if entry_ref_db is None:
            self.entry_ref_db = []
        else:
            self.entry_ref_db = entry_ref_db
        self.entry_ref_db_nsprefix_ = None
    def factory(*args_, **kwargs_):
        if CurrentSubclassModule_ is not None:
            subclass = getSubclassFromModule_(
                CurrentSubclassModule_, entry_ref_dbsType)
            if subclass is not None:
                return subclass(*args_, **kwargs_)
        if entry_ref_dbsType.subclass:
            return entry_ref_dbsType.subclass(*args_, **kwargs_)
        else:
            return entry_ref_dbsType(*args_, **kwargs_)
    factory = staticmethod(factory)
    def get_ns_prefix_(self):
        return self.ns_prefix_
    def set_ns_prefix_(self, ns_prefix):
        self.ns_prefix_ = ns_prefix
    def get_entry_ref_db(self):
        return self.entry_ref_db
    def set_entry_ref_db(self, entry_ref_db):
        self.entry_ref_db = entry_ref_db
    def add_entry_ref_db(self, value):
        self.entry_ref_db.append(value)
    def insert_entry_ref_db_at(self, index, value):
        self.entry_ref_db.insert(index, value)
    def replace_entry_ref_db_at(self, index, value):
        self.entry_ref_db[index] = value
    def hasContent_(self):
        if (
            self.entry_ref_db
        ):
            return True
        else:
            return False
    def export(self, outfile, level, namespaceprefix_='', namespacedef_='', name_='entry_ref_dbsType', pretty_print=True):
        imported_ns_def_ = GenerateDSNamespaceDefs_.get('entry_ref_dbsType')
        if imported_ns_def_ is not None:
            namespacedef_ = imported_ns_def_
        if pretty_print:
            eol_ = '\n'
        else:
            eol_ = ''
        if self.original_tagname_ is not None and name_ == 'entry_ref_dbsType':
            name_ = self.original_tagname_
        if UseCapturedNS_ and self.ns_prefix_:
            namespaceprefix_ = self.ns_prefix_ + ':'
        showIndent(outfile, level, pretty_print)
        outfile.write('<%s%s%s' % (namespaceprefix_, name_, namespacedef_ and ' ' + namespacedef_ or '', ))
        already_processed = set()
        self.exportAttributes(outfile, level, already_processed, namespaceprefix_, name_='entry_ref_dbsType')
        if self.hasContent_():
            outfile.write('>%s' % (eol_, ))
            self.exportChildren(outfile, level + 1, namespaceprefix_, namespacedef_, name_='entry_ref_dbsType', pretty_print=pretty_print)
            showIndent(outfile, level, pretty_print)
            outfile.write('</%s%s>%s' % (namespaceprefix_, name_, eol_))
        else:
            outfile.write('/>%s' % (eol_, ))
    def exportAttributes(self, outfile, level, already_processed, namespaceprefix_='', name_='entry_ref_dbsType'):
        pass
    def exportChildren(self, outfile, level, namespaceprefix_='', namespacedef_='', name_='entry_ref_dbsType', fromsubclass_=False, pretty_print=True):
        if pretty_print:
            eol_ = '\n'
        else:
            eol_ = ''
        for entry_ref_db_ in self.entry_ref_db:
            namespaceprefix_ = self.entry_ref_db_nsprefix_ + ':' if (UseCapturedNS_ and self.entry_ref_db_nsprefix_) else ''
            entry_ref_db_.export(outfile, level, namespaceprefix_, namespacedef_='', name_='entry_ref_db', pretty_print=pretty_print)
    def build(self, node, gds_collector_=None):
        self.gds_collector_ = gds_collector_
        if SaveElementTreeNode:
            self.gds_elementtree_node_ = node
        already_processed = set()
        self.ns_prefix_ = node.prefix
        self.buildAttributes(node, node.attrib, already_processed)
        for child in node:
            nodeName_ = Tag_pattern_.match(child.tag).groups()[-1]
            self.buildChildren(child, node, nodeName_, gds_collector_=gds_collector_)
        return self
    def buildAttributes(self, node, attrs, already_processed):
        pass
    def buildChildren(self, child_, node, nodeName_, fromsubclass_=False, gds_collector_=None):
        if nodeName_ == 'entry_ref_db':
            obj_ = entry_ref_dbType.factory(parent_object_=self)
            obj_.build(child_, gds_collector_=gds_collector_)
            self.entry_ref_db.append(obj_)
            obj_.original_tagname_ = 'entry_ref_db'
# end class entry_ref_dbsType


class entry_ref_dbType(GeneratedsSuper):
    """Database/resource nameUnique ID provided by the databaseResource where
    the cross reference is fetched from."""
    __hash__ = GeneratedsSuper.__hash__
    subclass = None
    superclass = None
    def __init__(self, source=None, accession_id=None, provenance=None, gds_collector_=None, **kwargs_):
        self.gds_collector_ = gds_collector_
        self.gds_elementtree_node_ = None
        self.original_tagname_ = None
        self.parent_object_ = kwargs_.get('parent_object_')
        self.ns_prefix_ = None
        self.source = _cast(None, source)
        self.source_nsprefix_ = None
        self.accession_id = _cast(None, accession_id)
        self.accession_id_nsprefix_ = None
        self.provenance = _cast(None, provenance)
        self.provenance_nsprefix_ = None
    def factory(*args_, **kwargs_):
        if CurrentSubclassModule_ is not None:
            subclass = getSubclassFromModule_(
                CurrentSubclassModule_, entry_ref_dbType)
            if subclass is not None:
                return subclass(*args_, **kwargs_)
        if entry_ref_dbType.subclass:
            return entry_ref_dbType.subclass(*args_, **kwargs_)
        else:
            return entry_ref_dbType(*args_, **kwargs_)
    factory = staticmethod(factory)
    def get_ns_prefix_(self):
        return self.ns_prefix_
    def set_ns_prefix_(self, ns_prefix):
        self.ns_prefix_ = ns_prefix
    def get_source(self):
        return self.source
    def set_source(self, source):
        self.source = source
    def get_accession_id(self):
        return self.accession_id
    def set_accession_id(self, accession_id):
        self.accession_id = accession_id
    def get_provenance(self):
        return self.provenance
    def set_provenance(self, provenance):
        self.provenance = provenance
    def validate_provenance_type(self, value):
        # Validate type provenance_type, a restriction on xsd:token.
        if value is not None and Validate_simpletypes_ and self.gds_collector_ is not None:
            if not isinstance(value, str):
                lineno = self.gds_get_node_lineno_()
                self.gds_collector_.add_message('Value "%(value)s"%(lineno)s is not of the correct base simple type (str)' % {"value": value, "lineno": lineno, })
                return False
            value = value
            enumerations = ['EMDB', 'UniProt', 'UniProtKB', 'PDBe', 'PDBe-KB', 'AlphaFold DB', 'EMPIAR', 'EuropePMC', 'Complex Portal', 'ChEMBL', 'ChEBI', 'DrugBank', 'PDBe-CCD', 'PubMed', 'PubMed Central', 'ISSN', 'DOI', 'GO', 'InterPro', 'Pfam', 'CATH', 'SCOP', 'SCOP2', 'SCOP2B', 'Rfam']
            if value not in enumerations:
                lineno = self.gds_get_node_lineno_()
                self.gds_collector_.add_message('Value "%(value)s"%(lineno)s does not match xsd enumeration restriction on provenance_type' % {"value" : encode_str_2_3(value), "lineno": lineno} )
                result = False
    def hasContent_(self):
        if (

        ):
            return True
        else:
            return False
    def export(self, outfile, level, namespaceprefix_='', namespacedef_='', name_='entry_ref_dbType', pretty_print=True):
        imported_ns_def_ = GenerateDSNamespaceDefs_.get('entry_ref_dbType')
        if imported_ns_def_ is not None:
            namespacedef_ = imported_ns_def_
        if pretty_print:
            eol_ = '\n'
        else:
            eol_ = ''
        if self.original_tagname_ is not None and name_ == 'entry_ref_dbType':
            name_ = self.original_tagname_
        if UseCapturedNS_ and self.ns_prefix_:
            namespaceprefix_ = self.ns_prefix_ + ':'
        showIndent(outfile, level, pretty_print)
        outfile.write('<%s%s%s' % (namespaceprefix_, name_, namespacedef_ and ' ' + namespacedef_ or '', ))
        already_processed = set()
        self.exportAttributes(outfile, level, already_processed, namespaceprefix_, name_='entry_ref_dbType')
        if self.hasContent_():
            outfile.write('>%s' % (eol_, ))
            self.exportChildren(outfile, level + 1, namespaceprefix_, namespacedef_, name_='entry_ref_dbType', pretty_print=pretty_print)
            outfile.write('</%s%s>%s' % (namespaceprefix_, name_, eol_))
        else:
            outfile.write('/>%s' % (eol_, ))
    def exportAttributes(self, outfile, level, already_processed, namespaceprefix_='', name_='entry_ref_dbType'):
        if self.source is not None and 'source' not in already_processed:
            already_processed.add('source')
            outfile.write(' source=%s' % (self.gds_encode(self.gds_format_string(quote_attrib(self.source), input_name='source')), ))
        if self.accession_id is not None and 'accession_id' not in already_processed:
            already_processed.add('accession_id')
            outfile.write(' accession_id=%s' % (self.gds_encode(self.gds_format_string(quote_attrib(self.accession_id), input_name='accession_id')), ))
        if self.provenance is not None and 'provenance' not in already_processed:
            already_processed.add('provenance')
            outfile.write(' provenance=%s' % (self.gds_encode(self.gds_format_string(quote_attrib(self.provenance), input_name='provenance')), ))
    def exportChildren(self, outfile, level, namespaceprefix_='', namespacedef_='', name_='entry_ref_dbType', fromsubclass_=False, pretty_print=True):
        pass
    def build(self, node, gds_collector_=None):
        self.gds_collector_ = gds_collector_
        if SaveElementTreeNode:
            self.gds_elementtree_node_ = node
        already_processed = set()
        self.ns_prefix_ = node.prefix
        self.buildAttributes(node, node.attrib, already_processed)
        for child in node:
            nodeName_ = Tag_pattern_.match(child.tag).groups()[-1]
            self.buildChildren(child, node, nodeName_, gds_collector_=gds_collector_)
        return self
    def buildAttributes(self, node, attrs, already_processed):
        value = find_attr_value_('source', node)
        if value is not None and 'source' not in already_processed:
            already_processed.add('source')
            self.source = value
        value = find_attr_value_('accession_id', node)
        if value is not None and 'accession_id' not in already_processed:
            already_processed.add('accession_id')
            self.accession_id = value
        value = find_attr_value_('provenance', node)
        if value is not None and 'provenance' not in already_processed:
            already_processed.add('provenance')
            self.provenance = value
            self.provenance = ' '.join(self.provenance.split())
            self.validate_provenance_type(self.provenance)    # validate type provenance_type
    def buildChildren(self, child_, node, nodeName_, fromsubclass_=False, gds_collector_=None):
        pass
# end class entry_ref_dbType


class primary_citationType(GeneratedsSuper):
    """Entry based citation annotations
    DOI for the publicationResource from where DOI is mapped"""
    __hash__ = GeneratedsSuper.__hash__
    subclass = None
    superclass = None
    def __init__(self, doi=None, provenance=None, ref_citation=None, authors=None, gds_collector_=None, **kwargs_):
        self.gds_collector_ = gds_collector_
        self.gds_elementtree_node_ = None
        self.original_tagname_ = None
        self.parent_object_ = kwargs_.get('parent_object_')
        self.ns_prefix_ = None
        self.doi = _cast(None, doi)
        self.doi_nsprefix_ = None
        self.provenance = _cast(None, provenance)
        self.provenance_nsprefix_ = None
        if ref_citation is None:
            self.ref_citation = []
        else:
            self.ref_citation = ref_citation
        self.ref_citation_nsprefix_ = None
        self.authors = authors
        self.authors_nsprefix_ = None
    def factory(*args_, **kwargs_):
        if CurrentSubclassModule_ is not None:
            subclass = getSubclassFromModule_(
                CurrentSubclassModule_, primary_citationType)
            if subclass is not None:
                return subclass(*args_, **kwargs_)
        if primary_citationType.subclass:
            return primary_citationType.subclass(*args_, **kwargs_)
        else:
            return primary_citationType(*args_, **kwargs_)
    factory = staticmethod(factory)
    def get_ns_prefix_(self):
        return self.ns_prefix_
    def set_ns_prefix_(self, ns_prefix):
        self.ns_prefix_ = ns_prefix
    def get_ref_citation(self):
        return self.ref_citation
    def set_ref_citation(self, ref_citation):
        self.ref_citation = ref_citation
    def add_ref_citation(self, value):
        self.ref_citation.append(value)
    def insert_ref_citation_at(self, index, value):
        self.ref_citation.insert(index, value)
    def replace_ref_citation_at(self, index, value):
        self.ref_citation[index] = value
    def get_authors(self):
        return self.authors
    def set_authors(self, authors):
        self.authors = authors
    def get_doi(self):
        return self.doi
    def set_doi(self, doi):
        self.doi = doi
    def get_provenance(self):
        return self.provenance
    def set_provenance(self, provenance):
        self.provenance = provenance
    def validate_provenance_type(self, value):
        # Validate type provenance_type, a restriction on xsd:token.
        if value is not None and Validate_simpletypes_ and self.gds_collector_ is not None:
            if not isinstance(value, str):
                lineno = self.gds_get_node_lineno_()
                self.gds_collector_.add_message('Value "%(value)s"%(lineno)s is not of the correct base simple type (str)' % {"value": value, "lineno": lineno, })
                return False
            value = value
            enumerations = ['EMDB', 'UniProt', 'UniProtKB', 'PDBe', 'PDBe-KB', 'AlphaFold DB', 'EMPIAR', 'EuropePMC', 'Complex Portal', 'ChEMBL', 'ChEBI', 'DrugBank', 'PDBe-CCD', 'PubMed', 'PubMed Central', 'ISSN', 'DOI', 'GO', 'InterPro', 'Pfam', 'CATH', 'SCOP', 'SCOP2', 'SCOP2B', 'Rfam']
            if value not in enumerations:
                lineno = self.gds_get_node_lineno_()
                self.gds_collector_.add_message('Value "%(value)s"%(lineno)s does not match xsd enumeration restriction on provenance_type' % {"value" : encode_str_2_3(value), "lineno": lineno} )
                result = False
    def hasContent_(self):
        if (
            self.ref_citation or
            self.authors is not None
        ):
            return True
        else:
            return False
    def export(self, outfile, level, namespaceprefix_='', namespacedef_='', name_='primary_citationType', pretty_print=True):
        imported_ns_def_ = GenerateDSNamespaceDefs_.get('primary_citationType')
        if imported_ns_def_ is not None:
            namespacedef_ = imported_ns_def_
        if pretty_print:
            eol_ = '\n'
        else:
            eol_ = ''
        if self.original_tagname_ is not None and name_ == 'primary_citationType':
            name_ = self.original_tagname_
        if UseCapturedNS_ and self.ns_prefix_:
            namespaceprefix_ = self.ns_prefix_ + ':'
        showIndent(outfile, level, pretty_print)
        outfile.write('<%s%s%s' % (namespaceprefix_, name_, namespacedef_ and ' ' + namespacedef_ or '', ))
        already_processed = set()
        self.exportAttributes(outfile, level, already_processed, namespaceprefix_, name_='primary_citationType')
        if self.hasContent_():
            outfile.write('>%s' % (eol_, ))
            self.exportChildren(outfile, level + 1, namespaceprefix_, namespacedef_, name_='primary_citationType', pretty_print=pretty_print)
            showIndent(outfile, level, pretty_print)
            outfile.write('</%s%s>%s' % (namespaceprefix_, name_, eol_))
        else:
            outfile.write('/>%s' % (eol_, ))
    def exportAttributes(self, outfile, level, already_processed, namespaceprefix_='', name_='primary_citationType'):
        if self.doi is not None and 'doi' not in already_processed:
            already_processed.add('doi')
            outfile.write(' doi=%s' % (self.gds_encode(self.gds_format_string(quote_attrib(self.doi), input_name='doi')), ))
        if self.provenance is not None and 'provenance' not in already_processed:
            already_processed.add('provenance')
            outfile.write(' provenance=%s' % (self.gds_encode(self.gds_format_string(quote_attrib(self.provenance), input_name='provenance')), ))
    def exportChildren(self, outfile, level, namespaceprefix_='', namespacedef_='', name_='primary_citationType', fromsubclass_=False, pretty_print=True):
        if pretty_print:
            eol_ = '\n'
        else:
            eol_ = ''
        for ref_citation_ in self.ref_citation:
            namespaceprefix_ = self.ref_citation_nsprefix_ + ':' if (UseCapturedNS_ and self.ref_citation_nsprefix_) else ''
            ref_citation_.export(outfile, level, namespaceprefix_, namespacedef_='', name_='ref_citation', pretty_print=pretty_print)
        if self.authors is not None:
            namespaceprefix_ = self.authors_nsprefix_ + ':' if (UseCapturedNS_ and self.authors_nsprefix_) else ''
            self.authors.export(outfile, level, namespaceprefix_, namespacedef_='', name_='authors', pretty_print=pretty_print)
    def build(self, node, gds_collector_=None):
        self.gds_collector_ = gds_collector_
        if SaveElementTreeNode:
            self.gds_elementtree_node_ = node
        already_processed = set()
        self.ns_prefix_ = node.prefix
        self.buildAttributes(node, node.attrib, already_processed)
        for child in node:
            nodeName_ = Tag_pattern_.match(child.tag).groups()[-1]
            self.buildChildren(child, node, nodeName_, gds_collector_=gds_collector_)
        return self
    def buildAttributes(self, node, attrs, already_processed):
        value = find_attr_value_('doi', node)
        if value is not None and 'doi' not in already_processed:
            already_processed.add('doi')
            self.doi = value
        value = find_attr_value_('provenance', node)
        if value is not None and 'provenance' not in already_processed:
            already_processed.add('provenance')
            self.provenance = value
            self.provenance = ' '.join(self.provenance.split())
            self.validate_provenance_type(self.provenance)    # validate type provenance_type
    def buildChildren(self, child_, node, nodeName_, fromsubclass_=False, gds_collector_=None):
        if nodeName_ == 'ref_citation':
            obj_ = ref_citationType.factory(parent_object_=self)
            obj_.build(child_, gds_collector_=gds_collector_)
            self.ref_citation.append(obj_)
            obj_.original_tagname_ = 'ref_citation'
        elif nodeName_ == 'authors':
            obj_ = authorsType.factory(parent_object_=self)
            obj_.build(child_, gds_collector_=gds_collector_)
            self.authors = obj_
            obj_.original_tagname_ = 'authors'
# end class primary_citationType


class ref_citationType(GeneratedsSuper):
    """Reference to citationDatabase/resource nameUnique ID provided by the
    database/resourceResource citation reference is fetched"""
    __hash__ = GeneratedsSuper.__hash__
    subclass = None
    superclass = None
    def __init__(self, source=None, accession_id=None, provenance=None, gds_collector_=None, **kwargs_):
        self.gds_collector_ = gds_collector_
        self.gds_elementtree_node_ = None
        self.original_tagname_ = None
        self.parent_object_ = kwargs_.get('parent_object_')
        self.ns_prefix_ = None
        self.source = _cast(None, source)
        self.source_nsprefix_ = None
        self.accession_id = _cast(None, accession_id)
        self.accession_id_nsprefix_ = None
        self.provenance = _cast(None, provenance)
        self.provenance_nsprefix_ = None
    def factory(*args_, **kwargs_):
        if CurrentSubclassModule_ is not None:
            subclass = getSubclassFromModule_(
                CurrentSubclassModule_, ref_citationType)
            if subclass is not None:
                return subclass(*args_, **kwargs_)
        if ref_citationType.subclass:
            return ref_citationType.subclass(*args_, **kwargs_)
        else:
            return ref_citationType(*args_, **kwargs_)
    factory = staticmethod(factory)
    def get_ns_prefix_(self):
        return self.ns_prefix_
    def set_ns_prefix_(self, ns_prefix):
        self.ns_prefix_ = ns_prefix
    def get_source(self):
        return self.source
    def set_source(self, source):
        self.source = source
    def get_accession_id(self):
        return self.accession_id
    def set_accession_id(self, accession_id):
        self.accession_id = accession_id
    def get_provenance(self):
        return self.provenance
    def set_provenance(self, provenance):
        self.provenance = provenance
    def validate_provenance_type(self, value):
        # Validate type provenance_type, a restriction on xsd:token.
        if value is not None and Validate_simpletypes_ and self.gds_collector_ is not None:
            if not isinstance(value, str):
                lineno = self.gds_get_node_lineno_()
                self.gds_collector_.add_message('Value "%(value)s"%(lineno)s is not of the correct base simple type (str)' % {"value": value, "lineno": lineno, })
                return False
            value = value
            enumerations = ['EMDB', 'UniProt', 'UniProtKB', 'PDBe', 'PDBe-KB', 'AlphaFold DB', 'EMPIAR', 'EuropePMC', 'Complex Portal', 'ChEMBL', 'ChEBI', 'DrugBank', 'PDBe-CCD', 'PubMed', 'PubMed Central', 'ISSN', 'DOI', 'GO', 'InterPro', 'Pfam', 'CATH', 'SCOP', 'SCOP2', 'SCOP2B', 'Rfam']
            if value not in enumerations:
                lineno = self.gds_get_node_lineno_()
                self.gds_collector_.add_message('Value "%(value)s"%(lineno)s does not match xsd enumeration restriction on provenance_type' % {"value" : encode_str_2_3(value), "lineno": lineno} )
                result = False
    def hasContent_(self):
        if (

        ):
            return True
        else:
            return False
    def export(self, outfile, level, namespaceprefix_='', namespacedef_='', name_='ref_citationType', pretty_print=True):
        imported_ns_def_ = GenerateDSNamespaceDefs_.get('ref_citationType')
        if imported_ns_def_ is not None:
            namespacedef_ = imported_ns_def_
        if pretty_print:
            eol_ = '\n'
        else:
            eol_ = ''
        if self.original_tagname_ is not None and name_ == 'ref_citationType':
            name_ = self.original_tagname_
        if UseCapturedNS_ and self.ns_prefix_:
            namespaceprefix_ = self.ns_prefix_ + ':'
        showIndent(outfile, level, pretty_print)
        outfile.write('<%s%s%s' % (namespaceprefix_, name_, namespacedef_ and ' ' + namespacedef_ or '', ))
        already_processed = set()
        self.exportAttributes(outfile, level, already_processed, namespaceprefix_, name_='ref_citationType')
        if self.hasContent_():
            outfile.write('>%s' % (eol_, ))
            self.exportChildren(outfile, level + 1, namespaceprefix_, namespacedef_, name_='ref_citationType', pretty_print=pretty_print)
            outfile.write('</%s%s>%s' % (namespaceprefix_, name_, eol_))
        else:
            outfile.write('/>%s' % (eol_, ))
    def exportAttributes(self, outfile, level, already_processed, namespaceprefix_='', name_='ref_citationType'):
        if self.source is not None and 'source' not in already_processed:
            already_processed.add('source')
            outfile.write(' source=%s' % (self.gds_encode(self.gds_format_string(quote_attrib(self.source), input_name='source')), ))
        if self.accession_id is not None and 'accession_id' not in already_processed:
            already_processed.add('accession_id')
            outfile.write(' accession_id=%s' % (self.gds_encode(self.gds_format_string(quote_attrib(self.accession_id), input_name='accession_id')), ))
        if self.provenance is not None and 'provenance' not in already_processed:
            already_processed.add('provenance')
            outfile.write(' provenance=%s' % (self.gds_encode(self.gds_format_string(quote_attrib(self.provenance), input_name='provenance')), ))
    def exportChildren(self, outfile, level, namespaceprefix_='', namespacedef_='', name_='ref_citationType', fromsubclass_=False, pretty_print=True):
        pass
    def build(self, node, gds_collector_=None):
        self.gds_collector_ = gds_collector_
        if SaveElementTreeNode:
            self.gds_elementtree_node_ = node
        already_processed = set()
        self.ns_prefix_ = node.prefix
        self.buildAttributes(node, node.attrib, already_processed)
        for child in node:
            nodeName_ = Tag_pattern_.match(child.tag).groups()[-1]
            self.buildChildren(child, node, nodeName_, gds_collector_=gds_collector_)
        return self
    def buildAttributes(self, node, attrs, already_processed):
        value = find_attr_value_('source', node)
        if value is not None and 'source' not in already_processed:
            already_processed.add('source')
            self.source = value
            self.source = ' '.join(self.source.split())
            self.validate_provenance_type(self.source)    # validate type provenance_type
        value = find_attr_value_('accession_id', node)
        if value is not None and 'accession_id' not in already_processed:
            already_processed.add('accession_id')
            self.accession_id = value
        value = find_attr_value_('provenance', node)
        if value is not None and 'provenance' not in already_processed:
            already_processed.add('provenance')
            self.provenance = value
            self.provenance = ' '.join(self.provenance.split())
            self.validate_provenance_type(self.provenance)    # validate type provenance_type
    def buildChildren(self, child_, node, nodeName_, fromsubclass_=False, gds_collector_=None):
        pass
# end class ref_citationType


class authorsType(GeneratedsSuper):
    """Primary publication authors list"""
    __hash__ = GeneratedsSuper.__hash__
    subclass = None
    superclass = None
    def __init__(self, author=None, gds_collector_=None, **kwargs_):
        self.gds_collector_ = gds_collector_
        self.gds_elementtree_node_ = None
        self.original_tagname_ = None
        self.parent_object_ = kwargs_.get('parent_object_')
        self.ns_prefix_ = None
        if author is None:
            self.author = []
        else:
            self.author = author
        self.author_nsprefix_ = None
    def factory(*args_, **kwargs_):
        if CurrentSubclassModule_ is not None:
            subclass = getSubclassFromModule_(
                CurrentSubclassModule_, authorsType)
            if subclass is not None:
                return subclass(*args_, **kwargs_)
        if authorsType.subclass:
            return authorsType.subclass(*args_, **kwargs_)
        else:
            return authorsType(*args_, **kwargs_)
    factory = staticmethod(factory)
    def get_ns_prefix_(self):
        return self.ns_prefix_
    def set_ns_prefix_(self, ns_prefix):
        self.ns_prefix_ = ns_prefix
    def get_author(self):
        return self.author
    def set_author(self, author):
        self.author = author
    def add_author(self, value):
        self.author.append(value)
    def insert_author_at(self, index, value):
        self.author.insert(index, value)
    def replace_author_at(self, index, value):
        self.author[index] = value
    def hasContent_(self):
        if (
            self.author
        ):
            return True
        else:
            return False
    def export(self, outfile, level, namespaceprefix_='', namespacedef_='', name_='authorsType', pretty_print=True):
        imported_ns_def_ = GenerateDSNamespaceDefs_.get('authorsType')
        if imported_ns_def_ is not None:
            namespacedef_ = imported_ns_def_
        if pretty_print:
            eol_ = '\n'
        else:
            eol_ = ''
        if self.original_tagname_ is not None and name_ == 'authorsType':
            name_ = self.original_tagname_
        if UseCapturedNS_ and self.ns_prefix_:
            namespaceprefix_ = self.ns_prefix_ + ':'
        showIndent(outfile, level, pretty_print)
        outfile.write('<%s%s%s' % (namespaceprefix_, name_, namespacedef_ and ' ' + namespacedef_ or '', ))
        already_processed = set()
        self.exportAttributes(outfile, level, already_processed, namespaceprefix_, name_='authorsType')
        if self.hasContent_():
            outfile.write('>%s' % (eol_, ))
            self.exportChildren(outfile, level + 1, namespaceprefix_, namespacedef_, name_='authorsType', pretty_print=pretty_print)
            showIndent(outfile, level, pretty_print)
            outfile.write('</%s%s>%s' % (namespaceprefix_, name_, eol_))
        else:
            outfile.write('/>%s' % (eol_, ))
    def exportAttributes(self, outfile, level, already_processed, namespaceprefix_='', name_='authorsType'):
        pass
    def exportChildren(self, outfile, level, namespaceprefix_='', namespacedef_='', name_='authorsType', fromsubclass_=False, pretty_print=True):
        if pretty_print:
            eol_ = '\n'
        else:
            eol_ = ''
        for author_ in self.author:
            namespaceprefix_ = self.author_nsprefix_ + ':' if (UseCapturedNS_ and self.author_nsprefix_) else ''
            author_.export(outfile, level, namespaceprefix_, namespacedef_='', name_='author', pretty_print=pretty_print)
    def build(self, node, gds_collector_=None):
        self.gds_collector_ = gds_collector_
        if SaveElementTreeNode:
            self.gds_elementtree_node_ = node
        already_processed = set()
        self.ns_prefix_ = node.prefix
        self.buildAttributes(node, node.attrib, already_processed)
        for child in node:
            nodeName_ = Tag_pattern_.match(child.tag).groups()[-1]
            self.buildChildren(child, node, nodeName_, gds_collector_=gds_collector_)
        return self
    def buildAttributes(self, node, attrs, already_processed):
        pass
    def buildChildren(self, child_, node, nodeName_, fromsubclass_=False, gds_collector_=None):
        if nodeName_ == 'author':
            obj_ = authorType.factory(parent_object_=self)
            obj_.build(child_, gds_collector_=gds_collector_)
            self.author.append(obj_)
            obj_.original_tagname_ = 'author'
# end class authorsType


class authorType(GeneratedsSuper):
    """Author’s detailAuthor's nameAuthor's ORCID idOrder of the author’s name
    as in publication Resource ORCID is fetched"""
    __hash__ = GeneratedsSuper.__hash__
    subclass = None
    superclass = None
    def __init__(self, name=None, orcid_id=None, order=None, provenance=None, gds_collector_=None, **kwargs_):
        self.gds_collector_ = gds_collector_
        self.gds_elementtree_node_ = None
        self.original_tagname_ = None
        self.parent_object_ = kwargs_.get('parent_object_')
        self.ns_prefix_ = None
        self.name = _cast(None, name)
        self.name_nsprefix_ = None
        self.orcid_id = _cast(None, orcid_id)
        self.orcid_id_nsprefix_ = None
        self.order = _cast(int, order)
        self.order_nsprefix_ = None
        self.provenance = _cast(None, provenance)
        self.provenance_nsprefix_ = None
    def factory(*args_, **kwargs_):
        if CurrentSubclassModule_ is not None:
            subclass = getSubclassFromModule_(
                CurrentSubclassModule_, authorType)
            if subclass is not None:
                return subclass(*args_, **kwargs_)
        if authorType.subclass:
            return authorType.subclass(*args_, **kwargs_)
        else:
            return authorType(*args_, **kwargs_)
    factory = staticmethod(factory)
    def get_ns_prefix_(self):
        return self.ns_prefix_
    def set_ns_prefix_(self, ns_prefix):
        self.ns_prefix_ = ns_prefix
    def get_name(self):
        return self.name
    def set_name(self, name):
        self.name = name
    def get_orcid_id(self):
        return self.orcid_id
    def set_orcid_id(self, orcid_id):
        self.orcid_id = orcid_id
    def get_order(self):
        return self.order
    def set_order(self, order):
        self.order = order
    def get_provenance(self):
        return self.provenance
    def set_provenance(self, provenance):
        self.provenance = provenance
    def validate_provenance_type(self, value):
        # Validate type provenance_type, a restriction on xsd:token.
        if value is not None and Validate_simpletypes_ and self.gds_collector_ is not None:
            if not isinstance(value, str):
                lineno = self.gds_get_node_lineno_()
                self.gds_collector_.add_message('Value "%(value)s"%(lineno)s is not of the correct base simple type (str)' % {"value": value, "lineno": lineno, })
                return False
            value = value
            enumerations = ['EMDB', 'UniProt', 'UniProtKB', 'PDBe', 'PDBe-KB', 'AlphaFold DB', 'EMPIAR', 'EuropePMC', 'Complex Portal', 'ChEMBL', 'ChEBI', 'DrugBank', 'PDBe-CCD', 'PubMed', 'PubMed Central', 'ISSN', 'DOI', 'GO', 'InterPro', 'Pfam', 'CATH', 'SCOP', 'SCOP2', 'SCOP2B', 'Rfam']
            if value not in enumerations:
                lineno = self.gds_get_node_lineno_()
                self.gds_collector_.add_message('Value "%(value)s"%(lineno)s does not match xsd enumeration restriction on provenance_type' % {"value" : encode_str_2_3(value), "lineno": lineno} )
                result = False
    def hasContent_(self):
        if (

        ):
            return True
        else:
            return False
    def export(self, outfile, level, namespaceprefix_='', namespacedef_='', name_='authorType', pretty_print=True):
        imported_ns_def_ = GenerateDSNamespaceDefs_.get('authorType')
        if imported_ns_def_ is not None:
            namespacedef_ = imported_ns_def_
        if pretty_print:
            eol_ = '\n'
        else:
            eol_ = ''
        if self.original_tagname_ is not None and name_ == 'authorType':
            name_ = self.original_tagname_
        if UseCapturedNS_ and self.ns_prefix_:
            namespaceprefix_ = self.ns_prefix_ + ':'
        showIndent(outfile, level, pretty_print)
        outfile.write('<%s%s%s' % (namespaceprefix_, name_, namespacedef_ and ' ' + namespacedef_ or '', ))
        already_processed = set()
        self.exportAttributes(outfile, level, already_processed, namespaceprefix_, name_='authorType')
        if self.hasContent_():
            outfile.write('>%s' % (eol_, ))
            self.exportChildren(outfile, level + 1, namespaceprefix_, namespacedef_, name_='authorType', pretty_print=pretty_print)
            outfile.write('</%s%s>%s' % (namespaceprefix_, name_, eol_))
        else:
            outfile.write('/>%s' % (eol_, ))
    def exportAttributes(self, outfile, level, already_processed, namespaceprefix_='', name_='authorType'):
        if self.name is not None and 'name' not in already_processed:
            already_processed.add('name')
            outfile.write(' name=%s' % (self.gds_encode(self.gds_format_string(quote_attrib(self.name), input_name='name')), ))
        if self.orcid_id is not None and 'orcid_id' not in already_processed:
            already_processed.add('orcid_id')
            outfile.write(' orcid_id=%s' % (self.gds_encode(self.gds_format_string(quote_attrib(self.orcid_id), input_name='orcid_id')), ))
        if self.order is not None and 'order' not in already_processed:
            already_processed.add('order')
            outfile.write(' order="%s"' % self.gds_format_integer(self.order, input_name='order'))
        if self.provenance is not None and 'provenance' not in already_processed:
            already_processed.add('provenance')
            outfile.write(' provenance=%s' % (self.gds_encode(self.gds_format_string(quote_attrib(self.provenance), input_name='provenance')), ))
    def exportChildren(self, outfile, level, namespaceprefix_='', namespacedef_='', name_='authorType', fromsubclass_=False, pretty_print=True):
        pass
    def build(self, node, gds_collector_=None):
        self.gds_collector_ = gds_collector_
        if SaveElementTreeNode:
            self.gds_elementtree_node_ = node
        already_processed = set()
        self.ns_prefix_ = node.prefix
        self.buildAttributes(node, node.attrib, already_processed)
        for child in node:
            nodeName_ = Tag_pattern_.match(child.tag).groups()[-1]
            self.buildChildren(child, node, nodeName_, gds_collector_=gds_collector_)
        return self
    def buildAttributes(self, node, attrs, already_processed):
        value = find_attr_value_('name', node)
        if value is not None and 'name' not in already_processed:
            already_processed.add('name')
            self.name = value
        value = find_attr_value_('orcid_id', node)
        if value is not None and 'orcid_id' not in already_processed:
            already_processed.add('orcid_id')
            self.orcid_id = value
        value = find_attr_value_('order', node)
        if value is not None and 'order' not in already_processed:
            already_processed.add('order')
            self.order = self.gds_parse_integer(value, node, 'order')
        value = find_attr_value_('provenance', node)
        if value is not None and 'provenance' not in already_processed:
            already_processed.add('provenance')
            self.provenance = value
            self.provenance = ' '.join(self.provenance.split())
            self.validate_provenance_type(self.provenance)    # validate type provenance_type
    def buildChildren(self, child_, node, nodeName_, fromsubclass_=False, gds_collector_=None):
        pass
# end class authorType


class weightsType(GeneratedsSuper):
    """Total weight of the assemblies. Calculated from both PDBe and author
    provided (experimental and theoretical) values"""
    __hash__ = GeneratedsSuper.__hash__
    subclass = None
    superclass = None
    def __init__(self, weight_info=None, gds_collector_=None, **kwargs_):
        self.gds_collector_ = gds_collector_
        self.gds_elementtree_node_ = None
        self.original_tagname_ = None
        self.parent_object_ = kwargs_.get('parent_object_')
        self.ns_prefix_ = None
        if weight_info is None:
            self.weight_info = []
        else:
            self.weight_info = weight_info
        self.weight_info_nsprefix_ = None
    def factory(*args_, **kwargs_):
        if CurrentSubclassModule_ is not None:
            subclass = getSubclassFromModule_(
                CurrentSubclassModule_, weightsType)
            if subclass is not None:
                return subclass(*args_, **kwargs_)
        if weightsType.subclass:
            return weightsType.subclass(*args_, **kwargs_)
        else:
            return weightsType(*args_, **kwargs_)
    factory = staticmethod(factory)
    def get_ns_prefix_(self):
        return self.ns_prefix_
    def set_ns_prefix_(self, ns_prefix):
        self.ns_prefix_ = ns_prefix
    def get_weight_info(self):
        return self.weight_info
    def set_weight_info(self, weight_info):
        self.weight_info = weight_info
    def add_weight_info(self, value):
        self.weight_info.append(value)
    def insert_weight_info_at(self, index, value):
        self.weight_info.insert(index, value)
    def replace_weight_info_at(self, index, value):
        self.weight_info[index] = value
    def hasContent_(self):
        if (
            self.weight_info
        ):
            return True
        else:
            return False
    def export(self, outfile, level, namespaceprefix_='', namespacedef_='', name_='weightsType', pretty_print=True):
        imported_ns_def_ = GenerateDSNamespaceDefs_.get('weightsType')
        if imported_ns_def_ is not None:
            namespacedef_ = imported_ns_def_
        if pretty_print:
            eol_ = '\n'
        else:
            eol_ = ''
        if self.original_tagname_ is not None and name_ == 'weightsType':
            name_ = self.original_tagname_
        if UseCapturedNS_ and self.ns_prefix_:
            namespaceprefix_ = self.ns_prefix_ + ':'
        showIndent(outfile, level, pretty_print)
        outfile.write('<%s%s%s' % (namespaceprefix_, name_, namespacedef_ and ' ' + namespacedef_ or '', ))
        already_processed = set()
        self.exportAttributes(outfile, level, already_processed, namespaceprefix_, name_='weightsType')
        if self.hasContent_():
            outfile.write('>%s' % (eol_, ))
            self.exportChildren(outfile, level + 1, namespaceprefix_, namespacedef_, name_='weightsType', pretty_print=pretty_print)
            showIndent(outfile, level, pretty_print)
            outfile.write('</%s%s>%s' % (namespaceprefix_, name_, eol_))
        else:
            outfile.write('/>%s' % (eol_, ))
    def exportAttributes(self, outfile, level, already_processed, namespaceprefix_='', name_='weightsType'):
        pass
    def exportChildren(self, outfile, level, namespaceprefix_='', namespacedef_='', name_='weightsType', fromsubclass_=False, pretty_print=True):
        if pretty_print:
            eol_ = '\n'
        else:
            eol_ = ''
        for weight_info_ in self.weight_info:
            namespaceprefix_ = self.weight_info_nsprefix_ + ':' if (UseCapturedNS_ and self.weight_info_nsprefix_) else ''
            weight_info_.export(outfile, level, namespaceprefix_, namespacedef_='', name_='weight_info', pretty_print=pretty_print)
    def build(self, node, gds_collector_=None):
        self.gds_collector_ = gds_collector_
        if SaveElementTreeNode:
            self.gds_elementtree_node_ = node
        already_processed = set()
        self.ns_prefix_ = node.prefix
        self.buildAttributes(node, node.attrib, already_processed)
        for child in node:
            nodeName_ = Tag_pattern_.match(child.tag).groups()[-1]
            self.buildChildren(child, node, nodeName_, gds_collector_=gds_collector_)
        return self
    def buildAttributes(self, node, attrs, already_processed):
        pass
    def buildChildren(self, child_, node, nodeName_, fromsubclass_=False, gds_collector_=None):
        if nodeName_ == 'weight_info':
            obj_ = weight_infoType.factory(parent_object_=self)
            obj_.build(child_, gds_collector_=gds_collector_)
            self.weight_info.append(obj_)
            obj_.original_tagname_ = 'weight_info'
# end class weightsType


class weight_infoType(GeneratedsSuper):
    """Sample weight by different methodsProtein Data Bank idNumber of
    biological assemblies in PDBWeight of the sampleSample weight
    UnitsResource sample weight is fetched"""
    __hash__ = GeneratedsSuper.__hash__
    subclass = None
    superclass = None
    def __init__(self, pdb_id=None, assemblies=None, weight=None, unit=None, provenance=None, gds_collector_=None, **kwargs_):
        self.gds_collector_ = gds_collector_
        self.gds_elementtree_node_ = None
        self.original_tagname_ = None
        self.parent_object_ = kwargs_.get('parent_object_')
        self.ns_prefix_ = None
        self.pdb_id = _cast(None, pdb_id)
        self.pdb_id_nsprefix_ = None
        self.assemblies = _cast(int, assemblies)
        self.assemblies_nsprefix_ = None
        self.weight = _cast(None, weight)
        self.weight_nsprefix_ = None
        self.unit = _cast(None, unit)
        self.unit_nsprefix_ = None
        self.provenance = _cast(None, provenance)
        self.provenance_nsprefix_ = None
    def factory(*args_, **kwargs_):
        if CurrentSubclassModule_ is not None:
            subclass = getSubclassFromModule_(
                CurrentSubclassModule_, weight_infoType)
            if subclass is not None:
                return subclass(*args_, **kwargs_)
        if weight_infoType.subclass:
            return weight_infoType.subclass(*args_, **kwargs_)
        else:
            return weight_infoType(*args_, **kwargs_)
    factory = staticmethod(factory)
    def get_ns_prefix_(self):
        return self.ns_prefix_
    def set_ns_prefix_(self, ns_prefix):
        self.ns_prefix_ = ns_prefix
    def get_pdb_id(self):
        return self.pdb_id
    def set_pdb_id(self, pdb_id):
        self.pdb_id = pdb_id
    def get_assemblies(self):
        return self.assemblies
    def set_assemblies(self, assemblies):
        self.assemblies = assemblies
    def get_weight(self):
        return self.weight
    def set_weight(self, weight):
        self.weight = weight
    def get_unit(self):
        return self.unit
    def set_unit(self, unit):
        self.unit = unit
    def get_provenance(self):
        return self.provenance
    def set_provenance(self, provenance):
        self.provenance = provenance
    def validate_provenance_type(self, value):
        # Validate type provenance_type, a restriction on xsd:token.
        if value is not None and Validate_simpletypes_ and self.gds_collector_ is not None:
            if not isinstance(value, str):
                lineno = self.gds_get_node_lineno_()
                self.gds_collector_.add_message('Value "%(value)s"%(lineno)s is not of the correct base simple type (str)' % {"value": value, "lineno": lineno, })
                return False
            value = value
            enumerations = ['EMDB', 'UniProt', 'UniProtKB', 'PDBe', 'PDBe-KB', 'AlphaFold DB', 'EMPIAR', 'EuropePMC', 'Complex Portal', 'ChEMBL', 'ChEBI', 'DrugBank', 'PDBe-CCD', 'PubMed', 'PubMed Central', 'ISSN', 'DOI', 'GO', 'InterPro', 'Pfam', 'CATH', 'SCOP', 'SCOP2', 'SCOP2B', 'Rfam']
            if value not in enumerations:
                lineno = self.gds_get_node_lineno_()
                self.gds_collector_.add_message('Value "%(value)s"%(lineno)s does not match xsd enumeration restriction on provenance_type' % {"value" : encode_str_2_3(value), "lineno": lineno} )
                result = False
    def hasContent_(self):
        if (

        ):
            return True
        else:
            return False
    def export(self, outfile, level, namespaceprefix_='', namespacedef_='', name_='weight_infoType', pretty_print=True):
        imported_ns_def_ = GenerateDSNamespaceDefs_.get('weight_infoType')
        if imported_ns_def_ is not None:
            namespacedef_ = imported_ns_def_
        if pretty_print:
            eol_ = '\n'
        else:
            eol_ = ''
        if self.original_tagname_ is not None and name_ == 'weight_infoType':
            name_ = self.original_tagname_
        if UseCapturedNS_ and self.ns_prefix_:
            namespaceprefix_ = self.ns_prefix_ + ':'
        showIndent(outfile, level, pretty_print)
        outfile.write('<%s%s%s' % (namespaceprefix_, name_, namespacedef_ and ' ' + namespacedef_ or '', ))
        already_processed = set()
        self.exportAttributes(outfile, level, already_processed, namespaceprefix_, name_='weight_infoType')
        if self.hasContent_():
            outfile.write('>%s' % (eol_, ))
            self.exportChildren(outfile, level + 1, namespaceprefix_, namespacedef_, name_='weight_infoType', pretty_print=pretty_print)
            outfile.write('</%s%s>%s' % (namespaceprefix_, name_, eol_))
        else:
            outfile.write('/>%s' % (eol_, ))
    def exportAttributes(self, outfile, level, already_processed, namespaceprefix_='', name_='weight_infoType'):
        if self.pdb_id is not None and 'pdb_id' not in already_processed:
            already_processed.add('pdb_id')
            outfile.write(' pdb_id=%s' % (self.gds_encode(self.gds_format_string(quote_attrib(self.pdb_id), input_name='pdb_id')), ))
        if self.assemblies is not None and 'assemblies' not in already_processed:
            already_processed.add('assemblies')
            outfile.write(' assemblies="%s"' % self.gds_format_integer(self.assemblies, input_name='assemblies'))
        if self.weight is not None and 'weight' not in already_processed:
            already_processed.add('weight')
            outfile.write(' weight=%s' % (self.gds_encode(self.gds_format_string(quote_attrib(self.weight), input_name='weight')), ))
        if self.unit is not None and 'unit' not in already_processed:
            already_processed.add('unit')
            outfile.write(' unit=%s' % (self.gds_encode(self.gds_format_string(quote_attrib(self.unit), input_name='unit')), ))
        if self.provenance is not None and 'provenance' not in already_processed:
            already_processed.add('provenance')
            outfile.write(' provenance=%s' % (self.gds_encode(self.gds_format_string(quote_attrib(self.provenance), input_name='provenance')), ))
    def exportChildren(self, outfile, level, namespaceprefix_='', namespacedef_='', name_='weight_infoType', fromsubclass_=False, pretty_print=True):
        pass
    def build(self, node, gds_collector_=None):
        self.gds_collector_ = gds_collector_
        if SaveElementTreeNode:
            self.gds_elementtree_node_ = node
        already_processed = set()
        self.ns_prefix_ = node.prefix
        self.buildAttributes(node, node.attrib, already_processed)
        for child in node:
            nodeName_ = Tag_pattern_.match(child.tag).groups()[-1]
            self.buildChildren(child, node, nodeName_, gds_collector_=gds_collector_)
        return self
    def buildAttributes(self, node, attrs, already_processed):
        value = find_attr_value_('pdb_id', node)
        if value is not None and 'pdb_id' not in already_processed:
            already_processed.add('pdb_id')
            self.pdb_id = value
        value = find_attr_value_('assemblies', node)
        if value is not None and 'assemblies' not in already_processed:
            already_processed.add('assemblies')
            self.assemblies = self.gds_parse_integer(value, node, 'assemblies')
        value = find_attr_value_('weight', node)
        if value is not None and 'weight' not in already_processed:
            already_processed.add('weight')
            self.weight = value
        value = find_attr_value_('unit', node)
        if value is not None and 'unit' not in already_processed:
            already_processed.add('unit')
            self.unit = value
        value = find_attr_value_('provenance', node)
        if value is not None and 'provenance' not in already_processed:
            already_processed.add('provenance')
            self.provenance = value
            self.provenance = ' '.join(self.provenance.split())
            self.validate_provenance_type(self.provenance)    # validate type provenance_type
    def buildChildren(self, child_, node, nodeName_, fromsubclass_=False, gds_collector_=None):
        pass
# end class weight_infoType


class sampleType(GeneratedsSuper):
    """Sample level annotations"""
    __hash__ = GeneratedsSuper.__hash__
    subclass = None
    superclass = None
    def __init__(self, name=None, cross_ref_dbs=None, supramolecules=None, macromolecules=None, gds_collector_=None, **kwargs_):
        self.gds_collector_ = gds_collector_
        self.gds_elementtree_node_ = None
        self.original_tagname_ = None
        self.parent_object_ = kwargs_.get('parent_object_')
        self.ns_prefix_ = None
        self.name = name
        self.name_nsprefix_ = None
        self.cross_ref_dbs = cross_ref_dbs
        self.cross_ref_dbs_nsprefix_ = None
        self.supramolecules = supramolecules
        self.supramolecules_nsprefix_ = None
        self.macromolecules = macromolecules
        self.macromolecules_nsprefix_ = None
    def factory(*args_, **kwargs_):
        if CurrentSubclassModule_ is not None:
            subclass = getSubclassFromModule_(
                CurrentSubclassModule_, sampleType)
            if subclass is not None:
                return subclass(*args_, **kwargs_)
        if sampleType.subclass:
            return sampleType.subclass(*args_, **kwargs_)
        else:
            return sampleType(*args_, **kwargs_)
    factory = staticmethod(factory)
    def get_ns_prefix_(self):
        return self.ns_prefix_
    def set_ns_prefix_(self, ns_prefix):
        self.ns_prefix_ = ns_prefix
    def get_name(self):
        return self.name
    def set_name(self, name):
        self.name = name
    def get_cross_ref_dbs(self):
        return self.cross_ref_dbs
    def set_cross_ref_dbs(self, cross_ref_dbs):
        self.cross_ref_dbs = cross_ref_dbs
    def get_supramolecules(self):
        return self.supramolecules
    def set_supramolecules(self, supramolecules):
        self.supramolecules = supramolecules
    def get_macromolecules(self):
        return self.macromolecules
    def set_macromolecules(self, macromolecules):
        self.macromolecules = macromolecules
    def hasContent_(self):
        if (
            self.name is not None or
            self.cross_ref_dbs is not None or
            self.supramolecules is not None or
            self.macromolecules is not None
        ):
            return True
        else:
            return False
    def export(self, outfile, level, namespaceprefix_='', namespacedef_='', name_='sampleType', pretty_print=True):
        imported_ns_def_ = GenerateDSNamespaceDefs_.get('sampleType')
        if imported_ns_def_ is not None:
            namespacedef_ = imported_ns_def_
        if pretty_print:
            eol_ = '\n'
        else:
            eol_ = ''
        if self.original_tagname_ is not None and name_ == 'sampleType':
            name_ = self.original_tagname_
        if UseCapturedNS_ and self.ns_prefix_:
            namespaceprefix_ = self.ns_prefix_ + ':'
        showIndent(outfile, level, pretty_print)
        outfile.write('<%s%s%s' % (namespaceprefix_, name_, namespacedef_ and ' ' + namespacedef_ or '', ))
        already_processed = set()
        self.exportAttributes(outfile, level, already_processed, namespaceprefix_, name_='sampleType')
        if self.hasContent_():
            outfile.write('>%s' % (eol_, ))
            self.exportChildren(outfile, level + 1, namespaceprefix_, namespacedef_, name_='sampleType', pretty_print=pretty_print)
            showIndent(outfile, level, pretty_print)
            outfile.write('</%s%s>%s' % (namespaceprefix_, name_, eol_))
        else:
            outfile.write('/>%s' % (eol_, ))
    def exportAttributes(self, outfile, level, already_processed, namespaceprefix_='', name_='sampleType'):
        pass
    def exportChildren(self, outfile, level, namespaceprefix_='', namespacedef_='', name_='sampleType', fromsubclass_=False, pretty_print=True):
        if pretty_print:
            eol_ = '\n'
        else:
            eol_ = ''
        if self.name is not None:
            namespaceprefix_ = self.name_nsprefix_ + ':' if (UseCapturedNS_ and self.name_nsprefix_) else ''
            showIndent(outfile, level, pretty_print)
            outfile.write('<%sname>%s</%sname>%s' % (namespaceprefix_ , self.gds_encode(self.gds_format_string(quote_xml(self.name), input_name='name')), namespaceprefix_ , eol_))
        if self.cross_ref_dbs is not None:
            namespaceprefix_ = self.cross_ref_dbs_nsprefix_ + ':' if (UseCapturedNS_ and self.cross_ref_dbs_nsprefix_) else ''
            self.cross_ref_dbs.export(outfile, level, namespaceprefix_, namespacedef_='', name_='cross_ref_dbs', pretty_print=pretty_print)
        if self.supramolecules is not None:
            namespaceprefix_ = self.supramolecules_nsprefix_ + ':' if (UseCapturedNS_ and self.supramolecules_nsprefix_) else ''
            self.supramolecules.export(outfile, level, namespaceprefix_, namespacedef_='', name_='supramolecules', pretty_print=pretty_print)
        if self.macromolecules is not None:
            namespaceprefix_ = self.macromolecules_nsprefix_ + ':' if (UseCapturedNS_ and self.macromolecules_nsprefix_) else ''
            self.macromolecules.export(outfile, level, namespaceprefix_, namespacedef_='', name_='macromolecules', pretty_print=pretty_print)
    def build(self, node, gds_collector_=None):
        self.gds_collector_ = gds_collector_
        if SaveElementTreeNode:
            self.gds_elementtree_node_ = node
        already_processed = set()
        self.ns_prefix_ = node.prefix
        self.buildAttributes(node, node.attrib, already_processed)
        for child in node:
            nodeName_ = Tag_pattern_.match(child.tag).groups()[-1]
            self.buildChildren(child, node, nodeName_, gds_collector_=gds_collector_)
        return self
    def buildAttributes(self, node, attrs, already_processed):
        pass
    def buildChildren(self, child_, node, nodeName_, fromsubclass_=False, gds_collector_=None):
        if nodeName_ == 'name':
            value_ = child_.text
            value_ = self.gds_parse_string(value_, node, 'name')
            value_ = self.gds_validate_string(value_, node, 'name')
            self.name = value_
            self.name_nsprefix_ = child_.prefix
        elif nodeName_ == 'cross_ref_dbs':
            obj_ = cross_ref_dbsType.factory(parent_object_=self)
            obj_.build(child_, gds_collector_=gds_collector_)
            self.cross_ref_dbs = obj_
            obj_.original_tagname_ = 'cross_ref_dbs'
        elif nodeName_ == 'supramolecules':
            obj_ = supramoleculesType.factory(parent_object_=self)
            obj_.build(child_, gds_collector_=gds_collector_)
            self.supramolecules = obj_
            obj_.original_tagname_ = 'supramolecules'
        elif nodeName_ == 'macromolecules':
            obj_ = macromoleculesType.factory(parent_object_=self)
            obj_.build(child_, gds_collector_=gds_collector_)
            self.macromolecules = obj_
            obj_.original_tagname_ = 'macromolecules'
# end class sampleType


class cross_ref_dbsType(GeneratedsSuper):
    """Annotations based on sample level"""
    __hash__ = GeneratedsSuper.__hash__
    subclass = None
    superclass = None
    def __init__(self, cross_ref_db=None, gds_collector_=None, **kwargs_):
        self.gds_collector_ = gds_collector_
        self.gds_elementtree_node_ = None
        self.original_tagname_ = None
        self.parent_object_ = kwargs_.get('parent_object_')
        self.ns_prefix_ = None
        if cross_ref_db is None:
            self.cross_ref_db = []
        else:
            self.cross_ref_db = cross_ref_db
        self.cross_ref_db_nsprefix_ = None
    def factory(*args_, **kwargs_):
        if CurrentSubclassModule_ is not None:
            subclass = getSubclassFromModule_(
                CurrentSubclassModule_, cross_ref_dbsType)
            if subclass is not None:
                return subclass(*args_, **kwargs_)
        if cross_ref_dbsType.subclass:
            return cross_ref_dbsType.subclass(*args_, **kwargs_)
        else:
            return cross_ref_dbsType(*args_, **kwargs_)
    factory = staticmethod(factory)
    def get_ns_prefix_(self):
        return self.ns_prefix_
    def set_ns_prefix_(self, ns_prefix):
        self.ns_prefix_ = ns_prefix
    def get_cross_ref_db(self):
        return self.cross_ref_db
    def set_cross_ref_db(self, cross_ref_db):
        self.cross_ref_db = cross_ref_db
    def add_cross_ref_db(self, value):
        self.cross_ref_db.append(value)
    def insert_cross_ref_db_at(self, index, value):
        self.cross_ref_db.insert(index, value)
    def replace_cross_ref_db_at(self, index, value):
        self.cross_ref_db[index] = value
    def hasContent_(self):
        if (
            self.cross_ref_db
        ):
            return True
        else:
            return False
    def export(self, outfile, level, namespaceprefix_='', namespacedef_='', name_='cross_ref_dbsType', pretty_print=True):
        imported_ns_def_ = GenerateDSNamespaceDefs_.get('cross_ref_dbsType')
        if imported_ns_def_ is not None:
            namespacedef_ = imported_ns_def_
        if pretty_print:
            eol_ = '\n'
        else:
            eol_ = ''
        if self.original_tagname_ is not None and name_ == 'cross_ref_dbsType':
            name_ = self.original_tagname_
        if UseCapturedNS_ and self.ns_prefix_:
            namespaceprefix_ = self.ns_prefix_ + ':'
        showIndent(outfile, level, pretty_print)
        outfile.write('<%s%s%s' % (namespaceprefix_, name_, namespacedef_ and ' ' + namespacedef_ or '', ))
        already_processed = set()
        self.exportAttributes(outfile, level, already_processed, namespaceprefix_, name_='cross_ref_dbsType')
        if self.hasContent_():
            outfile.write('>%s' % (eol_, ))
            self.exportChildren(outfile, level + 1, namespaceprefix_, namespacedef_, name_='cross_ref_dbsType', pretty_print=pretty_print)
            showIndent(outfile, level, pretty_print)
            outfile.write('</%s%s>%s' % (namespaceprefix_, name_, eol_))
        else:
            outfile.write('/>%s' % (eol_, ))
    def exportAttributes(self, outfile, level, already_processed, namespaceprefix_='', name_='cross_ref_dbsType'):
        pass
    def exportChildren(self, outfile, level, namespaceprefix_='', namespacedef_='', name_='cross_ref_dbsType', fromsubclass_=False, pretty_print=True):
        if pretty_print:
            eol_ = '\n'
        else:
            eol_ = ''
        for cross_ref_db_ in self.cross_ref_db:
            namespaceprefix_ = self.cross_ref_db_nsprefix_ + ':' if (UseCapturedNS_ and self.cross_ref_db_nsprefix_) else ''
            cross_ref_db_.export(outfile, level, namespaceprefix_, namespacedef_='', name_='cross_ref_db', pretty_print=pretty_print)
    def build(self, node, gds_collector_=None):
        self.gds_collector_ = gds_collector_
        if SaveElementTreeNode:
            self.gds_elementtree_node_ = node
        already_processed = set()
        self.ns_prefix_ = node.prefix
        self.buildAttributes(node, node.attrib, already_processed)
        for child in node:
            nodeName_ = Tag_pattern_.match(child.tag).groups()[-1]
            self.buildChildren(child, node, nodeName_, gds_collector_=gds_collector_)
        return self
    def buildAttributes(self, node, attrs, already_processed):
        pass
    def buildChildren(self, child_, node, nodeName_, fromsubclass_=False, gds_collector_=None):
        if nodeName_ == 'cross_ref_db':
            obj_ = cross_ref_db.factory(parent_object_=self)
            obj_.build(child_, gds_collector_=gds_collector_)
            self.cross_ref_db.append(obj_)
            obj_.original_tagname_ = 'cross_ref_db'
# end class cross_ref_dbsType


class supramoleculesType(GeneratedsSuper):
    """Supramolecule annotations"""
    __hash__ = GeneratedsSuper.__hash__
    subclass = None
    superclass = None
    def __init__(self, supramolecule=None, gds_collector_=None, **kwargs_):
        self.gds_collector_ = gds_collector_
        self.gds_elementtree_node_ = None
        self.original_tagname_ = None
        self.parent_object_ = kwargs_.get('parent_object_')
        self.ns_prefix_ = None
        if supramolecule is None:
            self.supramolecule = []
        else:
            self.supramolecule = supramolecule
        self.supramolecule_nsprefix_ = None
    def factory(*args_, **kwargs_):
        if CurrentSubclassModule_ is not None:
            subclass = getSubclassFromModule_(
                CurrentSubclassModule_, supramoleculesType)
            if subclass is not None:
                return subclass(*args_, **kwargs_)
        if supramoleculesType.subclass:
            return supramoleculesType.subclass(*args_, **kwargs_)
        else:
            return supramoleculesType(*args_, **kwargs_)
    factory = staticmethod(factory)
    def get_ns_prefix_(self):
        return self.ns_prefix_
    def set_ns_prefix_(self, ns_prefix):
        self.ns_prefix_ = ns_prefix
    def get_supramolecule(self):
        return self.supramolecule
    def set_supramolecule(self, supramolecule):
        self.supramolecule = supramolecule
    def add_supramolecule(self, value):
        self.supramolecule.append(value)
    def insert_supramolecule_at(self, index, value):
        self.supramolecule.insert(index, value)
    def replace_supramolecule_at(self, index, value):
        self.supramolecule[index] = value
    def hasContent_(self):
        if (
            self.supramolecule
        ):
            return True
        else:
            return False
    def export(self, outfile, level, namespaceprefix_='', namespacedef_='', name_='supramoleculesType', pretty_print=True):
        imported_ns_def_ = GenerateDSNamespaceDefs_.get('supramoleculesType')
        if imported_ns_def_ is not None:
            namespacedef_ = imported_ns_def_
        if pretty_print:
            eol_ = '\n'
        else:
            eol_ = ''
        if self.original_tagname_ is not None and name_ == 'supramoleculesType':
            name_ = self.original_tagname_
        if UseCapturedNS_ and self.ns_prefix_:
            namespaceprefix_ = self.ns_prefix_ + ':'
        showIndent(outfile, level, pretty_print)
        outfile.write('<%s%s%s' % (namespaceprefix_, name_, namespacedef_ and ' ' + namespacedef_ or '', ))
        already_processed = set()
        self.exportAttributes(outfile, level, already_processed, namespaceprefix_, name_='supramoleculesType')
        if self.hasContent_():
            outfile.write('>%s' % (eol_, ))
            self.exportChildren(outfile, level + 1, namespaceprefix_, namespacedef_, name_='supramoleculesType', pretty_print=pretty_print)
            showIndent(outfile, level, pretty_print)
            outfile.write('</%s%s>%s' % (namespaceprefix_, name_, eol_))
        else:
            outfile.write('/>%s' % (eol_, ))
    def exportAttributes(self, outfile, level, already_processed, namespaceprefix_='', name_='supramoleculesType'):
        pass
    def exportChildren(self, outfile, level, namespaceprefix_='', namespacedef_='', name_='supramoleculesType', fromsubclass_=False, pretty_print=True):
        if pretty_print:
            eol_ = '\n'
        else:
            eol_ = ''
        for supramolecule_ in self.supramolecule:
            namespaceprefix_ = self.supramolecule_nsprefix_ + ':' if (UseCapturedNS_ and self.supramolecule_nsprefix_) else ''
            supramolecule_.export(outfile, level, namespaceprefix_, namespacedef_='', name_='supramolecule', pretty_print=pretty_print)
    def build(self, node, gds_collector_=None):
        self.gds_collector_ = gds_collector_
        if SaveElementTreeNode:
            self.gds_elementtree_node_ = node
        already_processed = set()
        self.ns_prefix_ = node.prefix
        self.buildAttributes(node, node.attrib, already_processed)
        for child in node:
            nodeName_ = Tag_pattern_.match(child.tag).groups()[-1]
            self.buildChildren(child, node, nodeName_, gds_collector_=gds_collector_)
        return self
    def buildAttributes(self, node, attrs, already_processed):
        pass
    def buildChildren(self, child_, node, nodeName_, fromsubclass_=False, gds_collector_=None):
        if nodeName_ == 'supramolecule':
            obj_ = supramoleculeType.factory(parent_object_=self)
            obj_.build(child_, gds_collector_=gds_collector_)
            self.supramolecule.append(obj_)
            obj_.original_tagname_ = 'supramolecule'
# end class supramoleculesType


class supramoleculeType(GeneratedsSuper):
    """Annotations for a supramolecularSample type either
    Complexes/proteinsSupramolecule id numberNumber of supramolecules
    Resource supramolecular cross reference is fetched"""
    __hash__ = GeneratedsSuper.__hash__
    subclass = None
    superclass = None
    def __init__(self, type_=None, id=None, copies=None, provenance=None, name=None, cross_ref_dbs=None, gds_collector_=None, **kwargs_):
        self.gds_collector_ = gds_collector_
        self.gds_elementtree_node_ = None
        self.original_tagname_ = None
        self.parent_object_ = kwargs_.get('parent_object_')
        self.ns_prefix_ = None
        self.type_ = _cast(None, type_)
        self.type__nsprefix_ = None
        self.id = _cast(int, id)
        self.id_nsprefix_ = None
        self.copies = _cast(int, copies)
        self.copies_nsprefix_ = None
        self.provenance = _cast(None, provenance)
        self.provenance_nsprefix_ = None
        self.name = name
        self.name_nsprefix_ = None
        self.cross_ref_dbs = cross_ref_dbs
        self.cross_ref_dbs_nsprefix_ = None
    def factory(*args_, **kwargs_):
        if CurrentSubclassModule_ is not None:
            subclass = getSubclassFromModule_(
                CurrentSubclassModule_, supramoleculeType)
            if subclass is not None:
                return subclass(*args_, **kwargs_)
        if supramoleculeType.subclass:
            return supramoleculeType.subclass(*args_, **kwargs_)
        else:
            return supramoleculeType(*args_, **kwargs_)
    factory = staticmethod(factory)
    def get_ns_prefix_(self):
        return self.ns_prefix_
    def set_ns_prefix_(self, ns_prefix):
        self.ns_prefix_ = ns_prefix
    def get_name(self):
        return self.name
    def set_name(self, name):
        self.name = name
    def get_cross_ref_dbs(self):
        return self.cross_ref_dbs
    def set_cross_ref_dbs(self, cross_ref_dbs):
        self.cross_ref_dbs = cross_ref_dbs
    def get_type(self):
        return self.type_
    def set_type(self, type_):
        self.type_ = type_
    def get_id(self):
        return self.id
    def set_id(self, id):
        self.id = id
    def get_copies(self):
        return self.copies
    def set_copies(self, copies):
        self.copies = copies
    def get_provenance(self):
        return self.provenance
    def set_provenance(self, provenance):
        self.provenance = provenance
    def validate_sample_kind(self, value):
        # Validate type sample_kind, a restriction on xsd:token.
        if value is not None and Validate_simpletypes_ and self.gds_collector_ is not None:
            if not isinstance(value, str):
                lineno = self.gds_get_node_lineno_()
                self.gds_collector_.add_message('Value "%(value)s"%(lineno)s is not of the correct base simple type (str)' % {"value": value, "lineno": lineno, })
                return False
            value = value
            enumerations = ['complex', 'protein', 'ligand', 'rna']
            if value not in enumerations:
                lineno = self.gds_get_node_lineno_()
                self.gds_collector_.add_message('Value "%(value)s"%(lineno)s does not match xsd enumeration restriction on sample_kind' % {"value" : encode_str_2_3(value), "lineno": lineno} )
                result = False
    def validate_provenance_type(self, value):
        # Validate type provenance_type, a restriction on xsd:token.
        if value is not None and Validate_simpletypes_ and self.gds_collector_ is not None:
            if not isinstance(value, str):
                lineno = self.gds_get_node_lineno_()
                self.gds_collector_.add_message('Value "%(value)s"%(lineno)s is not of the correct base simple type (str)' % {"value": value, "lineno": lineno, })
                return False
            value = value
            enumerations = ['EMDB', 'UniProt', 'UniProtKB', 'PDBe', 'PDBe-KB', 'AlphaFold DB', 'EMPIAR', 'EuropePMC', 'Complex Portal', 'ChEMBL', 'ChEBI', 'DrugBank', 'PDBe-CCD', 'PubMed', 'PubMed Central', 'ISSN', 'DOI', 'GO', 'InterPro', 'Pfam', 'CATH', 'SCOP', 'SCOP2', 'SCOP2B', 'Rfam']
            if value not in enumerations:
                lineno = self.gds_get_node_lineno_()
                self.gds_collector_.add_message('Value "%(value)s"%(lineno)s does not match xsd enumeration restriction on provenance_type' % {"value" : encode_str_2_3(value), "lineno": lineno} )
                result = False
    def hasContent_(self):
        if (
            self.name is not None or
            self.cross_ref_dbs is not None
        ):
            return True
        else:
            return False
    def export(self, outfile, level, namespaceprefix_='', namespacedef_='', name_='supramoleculeType', pretty_print=True):
        imported_ns_def_ = GenerateDSNamespaceDefs_.get('supramoleculeType')
        if imported_ns_def_ is not None:
            namespacedef_ = imported_ns_def_
        if pretty_print:
            eol_ = '\n'
        else:
            eol_ = ''
        if self.original_tagname_ is not None and name_ == 'supramoleculeType':
            name_ = self.original_tagname_
        if UseCapturedNS_ and self.ns_prefix_:
            namespaceprefix_ = self.ns_prefix_ + ':'
        showIndent(outfile, level, pretty_print)
        outfile.write('<%s%s%s' % (namespaceprefix_, name_, namespacedef_ and ' ' + namespacedef_ or '', ))
        already_processed = set()
        self.exportAttributes(outfile, level, already_processed, namespaceprefix_, name_='supramoleculeType')
        if self.hasContent_():
            outfile.write('>%s' % (eol_, ))
            self.exportChildren(outfile, level + 1, namespaceprefix_, namespacedef_, name_='supramoleculeType', pretty_print=pretty_print)
            showIndent(outfile, level, pretty_print)
            outfile.write('</%s%s>%s' % (namespaceprefix_, name_, eol_))
        else:
            outfile.write('/>%s' % (eol_, ))
    def exportAttributes(self, outfile, level, already_processed, namespaceprefix_='', name_='supramoleculeType'):
        if self.type_ is not None and 'type_' not in already_processed:
            already_processed.add('type_')
            outfile.write(' type=%s' % (self.gds_encode(self.gds_format_string(quote_attrib(self.type_), input_name='type')), ))
        if self.id is not None and 'id' not in already_processed:
            already_processed.add('id')
            outfile.write(' id="%s"' % self.gds_format_integer(self.id, input_name='id'))
        if self.copies is not None and 'copies' not in already_processed:
            already_processed.add('copies')
            outfile.write(' copies="%s"' % self.gds_format_integer(self.copies, input_name='copies'))
        if self.provenance is not None and 'provenance' not in already_processed:
            already_processed.add('provenance')
            outfile.write(' provenance=%s' % (self.gds_encode(self.gds_format_string(quote_attrib(self.provenance), input_name='provenance')), ))
    def exportChildren(self, outfile, level, namespaceprefix_='', namespacedef_='', name_='supramoleculeType', fromsubclass_=False, pretty_print=True):
        if pretty_print:
            eol_ = '\n'
        else:
            eol_ = ''
        if self.name is not None:
            namespaceprefix_ = self.name_nsprefix_ + ':' if (UseCapturedNS_ and self.name_nsprefix_) else ''
            showIndent(outfile, level, pretty_print)
            outfile.write('<%sname>%s</%sname>%s' % (namespaceprefix_ , self.gds_encode(self.gds_format_string(quote_xml(self.name), input_name='name')), namespaceprefix_ , eol_))
        if self.cross_ref_dbs is not None:
            namespaceprefix_ = self.cross_ref_dbs_nsprefix_ + ':' if (UseCapturedNS_ and self.cross_ref_dbs_nsprefix_) else ''
            self.cross_ref_dbs.export(outfile, level, namespaceprefix_, namespacedef_='', name_='cross_ref_dbs', pretty_print=pretty_print)
    def build(self, node, gds_collector_=None):
        self.gds_collector_ = gds_collector_
        if SaveElementTreeNode:
            self.gds_elementtree_node_ = node
        already_processed = set()
        self.ns_prefix_ = node.prefix
        self.buildAttributes(node, node.attrib, already_processed)
        for child in node:
            nodeName_ = Tag_pattern_.match(child.tag).groups()[-1]
            self.buildChildren(child, node, nodeName_, gds_collector_=gds_collector_)
        return self
    def buildAttributes(self, node, attrs, already_processed):
        value = find_attr_value_('type', node)
        if value is not None and 'type' not in already_processed:
            already_processed.add('type')
            self.type_ = value
            self.type_ = ' '.join(self.type_.split())
            self.validate_sample_kind(self.type_)    # validate type sample_kind
        value = find_attr_value_('id', node)
        if value is not None and 'id' not in already_processed:
            already_processed.add('id')
            self.id = self.gds_parse_integer(value, node, 'id')
            if self.id < 0:
                raise_parse_error(node, 'Invalid NonNegativeInteger')
        value = find_attr_value_('copies', node)
        if value is not None and 'copies' not in already_processed:
            already_processed.add('copies')
            self.copies = self.gds_parse_integer(value, node, 'copies')
            if self.copies < 0:
                raise_parse_error(node, 'Invalid NonNegativeInteger')
        value = find_attr_value_('provenance', node)
        if value is not None and 'provenance' not in already_processed:
            already_processed.add('provenance')
            self.provenance = value
            self.provenance = ' '.join(self.provenance.split())
            self.validate_provenance_type(self.provenance)    # validate type provenance_type
    def buildChildren(self, child_, node, nodeName_, fromsubclass_=False, gds_collector_=None):
        if nodeName_ == 'name':
            value_ = child_.text
            value_ = self.gds_parse_string(value_, node, 'name')
            value_ = self.gds_validate_string(value_, node, 'name')
            self.name = value_
            self.name_nsprefix_ = child_.prefix
        elif nodeName_ == 'cross_ref_dbs':
            obj_ = cross_ref_dbsType1.factory(parent_object_=self)
            obj_.build(child_, gds_collector_=gds_collector_)
            self.cross_ref_dbs = obj_
            obj_.original_tagname_ = 'cross_ref_dbs'
# end class supramoleculeType


class cross_ref_dbsType1(GeneratedsSuper):
    """List of cross references for supramolecule"""
    __hash__ = GeneratedsSuper.__hash__
    subclass = None
    superclass = None
    def __init__(self, cross_ref_db=None, gds_collector_=None, **kwargs_):
        self.gds_collector_ = gds_collector_
        self.gds_elementtree_node_ = None
        self.original_tagname_ = None
        self.parent_object_ = kwargs_.get('parent_object_')
        self.ns_prefix_ = None
        if cross_ref_db is None:
            self.cross_ref_db = []
        else:
            self.cross_ref_db = cross_ref_db
        self.cross_ref_db_nsprefix_ = None
    def factory(*args_, **kwargs_):
        if CurrentSubclassModule_ is not None:
            subclass = getSubclassFromModule_(
                CurrentSubclassModule_, cross_ref_dbsType1)
            if subclass is not None:
                return subclass(*args_, **kwargs_)
        if cross_ref_dbsType1.subclass:
            return cross_ref_dbsType1.subclass(*args_, **kwargs_)
        else:
            return cross_ref_dbsType1(*args_, **kwargs_)
    factory = staticmethod(factory)
    def get_ns_prefix_(self):
        return self.ns_prefix_
    def set_ns_prefix_(self, ns_prefix):
        self.ns_prefix_ = ns_prefix
    def get_cross_ref_db(self):
        return self.cross_ref_db
    def set_cross_ref_db(self, cross_ref_db):
        self.cross_ref_db = cross_ref_db
    def add_cross_ref_db(self, value):
        self.cross_ref_db.append(value)
    def insert_cross_ref_db_at(self, index, value):
        self.cross_ref_db.insert(index, value)
    def replace_cross_ref_db_at(self, index, value):
        self.cross_ref_db[index] = value
    def hasContent_(self):
        if (
            self.cross_ref_db
        ):
            return True
        else:
            return False
    def export(self, outfile, level, namespaceprefix_='', namespacedef_='', name_='cross_ref_dbsType1', pretty_print=True):
        imported_ns_def_ = GenerateDSNamespaceDefs_.get('cross_ref_dbsType1')
        if imported_ns_def_ is not None:
            namespacedef_ = imported_ns_def_
        if pretty_print:
            eol_ = '\n'
        else:
            eol_ = ''
        if self.original_tagname_ is not None and name_ == 'cross_ref_dbsType1':
            name_ = self.original_tagname_
        if UseCapturedNS_ and self.ns_prefix_:
            namespaceprefix_ = self.ns_prefix_ + ':'
        showIndent(outfile, level, pretty_print)
        outfile.write('<%s%s%s' % (namespaceprefix_, name_, namespacedef_ and ' ' + namespacedef_ or '', ))
        already_processed = set()
        self.exportAttributes(outfile, level, already_processed, namespaceprefix_, name_='cross_ref_dbsType1')
        if self.hasContent_():
            outfile.write('>%s' % (eol_, ))
            self.exportChildren(outfile, level + 1, namespaceprefix_, namespacedef_, name_='cross_ref_dbsType1', pretty_print=pretty_print)
            showIndent(outfile, level, pretty_print)
            outfile.write('</%s%s>%s' % (namespaceprefix_, name_, eol_))
        else:
            outfile.write('/>%s' % (eol_, ))
    def exportAttributes(self, outfile, level, already_processed, namespaceprefix_='', name_='cross_ref_dbsType1'):
        pass
    def exportChildren(self, outfile, level, namespaceprefix_='', namespacedef_='', name_='cross_ref_dbsType1', fromsubclass_=False, pretty_print=True):
        if pretty_print:
            eol_ = '\n'
        else:
            eol_ = ''
        for cross_ref_db_ in self.cross_ref_db:
            namespaceprefix_ = self.cross_ref_db_nsprefix_ + ':' if (UseCapturedNS_ and self.cross_ref_db_nsprefix_) else ''
            cross_ref_db_.export(outfile, level, namespaceprefix_, namespacedef_='', name_='cross_ref_db', pretty_print=pretty_print)
    def build(self, node, gds_collector_=None):
        self.gds_collector_ = gds_collector_
        if SaveElementTreeNode:
            self.gds_elementtree_node_ = node
        already_processed = set()
        self.ns_prefix_ = node.prefix
        self.buildAttributes(node, node.attrib, already_processed)
        for child in node:
            nodeName_ = Tag_pattern_.match(child.tag).groups()[-1]
            self.buildChildren(child, node, nodeName_, gds_collector_=gds_collector_)
        return self
    def buildAttributes(self, node, attrs, already_processed):
        pass
    def buildChildren(self, child_, node, nodeName_, fromsubclass_=False, gds_collector_=None):
        if nodeName_ == 'cross_ref_db':
            obj_ = cross_ref_db.factory(parent_object_=self)
            obj_.build(child_, gds_collector_=gds_collector_)
            self.cross_ref_db.append(obj_)
            obj_.original_tagname_ = 'cross_ref_db'
# end class cross_ref_dbsType1


class macromoleculesType(GeneratedsSuper):
    """Macromolecule annotations"""
    __hash__ = GeneratedsSuper.__hash__
    subclass = None
    superclass = None
    def __init__(self, macromolecule=None, gds_collector_=None, **kwargs_):
        self.gds_collector_ = gds_collector_
        self.gds_elementtree_node_ = None
        self.original_tagname_ = None
        self.parent_object_ = kwargs_.get('parent_object_')
        self.ns_prefix_ = None
        if macromolecule is None:
            self.macromolecule = []
        else:
            self.macromolecule = macromolecule
        self.macromolecule_nsprefix_ = None
    def factory(*args_, **kwargs_):
        if CurrentSubclassModule_ is not None:
            subclass = getSubclassFromModule_(
                CurrentSubclassModule_, macromoleculesType)
            if subclass is not None:
                return subclass(*args_, **kwargs_)
        if macromoleculesType.subclass:
            return macromoleculesType.subclass(*args_, **kwargs_)
        else:
            return macromoleculesType(*args_, **kwargs_)
    factory = staticmethod(factory)
    def get_ns_prefix_(self):
        return self.ns_prefix_
    def set_ns_prefix_(self, ns_prefix):
        self.ns_prefix_ = ns_prefix
    def get_macromolecule(self):
        return self.macromolecule
    def set_macromolecule(self, macromolecule):
        self.macromolecule = macromolecule
    def add_macromolecule(self, value):
        self.macromolecule.append(value)
    def insert_macromolecule_at(self, index, value):
        self.macromolecule.insert(index, value)
    def replace_macromolecule_at(self, index, value):
        self.macromolecule[index] = value
    def hasContent_(self):
        if (
            self.macromolecule
        ):
            return True
        else:
            return False
    def export(self, outfile, level, namespaceprefix_='', namespacedef_='', name_='macromoleculesType', pretty_print=True):
        imported_ns_def_ = GenerateDSNamespaceDefs_.get('macromoleculesType')
        if imported_ns_def_ is not None:
            namespacedef_ = imported_ns_def_
        if pretty_print:
            eol_ = '\n'
        else:
            eol_ = ''
        if self.original_tagname_ is not None and name_ == 'macromoleculesType':
            name_ = self.original_tagname_
        if UseCapturedNS_ and self.ns_prefix_:
            namespaceprefix_ = self.ns_prefix_ + ':'
        showIndent(outfile, level, pretty_print)
        outfile.write('<%s%s%s' % (namespaceprefix_, name_, namespacedef_ and ' ' + namespacedef_ or '', ))
        already_processed = set()
        self.exportAttributes(outfile, level, already_processed, namespaceprefix_, name_='macromoleculesType')
        if self.hasContent_():
            outfile.write('>%s' % (eol_, ))
            self.exportChildren(outfile, level + 1, namespaceprefix_, namespacedef_, name_='macromoleculesType', pretty_print=pretty_print)
            showIndent(outfile, level, pretty_print)
            outfile.write('</%s%s>%s' % (namespaceprefix_, name_, eol_))
        else:
            outfile.write('/>%s' % (eol_, ))
    def exportAttributes(self, outfile, level, already_processed, namespaceprefix_='', name_='macromoleculesType'):
        pass
    def exportChildren(self, outfile, level, namespaceprefix_='', namespacedef_='', name_='macromoleculesType', fromsubclass_=False, pretty_print=True):
        if pretty_print:
            eol_ = '\n'
        else:
            eol_ = ''
        for macromolecule_ in self.macromolecule:
            namespaceprefix_ = self.macromolecule_nsprefix_ + ':' if (UseCapturedNS_ and self.macromolecule_nsprefix_) else ''
            macromolecule_.export(outfile, level, namespaceprefix_, namespacedef_='', name_='macromolecule', pretty_print=pretty_print)
    def build(self, node, gds_collector_=None):
        self.gds_collector_ = gds_collector_
        if SaveElementTreeNode:
            self.gds_elementtree_node_ = node
        already_processed = set()
        self.ns_prefix_ = node.prefix
        self.buildAttributes(node, node.attrib, already_processed)
        for child in node:
            nodeName_ = Tag_pattern_.match(child.tag).groups()[-1]
            self.buildChildren(child, node, nodeName_, gds_collector_=gds_collector_)
        return self
    def buildAttributes(self, node, attrs, already_processed):
        pass
    def buildChildren(self, child_, node, nodeName_, fromsubclass_=False, gds_collector_=None):
        if nodeName_ == 'macromolecule':
            obj_ = macromoleculeType.factory(parent_object_=self)
            obj_.build(child_, gds_collector_=gds_collector_)
            self.macromolecule.append(obj_)
            obj_.original_tagname_ = 'macromolecule'
# end class macromoleculesType


class macromoleculeType(GeneratedsSuper):
    """Annotations for a macromolecularSample type either
    proteins/ligandsMacromolecule id numberNumber of copies of the
    macromolecular in the sampleResource macromolecular cross reference is
    fetched"""
    __hash__ = GeneratedsSuper.__hash__
    subclass = None
    superclass = None
    def __init__(self, type_=None, id=None, copies=None, provenance=None, name=None, ccd_id=None, cross_ref_dbs=None, gds_collector_=None, **kwargs_):
        self.gds_collector_ = gds_collector_
        self.gds_elementtree_node_ = None
        self.original_tagname_ = None
        self.parent_object_ = kwargs_.get('parent_object_')
        self.ns_prefix_ = None
        self.type_ = _cast(None, type_)
        self.type__nsprefix_ = None
        self.id = _cast(int, id)
        self.id_nsprefix_ = None
        self.copies = _cast(int, copies)
        self.copies_nsprefix_ = None
        self.provenance = _cast(None, provenance)
        self.provenance_nsprefix_ = None
        self.name = name
        self.name_nsprefix_ = None
        self.ccd_id = ccd_id
        self.ccd_id_nsprefix_ = None
        self.cross_ref_dbs = cross_ref_dbs
        self.cross_ref_dbs_nsprefix_ = None
    def factory(*args_, **kwargs_):
        if CurrentSubclassModule_ is not None:
            subclass = getSubclassFromModule_(
                CurrentSubclassModule_, macromoleculeType)
            if subclass is not None:
                return subclass(*args_, **kwargs_)
        if macromoleculeType.subclass:
            return macromoleculeType.subclass(*args_, **kwargs_)
        else:
            return macromoleculeType(*args_, **kwargs_)
    factory = staticmethod(factory)
    def get_ns_prefix_(self):
        return self.ns_prefix_
    def set_ns_prefix_(self, ns_prefix):
        self.ns_prefix_ = ns_prefix
    def get_name(self):
        return self.name
    def set_name(self, name):
        self.name = name
    def get_ccd_id(self):
        return self.ccd_id
    def set_ccd_id(self, ccd_id):
        self.ccd_id = ccd_id
    def get_cross_ref_dbs(self):
        return self.cross_ref_dbs
    def set_cross_ref_dbs(self, cross_ref_dbs):
        self.cross_ref_dbs = cross_ref_dbs
    def get_type(self):
        return self.type_
    def set_type(self, type_):
        self.type_ = type_
    def get_id(self):
        return self.id
    def set_id(self, id):
        self.id = id
    def get_copies(self):
        return self.copies
    def set_copies(self, copies):
        self.copies = copies
    def get_provenance(self):
        return self.provenance
    def set_provenance(self, provenance):
        self.provenance = provenance
    def validate_sample_kind(self, value):
        # Validate type sample_kind, a restriction on xsd:token.
        if value is not None and Validate_simpletypes_ and self.gds_collector_ is not None:
            if not isinstance(value, str):
                lineno = self.gds_get_node_lineno_()
                self.gds_collector_.add_message('Value "%(value)s"%(lineno)s is not of the correct base simple type (str)' % {"value": value, "lineno": lineno, })
                return False
            value = value
            enumerations = ['complex', 'protein', 'ligand', 'rna']
            if value not in enumerations:
                lineno = self.gds_get_node_lineno_()
                self.gds_collector_.add_message('Value "%(value)s"%(lineno)s does not match xsd enumeration restriction on sample_kind' % {"value" : encode_str_2_3(value), "lineno": lineno} )
                result = False
    def validate_provenance_type(self, value):
        # Validate type provenance_type, a restriction on xsd:token.
        if value is not None and Validate_simpletypes_ and self.gds_collector_ is not None:
            if not isinstance(value, str):
                lineno = self.gds_get_node_lineno_()
                self.gds_collector_.add_message('Value "%(value)s"%(lineno)s is not of the correct base simple type (str)' % {"value": value, "lineno": lineno, })
                return False
            value = value
            enumerations = ['EMDB', 'UniProt', 'UniProtKB', 'PDBe', 'PDBe-KB', 'AlphaFold DB', 'EMPIAR', 'EuropePMC', 'Complex Portal', 'ChEMBL', 'ChEBI', 'DrugBank', 'PDBe-CCD', 'PubMed', 'PubMed Central', 'ISSN', 'DOI', 'GO', 'InterPro', 'Pfam', 'CATH', 'SCOP', 'SCOP2', 'SCOP2B', 'Rfam']
            if value not in enumerations:
                lineno = self.gds_get_node_lineno_()
                self.gds_collector_.add_message('Value "%(value)s"%(lineno)s does not match xsd enumeration restriction on provenance_type' % {"value" : encode_str_2_3(value), "lineno": lineno} )
                result = False
    def hasContent_(self):
        if (
            self.name is not None or
            self.ccd_id is not None or
            self.cross_ref_dbs is not None
        ):
            return True
        else:
            return False
    def export(self, outfile, level, namespaceprefix_='', namespacedef_='', name_='macromoleculeType', pretty_print=True):
        imported_ns_def_ = GenerateDSNamespaceDefs_.get('macromoleculeType')
        if imported_ns_def_ is not None:
            namespacedef_ = imported_ns_def_
        if pretty_print:
            eol_ = '\n'
        else:
            eol_ = ''
        if self.original_tagname_ is not None and name_ == 'macromoleculeType':
            name_ = self.original_tagname_
        if UseCapturedNS_ and self.ns_prefix_:
            namespaceprefix_ = self.ns_prefix_ + ':'
        showIndent(outfile, level, pretty_print)
        outfile.write('<%s%s%s' % (namespaceprefix_, name_, namespacedef_ and ' ' + namespacedef_ or '', ))
        already_processed = set()
        self.exportAttributes(outfile, level, already_processed, namespaceprefix_, name_='macromoleculeType')
        if self.hasContent_():
            outfile.write('>%s' % (eol_, ))
            self.exportChildren(outfile, level + 1, namespaceprefix_, namespacedef_, name_='macromoleculeType', pretty_print=pretty_print)
            showIndent(outfile, level, pretty_print)
            outfile.write('</%s%s>%s' % (namespaceprefix_, name_, eol_))
        else:
            outfile.write('/>%s' % (eol_, ))
    def exportAttributes(self, outfile, level, already_processed, namespaceprefix_='', name_='macromoleculeType'):
        if self.type_ is not None and 'type_' not in already_processed:
            already_processed.add('type_')
            outfile.write(' type=%s' % (self.gds_encode(self.gds_format_string(quote_attrib(self.type_), input_name='type')), ))
        if self.id is not None and 'id' not in already_processed:
            already_processed.add('id')
            outfile.write(' id="%s"' % self.gds_format_integer(self.id, input_name='id'))
        if self.copies is not None and 'copies' not in already_processed:
            already_processed.add('copies')
            outfile.write(' copies="%s"' % self.gds_format_integer(self.copies, input_name='copies'))
        if self.provenance is not None and 'provenance' not in already_processed:
            already_processed.add('provenance')
            outfile.write(' provenance=%s' % (self.gds_encode(self.gds_format_string(quote_attrib(self.provenance), input_name='provenance')), ))
    def exportChildren(self, outfile, level, namespaceprefix_='', namespacedef_='', name_='macromoleculeType', fromsubclass_=False, pretty_print=True):
        if pretty_print:
            eol_ = '\n'
        else:
            eol_ = ''
        if self.name is not None:
            namespaceprefix_ = self.name_nsprefix_ + ':' if (UseCapturedNS_ and self.name_nsprefix_) else ''
            showIndent(outfile, level, pretty_print)
            outfile.write('<%sname>%s</%sname>%s' % (namespaceprefix_ , self.gds_encode(self.gds_format_string(quote_xml(self.name), input_name='name')), namespaceprefix_ , eol_))
        if self.ccd_id is not None:
            namespaceprefix_ = self.ccd_id_nsprefix_ + ':' if (UseCapturedNS_ and self.ccd_id_nsprefix_) else ''
            showIndent(outfile, level, pretty_print)
            outfile.write('<%sccd_id>%s</%sccd_id>%s' % (namespaceprefix_ , self.gds_encode(self.gds_format_string(quote_xml(self.ccd_id), input_name='ccd_id')), namespaceprefix_ , eol_))
        if self.cross_ref_dbs is not None:
            namespaceprefix_ = self.cross_ref_dbs_nsprefix_ + ':' if (UseCapturedNS_ and self.cross_ref_dbs_nsprefix_) else ''
            self.cross_ref_dbs.export(outfile, level, namespaceprefix_, namespacedef_='', name_='cross_ref_dbs', pretty_print=pretty_print)
    def build(self, node, gds_collector_=None):
        self.gds_collector_ = gds_collector_
        if SaveElementTreeNode:
            self.gds_elementtree_node_ = node
        already_processed = set()
        self.ns_prefix_ = node.prefix
        self.buildAttributes(node, node.attrib, already_processed)
        for child in node:
            nodeName_ = Tag_pattern_.match(child.tag).groups()[-1]
            self.buildChildren(child, node, nodeName_, gds_collector_=gds_collector_)
        return self
    def buildAttributes(self, node, attrs, already_processed):
        value = find_attr_value_('type', node)
        if value is not None and 'type' not in already_processed:
            already_processed.add('type')
            self.type_ = value
            self.type_ = ' '.join(self.type_.split())
            self.validate_sample_kind(self.type_)    # validate type sample_kind
        value = find_attr_value_('id', node)
        if value is not None and 'id' not in already_processed:
            already_processed.add('id')
            self.id = self.gds_parse_integer(value, node, 'id')
            if self.id < 0:
                raise_parse_error(node, 'Invalid NonNegativeInteger')
        value = find_attr_value_('copies', node)
        if value is not None and 'copies' not in already_processed:
            already_processed.add('copies')
            self.copies = self.gds_parse_integer(value, node, 'copies')
            if self.copies < 0:
                raise_parse_error(node, 'Invalid NonNegativeInteger')
        value = find_attr_value_('provenance', node)
        if value is not None and 'provenance' not in already_processed:
            already_processed.add('provenance')
            self.provenance = value
            self.provenance = ' '.join(self.provenance.split())
            self.validate_provenance_type(self.provenance)    # validate type provenance_type
    def buildChildren(self, child_, node, nodeName_, fromsubclass_=False, gds_collector_=None):
        if nodeName_ == 'name':
            value_ = child_.text
            value_ = self.gds_parse_string(value_, node, 'name')
            value_ = self.gds_validate_string(value_, node, 'name')
            self.name = value_
            self.name_nsprefix_ = child_.prefix
        elif nodeName_ == 'ccd_id':
            value_ = child_.text
            if value_:
                value_ = re_.sub(String_cleanup_pat_, " ", value_).strip()
            else:
                value_ = ""
            value_ = self.gds_parse_string(value_, node, 'ccd_id')
            value_ = self.gds_validate_string(value_, node, 'ccd_id')
            self.ccd_id = value_
            self.ccd_id_nsprefix_ = child_.prefix
        elif nodeName_ == 'cross_ref_dbs':
            obj_ = cross_ref_dbsType2.factory(parent_object_=self)
            obj_.build(child_, gds_collector_=gds_collector_)
            self.cross_ref_dbs = obj_
            obj_.original_tagname_ = 'cross_ref_dbs'
# end class macromoleculeType


class cross_ref_dbsType2(GeneratedsSuper):
    __hash__ = GeneratedsSuper.__hash__
    subclass = None
    superclass = None
    def __init__(self, cross_ref_db=None, gds_collector_=None, **kwargs_):
        self.gds_collector_ = gds_collector_
        self.gds_elementtree_node_ = None
        self.original_tagname_ = None
        self.parent_object_ = kwargs_.get('parent_object_')
        self.ns_prefix_ = None
        if cross_ref_db is None:
            self.cross_ref_db = []
        else:
            self.cross_ref_db = cross_ref_db
        self.cross_ref_db_nsprefix_ = None
    def factory(*args_, **kwargs_):
        if CurrentSubclassModule_ is not None:
            subclass = getSubclassFromModule_(
                CurrentSubclassModule_, cross_ref_dbsType2)
            if subclass is not None:
                return subclass(*args_, **kwargs_)
        if cross_ref_dbsType2.subclass:
            return cross_ref_dbsType2.subclass(*args_, **kwargs_)
        else:
            return cross_ref_dbsType2(*args_, **kwargs_)
    factory = staticmethod(factory)
    def get_ns_prefix_(self):
        return self.ns_prefix_
    def set_ns_prefix_(self, ns_prefix):
        self.ns_prefix_ = ns_prefix
    def get_cross_ref_db(self):
        return self.cross_ref_db
    def set_cross_ref_db(self, cross_ref_db):
        self.cross_ref_db = cross_ref_db
    def add_cross_ref_db(self, value):
        self.cross_ref_db.append(value)
    def insert_cross_ref_db_at(self, index, value):
        self.cross_ref_db.insert(index, value)
    def replace_cross_ref_db_at(self, index, value):
        self.cross_ref_db[index] = value
    def hasContent_(self):
        if (
            self.cross_ref_db
        ):
            return True
        else:
            return False
    def export(self, outfile, level, namespaceprefix_='', namespacedef_='', name_='cross_ref_dbsType2', pretty_print=True):
        imported_ns_def_ = GenerateDSNamespaceDefs_.get('cross_ref_dbsType2')
        if imported_ns_def_ is not None:
            namespacedef_ = imported_ns_def_
        if pretty_print:
            eol_ = '\n'
        else:
            eol_ = ''
        if self.original_tagname_ is not None and name_ == 'cross_ref_dbsType2':
            name_ = self.original_tagname_
        if UseCapturedNS_ and self.ns_prefix_:
            namespaceprefix_ = self.ns_prefix_ + ':'
        showIndent(outfile, level, pretty_print)
        outfile.write('<%s%s%s' % (namespaceprefix_, name_, namespacedef_ and ' ' + namespacedef_ or '', ))
        already_processed = set()
        self.exportAttributes(outfile, level, already_processed, namespaceprefix_, name_='cross_ref_dbsType2')
        if self.hasContent_():
            outfile.write('>%s' % (eol_, ))
            self.exportChildren(outfile, level + 1, namespaceprefix_, namespacedef_, name_='cross_ref_dbsType2', pretty_print=pretty_print)
            showIndent(outfile, level, pretty_print)
            outfile.write('</%s%s>%s' % (namespaceprefix_, name_, eol_))
        else:
            outfile.write('/>%s' % (eol_, ))
    def exportAttributes(self, outfile, level, already_processed, namespaceprefix_='', name_='cross_ref_dbsType2'):
        pass
    def exportChildren(self, outfile, level, namespaceprefix_='', namespacedef_='', name_='cross_ref_dbsType2', fromsubclass_=False, pretty_print=True):
        if pretty_print:
            eol_ = '\n'
        else:
            eol_ = ''
        for cross_ref_db_ in self.cross_ref_db:
            namespaceprefix_ = self.cross_ref_db_nsprefix_ + ':' if (UseCapturedNS_ and self.cross_ref_db_nsprefix_) else ''
            cross_ref_db_.export(outfile, level, namespaceprefix_, namespacedef_='', name_='cross_ref_db', pretty_print=pretty_print)
    def build(self, node, gds_collector_=None):
        self.gds_collector_ = gds_collector_
        if SaveElementTreeNode:
            self.gds_elementtree_node_ = node
        already_processed = set()
        self.ns_prefix_ = node.prefix
        self.buildAttributes(node, node.attrib, already_processed)
        for child in node:
            nodeName_ = Tag_pattern_.match(child.tag).groups()[-1]
            self.buildChildren(child, node, nodeName_, gds_collector_=gds_collector_)
        return self
    def buildAttributes(self, node, attrs, already_processed):
        pass
    def buildChildren(self, child_, node, nodeName_, fromsubclass_=False, gds_collector_=None):
        if nodeName_ == 'cross_ref_db':
            obj_ = cross_ref_db.factory(parent_object_=self)
            obj_.build(child_, gds_collector_=gds_collector_)
            self.cross_ref_db.append(obj_)
            obj_.original_tagname_ = 'cross_ref_db'
# end class cross_ref_dbsType2


GDSClassesMapping = {
}


USAGE_TEXT = """
Usage: python <Parser>.py [ -s ] <in_xml_file>
"""


def usage():
    print(USAGE_TEXT)
    sys.exit(1)


def get_root_tag(node):
    tag = Tag_pattern_.match(node.tag).groups()[-1]
    rootClass = GDSClassesMapping.get(tag)
    if rootClass is None:
        rootClass = globals().get(tag)
    return tag, rootClass


def get_required_ns_prefix_defs(rootNode):
    '''Get all name space prefix definitions required in this XML doc.
    Return a dictionary of definitions and a char string of definitions.
    '''
    nsmap = {
        prefix: uri
        for node in rootNode.iter()
        for (prefix, uri) in node.nsmap.items()
        if prefix is not None
    }
    namespacedefs = ' '.join([
        'xmlns:{}="{}"'.format(prefix, uri)
        for prefix, uri in nsmap.items()
    ])
    return nsmap, namespacedefs


def parse(inFileName, silence=False, print_warnings=True):
    global CapturedNsmap_
    gds_collector = GdsCollector_()
    parser = None
    doc = parsexml_(inFileName, parser)
    rootNode = doc.getroot()
    rootTag, rootClass = get_root_tag(rootNode)
    if rootClass is None:
        rootTag = 'emicss'
        rootClass = emicss
    rootObj = rootClass.factory()
    rootObj.build(rootNode, gds_collector_=gds_collector)
    CapturedNsmap_, namespacedefs = get_required_ns_prefix_defs(rootNode)
    if not SaveElementTreeNode:
        doc = None
        rootNode = None
    if not silence:
        sys.stdout.write('<?xml version="1.0" ?>\n')
        rootObj.export(
            sys.stdout, 0, name_=rootTag,
            namespacedef_=namespacedefs,
            pretty_print=True)
    if print_warnings and len(gds_collector.get_messages()) > 0:
        separator = ('-' * 50) + '\n'
        sys.stderr.write(separator)
        sys.stderr.write('----- Warnings -- count: {} -----\n'.format(
            len(gds_collector.get_messages()), ))
        gds_collector.write_messages(sys.stderr)
        sys.stderr.write(separator)
    return rootObj


def parseEtree(inFileName, silence=False, print_warnings=True,
               mapping=None, nsmap=None):
    parser = None
    doc = parsexml_(inFileName, parser)
    gds_collector = GdsCollector_()
    rootNode = doc.getroot()
    rootTag, rootClass = get_root_tag(rootNode)
    if rootClass is None:
        rootTag = 'emicss'
        rootClass = emicss
    rootObj = rootClass.factory()
    rootObj.build(rootNode, gds_collector_=gds_collector)
    # Enable Python to collect the space used by the DOM.
    if mapping is None:
        mapping = {}
    rootElement = rootObj.to_etree(
        None, name_=rootTag, mapping_=mapping, nsmap_=nsmap)
    reverse_mapping = rootObj.gds_reverse_node_mapping(mapping)
    if not SaveElementTreeNode:
        doc = None
        rootNode = None
    if not silence:
        content = etree_.tostring(
            rootElement, pretty_print=True,
            xml_declaration=True, encoding="utf-8")
        sys.stdout.write(str(content))
        sys.stdout.write('\n')
    if print_warnings and len(gds_collector.get_messages()) > 0:
        separator = ('-' * 50) + '\n'
        sys.stderr.write(separator)
        sys.stderr.write('----- Warnings -- count: {} -----\n'.format(
            len(gds_collector.get_messages()), ))
        gds_collector.write_messages(sys.stderr)
        sys.stderr.write(separator)
    return rootObj, rootElement, mapping, reverse_mapping


def parseString(inString, silence=False, print_warnings=True):
    '''Parse a string, create the object tree, and export it.

    Arguments:
    - inString -- A string.  This XML fragment should not start
      with an XML declaration containing an encoding.
    - silence -- A boolean.  If False, export the object.
    Returns -- The root object in the tree.
    '''
    parser = None
    rootNode= parsexmlstring_(inString, parser)
    gds_collector = GdsCollector_()
    rootTag, rootClass = get_root_tag(rootNode)
    if rootClass is None:
        rootTag = 'emicss'
        rootClass = emicss
    rootObj = rootClass.factory()
    rootObj.build(rootNode, gds_collector_=gds_collector)
    if not SaveElementTreeNode:
        rootNode = None
    if not silence:
        sys.stdout.write('<?xml version="1.0" ?>\n')
        rootObj.export(
            sys.stdout, 0, name_=rootTag,
            namespacedef_='')
    if print_warnings and len(gds_collector.get_messages()) > 0:
        separator = ('-' * 50) + '\n'
        sys.stderr.write(separator)
        sys.stderr.write('----- Warnings -- count: {} -----\n'.format(
            len(gds_collector.get_messages()), ))
        gds_collector.write_messages(sys.stderr)
        sys.stderr.write(separator)
    return rootObj


def parseLiteral(inFileName, silence=False, print_warnings=True):
    parser = None
    doc = parsexml_(inFileName, parser)
    gds_collector = GdsCollector_()
    rootNode = doc.getroot()
    rootTag, rootClass = get_root_tag(rootNode)
    if rootClass is None:
        rootTag = 'emicss'
        rootClass = emicss
    rootObj = rootClass.factory()
    rootObj.build(rootNode, gds_collector_=gds_collector)
    # Enable Python to collect the space used by the DOM.
    if not SaveElementTreeNode:
        doc = None
        rootNode = None
    if not silence:
        sys.stdout.write('#from EMICSS import *\n\n')
        sys.stdout.write('import EMICSS as model_\n\n')
        sys.stdout.write('rootObj = model_.rootClass(\n')
        rootObj.exportLiteral(sys.stdout, 0, name_=rootTag)
        sys.stdout.write(')\n')
    if print_warnings and len(gds_collector.get_messages()) > 0:
        separator = ('-' * 50) + '\n'
        sys.stderr.write(separator)
        sys.stderr.write('----- Warnings -- count: {} -----\n'.format(
            len(gds_collector.get_messages()), ))
        gds_collector.write_messages(sys.stderr)
        sys.stderr.write(separator)
    return rootObj


def main():
    args = sys.argv[1:]
    if len(args) == 1:
        parse(args[0])
    else:
        usage()


if __name__ == '__main__':
    #import pdb; pdb.set_trace()
    main()

RenameMappings_ = {
}

#
# Mapping of namespaces to types defined in them
# and the file in which each is defined.
# simpleTypes are marked "ST" and complexTypes "CT".
NamespaceToDefMappings_ = {}

__all__ = [
    "authorType",
    "authorsType",
    "cross_ref_db",
    "cross_ref_dbsType",
    "cross_ref_dbsType1",
    "cross_ref_dbsType2",
    "dbType",
    "dbsType",
    "emicss",
    "entry_ref_dbType",
    "entry_ref_dbsType",
    "macromoleculeType",
    "macromoleculesType",
    "primary_citationType",
    "ref_citationType",
    "sampleType",
    "supramoleculeType",
    "supramoleculesType",
    "weight_infoType",
    "weightsType"
]
