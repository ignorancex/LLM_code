
from collections import defaultdict, Counter
from datetime import date
from kopl.data import KB
from kopl.util import ValueClass, comp


class KoPLContext(object):
    '''
    Modified code from `kopl.KoPLEngine`.
    self._iterate is used for every "for" expression.
    '''

    def __init__(self, kb):
        self.kb = KB(kb)

    def _iterate(self, iterator):
        return iterator

    def _parse_key_value(self, key, value, typ=None):
        if typ is None:
            typ = self.kb.key_type[key]
        if typ=='string':
            value = ValueClass('string', value)
        elif typ=='quantity':
            if ' ' in value:
                vs = value.split()
                v = vs[0]
                unit = ' '.join(vs[1:])
            else:
                v = value
                unit = '1'
            value = ValueClass('quantity', float(v), unit)
        else:
            if '/' in value or ('-' in value and '-' != value[0]):
                split_char = '/' if '/' in value else '-'
                p1, p2 = value.find(split_char), value.rfind(split_char)
                y, m, d = int(value[:p1]), int(value[p1+1:p2]), int(value[p2+1:])
                value = ValueClass('date', date(y, m, d))
            else:
                value = ValueClass('year', int(value))
        return value

    def forward(self, program, inputs,
                ignore_error=False, show_details=False):
        memory = []
        program = ['<START>'] + program + ['<END>']
        inputs = [[]] + inputs + [[]]
        try:
            # infer the dependency based on the function definition
            dependency = []
            branch_stack = []
            for i, p in enumerate(program):
                if p in {'<START>', '<END>', '<PAD>'}:
                    dep = []
                elif p in {'FindAll', 'Find'}:
                    dep = []
                    branch_stack.append(i - 1)
                elif p in {'And', 'Or', 'SelectBetween', 'QueryRelation', 'QueryRelationQualifier'}:
                    dep = [branch_stack[-1], i-1]
                    branch_stack = branch_stack[:-1]
                else:
                    dep = [i-1]
                dependency.append(dep)

            memory = []
            for p, dep, inp in zip(program, dependency, inputs):
                if p == 'What':
                    p = 'QueryName'
                if p == '<START>':
                    res = None
                elif p == '<END>':
                    break
                else:
                    fun_args = [memory[x] for x in dep]
                    func = getattr(self, p)
                    res = func(*fun_args, *inp)

                memory.append(res)
                if show_details:
                    print(p, dep, inp)
                    print(res)
            return [str(_) for _ in memory[-1]] if isinstance(memory[-1], list) else str(memory[-1])
        except Exception as e:
            if ignore_error:
                return None
            else:
                raise

    def FindAll(self):
        """
        Return all entities in the knowledge base

        Returns:
            :obj:`tuple`: returns a tuple, the first member is a list of entities, and the second member is None
        """
        entity_ids = list(self.kb.entities.keys())
        return (entity_ids, None)

    def Find(self, name):
        """
        Find all entities with a specific name

        Args:
            name (string): the name of the entity
        Returns:
            :obj:`tuple`: returns a tuple, the first member is a list of entities, and the second member is None
        """
        entity_ids = self.kb.name_to_id[name]
        return (entity_ids, None)

    def FilterConcept(self, entities, concept_name):
        """
        Find all entities belonging to a specific concept

        Args:
            entities (tuple): 2-tuple, the first member is a list of entities, the second is a list of triples
            concept_name (string): The given concept label

        Returns:
            :obj:`tuple`: returns a tuple, the first member is the intersection of the input entity sets, and the second member is None
        """
        entity_ids, _ = entities
        concept_ids = self.kb.name_to_id[concept_name]
        entity_ids_2 = []
        for i in self._iterate(concept_ids):
            entity_ids_2 += self.kb.concept_to_entity.get(i, [])
        entity_ids = list(set(entity_ids) & set(entity_ids_2))
        return (entity_ids, None)

    def _filter_attribute(self, entity_ids, tgt_key, tgt_value, op, typ):
        tgt_value = self._parse_key_value(tgt_key, tgt_value, typ)
        res_ids = []
        res_facts = []
        entity_ids = set(entity_ids) & set(self.kb.attribute_inv_index[tgt_key].keys())
        for ent_id in self._iterate(entity_ids):
            for idx in self._iterate(self.kb.attribute_inv_index[tgt_key][ent_id]):
                attr_info = self.kb.entities[ent_id]['attributes'][idx]
                k, v = attr_info['key'], attr_info['value']
                if k==tgt_key and v.can_compare(tgt_value) and comp(v, tgt_value, op):
                    res_ids.append(ent_id)
                    res_facts.append(attr_info)
        return (res_ids, res_facts)

    def FilterStr(self, entities, key, value):
        """
        For attributes of string type, filter out all entities that meet the condition according to the key and value, and return the triplet of the entity and the corresponding attribute type

        Args:
            entities (tuple): 2-tuple, the first member is a list of entities, the second is a list of triples
            key (string): property key
            value (string): attribute value

        Returns:
            :obj:`tuple`: returns a tuple, the first member is a list of entities, and the second is a list of triples
        """
        entity_ids, _ = entities
        op = '='
        return self._filter_attribute(entity_ids, key, value, op, 'string')

    def FilterNum(self, entities, key, value, op):
        """
        For the attribute of numeric type, key and value specify the key value, op specifies the comparison operator, and returns the triplet of the entity and the corresponding attribute type

        Args:
            entities (tuple): 2-tuple, the first member is a list of entities, the second is a list of triples
            key (string): property key
            value (string): attribute value, which is a numeric type
            op (string): comparison operator, one of "=", "!=", "<", ">"

        Returns:
            :obj:`tuple`: returns a tuple, the first member is a list of entities, and the second is a list of triples
        """
        entity_ids, _ = entities
        return self._filter_attribute(entity_ids, key, value, op, 'quantity')

    def FilterYear(self, entities, key, value, op):
        """
        For the year type attribute, key and value specify the key value, op specifies the comparison operator, and returns the entity and the corresponding attribute type triplet

        Args:
            entities (tuple): 2-tuple, the first member is a list of entities, the second is a list of triples
            key (string): property key
            value (string): attribute value, which is the year type
            op (string): comparison operator, one of "=", "!=", "<", ">"

        Returns:
            :obj:`tuple`: returns a tuple, the first member is a list of entities, and the second is a list of triples
        """
        entity_ids, _ = entities
        return self._filter_attribute(entity_ids, key, value, op, 'year')

    def FilterDate(self, entities, key, value, op):
        """
        For attributes of the date type, key and value specify the key value, op specifies the comparison operator, and returns the triplet of the entity and the corresponding attribute type

        Args:
            entities (tuple): 2-tuple, the first member is a list of entities, the second is a list of triples
            key (string): property key
            value (string): attribute value, date type
            op (string): comparison operator, one of "=", "!=", "<", ">"

        Returns:
            :obj:`tuple`: returns a tuple, the first member is a list of entities, and the second is a list of triples
        """
        entity_ids, _ = entities
        return self._filter_attribute(entity_ids, key, value, op, 'date')

    def _filter_qualifier(self, entity_ids, facts, tgt_key, tgt_value, op, typ):
        tgt_value = self._parse_key_value(tgt_key, tgt_value, typ)
        res_ids = []
        res_facts = []
        for i, f in self._iterate(zip(entity_ids, facts)):
            for qk, qvs in self._iterate(f['qualifiers'].items()):
                if qk == tgt_key:
                    for qv in self._iterate(qvs):
                        if qv.can_compare(tgt_value) and comp(qv, tgt_value, op):
                            res_ids.append(i)
                            res_facts.append(f)
                            break
        return (res_ids, res_facts)

    def QFilterStr(self, entities, qkey, qvalue):
        """
        Use the modifier key qkey and modifier value qvalue to filter triples, and filter out qualified triples and corresponding entities

        Args:
            entities (tuple): 2-tuple, the first member is a list of entities, the second is a list of triples
            qkey (string): modifier key
            qvalue (string): modified value, string type

        Returns:
            :obj:`tuple`: returns a tuple, the first member is a list of entities, and the second is a list of triples
        """
        entity_ids, facts = entities
        op = '='
        return self._filter_qualifier(entity_ids, facts, qkey, qvalue, op, 'string')

    def QFilterNum(self, entities, qkey, qvalue, op):
        """
        Similar to QFilterStr, but for modified values of numeric types, op specifies a comparison operator

        Args:
            entities (tuple): 2-tuple, the first member is a list of entities, the second is a list of triples
            qkey (string): modifier key
            qvalue (string): modified value, which is a numeric type
            op (string): comparison operator, one of "=", "!=", "<", ">"
        Returns:
            :obj:`tuple`: returns a tuple, the first member is a list of entities, and the second is a list of triples
        """
        entity_ids, facts = entities
        return self._filter_qualifier(entity_ids, facts, qkey, qvalue, op, 'quantity')

    def QFilterYear(self, entities, qkey, qvalue, op):
        """
        Similar to QFilterStr, but for modified values of numeric types, op specifies a comparison operator

        Args:
            entities (tuple): 2-tuple, the first member is a list of entities, the second is a list of triples
            qkey (string): modifier key
            qvalue (string): modified value, which is the year type
            op (string): comparison operator, one of "=", "!=", "<", ">"
        Returns:
            :obj:`tuple`: returns a tuple, the first member is a list of entities, and the second is a list of triples
        """
        entity_ids, facts = entities
        return self._filter_qualifier(entity_ids, facts, qkey, qvalue, op, 'year')

    def QFilterDate(self, entities, qkey, qvalue, op):
        """
        Similar to QFilterStr, but for modified values of numeric types, op specifies a comparison operator

        Args:
            entities (tuple): 2-tuple, the first member is a list of entities, the second is a list of triples
            qkey (string): modifier key
            qvalue (string): modified value, date type
            op (string): comparison operator, one of "=", "!=", "<", ">"
        Returns:
            :obj:`tuple`: returns a tuple, the first member is a list of entities, and the second is a list of triples
        """
        entity_ids, facts = entities
        return self._filter_qualifier(entity_ids, facts, qkey, qvalue, op, 'date')

    def Relate(self, entities, relation, direction):
        """
        Find all entities and corresponding triples that have a specific relationship with the input entity

        Args:
            entities (tuple): 2-tuple, the first member is a list of entities, the second is None or a list of triples
            relation (string): relation label
            direction (string): "forward" or "backward", indicating that the input entity is the head entity or tail entity of the relationship

        Returns:
            :obj:`tuple`: returns a tuple, the first member is a list of entities that have a specific relationship with the input entity, and the second member is a list of corresponding triples
        """
        entity_ids, _ = entities
        res_ids = []
        res_facts = []
        entity_ids = set(entity_ids) & set(self.kb.relation_inv_index[(relation,direction)].keys())
        for ent_id in self._iterate(entity_ids):
            for idx in self._iterate(self.kb.relation_inv_index[(relation,direction)][ent_id]):
                rel_info = self.kb.entities[ent_id]['relations'][idx]
                res_ids.append(rel_info['object'])
                res_facts.append(rel_info)
        return (res_ids, res_facts)

    def And(self, l_entities, r_entities):
        """
        Returns the intersection of two entity collections

        Args:
            l_entities (tuple): Two-tuple, the first member is a list of entities, the second member is None or a list of triples
            r_entities (tuple): Two-tuple, the first member is a list of entities, the second member is None or a list of triples

        Returns:
            :obj:`tuple`: returns a tuple, the first member is the intersection of the input entity sets, and the second member is None
        """
        entity_ids_1, _ = l_entities
        entity_ids_2, _ = r_entities
        return (list(set(entity_ids_1) & set(entity_ids_2)), None)

    def Or(self, l_entities, r_entities):
        """
        Returns the union of two entity collections

        Args:
            l_entities (tuple): Two-tuple, the first member is a list of entities, the second member is None or a list of triples
            r_entities (tuple): Two-tuple, the first member is a list of entities, the second member is None or a list of triples

        Returns:
            :obj:`tuple`: returns a tuple, the first member is the union of the input entity sets, and the second member is None
        """
        entity_ids_1, _ = l_entities
        entity_ids_2, _ = r_entities
        return (list(set(entity_ids_1) | set(entity_ids_2)), None)

    def QueryName(self, entities):
        """
        query entity name

        Args:
            entities (tuple): two-tuple, the first member is a list of entities, the second member is None or a list of triples

        Returns:
            :obj:`list`: returns a list, each element is a string, corresponding to the name of the input entity
        """
        entity_ids, _ = entities
        res = []
        for entity_id in self._iterate(entity_ids):
            name = self.kb.entities[entity_id]['name']
            res.append(name)
        return res

    def Count(self, entities):
        """
        Query the number of entity collections

        Args:
            entities (tuple): two-tuple, the first member is a list of entities, the second member is None or a list of triples

        Returns:
            :obj:`int`: Returns the size of the input entity list
        """
        entity_ids, _ = entities
        return len(entity_ids)

    def SelectBetween(self, l_entities, r_entities, key, op):
        """
        Between two entities, query for entities with greater/smaller values for a specific property

        Args:
            l_entities (tuple): Two-tuple, the first member is a list of entities, the second member is None or a list of triples
            r_entities (tuple): Two-tuple, the first member is a list of entities, the second member is None or a list of triples
            key (string): attribute, requiring its attribute value to be a numeric type, such as "height"
            op (string): Comparator, "less" or "greater", which means querying entities with smaller or larger attribute values

        Returns:
            :obj:`str`: Returns the name of the entity

        """
        entity_ids_1, _ = l_entities
        entity_ids_2, _ = r_entities
        candidates = []
        for ent_id in self._iterate(entity_ids_1):
            for idx in self._iterate(self.kb.attribute_inv_index[key][ent_id]):
                attr_info = self.kb.entities[ent_id]['attributes'][idx]
                candidates.append((ent_id, attr_info['value']))
        for ent_id in self._iterate(entity_ids_2):
            for idx in self._iterate(self.kb.attribute_inv_index[key][ent_id]):
                attr_info = self.kb.entities[ent_id]['attributes'][idx]
                candidates.append((ent_id, attr_info['value']))
        candidates = list(filter(lambda x: x[1].type=='quantity', candidates))
        unit_cnt = defaultdict(int)
        for x in self._iterate(candidates):
            unit_cnt[x[1].unit] += 1
        common_unit = Counter(unit_cnt).most_common()[0][0]
        candidates = list(filter(lambda x: x[1].unit==common_unit, candidates))
        sort = sorted(candidates, key=lambda x: x[1])
        i = sort[0][0] if op=='less' else sort[-1][0]
        name = self.kb.entities[i]['name']
        return name

    def SelectAmong(self, entities, key, op):
        """
        In an entity collection, query the entity with the largest/smallest value of a specific attribute

        Args:
            entities (tuple): two-tuple, the first member is a list of entities, the second member is None or a list of triples
            key (string): attribute, requiring its attribute value to be a numeric type, such as "height"
            op (string): Comparator, "smallest" or "largest", representing the entity with the smallest or largest attribute value

        Returns:
            :obj:`list`: returns a list of entity names, each element is a string (can have multiple max/min values)
        """
        entity_ids, _ = entities
        entity_ids = set(entity_ids)
        candidates = []
        for ent_id in self._iterate(entity_ids):
            for idx in self._iterate(self.kb.attribute_inv_index[key][ent_id]):
                attr_info = self.kb.entities[ent_id]['attributes'][idx]
                candidates.append((ent_id, attr_info['value']))
        candidates = list(filter(lambda x: x[1].type=='quantity', candidates))
        unit_cnt = defaultdict(int)
        for x in self._iterate(candidates):
            unit_cnt[x[1].unit] += 1
        common_unit = Counter(unit_cnt).most_common()[0][0]
        candidates = list(filter(lambda x: x[1].unit==common_unit, candidates))
        sort = sorted(candidates, key=lambda x: x[1])
        value = sort[0][1] if op=='smallest' else sort[-1][1]
        names = list(set([self.kb.entities[i]['name'] for i,v in self._iterate(candidates) if v==value])) # 可以有多个最大/最小值
        return names

    def QueryAttr(self, entities, key):
        """
        Query specific property values of an entity

        Args:
            entities (tuple): two-tuple, the first member is a list of entities, the second member is None or a list of triples
            key (string): attribute

        Returns:
            :obj:`list`: returns a list of attribute values, each element is a string, which is the attribute value of the specified attribute of the corresponding entity
        """
        entity_ids, _ = entities
        res = []
        for ent_id in self._iterate(entity_ids):
            for idx in self._iterate(self.kb.attribute_inv_index[key][ent_id]):
                attr_info = self.kb.entities[ent_id]['attributes'][idx]
                res.append(attr_info['value'])
        return res

    def QueryAttrUnderCondition(self, entities, key, qkey, qvalue):
        """
        Returns the attribute value of the input entity under certain modification conditions

        Args:
            entities (tuple): two-tuple, the first member is a list of entities, the second member is None or a list of triples
            key (string): property key
            qkey (string): modifier key
            qvalue (string): modifier value

        Returns:
            :obj:`list`: returns a list of attribute values that satisfy the condition, each attribute value is a ValueClass
        """
        entity_ids, _ = entities
        qvalue = self._parse_key_value(qkey, qvalue)
        res = []
        for ent_id in self._iterate(entity_ids):
            for idx in self._iterate(self.kb.attribute_inv_index[key][ent_id]):
                attr_info = self.kb.entities[ent_id]['attributes'][idx]
                flag = False
                for qk, qvs in self._iterate(attr_info['qualifiers'].items()):
                    if qk == qkey:
                        for qv in self._iterate(qvs):
                            if qv.can_compare(qvalue) and comp(qv, qvalue, "="):
                                flag = True
                                break
                    if flag:
                        break
                if flag:
                    v = attr_info['value']
                    res.append(v)
        return res

    def _verify(self, s_value, t_value, op, typ):
        attr_values = s_value
        value = self._parse_key_value(None, t_value, typ)
        match = []
        for attr_value in self._iterate(attr_values):
            if attr_value.can_compare(value) and comp(attr_value, value, op):
                match.append(1)
            else:
                match.append(0)
        if sum(match) >= 1 and sum(match) == len(match):
            answer = 'yes'
        elif sum(match) == 0:
            answer = 'no'
        else:
            answer = 'not sure'
        return answer

    def VerifyStr(self, s_value, t_value):
        """
        Verifies that the output of the QueryAttr or QueryAttrUnderCondition function is equal to the given string

        Args:
            s_value (list): A list of ValueClass instances, output from the QueryAttr or QueryAttrUnderCondition functions
            t_value (string): The given attribute value, which is a string type

        Returns:
            :obj:`string`: "yes", or "no" or "not sure" indicates that the attribute value is the same as the given string, different, or not sure
        """
        op = '='
        return self._verify(s_value, t_value, op, 'string')

    def VerifyNum(self, s_value, t_value, op):
        """
        Similar to VerifyStr, but for numeric types, verify whether the attribute value meets certain conditions, and Op specifies the comparison operator

        Args:
            s_value (list): A list of ValueClass instances, output from the QueryAttr or QueryAttrUnderCondition functions
            t_value (string): The given attribute value, which is a numeric type
            op (string): comparison operator, one of "=", "!=", "<", ">"


        Returns:
            :obj:`string`: "yes", or "no" or "not sure" indicates that the attribute value is the same as the given string, different, or not sure
        """
        return self._verify(s_value, t_value, op, 'quantity')

    def VerifyYear(self, s_value, t_value, op,):
        """
        Similar to VerifyStr, but for year types

        Args:
            s_value (list): A list of ValueClass instances, output from the QueryAttr or QueryAttrUnderCondition functions
            t_value (string): The given attribute value, which is the year type
            op (string): comparison operator, one of "=", "!=", "<", ">"


        Returns:
            :obj:`string`: "yes", or "no" or "not sure" indicates that the attribute value is the same as the given string, different, or not sure
        """
        return self._verify(s_value, t_value, op, 'year')

    def VerifyDate(self, s_value, t_value, op,):
        """
        Similar to VerifyStr, but for date types

        Args:
            s_value (list): A list of ValueClass instances, output from the QueryAttr or QueryAttrUnderCondition functions
            t_value (string): given attribute value, date type
            op (string): comparison operator, one of "=", "!=", "<", ">"


        Returns:
            :obj:`string`: "yes", or "no" or "not sure" indicates that the attribute value is the same as the given string, different, or not sure
        """
        return self._verify(s_value, t_value, op, 'date')

    def QueryRelation(self, s_entities, t_entities):
        """
        Query relationships between entities

        Args:
            s_entities (tuple): Two-tuple, the first member is a list of entities, the second member is None or a list of triples
            t_entities (tuple): Two-tuple, the first member is a list of entities, the second member is None or a list of triples

        Returns:
            :obj:`list`: returns a list of relations, each relation is a string
        """
        entity_ids_1, _ = s_entities
        entity_ids_2, _ = t_entities
        res = []
        for entity_id_1 in self._iterate(entity_ids_1):
            for entity_id_2 in self._iterate(entity_ids_2):
                for idx in self._iterate(self.kb.forward_relation_index[(entity_id_1, entity_id_2)]):
                    rel_info = self.kb.entities[entity_id_1]['relations'][idx]
                    res.append(rel_info['relation'])
        return res

    def QueryAttrQualifier(self, entities, key, value, qkey):
        """
        Query a specific modifier value for a property of a given entity

        Args:
            entities (tuple): two-tuple, the first member is a list of entities, the second member is None or a list of triples
            key (string): attribute label
            value (string): attribute value
            qkey (string): Modifier label

        Returns:
            :obj:`list`: returns a list of modified values, each modified value is a string
        """
        entity_ids, _ = entities
        value = self._parse_key_value(key, value)
        res = []
        for ent_id in self._iterate(entity_ids):
            for idx in self._iterate(self.kb.attribute_inv_index[key][ent_id]):
                attr_info = self.kb.entities[ent_id]['attributes'][idx]
                if attr_info['key']==key and attr_info['value'].can_compare(value) and \
                    comp(attr_info['value'], value, '='):
                    for qk, qvs in self._iterate(attr_info['qualifiers'].items()):
                        if qk == qkey:
                            res += qvs
        return res

    def QueryRelationQualifier(self, s_entities, t_entities, relation, qkey):
        """
        Queries specific modifier values for relation triples between given entities

        Args:
            s_entities (tuple): Two-tuple, the first member is a list of entities, the second member is None or a list of triples
            t_entities (tuple): Two-tuple, the first member is a list of entities, the second member is None or a list of triples
            relation (string): relation label
            qkey (string): Modifier label

        Returns:
            :obj:`list`: returns a list of modified values, each modified value is a string
        """
        entity_ids_1, _ = s_entities
        entity_ids_2, _ = t_entities
        res = []
        for entity_id_1 in self._iterate(entity_ids_1):
            for entity_id_2 in self._iterate(entity_ids_2):
                for idx in self._iterate(self.kb.forward_relation_index[(entity_id_1, entity_id_2)]):
                    rel_info = self.kb.entities[entity_id_1]['relations'][idx]
                    if rel_info['relation']==relation:
                        for qk, qvs in self._iterate(rel_info['qualifiers'].items()):
                            if qk == qkey:
                                res += qvs
        return res
