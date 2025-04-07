import numbers

class InputVariable:
	SMTLIB_TYPE_DICT = {
		'integer': {'smtlib_type':'Int', 'bound_type':numbers.Integral},
		'real': {'smtlib_type':'Real', 'bound_type':numbers.Real},
		'bool': {'smtlib_type':'Bool', 'bound_type':type(None)}
	}

	def __init__(self, name, type = 'real', lower_bound = 0.0, upper_bound = 1.0, pert_bound=None, protected=False):
		assert type in self.SMTLIB_TYPE_DICT, 'Parameter \'type\' should be one of [{}].'.format(', '.join(self.SMTLIB_TYPE_DICT.keys()))
		assert isinstance(lower_bound, self.SMTLIB_TYPE_DICT[type]['bound_type']) and isinstance(upper_bound, self.SMTLIB_TYPE_DICT[type]['bound_type']), 'Parameters \'lower_bound\' and \'upper_bound\' should be of appropriate type.'
		
		self.name = name
		self.type = type
		self.lower_bound = lower_bound
		self.upper_bound = upper_bound
		self.is_protected = protected
		
		if pert_bound is None:
			if protected == False:
				self.pert_bound = 0.
			else:
				self.pert_bound = None
		else:
			self.pert_bound = pert_bound
	# End __init__

	def __str__(self):
		tc = self.type[0]
		return '%c[%f, %f]' % (tc, self.lower_bound, self.upper_bound)

	def __copy__(self):
		return InputVariable(self.name, self.type, self.lower_bound, self.upper_bound, self.pert_bound, self.is_protected)

	def to_dict(self):
		return {'name':self.name, 'type':self.type, 'lower_bound':self.lower_bound,
			'upper_bound':self.upper_bound, 'pert_bound':self.pert_bound, 'is_protected':self.is_protected}

	def from_dict(dict):
		return InputVariable(dict['name'], dict['type'], dict['lower_bound'], dict['upper_bound'], dict['pert_bound'], dict['is_protected'])
# End class InputVariable
