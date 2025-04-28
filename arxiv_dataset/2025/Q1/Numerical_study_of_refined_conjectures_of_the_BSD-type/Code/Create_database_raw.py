

# This code is for transparency reason, but it is not advice to use, because it is not ease of use. Use the file "Create_database.sage" instead.
from sage.all_cmdline import *   # import sage library

_sage_const_0 = Integer(0); _sage_const_1 = Integer(1); _sage_const_2 = Integer(2); _sage_const_3 = Integer(3); _sage_const_4 = Integer(4); _sage_const_5 = Integer(5); _sage_const_6 = Integer(6); _sage_const_7 = Integer(7)


import os
import sys
import time
import sage.parallel.multiprocessing_sage
import itertools
from multiprocessing import Pool
from sage.databases.cremona import cremona_to_lmfdb, lmfdb_to_cremona

file_names = ['00000-09999', '10000-19999', '20000-29999','30000-39999','40000-49999','50000-59999','60000-69999','70000-79999','80000-89999','90000-99999']

class GroupAlebra:

	def __init__(self, Level):
		self.Level = Level
		self.Coefficients = {}
		self.Base = Level.coprime_integers(Level)
		for index in self.Base:
			self.Coefficients[index] = _sage_const_0 
	
	def ScalarMultiplication(self, Scalar):
		for Index in self.Base:
			self.Coefficients[Index] = self.Coefficients[Index] * Scalar
		
	def Add(self, Other):
		if self.Level != Other.Level:
			print('Not same Level')
			return _sage_const_0 
		Sum = GroupAlebra(self.Level)
		for Index in self.Base:
			Sum.Coefficients[Index] = self.Coefficients[Index] + Other.Coefficients[Index]
		return Sum
	
	def Multiply(self, Other):
		if self.Level != Other.Level:
			print('Not same Level')
			return _sage_const_0 
		MonomialProduct = []
		for index1 in self.Base:
			for index2 in Other.Base:
				MonomialProduct.append([self.Coefficients[index1] * Other.Coefficients[index2], index1 * index2 % self.Level])

		Product = GroupAlebra(self.Level)
		for Monomial in MonomialProduct:
			Product.Coefficients[Monomial[_sage_const_1 ]] = Product.Coefficients[Monomial[_sage_const_1 ]] + Monomial[_sage_const_0 ]
		return Product
	
	def SetCoefficients(self, Coefficients):
		for Coefficient in Coefficients:
			self.Coefficients[Coefficient[_sage_const_1 ]] = Coefficient[_sage_const_0 ]
	
	def SetCoefficientsReturn(self, Coefficients):
		for Coefficient in Coefficients:
			self.Coefficients[Coefficient[_sage_const_1 ]] = Coefficient[_sage_const_0 ]
		return self
	
	def Print(self):
		print(self.Coefficients)
	
	def OnlyCoefficients(self):
		OnlyCoefficients = []
		for BaseElement in self.Base:
			OnlyCoefficients.append(self.Coefficients[BaseElement])
		return OnlyCoefficients

	
class AugmentedIdealPower:

	def __init__(self, Level):
		self.Level = Level
		self.ElementsInGroup = Level.coprime_integers(Level)
		self.Bases = {}

	def SetBaseGivenDepth(self, Depth):
		tiempo_2 = time.time()
		Base = []
		for index in range(_sage_const_1 ,len(self.ElementsInGroup)):
			Base.append(GroupAlebra(self.Level).SetCoefficientsReturn([[-_sage_const_1 ,_sage_const_1 ], [_sage_const_1 , self.ElementsInGroup[index]]]))
		if Depth > _sage_const_1 :
			DepthStep = Depth
			TempBase = Base.copy()
			while DepthStep > _sage_const_1 :
				PrimalBase = TempBase.copy()
				TempBase = []
				DepthStep -= _sage_const_1 
				for StepElement in PrimalBase:
					for PrimalElement in Base:
						TempBase.append(PrimalElement.Multiply(StepElement))
			IndependentElementsVectors = []
			for TempElement in TempBase:
				IndependentElementsVectors.append(TempElement.OnlyCoefficients())
			FreeModule = span(IndependentElementsVectors, ZZ)
			Base = []
			print(FreeModule.gens(), 'ren')
			quit()
			for BaseVector in FreeModule.gens():
				print(BaseVector)
				TempElement = GroupAlebra(self.Level)
				Base.append(TempElement.SetCoefficientsReturn(BaseVector))

		self.Bases[Depth] = Base

	def FindAllBases(self,MaxDepth):
		for AugmentedPower in range(_sage_const_1 , Depth):
			self.AllBases[AugmentedPower] = FindBaseGivenDepth(AugmentedPower)
		return _sage_const_1 
	
	def IsZero(self, Vector, Depth):
		FreeModule = span(self.Bases[Depth], ZZ)
		try:
			FreeModule.coordinate_vector(Vector, check=True)
			return True
		except ArithmeticError:
			return False

class EllipticCurveClass:
	
	def __init__(self, CremonaName, Conductor, EquationCoefficients, AlgebraicRank, Torsion, TamagawaNumber, Sha, ModularDegree):

		# Number from Cremona table
		self.EquationCoefficients = EquationCoefficients
		self.SplitPrimes = []
		self.NonSplitPrimes = []
		self.FindNonSplitPrimes()
		self.FindSplitPrimes()
	
	def ReturnAllData(self):
		return [self.CremonaName, self.Conductor, self.EquationCoefficients, self.AlgebraicRank, self.Torsion, self.TamagawaNumber, self.Sha, self.ModularDegree]
	
	def FindSplitPrimes(self):
		Factorization = list(factor(self.Conductor))
		for Prime in Factorization:
			if EllipticCurve(self.EquationCoefficients).has_split_multiplicative_reduction(Prime[_sage_const_0 ]):
				self.SplitPrimes.append(Prime[_sage_const_0 ])

	def FindNonSplitPrimes(self):
		Factorization = list(factor(self.Conductor))
		for Prime in Factorization:
			if EllipticCurve(self.EquationCoefficients).has_nonsplit_multiplicative_reduction(Prime[_sage_const_0 ]):
				self.NonSplitPrimes.append(Prime[_sage_const_0 ])
	

def DumpData(ListEllipticCurvesObjects):
	DumpDataFile = open('./Data/DumpData.txt')
	for EllipticCurve in ListEllipticCurvesObjects:
		DumpDataFile.write(Elliptic_curve.ReturnAllData() + '\n')
	DumpData.close()
	return True

def ReadData():
	ListEllipticCurvesObjects = []
	DumpDataFile = open('./Data/DumpData.txt')
	for EllipticCurveLineComplete in DumpDataFile.readlines():
		EllipticCurveLine = EllipticCurveLineComplete[:-_sage_const_1 ].split(';')
		ListEllipticCurvesObjects.append(EllipticCurveClass(EllipticCurveLine[_sage_const_0 ], eval(EllipticCurveLine[_sage_const_1 ]), eval(EllipticCurveLine[_sage_const_2 ]), eval(EllipticCurveLine[_sage_const_3 ]), eval(EllipticCurveLine[_sage_const_4 ]), eval(EllipticCurveLine[_sage_const_5 ]), eval(EllipticCurveLine[_sage_const_6 ]), eval(EllipticCurveLine[_sage_const_7 ])))
	return ListEllipticCurvesObjects

def FindGenerator(Level):
	for PossibleGenerator in range(_sage_const_1 ,Level):
		for Exponent in range(_sage_const_1 ,Level):
				if Exponent == Level - _sage_const_1 :
					return PossibleGenerator
				elif PossibleGenerator**Exponent % Level == _sage_const_1 :
					break

def MorphismFromUnits2Cyclic(Level, Generator, Number):
	for PossibleImage in range(_sage_const_1 , Level):
		if (Generator**PossibleImage - Number) % Level == _sage_const_0 :
			return PossibleImage

def TestRefinedConjeture(EllipticCurveObject, SplitPrimesWithExponent, Level, Depth):
	EllipticCurveSage = EllipticCurve(EllipticCurveObject.EquationCoefficients)
	jInvariant = EllipticCurveSage.j_invariant()
	ModularSymbol = EllipticCurveSage.modular_symbol()
	for SplitPrimeData in SplitPrimesWithExponent:
		TateCurve = EllipticCurveSage.tate_curve(SplitPrimeData[_sage_const_0 ])
		pAdicParameter = TateCurve.parameter()
		OrderpAdicParameter = jInvariant.denominator().valuation(SplitPrimeData[_sage_const_0 ])
		TorsionpAdicParameter = (pAdicParameter * SplitPrimeData[_sage_const_0 ]**(-_sage_const_1  * OrderpAdicParameter)).unit_part().expansion()[_sage_const_0 ]
	ListValuesModularSymbols = ComputatioComputationModularSymbols(EllipticCurveObject, EllipticCurveSage, Level)
	ListCommonDenominators = []	
	CommonDenominatorModularSymbols = lcm(ListValuesModularSymbols)
	LeftSide = []
	LeftSideElement = GroupAlebra(Level)
	for a, index in zip(Level.coprime_integers(Level), range(_sage_const_0 , euler_phi(Level))):
		LeftSide.append([CommonDenominatorModularSymbols * OrderpAdicParameter * ListValuesModularSymbols[index], a])
	LeftSideElement.SetCoefficients(LeftSide)
	RightSideElement = GroupAlebra(Level)
	RightSide = ListValuesModularSymbols[-_sage_const_1 ] * CommonDenominatorModularSymbols
	RightSideElement.SetCoefficients([[ListValuesModularSymbols[-_sage_const_1 ] * CommonDenominatorModularSymbols, _sage_const_1 ],[ListValuesModularSymbols[-_sage_const_1 ] * CommonDenominatorModularSymbols, TorsionpAdicParameter]])


	
def ComputatioComputationModularSymbols(EllipticCurveObject, EllipticCurveSage, Level):
	ListModularSymbolsValues = []
	if os.path.exists('./ModularSymbols/{}/{}'.format(EllipticCurveObject.CremonaName, Level)):
		ModularSymbolsFile = open('./ModularSymbols/{}_{}'.format(EllipticCurveObject.CremonaName, Level), 'r').readlines()[_sage_const_0 ][_sage_const_1 :-_sage_const_1 ].split(',')
		for ValueModularSymbol in ModularSymbolsFile:
			ListModularSymbolsValues.append(Rational(ValueModularSymbol))
		return ListModularSymbolsValues
	ModularSymbol = EllipticCurveSage.modular_symbol()
	for a in range(_sage_const_1 , Level):
		if gcd(a, Level) > _sage_const_1 :
			continue
		ListModularSymbolsValues.append(ModularSymbol(a/Level))
	ListModularSymbolsValues.append(ModularSymbol(_sage_const_0 ))
	ModularSymbolsFile = open('./ModularSymbols/{}/{}'.format(EllipticCurveObject.CremonaName, Level), 'w')
	ModularSymbolsFile.write(str(ListModularSymbolsValues))
	return ListModularSymbolsValues

def MultiprocessingComputatioComputationModularSymbols(Index):
	Numbers = file_names[Index]
	if not os.path.exists('./ToCalculate/{}.txt'.format(Numbers)):
		return None
	filesomething = open('./ToCalculate/{}.txt'.format(Numbers), 'r')
	Done = open('./Done/{}'.format(Numbers), 'w')
	Done.close()
	Done = open('./Done/{}'.format(Numbers), 'w')
	RandomValuesFile = open('./RandomValues/{}'.format(Numbers), 'w')
	RandomValuesFile.close()
	RandomValuesFile = open('./RandomValues/{}'.format(Numbers), 'w')
	for line in filesomething.readlines():
		lines = line.split(';')
		ListModularSymbolsValues = []
		equation = eval(lines[_sage_const_0 ])
		E = EllipticCurve(equation)
		ModularSymbol = E.modular_symbol()
		Level = eval(lines[_sage_const_1 ])
		for a in range(_sage_const_1 , Level):
			if gcd(a, Level) > _sage_const_1 :
				continue
			ListModularSymbolsValues.append(ModularSymbol(a/Level))
		ListModularSymbolsValues.append(ModularSymbol(_sage_const_0 ))
		RandomValuesFile.write(str(ListModularSymbolsValues)+';'+line)
		RandomValuesFile.close()
		RandomValuesFile = open('./RandomValues/{}'.format(Numbers), 'a')
		Done.write(line)
		Done.close()
		Done = open('./Done/{}'.format(Numbers), 'a')

def TestConjecture6():
	file = open('./Results.txt', 'w')
	for i in ['00000-09999', '10000-19999', '20000-29999','30000-39999','40000-49999','50000-59999','60000-69999','70000-79999','80000-89999','90000-99999']:
		ModularSymbolsfile = eval(open('./ModularSymbolsReady/{}.txt'.format(i), 'r').readlines()[_sage_const_0 ])
		print(i)
		for key in ModularSymbolsfile.keys():
			for prime in ModularSymbolsfile[key].keys():
				EllipticCurveSage = EllipticCurve(eval(key))
				CremonaEllipticCurve = CremonaDatabase().data_from_coefficients(eval(key))
				if CremonaEllipticCurve['rank'] != _sage_const_1 :
					continue
				prime_2 = Integer(prime)
				if not EllipticCurveSage.has_split_multiplicative_reduction(prime_2):
					continue
				listmodularsymbolsstring = ModularSymbolsfile[key][prime]
				denominators = []
				ModularSymbolsraw = listmodularsymbolsstring[_sage_const_1 :-_sage_const_1 ].split(',')
				ModularSymbols = []
				for a in ModularSymbolsraw[:-_sage_const_1 ]:
					ModularSymbols.append(Rational(a))
				for a in ModularSymbols[:-_sage_const_1 ]:
					denominators.append(a.denominator())
				CommonDenominatorModularSymbols = lcm(denominators)
				LeftSide = _sage_const_1 
				index = _sage_const_0 
				for a in ModularSymbols[:-_sage_const_1 ]:
					index += _sage_const_1 
					LeftSide *= index**(CommonDenominatorModularSymbols * a) % prime_2
				if LeftSide % prime_2 != _sage_const_1  and LeftSide % prime_2 != -_sage_const_1  % prime_2:
					print( str(key)+  '; ' + str(prime_2) )

def TestConjecture5():
	file = open('./Results.txt', 'w')
	for i in ['00000-09999', '10000-19999', '20000-29999','30000-39999','40000-49999','50000-59999','60000-69999','70000-79999','80000-89999','90000-99999']:
		ModularSymbolsfile = eval(open('./ModularSymbolsReady/{}.txt'.format(i), 'r').readlines()[_sage_const_0 ])
		print(i)
		for key in ModularSymbolsfile.keys():
			for prime in ModularSymbolsfile[key].keys():
				EllipticCurveSage = EllipticCurve(eval(key))
				prime_2 = Integer(prime)
				if not EllipticCurveSage.has_split_multiplicative_reduction(prime_2):
					continue
				listmodularsymbolsstring = ModularSymbolsfile[key][prime]
				denominators = []
				ModularSymbolsraw = listmodularsymbolsstring[_sage_const_1 :-_sage_const_1 ].split(',')
				ModularSymbols = []
				for a in ModularSymbolsraw[:-_sage_const_1 ]:
					ModularSymbols.append(Rational(a))
				for a in ModularSymbols[:-_sage_const_1 ]:
					denominators.append(a.denominator())
				TateCurve = EllipticCurveSage.tate_curve(prime_2)
				pro_to_mod_p = EllipticCurveSage.j_invariant().denominator().valuation(int(prime))
				period = TateCurve.parameter()
				torsion = Integer(period.unit_part().expansion()[_sage_const_0 ]) 
				CommonDenominatorModularSymbols = lcm(denominators)
				CremonaEllipticCurve = CremonaDatabase().data_from_coefficients(eval(key))
				LeftSide = _sage_const_1 
				index = _sage_const_0 
				if CremonaEllipticCurve['rank'] > _sage_const_0 :
					continue
				for a in ModularSymbols[:-_sage_const_1 ]:
					index += _sage_const_1 
					LeftSide *= index**(CommonDenominatorModularSymbols * a * pro_to_mod_p * CremonaEllipticCurve['torsion_order'] * CremonaEllipticCurve['torsion_order']) % prime_2
				RightSide = torsion**(_sage_const_2  * CommonDenominatorModularSymbols * CremonaEllipticCurve['db_extra'][_sage_const_0 ] * Integer(CremonaEllipticCurve['db_extra'][-_sage_const_1 ]))
				if LeftSide % prime_2 != RightSide % prime_2 and LeftSide % prime_2 != - RightSide % prime_2:
					file.write(cremona_to_lmfdb(str(EllipticCurveSage.cremona_label())) + ";" + str(prime_2) + ';' + str(LeftSide % prime_2) + ';' + str(RightSide % prime_2) + ';' + str(CremonaEllipticCurve['rank']) + ';' +'${}\\cdot{}^{} + O({}^{})$'.format(torsion, prime_2, pro_to_mod_p, prime_2, pro_to_mod_p + _sage_const_1 ) + ';' + str(EllipticCurveSage.modular_degree()) +  '\n')
					print('here')
def TestConjecture5local():
	file = open('./Results.txt', 'w')
	for i in ['00000-09999', '10000-19999', '20000-29999','30000-39999','40000-49999','50000-59999','60000-69999','70000-79999','80000-89999','90000-99999']:
		ModularSymbolsfile = eval(open('./ModularSymbolsReady/{}.txt'.format(i), 'r').readlines()[_sage_const_0 ])
		print(i)
		for key in ModularSymbolsfile.keys():
			for prime in ModularSymbolsfile[key].keys():
				EllipticCurveSage = EllipticCurve(eval(key))
				prime_2 = Integer(prime)
				if not EllipticCurveSage.has_split_multiplicative_reduction(prime_2):
					continue
				listmodularsymbolsstring = ModularSymbolsfile[key][prime]
				denominators = []
				ModularSymbolsraw = listmodularsymbolsstring[_sage_const_1 :-_sage_const_1 ].split(',')
				ModularSymbols = []
				for a in ModularSymbolsraw[:-_sage_const_1 ]:
					ModularSymbols.append(Rational(a))
				for a in ModularSymbols[:-_sage_const_1 ]:
					denominators.append(a.denominator())
				TateCurve = EllipticCurveSage.tate_curve(prime_2)
				pro_to_mod_p = EllipticCurveSage.j_invariant().denominator().valuation(int(prime))
				period = TateCurve.parameter()
				torsion = Integer(period.unit_part().expansion()[_sage_const_0 ]) 
				CommonDenominatorModularSymbols = lcm(denominators)
				CremonaEllipticCurve = CremonaDatabase().data_from_coefficients(eval(key))
				LeftSide = _sage_const_1 
				index = _sage_const_0 
				if CremonaEllipticCurve['rank'] > _sage_const_0 :
					continue
				if CremonaEllipticCurve['torsion_order'] > _sage_const_0 :
					continue
				for a in ModularSymbols[:-_sage_const_1 ]:
					index += _sage_const_1 
					LeftSide *= index**(CommonDenominatorModularSymbols * a * pro_to_mod_p * CremonaEllipticCurve['torsion_order']*CremonaEllipticCurve['torsion_order']) % prime_2
				RightSide = torsion**(_sage_const_2  * CommonDenominatorModularSymbols * CremonaEllipticCurve['db_extra'][_sage_const_0 ] * Integer(CremonaEllipticCurve['db_extra'][-_sage_const_1 ]))
				if LeftSide % prime_2 != RightSide % prime_2 and LeftSide % prime_2 != - RightSide % prime_2:
					good_primes = prime_2
					for ell in list(factor(good_primes - _sage_const_1 )):
						if ell[_sage_const_0 ] < _sage_const_5 :
							continue
						if gcd(ell[_sage_const_0 ], EllipticCurveSage.modular_degree()) > _sage_const_1 :
							continue
						generator = FindGenerator(good_primes)
						Image_Left = MorphismFromUnits2Cyclic(good_primes, generator, LeftSide)
						Image_Right = MorphismFromUnits2Cyclic(good_primes, generator, RightSide)
						if Image_Left % ell[_sage_const_0 ]**(ell[_sage_const_1 ]) != Image_Right % ell[_sage_const_0 ]**(ell[_sage_const_1 ]) and Image_Left % ell[_sage_const_0 ]**(ell[_sage_const_1 ]) != - Image_Right % ell[_sage_const_0 ]**(ell[_sage_const_1 ]):
							file.write(str(Integer(CremonaEllipticCurve['db_extra'][-_sage_const_1 ])) + ";" + str(CremonaEllipticCurve['db_extra'][_sage_const_0 ]) + ';' + str(LeftSide % good_primes) + ';' + str(RightSide % good_primes) + ';' + str(ell) + ';' +'${}\\cdot{}^{} + O({}^{})$'.format(torsion, good_primes, pro_to_mod_p, good_primes, pro_to_mod_p + _sage_const_1 ) + ';' + str(_sage_const_2 ) +  '\n')
							print('here')
							quit()


def TestConjecture6Local():
	file = open('./listcounterexamples', 'w')
	for i in ['00000-09999', '10000-19999', '20000-29999','30000-39999','40000-49999','50000-59999','60000-69999','70000-79999','80000-89999','90000-99999']:
		print(i)
		ModularSymbolsfile = eval(open('./ModularSymbolsReady/{}.txt'.format(i), 'r').readlines()[_sage_const_0 ])
		for key in ModularSymbolsfile.keys():
			for prime in ModularSymbolsfile[key].keys():
				EllipticCurveSage = EllipticCurve(eval(key))
				good_primes = Integer(prime)
				if not EllipticCurveSage.has_split_multiplicative_reduction(good_primes):
					continue
				listmodularsymbolsstring = ModularSymbolsfile[key][prime]
				denominators = []
				ModularSymbolsraw = listmodularsymbolsstring[_sage_const_1 :-_sage_const_1 ].split(',')
				ModularSymbols = []
				for a in ModularSymbolsraw:
					ModularSymbols.append(Rational(a))
				for a in ModularSymbols:
					denominators.append(a.denominator())

				TateCurve = EllipticCurveSage.tate_curve(good_primes)
				pro_to_mod_p = EllipticCurveSage.j_invariant().denominator().valuation(int(prime))
				period = TateCurve.parameter()
				torsion = Integer(period.unit_part().expansion()[_sage_const_0 ])
				CommonDenominatorModularSymbols = lcm(denominators)
				LeftSide = _sage_const_1 
				index = _sage_const_0 
				for a in ModularSymbols[:-_sage_const_1 ]:
					index += _sage_const_1 
					LeftSide *= index**(CommonDenominatorModularSymbols * a * pro_to_mod_p) % good_primes
				RightSide = torsion**(ModularSymbols[-_sage_const_1 ] * CommonDenominatorModularSymbols)
				if LeftSide % good_primes != RightSide % good_primes and LeftSide % good_primes != - RightSide % good_primes:
					LeftSide = LeftSide % good_primes
					RightSide = RightSide % good_primes
					for ell in list(factor(good_primes - _sage_const_1 )):
						if ell[_sage_const_0 ] < _sage_const_5 :
							continue
						if gcd(ell[_sage_const_0 ], EllipticCurveSage.modular_degree()) > _sage_const_1 :
							continue
						generator = FindGenerator(good_primes)
						Image_Left = MorphismFromUnits2Cyclic(good_primes, generator, LeftSide)
						Image_Right = MorphismFromUnits2Cyclic(good_primes, generator, RightSide)
						if Image_Left % ell[_sage_const_0 ]**(ell[_sage_const_1 ]) != Image_Right % ell[_sage_const_0 ]**(ell[_sage_const_1 ]) and Image_Left % ell[_sage_const_0 ]**(ell[_sage_const_1 ]) != - Image_Right % ell[_sage_const_0 ]**(ell[_sage_const_1 ]):
							file.write(cremona_to_lmfdb(str(EllipticCurveSage.cremona_label())) + ";" + str(generator) + ';' + str(LeftSide % good_primes) + ';' + str(RightSide % good_primes) + ';' + str(ell) + ';' +'${}\\cdot{}^{} + O({}^{})$'.format(torsion, good_primes, pro_to_mod_p, good_primes, pro_to_mod_p + _sage_const_1 ) + ';' + str(_sage_const_2 ) +  '\n')
							print('here')
							quit()

def CleanData():
	for nameFile in file_names:
		if not os.path.exists('./ModularSymbolsReady/{}.txt'.format(nameFile)):
			Dictionary = {}
		else:
			Dictionary = eval(open('./ModularSymbolsReady/{}.txt'.format(nameFile), 'r').readlines()[_sage_const_0 ])
		if not os.path.exists('./RandomValues/{}'.format(nameFile)):
			continue
		ToAdd = open('./RandomValues/{}'.format(nameFile), 'r')
		for line in ToAdd.readlines():
			lines = line[:-_sage_const_1 ].split(';')
			if lines[_sage_const_1 ] in Dictionary:
				Dictionary[lines[_sage_const_1 ]][lines[_sage_const_2 ]] = lines[_sage_const_0 ]
			else:
				Dictionary[lines[_sage_const_1 ]] = {lines[_sage_const_2 ]:lines[_sage_const_0 ]}
		ReadyData = open('./ModularSymbolsReady/{}.txt'.format(nameFile), 'w')
		ReadyData.write(str(Dictionary))
		ReadyData.close()

	return None

#184000
if __name__ == '__main__':
	TestConjecture6()
	quit()
	ListEllipticCurves = []
	for nameFile in file_names:
		ListEllipticCurves = open('./allbsd/allbsd.{}'.format(nameFile), 'r').readlines()
		Output = open('./Trash/{}.txt'.format(nameFile), 'w')
		for lines in ListEllipticCurves:
			splitlines = lines.split(' ')
			for prime in list(factor(eval(splitlines[_sage_const_0 ]))):
				if prime[_sage_const_1 ] > _sage_const_1 :
					continue
				Eliiptic = EllipticCurve(eval(splitlines[_sage_const_3 ]))
				if not Eliiptic.has_split_multiplicative_reduction(prime[_sage_const_0 ]):
					continue
				Output.write(str(splitlines[_sage_const_3 ]) + ';' +  str(prime[_sage_const_0 ]) + '\n')

		Output.close()
	quit()
	for nameFile in file_names:
		if not os.path.exists('./ModularSymbolsReady/{}.txt'.format(nameFile)):
			Dictionary = {}
		else:
			Dictionary =  eval(open('./ModularSymbolsReady/{}.txt'.format(nameFile), 'r').readlines()[_sage_const_0 ])
		NewValues = []
		TempFileTrash = open('./Trash/{}.txt'.format(nameFile), 'r')
		for line in TempFileTrash.readlines():
			lines = line[:-_sage_const_1 ].split(';')
			if not lines[_sage_const_0 ] in Dictionary.keys():
				NewValues.append(line)
			elif not lines[_sage_const_1 ] in Dictionary[lines[_sage_const_0 ]]:
				NewValues.append(line)
		if len(NewValues) > _sage_const_0 :
			TempFile = open('./ToCalculate/{}.txt'.format(nameFile), 'w')
			for NewValue in NewValues:
				TempFile.write(NewValue)
			TempFile.close()
	tiempo = time.time()
	Number = range(len(file_names))
	p = Pool(processes=_sage_const_7 )
	resuls = p.map(MultiprocessingComputatioComputationModularSymbols, Number)
	p.close()
	p.join()
	print(int(time.time() - tiempo))

