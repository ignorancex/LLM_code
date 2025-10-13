from numba import types, typed, typeof, njit, objmode
from numba.experimental import jitclass
import numpy as np
from typing import Generator, Tuple, Callable, List, Iterable, Dict

I_2 = np.eye(
    2,
    dtype=np.complex128
)
PAULI_X = np.array(
    [
        [0, 1],
        [1, 0]
    ],
    dtype=np.complex128
)
PAULI_Y = np.array(
    [
        [0, -1j],
        [1j, 0]
    ],
    dtype=np.complex128
)
PAULI_Z = np.array(
    [
        [1, 0],
        [0, -1]
    ],
    dtype=np.complex128
)

pauli_dict = typed.Dict(zip('ixyz', (I_2, PAULI_X, PAULI_Y, PAULI_Z)))
basis_spec = [
    ('__signature', types.unicode_type),
    ('__weight', types.uint16),
    ('__pauli_dict', typeof(pauli_dict)),
    ('__base', typeof(I_2)),
    ('__n_qubits', types.uint16)
]

def is_power_of_2(number:int) -> bool:

    if number < 1:
        return False
    else:
        return (number & ~(number - 1)) == number

@njit
def n_choose_2(
    sliceable:list | tuple | str,
    /
) -> Generator[tuple, None, None]:
    iter_length = len(sliceable)
    for ii, item1 in enumerate(sliceable):
        for item2 in sliceable[slice(ii+1, iter_length)]:
            yield item1, item2

@njit 
def two_product(
    iterable:Iterable
) -> Generator[tuple, None, None]:
    
    for item1 in iterable:
        for item2 in iterable:
            yield item1, item2

@njit
def product(
    string:str,
    repeats:int=1
):
    if repeats < 1:
        yield ''
    else:
        indexes = [0 for _ in range(repeats)]
        while True:
            yield ''.join([string[index] for index in indexes])
            j = len(indexes) - 1
            while True:
                indexes[j] += 1
                if indexes[j] < len(string): 
                    break
                indexes[j] = 0
                j -= 1
                if j < 0: 
                    return

@njit
def get_n_qubits(
    a:np.ndarray
) -> int:
    
    num_rows = a.shape[0]
    n_qubits = 0
    while num_rows > 1:
        num_rows >>= 1
        n_qubits += 1

    return n_qubits

@jitclass(spec=basis_spec)
class PauliBasis(object):

    def __init__(
        self,
        signature:str,
        pauli_dict:dict=pauli_dict
    ) -> None:
        
        self.__signature = signature
        self.__n_qubits = len(signature)
        self.__weight = self.__n_qubits - signature.count('i')
        self.__pauli_dict = pauli_dict
        self.__base = self.__construct_basis([self.__pauli_dict[sign] for sign in signature])

    @property
    def weight(
        self
    ) -> int:
        return self.__weight

    @property
    def base(
        self
    ) -> np.ndarray:
        
        return self.__base

    @property
    def shape(
        self
    ) -> tuple:
        return self.__base.shape

    @property
    def n_qubits(
        self
    ) -> int:
        return self.__n_qubits

    @property
    def signture(
        self
    ) -> str:
        return self.__signature

    def trace(
        self
    ) -> complex:
        
        return np.trace(self.__base)

    def __construct_basis(
        self,
        pauli_matrices
    ) -> np.ndarray:
        
        kron_product = np.array(
            [[1]], 
            dtype=np.complex128
        )
        for pauli_matrix in pauli_matrices:
            kron_product = np.kron(kron_product, pauli_matrix)
        return kron_product
    
    def __add__(
        self,
        other
    ) -> np.ndarray:

        if isinstance(other, PauliBasis):
            return self.base + other.base
        else:
            return self.base + other
        
    def __matmul__(
        self,
        other
    ) -> np.ndarray:
        
        if isinstance(other, PauliBasis):
            return self.base@other.base
        else:
            return self.base@other
    
    def __eq__(
        self,
        other
    ) -> bool:
        
        if isinstance(other, PauliBasis):
            return self.signture == other.signture
        else:
            return False

    def __mul__(
        self,
        other
    ) -> np.ndarray:
        
        if isinstance(other, PauliBasis):
            return self.base*other.base
        else:
            return self.base*other
        
    def __rmul__(
        self,
        other
    ) -> np.ndarray:
        
        if isinstance(other, PauliBasis):
            return self.base*other.base
        else:
            return self.base*other

    def __str__(
        self
    ) -> str:
        
        return str(self.base)



pauli_spec =[
    ('__coeffs', typeof([1+0j])),
    ('__bases', typeof([PauliBasis('ix')])),
    ('__base', typeof(I_2)),
    ('__n_qubits', types.uint16)
]

@jitclass(spec=pauli_spec)
class PauliMatrix(object):

    def __init__(
        self,
        base_array:np.ndarray=PauliBasis('').base
    ) -> None:
        
        if is_power_of_2(base_array.shape[0]):
            self.__base = base_array
            self.__n_qubits = get_n_qubits(base_array)
        else:
            raise ValueError("Base array is not a Pauli representable array!")

    def generate_P_signatures(
        self
    ) -> Generator[str, None, None]:

        tuple_range = list(range(self.__n_qubits))

        for ii in tuple_range:

            for character in 'xyz':

                yield ii*'i' + character + (self.__n_qubits-ii-1)*'i'

        for (index1, index2) in n_choose_2(tuple_range):
            
            for (char1, char2) in two_product('xyz'):

                yield index1*'i' + char1 + (index2 - index1 - 1)*'i' + char2 + (self.__n_qubits - index2 - 1)*'i'

    def generate_all_signatures(
        self
    ) -> Generator[str, None, None]:
        
        all_signatures = product('ixyz', repeats=self.__n_qubits)
        next(all_signatures, None)
        for signature in all_signatures:
            yield signature

def string_product(
    string:str,
    repeat:int=1,
):
    if repeat < 2:
        for character in string:
            yield character
    else:
        for string_p in string_product(string, repeat=repeat-1):
            for character in string:
                yield character + string_p

if __name__ == '__main__':

    print(is_power_of_2(1))

else:

    PauliBasis('ix')
    PauliMatrix()