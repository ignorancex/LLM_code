# -*- coding: utf-8 -*-
import io
import logging
from collections.abc import Iterable
from numbers import Integral

import numpy as np
import numpy.lib.mixins
from flex.extensions.bindata import MultipleDataExtension
from flex.flex import FlexExtension

logger = logging.getLogger(__name__)

# TODO make this a proper subclass of np.ndarray
# see also https://numpy.org/devdocs/user/basics.subclassing.html
HANDLED_FUNCTIONS = {}


def implements(np_function):
    "Register an __array_function__ implementation for DiagonalArray objects."

    def decorator(func):
        HANDLED_FUNCTIONS[np_function] = func
        return func

    return decorator


class Iliffe_vector(numpy.lib.mixins.NDArrayOperatorsMixin, MultipleDataExtension):
    """
    Illiffe vectors are multidimensional (here 2D) but not necessarily rectangular
    Instead the index is a pointer to segments of a 1D array with varying sizes
    """

    def __init__(self, values, offsets=None, dtype=None):
        FlexExtension.__init__(self, cls=MultipleDataExtension)
        if offsets is not None:
            if offsets[0] != 0:
                offsets = [0, *offsets]
            self.data = np.asarray(values, dtype=dtype)
            self.offsets = np.asarray(offsets, dtype=int)
            return

        sizes = [len(v) for v in values]
        offsets = np.array([0, *np.cumsum(sizes)])

        data = np.concatenate(values)
        if dtype is not None:
            data = data.astype(dtype)

        self.data = data
        self.offsets = offsets

    def __getitem__(self, key):
        if isinstance(key, Integral):
            return self.__getsegment__(key)
        if isinstance(key, (slice, Iterable)) and not isinstance(key, tuple):
            if isinstance(key, slice):
                key = range(self.nseg)[key]
            values = [self.__getsegment__(k) for k in key]
            return self.__class__(values)
        if isinstance(key, tuple):
            if isinstance(key[0], Integral):
                return self[key[0]][key[1]]
            if isinstance(key[0], (slice, Iterable)):
                if isinstance(key[0], slice):
                    key0 = range(self.nseg)[key[0]]
                else:
                    key0 = key[0]
                values = [self.__getsegment__(k) for k in key0]
                values = [v[key[1]] for v in values]
                if isinstance(key[1], Integral):
                    return np.array(values)
                if isinstance(key[1], (slice, Iterable)):
                    return self.__class__(values)
        if isinstance(key, Iliffe_vector):
            data = [w[m] for w, m in zip(self.segments, key.segments)]
            return self.__class__(data)
        raise KeyError

    def __setitem__(self, key, value):
        isscalar = np.isscalar(value)
        if not isscalar:
            value = np.asarray(value)
        if isinstance(key, Integral):
            return self.__setsegment__(key, value)
        if isinstance(key, (slice, Iterable)) and not isinstance(key, tuple):
            if isinstance(key, slice):
                key = range(self.nseg)[key]
            if isscalar or (value.ndim == 1 and value.dtype != "O"):
                for k in key:
                    self.__setsegment__(k, value)
            else:
                for i, k in enumerate(key):
                    self.__setsegment__(k, value[i])
            return
        if isinstance(key, tuple):
            if isinstance(key[0], Integral):
                data = self.__getsegment__(key[0])
                data[key[1]] = value
                return
            if isinstance(key[0], (slice, Iterable)):
                if isinstance(key[0], slice):
                    key0 = range(self.nseg)[key[0]]
                else:
                    key0 = key[0]
                if isscalar or (value.ndim == 1 and value.dtype != "O"):
                    for k in key0:
                        data = self.__getsegment__(k)
                        data[key[1]] = value
                else:
                    for i, k in enumerate(key0):
                        data = self.__getsegment__(k)
                        data[key[1]] = value
                return
        if isinstance(key, Iliffe_vector):
            self.data[key.data] = value
            return
        raise KeyError

    def __getsegment__(self, seg):
        while seg < 0:
            seg = self.nseg - seg
        if seg > self.nseg - 1:
            raise IndexError
        low, upp = self.offsets[seg : seg + 2]
        return self.data[low:upp]

    def __setsegment__(self, seg, value):
        while seg < 0:
            seg = self.nseg - seg
        if seg > self.nseg - 1:
            raise IndexError
        low, upp = self.offsets[seg : seg + 2]
        if np.isscalar(value) or value.size == upp - low:
            # Keep using the existing memory
            self.data[low:upp] = value
        else:
            # Need to set new memory
            data = [s if i != seg else value for i, s in enumerate(self.segments)]
            self.offsets = np.array([0, *np.cumsum([len(s) for s in data])])
            self.data = np.concatenate(data)

    def __array__(self, dtype=None):
        arr = self.data
        if dtype is not None:
            arr = arr.astype(dtype)
        return arr

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method == "__call__":
            inputs = list(inputs)
            for i, input in enumerate(inputs):
                if isinstance(input, self.__class__):
                    inputs[i] = input.__array__()
            if "out" in kwargs:
                out = []
                for i, o in enumerate(kwargs["out"]):
                    if isinstance(o, self.__class__):
                        out.append(o.__array__())
                    else:
                        out.append(o)
                kwargs["out"] = tuple(out)
            arr = ufunc(*inputs, **kwargs)
            return self.__class__.from_offsets(arr, self.offsets)
        else:
            return NotImplemented

    def __array_function__(self, func, types, args, kwargs):
        if func not in HANDLED_FUNCTIONS:
            return func(self.data, *args, **kwargs)
        # Note: this allows subclasses that don't override
        # __array_function__ to handle DiagonalArray objects.
        if not all(issubclass(t, self.__class__) for t in types):
            return NotImplemented
        return HANDLED_FUNCTIONS[func](*args, **kwargs)

    def __array_function_axis__(self, func, axis, args, kwargs):
        if axis is None:
            return func(self.data, *args, **kwargs)
        elif axis == 0:
            data = [func(s, *args, **kwargs) for s in self.segments]
            data = np.array(data)
            return data
        else:
            return NotImplemented

    def __repr__(self):
        return f"{self.__class__.__name__}({self.segments})"

    def __len__(self):
        return self.nseg

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def ndim(self):
        return 2

    @property
    def shape(self):
        return (self.nseg, self.sizes)

    @property
    def size(self):
        return self.data.size

    @property
    def sizes(self):
        return np.diff(self.offsets)

    @property
    def nseg(self):
        return len(self.offsets) - 1

    @property
    def segments(self):
        return [self.data[l:u] for l, u in zip(self.offsets[:-1], self.offsets[1:])]

    @implements(np.ravel)
    def ravel(self, *args, **kwargs):
        return np.ravel(self.data, *args, **kwargs)

    @implements(np.ndarray.flatten)
    def flatten(self, *args, **kwargs):
        return self.data.flatten(*args, **kwargs)

    @implements(np.copy)
    def copy(self, *args, **kwargs):
        data = np.copy(self.data, *args, **kwargs)
        offsets = np.copy(self.offsets, *args, **kwargs)
        return self.__class__(data, offsets=offsets)

    @implements(np.where)
    def where(self, *args, **kwargs):
        data = np.where(self.data, *args, **kwargs)
        res = self.__class__(data, offsets=self.offsets)
        return res

    @implements(np.all)
    def all(self, *args, axis=None, **kwargs):
        return self.__array_function_axis__(np.all, axis, args, kwargs)

    @implements(np.any)
    def any(self, *args, axis=None, **kwargs):
        return self.__array_function_axis__(np.any, axis, args, kwargs)

    @implements(np.min)
    def min(self, *args, axis=None, **kwargs):
        return self.__array_function_axis__(np.min, axis, args, kwargs)

    @implements(np.max)
    def max(self, *args, axis=None, **kwargs):
        return self.__array_function_axis__(np.max, axis, args, kwargs)

    @implements(np.mean)
    def mean(self, *args, axis=None, **kwargs):
        return self.__array_function_axis__(np.mean, axis, args, kwargs)

    @classmethod
    def from_offsets(cls, array, offsets):
        self = cls(array, offsets=offsets)
        return self

    @classmethod
    def from_indices(cls, array, indices):
        if indices is None:
            return cls([array])
        else:
            offsets = [0, *np.cumsum(indices)]
            arr = [array[l:u] for l, u in zip(offsets[:-1], offsets[1:])]
            return cls(arr)

    def astype(self, dtype):
        data = self.data.astype(dtype)
        offsets = self.offsets
        return Iliffe_vector(data, offsets, dtype=dtype)

    # For IO with Flex
    def _prepare(self, name: str):
        cls = self.__class__

        header_fname = f"{name}/header.json"
        header_info, header_bio = cls._prepare_json(header_fname, self.header)
        result = [(header_info, header_bio)]

        for key, value in enumerate(self.segments):
            data_fname = f"{name}/{key}.npy"
            data_info, data_bio = cls._prepare_npy(data_fname, value)
            result += [(data_info, data_bio)]

        return result

    @classmethod
    def _parse(cls, header: dict, members: dict):
        data = {key[:-4]: cls._parse_npy(bio) for key, bio in members.items()}
        data = [data[str(i)] for i in range(len(data))]
        ext = cls(values=data)
        return ext

    def to_dict(self):
        data = {"data": self.data, "offsets": self.offsets}
        return data

    @classmethod
    def from_dict(cls, data):
        obj = cls(values=data["data"], offsets=data["offsets"])
        return obj

    # def to_dict(self):
    #     cls = self.__class__
    #     obj = {"header": self.header}
    #     for i, v in enumerate(self.segments):
    #         obj[str(i)] = cls._np_to_dict(v)
    #     return obj

    # @classmethod
    # def from_dict(cls, header: dict, data: dict):
    #     data = {name: cls._np_from_dict(d) for name, d in data.items()}
    #     data = [data[str(i)] for i in range(len(data))]
    #     obj = cls(values=data)
    #     return obj

    def _save(self):
        data = {str(i): v for i, v in enumerate(self.segments)}
        ext = MultipleDataExtension(data=data)
        return ext

    @classmethod
    def _load(cls, ext: MultipleDataExtension):
        data = ext.data
        values = [data[str(i)] for i in range(len(data))]
        iv = cls(values=values)
        return iv

    def _save_v1(self, file, folder=""):
        """
        Creates a npz structure, representing the vector

        Returns
        -------
        data : bytes
            data to use
        """
        b = io.BytesIO()
        np.savez(b, *self.segments)
        file.writestr(f"{folder}.npz", b.getvalue())

    @classmethod
    def _load_v1(cls, file):
        # file: npzfile
        names = file.files
        values = [file[n] for n in names]
        return cls(values=values)
