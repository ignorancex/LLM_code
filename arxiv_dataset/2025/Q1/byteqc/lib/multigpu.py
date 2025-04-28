# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# This file is part of ByteQC.
#
# Licensed under the Apache License, Version 2.0 (the "License")
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https: // www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from threading import Lock, Barrier
from multiprocessing.pool import ThreadPool
import cupy
import os
import cupy_backends
from cupyx.distributed import NCCLBackend
from cupyx.distributed._nccl_comm import _get_nccl_dtype_and_count
from numbers import Integral

os.unsetenv("NCCL_DEBUG")  # this screen the output of NCCL lib
# os.environ["NCCL_DEBUG"] = "TRACE"


class LabelLock():
    def __init__(self):
        self.label = dict()

    def __call__(self, l=None):
        if l not in self.label:
            self.label[l] = Lock()
        return self.label[l]


class TrivalLabelLock():
    def __call__(self, *args):
        return TrivalLock()


class TrivalLock():
    def __enter__(self):
        pass

    def __exit__(self, *args):
        pass


class Params:
    def __init__(self, *params, length=None):
        if length is None:
            self.len = 0
            for p in params:
                if hasattr(p, '__iter__'):
                    self.len = max(self.len, len(p))
            assert self.len != 0
        else:
            self.len = length

        if params == []:
            self.params = [[]] * self.len
        else:
            self.params = [[]] * self.len
            for i in range(self.len):
                self.params[i] = [p[i] if hasattr(
                    p, '__getitem__') else p for p in params]

    def __getitem__(self, key):
        return self.params[key]

    def __len__(self):
        return self.len


class Singlegpu:
    def __init__(self, gpu=[0], active=True):
        self.ngpu = 1
        if isinstance(gpu, Integral):
            self.gpus = [gpu]
        else:
            self.gpus = gpu
        self.lock = TrivalLabelLock()
        self.isactive = active
        if active:
            self.active()

    def active(self):
        self.isactive = True
        cupy.cuda.runtime.setDevice(self.gpus[0])

    def getgid(self):
        return 0

    def reduce(self, f, *params, op='sum', batch=1):
        result = None
        with cupy.cuda.Device(self.gpus[0]):
            for p in Params(*params):
                result = f(result, *p)
            return result

    def sum(self, arrs, root=None, coeff=None):
        if coeff is not None:
            arrs[0] *= coeff
        return arrs[0]

    def map(self, f, *params, batch=1):
        with cupy.cuda.Device(self.gpus[0]):
            return [f(*p) for p in Params(*params)]

    def broadcast(self, *arr, root=None):
        with cupy.cuda.Device(self.gpus[0]):
            if len(arr) == 1:
                arr = arr[0]
                if getattr(arr, '__broad_as__', None) is None:
                    return [cupy.asarray(arr)]
                else:
                    return [arr.__broad_as__()[1]]
            else:
                r = [None] * len(arr)
                for i, a in enumerate(arr):
                    if getattr(a, '__broad_as__', None) is None:
                        r[i] = [cupy.asarray(a)]
                    else:
                        r[i] = [a.__broad_as__()[1]]
                # return [[cupy.asarray(a)] for a in arr]
                return r

    def mapgpu(self, f, *params):
        with cupy.cuda.Device(self.gpus[0]):
            return [f(*Params(*params, length=1)[0])]


class Multigpu:
    def __init__(self, gpus, active=True):
        if isinstance(gpus, Integral):
            self.ngpu = gpus
            self.gpus = list(range(gpus))
        else:
            self.ngpu = len(gpus)
            self.gpus = list(gpus)
        self.lock = LabelLock()
        ngpu = cupy.cuda.runtime.getDeviceCount(
        )
        assert self.ngpu <= ngpu, "Only %d GPUs detected while %d GPUs is demanded" % (
            cupy.cuda.runtime.getDeviceCount(), self.ngpu)
        for i in self.gpus:
            assert i in range(
                ngpu), "#%d GPUs not exist in total %d GPUs" % (i, ngpu)
        self.isactive = active
        if active:
            self.active()

    def __del__(self):
        if getattr(self, 'pool', None) is not None:
            self.pool.terminate()
            self.pool.join()

    def active(self):
        cupy.cuda.runtime.setDevice(self.gpus[0])
        self.bar = Barrier(self.ngpu)
        if getattr(self, 'pool', None) is not None:
            self.pool.terminate()
            self.pool.join()
        self.pool = ThreadPool(
            self.ngpu, lambda: cupy.cuda.runtime.setDevice(self.gpus[0]))
        self.comms = cupy.cuda.nccl.NcclCommunicator.initAll(self.gpus)
        self.isactive = True

    def getgid(self):
        igpu = cupy.cuda.runtime.getDevice()
        return self.gpus.index(igpu)

    def barrier(self):
        self.bar.wait()

    def _reduce(self, i, inarr, outarr, root, op='sum'):
        NCCLBackend._check_contiguous(None, inarr)
        if i == root:
            NCCLBackend._check_contiguous(None, outarr)
        stream = NCCLBackend._get_stream(None, None)
        dtype, count = _get_nccl_dtype_and_count(inarr)
        op = NCCLBackend._get_op(None, op, inarr.dtype.char)
        self.comms[i].reduce(
            inarr.data.ptr, outarr.data.ptr,
            count, dtype, op, root, stream)

    def _allreduce(self, i, inarr, outarr, op):
        NCCLBackend._check_contiguous(None, inarr)
        NCCLBackend._check_contiguous(None, outarr)
        stream = NCCLBackend._get_stream(None, None)
        dtype, count = _get_nccl_dtype_and_count(inarr)
        op = NCCLBackend._get_op(None, op, inarr.dtype.char)
        self.barrier()
        self.comms[i].allReduce(
            inarr.data.ptr, outarr.data.ptr, count, dtype, op, stream)

    def _broadcast(self, i, arr, root):
        NCCLBackend._check_contiguous(None, arr)
        stream = NCCLBackend._get_stream(None, None)
        dtype, count = _get_nccl_dtype_and_count(arr)
        self.barrier()
        self.comms[i].broadcast(
            arr.data.ptr, arr.data.ptr,
            count, dtype, root, stream)

    def task_reduce(self, i, root, op, batch=1):
        with cupy.cuda.Device(self.gpus[i]):
            result = None
            work = 0

            notDone = True
            while notDone:
                with self.lock():
                    work = self.i
                    self.i += batch
                for w in range(work, work + batch):
                    if w >= self.count:
                        notDone = False
                        break
                    result = self.f(result, *self.p[w])
                cupy.cuda.Device(self.gpus[i]).synchronize()

            if result is None:
                n = len(*self.p[0])
                result = self.f(result, *([None] * (n - 1)))

            self._reduce(i, result, result, root, op)
            self.barrier()
            if i == root:
                return result

    def reduce(self, f, *params, root=None, op='sum', batch=1):
        if root is None:
            root = self.getgid()
        self.p = Params(*params)
        self.count = len(self.p)
        self.i = 0
        self.f = f

        r = self.pool.starmap(
            self.task_reduce, [(i, root, op, batch) for i in range(self.ngpu)])

        self.p = self.f = None
        return r[root]

    def task_sum(self, i, arr, coeff=None):
        with cupy.cuda.Device(self.gpus[i]):
            self.barrier()
            self._allreduce(i, arr, arr, 'sum')
            if coeff is not None:
                arr *= coeff
            self.barrier()

    def sum(self, arrs, root=None, coeff=None):
        if root is None:
            root = self.getgid()
        self.pool.starmap(
            self.task_sum, [(i, arrs[i], coeff) for i in range(self.ngpu)])
        return arrs[root]

    def task_map(self, i, batch):
        with cupy.cuda.Device(self.gpus[i]):
            work = 0
            notDone = True
            while notDone:
                with self.lock():
                    work = self.i
                    self.i += batch
                for w in range(work, work + batch):
                    if w >= self.count:
                        notDone = False
                        break
                    self.r[w] = self.f(*self.p[w])
                    cupy.cuda.Device(self.gpus[i]).synchronize()

    def map(self, f, *params, batch=1):
        self.p = Params(*params)
        self.count = len(self.p)
        self.i = 0
        self.f = f
        self.r = [None] * self.count

        self.pool.starmap(self.task_map, [(i, batch)
                          for i in range(self.ngpu)])

        r = self.r
        self.r = self.p = self.f = None
        return r

    def broadcast(self, *arrs, root=None):
        root = self.getgid()
        narr = len(arrs)
        r = [[None] * self.ngpu for _ in range(narr)]
        wrap = [[None] * self.ngpu for _ in range(narr)]
        with cupy.cuda.Device(self.gpus[root]):
            for iarr in range(narr):
                arr = arrs[iarr]
                if getattr(arr, '__broad_as__', None) is None:
                    assert arr.flags.c_contiguous
                    wrap[iarr][root] = r[iarr][root] = cupy.asarray(arr)
                else:
                    r[iarr][root], wrap[iarr][root] = arr.__broad_as__()

        def task(i):
            with cupy.cuda.Device(self.gpus[i]):
                if i != root:
                    for iarr, arr in enumerate(arrs):
                        if getattr(arr, '__broad_new__', None) is None:
                            r[iarr][i] = cupy.empty(arr.shape, dtype=arr.dtype)
                            wrap[iarr][i] = r[iarr][i]
                        else:
                            r[iarr][i], wrap[iarr][i] = arr.__broad_new__()
                for iarr in range(narr):
                    if isinstance(r[iarr][i], tuple):
                        for subarr in r[iarr][i]:
                            self._broadcast(i, subarr, root)
                    else:
                        self._broadcast(i, r[iarr][i], root)
                self.barrier()

        self.pool.starmap(task, [(i,) for i in range(self.ngpu)])

        if narr == 1:
            return wrap[0]
        else:
            return wrap

    def mapgpu(self, f, *params):
        self.p = Params(*params, length=self.ngpu)

        def task(i):
            with cupy.cuda.Device(self.gpus[i]):
                r = f(*self.p[i])
                return r
        r = self.pool.starmap(task, [(i,) for i in range(self.ngpu)])
        self.p = None
        return r


class LazyGpus():
    '''
    Lazy object to avoid active CUDA environment in advance.
    '''

    def __init__(self, *args, active=True):
        self.args = args
        self.d = Gpus(*self.args, active=active)

    set_gpus = __init__

    def __getattr__(self, name):
        if self.d.isactive is False:
            self.d.active()
        return self.d.__getattribute__(name)


def Gpus(ngpu=None, active=True):
    '''
    Dispatch between multiple GPUs and single GPU.
    '''
    if ngpu is None:
        try:
            ngpu = cupy.cuda.runtime.getDeviceCount()
        except cupy_backends.cuda.api.runtime.CUDARuntimeError:
            # for case without GPU
            ngpu = 1
    if isinstance(ngpu, Integral):
        if ngpu == 1:
            return Singlegpu(active=active)
    else:
        if len(ngpu) == 1:
            return Singlegpu(ngpu, active=active)
    return Multigpu(ngpu, active=active)


# The globle multigpu object to control the multigpu behavior.
Mg = LazyGpus(eval(os.getenv('DEFAULT_MG', 'None')), active=False)

if __name__ == "__main__":
    import time
    n = 100
    a = Mg

    def task_reduce(r, p):
        if r is None:
            r = cupy.zeros((n, n))
        r += p
        return r

    r = a.reduce(task_reduce, range(100), batch=13)
    print(r[0, 0], 'should be 4950')

    def task_map(i, *args):
        d = cupy.zeros((n, n))
        d[:] = i
        time.sleep(0.1)
        return i, d

    r = a.map(task_map, range(80))
    for i in r:
        print("%d-%d" % (i[0], int(i[1][0, 0])), end=" ")
    print("should be exact paired")

    arr = cupy.random.rand(1)
    r = a.broadcast(arr)
    print(r, 'should be same among GPUs')

    arr2 = cupy.random.rand(1)
    r1, r2 = a.broadcast(arr, arr2)
    print(r1, 'should be same as above')
    print(r2, 'should be same among GPUs')

    def task_map_gpu():
        d = cupy.zeros((n, n))
        d[:] = cupy.cuda.runtime.getDevice()
        time.sleep(0.1)
        return d

    r = a.mapgpu(task_map_gpu)
    for i in r:
        print("%f" % (i[0, 0],), end=" ")
    print("should be 0:%d" % (Mg.ngpu - 1))

    rs = a.sum(r)
    print(rs, 'should be all', Mg.ngpu * (Mg.ngpu - 1) // 2)
