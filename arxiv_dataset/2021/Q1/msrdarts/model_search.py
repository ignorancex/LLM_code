import torch
import torch.nn as nn
import torch.nn.functional as F
from operations_spectral_norm import *
from torch.autograd import Variable
from genotypes import PRIMITIVES_SPECTRAL
from genotypes import Genotype, Genotype_normal, Genotype_reduce
from spectralnorm.spectral_norm_utils import *


class MixedOp(nn.Module):

  def __init__(self, C, stride, shape):
    super(MixedOp, self).__init__()
    self.shape = shape
    self.last_conv_shape = shape if stride == 1 else (shape[0]//2, shape[1]//2)
    self._ops = nn.ModuleList()
    for primitive in PRIMITIVES_SPECTRAL:
      op = OPS_SPECTORAL_NORM[primitive](C, stride, False, shape)
      if 'pool' in primitive:
        op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
      self._ops.append(op)

    self.stable_ranks = self.calc_stable_ranks()

  def forward(self, x):
    return sum(w * op(x) for w, op in zip(F.softmax(-torch.Tensor(self.stable_ranks)), self._ops))

  def calc_stable_ranks(self):
    stable_ranks = []
    for module in self._ops:
      last_conv = [m for _,m in module.op.named_modules() if isinstance(m, nn.Conv2d)][-1]
      stable_ranks.append(calculate_stable_rank(last_conv, self.last_conv_shape))
    return stable_ranks

  def update_stable_ranks(self):
    self.stable_ranks = self.calc_stable_ranks()

class Cell(nn.Module):

  def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev, shape):
    super(Cell, self).__init__()
    self.reduction = reduction
    self.shape = shape

    if reduction_prev:
      self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
    else:
      self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
    self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
    self._steps = steps
    self._multiplier = multiplier

    self._ops = nn.ModuleList()
    self._bns = nn.ModuleList()
    for i in range(self._steps):
      for j in range(2+i):
        stride = 2 if reduction and j < 2 else 1
        op = MixedOp(C, stride, shape)
        self._ops.append(op)

  def forward(self, s0, s1):
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)

    states = [s0, s1]
    offset = 0
    for i in range(self._steps):
      s = sum(self._ops[offset+j](h) for j, h in enumerate(states))
      offset += len(states)
      states.append(s)

    return torch.cat(states[-self._multiplier:], dim=1)

  def extract_arch(self, sr_max=False):
    def _parse(weights):
      gene = []
      n = 2
      start = 0
      for i in range(self._steps):
        end = start + n
        W = weights[start:end].copy()

        if sr_max:
          edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x]))))[:2]
          for j in edges:
            k_best = None
            for k in range(len(W[j])):
              if k_best is None or W[j][k] > W[j][k_best]:
                k_best = k
            gene.append((PRIMITIVES_SPECTRAL[k_best], j))
          start = end
          n += 1
        else:
          edges = sorted(range(i + 2), key=lambda x: min(W[x][k] for k in range(len(W[x]))))[:2]
          for j in edges:
            k_best = None
            for k in range(len(W[j])):
              if k_best is None or W[j][k] < W[j][k_best]:
                k_best = k
            gene.append((PRIMITIVES_SPECTRAL[k_best], j))
          start = end
          n += 1

      return gene

    srs_matrix = []
    for mixed_op in self._ops:
      srs_matrix.append(mixed_op.calc_stable_ranks())
    srs_matrix = np.array(srs_matrix)
    return _parse(srs_matrix), srs_matrix

  def update_stable_ranks(self):
    for op in self._ops:
      op.update_stable_ranks()

class NetworkCIFAR(nn.Module):

  def __init__(self, C, num_classes, layers, criterion, steps=4, multiplier=4, stem_multiplier=3, shape=(32,32)):
    super(NetworkCIFAR, self).__init__()
    self._C = C
    self._num_classes = num_classes
    self._layers = layers
    self._criterion = criterion
    self._steps = steps
    self._multiplier = multiplier
    self.in_shapes = []

    C_curr = stem_multiplier*C
    self.stem = nn.Sequential(
      nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
      nn.BatchNorm2d(C_curr)
    )

    C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
    self.cells = nn.ModuleList()
    reduction_prev = False
    h, w = shape
    for i in range(layers):
      # for in_shapes
      if reduction_prev:
        h = h // 2
        w = w // 2
      self.in_shapes.append((h,w))

      if i in [layers//3, 2*layers//3]:
        C_curr *= 2
        reduction = True
      else:
        reduction = False
      cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, self.in_shapes[-1])
      reduction_prev = reduction
      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, multiplier*C_curr

    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(C_prev, num_classes)

  def new(self):
    model_new = NetworkCIFAR(self._C, self._num_classes, self._layers, self._criterion).cuda()
    for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
        x.data.copy_(y.data)
    return model_new

  def forward(self, input):
    s0 = s1 = self.stem(input)
    for i, cell in enumerate(self.cells):
      s0, s1 = s1, cell(s0, s1)
    out = self.global_pooling(s1)
    logits = self.classifier(out.view(out.size(0),-1))
    return logits

  def genotype(self, sr_max=False, print_srs=False):
    def _parse(weights):
      gene = []
      n = 2
      start = 0
      for i in range(self._steps):
        end = start + n
        W = weights[start:end].copy()

        edges = sorted(range(i + 2), key=lambda x: min(W[x][k] for k in range(len(W[x]))))[:2]
        for j in edges:
          k_best = None
          for k in range(len(W[j])):
            if k_best is None or W[j][k] < W[j][k_best]:
              k_best = k
          gene.append((PRIMITIVES_SPECTRAL[k_best], j))
        start = end
        n += 1
      return gene

    ret_normal = []
    ret_reduce = []
    srs_normal = []
    srs_reduce = []
    concat = range(2+self._steps-self._multiplier, self._steps+2)
    for i, cell in enumerate(self.cells):
      cell_arch, srs = cell.extract_arch(sr_max=sr_max)
      if print_srs:
        print("cell {}".format(i))
        print(srs)
      if cell.reduction:
        ret_reduce.append(Genotype_reduce(reduce=cell_arch, reduce_concat=concat))
        srs_reduce.append(srs)
      else:
        ret_normal.append(Genotype_normal(normal=cell_arch, normal_concat=concat))
        srs_normal.append(srs)

    if not sr_max:
      self.normal = ret_normal
      self.reduce = ret_reduce
      self.normal_mean = np.mean(srs_normal,axis=0)
      self.reduce_mean = np.mean(srs_reduce,axis=0)
      if print_srs:
        print("mean information")
        print(self.normal_mean)
        print(self.reduce_mean)
      genotypes = Genotype(normal=_parse(self.normal_mean), normal_concat=concat,reduce=_parse(self.reduce_mean), reduce_concat=concat)
    return ret_normal, ret_reduce, genotypes

  def update_stable_ranks(self):
    for cell in self.cells:
      cell.update_stable_ranks()

class NetworkImageNet(nn.Module):

  def __init__(self, C, num_classes, layers, criterion, steps=4, multiplier=4, shape=(56,56)):
    super(NetworkImageNet, self).__init__()
    self._C = C
    self._num_classes = num_classes
    self._layers = layers
    self._criterion = criterion
    self._steps = steps
    self._multiplier = multiplier
    self.in_shapes = []

    self.stem0 = nn.Sequential(
      nn.Conv2d(3, C // 2, kernel_size=3, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(C // 2),
      nn.ReLU(inplace=True),
      nn.Conv2d(C // 2, C, 3, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(C),
    )

    self.stem1 = nn.Sequential(
      nn.ReLU(inplace=True),
      nn.Conv2d(C, C, 3, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(C),
    )

    C_prev_prev, C_prev, C_curr = C, C, C
    self.cells = nn.ModuleList()
    reduction_prev = True
    h, w = shape
    for i in range(layers):
      # for in_shapes
      if reduction_prev:
        h = h // 2
        w = w // 2
      self.in_shapes.append((h,w))

      if i in [layers//3, 2*layers//3]:
        C_curr *= 2
        reduction = True
      else:
        reduction = False
      cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, self.in_shapes[-1])
      reduction_prev = reduction
      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, multiplier*C_curr

    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(C_prev, num_classes)

  def new(self):
    model_new = NetworkImageNet(self._C, self._num_classes, self._layers, self._criterion).cuda()
    for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
        x.data.copy_(y.data)
    return model_new

  def forward(self, input):
    s0 = self.stem0(input)
    s1 = self.stem1(s0)
    for i, cell in enumerate(self.cells):
      s0, s1 = s1, cell(s0, s1)
    out = self.global_pooling(s1)
    logits = self.classifier(out.view(out.size(0),-1))
    return logits

  def genotype(self, sr_max=False, print_srs=False):
    ret_normal = []
    ret_reduce = []
    concat = range(2+self._steps-self._multiplier, self._steps+2)
    for i, cell in enumerate(self.cells):
      cell_arch, svs = cell.extract_arch(sr_max=sr_max)
      if print_srs:
        print("cell {}".format(i))
        print(svs)
      if cell.reduction:
        ret_reduce.append(Genotype_reduce(reduce=cell_arch, reduce_concat=concat))
      else:
        ret_normal.append(Genotype_normal(normal=cell_arch, normal_concat=concat))

    if not sr_max:
      self.normal = ret_normal
      self.reduce = ret_reduce
    return ret_normal, ret_reduce
