from sao.ramanujan_constructions import Ramanujan_Constructions
from sao.delta import *


class ECO_Module(Ramanujan_Constructions, Base):
    def __init__(
        self,
        module: nn.Module,
        gain: int = 1,
        sparsity: float = None,
        degree: int = None,
        method: str = "SAO",
        activation: str = "relu",
        in_channels_0: int = 3,
        num_classes: int = 100,
    ):
        self.module = module
        self.kernel_size = module.kernel_size[0]
        self.in_channels = module.in_channels
        self.out_channels = module.out_channels
        self.in_channels_0 = in_channels_0
        self.sparsity = sparsity
        self.degree = degree
        self.num_classes = num_classes
        self.method = method
        self.gain = gain
        self.activation = activation

    def _unique_ortho_tensor(self) -> torch.tensor:
        """
        Generates the unique orthogonal matrices, where the number is parameterized
        by the kernel size. If the "pruning method" is SAO, the matrices generated will be
        SAO matrices.

        Returns:
            torch.tensor: Tensor containing the unique orthogonal matrices
        """

        L = (self.kernel_size**2 + 1) // 2
        ortho_tensor = torch.zeros(L, self.out_channels, self.in_channels)

        if self.degree is not None and self.in_channels != self.in_channels_0:
            constructor = self._ramanujan_structure()

        for i in range(L):
            ortho_tensor[i] = (
                self._ortho_generator()
                if (self.degree is None or self.in_channels == self.in_channels_0)
                else constructor()[0]  # Get only the weights given by the constructor
            )

        return ortho_tensor.to("cuda")

    def _give_equiv(self, i: int, j: int):
        """
        For IDF_2D to yield only real values, the matrices of tensor P should follow
        a certain relation, i.e., some matrices should be identical. This function
        provides the indices of the matrices that should have the same values as the matrix
        with indices i and j.

        Args:
            i (int): Row index of the matrix in P
            j (int): Column index of the matrix in P

        Returns:
            tuple[int, int]: The row and column indices of the matrix in P identical to the matrix with indices i and j.
        """

        i_equiv = (self.kernel_size - i) % self.kernel_size
        j_equiv = (self.kernel_size - j) % self.kernel_size
        return i_equiv, j_equiv

    def _ortho_conv(self) -> torch.tensor:
        k = self.kernel_size
        List1 = []
        List2 = []

        for i, j in product(range(k), range(k)):
            eqi, eqj = self._give_equiv(i, j)
            List1.append([i, j])
            List2.append([eqi, eqj])

        for i in List1:
            index1 = List1.index(i)
            index2 = List2.index(i)

            if index1 > index2:
                List1[index1] = -1

        List1 = [x for x in List1 if x != -1]
        List2 = [x for x in List2 if x not in List1]

        ortho_tensor = self._unique_ortho_tensor()
        A = torch.zeros(k, k, self.out_channels, self.in_channels)

        for i in range(len(List1)):
            p, q = List1[i]
            A[p, q] = ortho_tensor[i]

        for i in range(len(List2)):
            p, q = List2[i]
            equivi, equivj = self._give_equiv(p, q)
            A[p, q] = A[equivi, equivj]

        weight_mat = torch.zeros(self.out_channels, self.in_channels, k, k)

        for i, j in product(range(self.out_channels), range(self.in_channels)):
            weight_mat[i, j] = torch.fft.ifft2(A[:, :, i, j])

        return weight_mat.to("cuda")

    def __call__(self) -> torch.tensor:
        return self._ortho_conv()


def ECO_Constructor(module, **kwargs):
    return ECO_Module(module, **kwargs)()
