import torch
import torch.nn as nn
import torch.nn.functional as F
from resnet import resnet34

import itertools
import numpy as np
from scipy.special import comb
import galois
import sys

class SmallNetwork(nn.Module):
    """Implements the small MNIST-optimised network from 
    
        P. Y. Simard, D. Steinkraus and J. C. Platt, 
        "Best practices for convolutional neural networks applied to visual document analysis," 
        Seventh International Conference on Document Analysis and Recognition, 2003.
    
    Parameter: out_dim specifies the output dimension of the network.
    """
    def __init__(self, out_dim):
        super(SmallNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=5, kernel_size=5, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=5, out_channels=50, kernel_size=5, stride=2, padding=0),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(in_features=50*5*5, out_features=100),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=100, out_features=out_dim),
        )
    def forward(self, x):
        out = self.net(x)
        return out

class ResNetD(nn.Module):
    """
    Implements the ResNet-34 model, as in Kasarla et al. (2022). 

    See the `resnet.py` for more details (authored by Kasarla et al.)
    """
    def __init__(self, depth, out_dim):
        super(ResNetD, self).__init__()
        if depth == 34:
            self.net = resnet34(out_dim)
        else:
            raise NotImplementedError(f"ResNet with depth {depth} is not implemented yet.")

    def forward(self, x):
        out = self.net(x)
        return out

class ReedMullerCode(nn.Module):
    """
    Implements hyperspherical prototypes based on Reed-Muller (RM) codes.
    The RM code implementation is based on
        
        E. Abbe, A. Shpilka and M. Ye, 
        "Reed–Muller Codes: Theory and Algorithms,"
        IEEE Transactions on Information Theory, vol. 67, no. 6, pp. 3251-3277, June 2021

    Parameters: 
        latent_dim: the latent space dimension, required to be a multiple of 2    
        num_classes: the number of classes in the classification problem. 

    There are additional requirements on the relation between latent space dimension and number of classes 
    (i.e., that the dimension is sufficiently large), which is explained in the paper.
    """
    def __init__(self, latent_dim, num_classes, randomise=False):
        super(ReedMullerCode, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        # Check if given latent_dim is a power of 2
        log2_latent_dim = np.log2(latent_dim)
        assert np.isclose(log2_latent_dim % 1, 0), "Expected a latent dimension which is a power of 2."

        # Create the generator matrix of the corresponding ReedMuller code
        rm_generator_matrix, code_dim = self.__get_generator_matrix(latent_dim, num_classes)
        if rm_generator_matrix is None:
            self.codebook = None
        else:
            # Enumerate the codebook from the generator matrix; 
            # this is of shape {0,1}^(num_classes, latent_dim)
            rm_codebook = self.__enumerate_codewords(rm_generator_matrix, code_dim, num_classes)
            # Rescale to unit length
            codebook = self.__rescale_to_unit_length(rm_codebook, latent_dim)
            # Optional: randomise the mapping between prototype and class
            if randomise:
                randperm = torch.randperm(num_classes)
                codebook = codebook[randperm]
            # Create nn.Parameter from the np-array codebook
            self.codebook = nn.Parameter(torch.tensor(codebook, dtype=torch.float), requires_grad=False)

    def __get_generator_matrix(self, latent_dim, num_classes):
        """Returns the RM(m,r) generator matrix, where m and r are such that:
            latent_dim = 2**m,
            r = smallest integer r such that sum_{i=0}^r nCk(m,i) >= num_classes.
        """
        # First check if there are enough binary vectors
        if latent_dim < np.log2(num_classes):
            return None, None

        # Since there are enough binary vectors, generate some
        # Get m and r of the desired code
        m = int(np.log2(latent_dim))
        # Calculate r: need the code dim >= log2(num_classes), and we choose the smallest r where this is true:
        terms_in_codedim_sum = [comb(m,i) for i in range(m+1)]
        nth_partial_sum = np.cumsum(terms_in_codedim_sum)
        r = np.min(np.where(nth_partial_sum >= np.log2(num_classes)))

        # Construct generator matrix as seen in Abbe et al, in IEEE Trans IT 2021, or The Theory of Error Correcting Codes by MacWilliams et al, North-Holland 1977.
        # The exact generator matrix look a bit different, but the codes are equivalent.
        code_len = int(latent_dim)
        code_dim = np.sum([comb(m,i) for i in range(r+1)], dtype=int)
        gen_mat = np.zeros((code_dim, code_len), dtype=int)
        gen_mat[0,:] = np.ones((code_len,), dtype=int) #all codes contain the all-1 codeword, i.e., a repetition code
        if r > 0:
            # all codes r > 0 contain the m codewords with Hamming distance 2**(m-1) to the all-1 codeword
            for i in range(1,m+1):
                x_i = np.array(([0]*2**(m-i) + [1]*2**(m-i))*2**(i-1), dtype=int)
                gen_mat[i,:] = x_i
            
            # We now have to add all polynomials of order r > 1 in GF(2), i.e., combinations of {x_i}
            next_cw_idx = m+1
            for poly_order in range(2, r+1):
                # Combinations of order i over {x_i}
                all_polynomials = np.arange(1,m+1, dtype=int)
                combinations = np.array([elem for elem in itertools.combinations(all_polynomials, poly_order)])
                for combination in combinations:
                    # For each combination, do modulo 2 multiplication and add to codebook
                    gen_mat[next_cw_idx, :] = np.prod(gen_mat[combination], axis=0, dtype=int)
                    next_cw_idx += 1

        return gen_mat, code_dim

    def __enumerate_codewords(self, rm_generator_matrix, code_dim, num_classes):
        """Enumerate all codewords from the generator matrix.
        This is done in the following manner. For all classes k = {0, 1, ..., num_classes},
        the corresponding codeword is given by mod-2 addition of the rows of the generator matrix
        corresponding to 1:s in the binary representation of k.

        We adopt the convention that the LSB is first.
        """
        code_dim = rm_generator_matrix.shape[0]
        latent_dim = rm_generator_matrix.shape[1]
        rm_codebook = np.zeros((num_classes, latent_dim), dtype=int)
        for k in range(num_classes):
            # Convert k to an array of length code_dim with the binary representation of k
            # LSB is first
            bin_rep = np.array(list(np.binary_repr(k).zfill(code_dim))).astype(int)
            bin_rep = np.flip(bin_rep).reshape(1,-1)
            # Do mod 2 addition of the rows which correspond to the codeword and put in codebook
            ck = np.sum(bin_rep @ rm_generator_matrix, axis=0, dtype=int) % 2
            rm_codebook[k,:] = ck

        return rm_codebook
    
    def __rescale_to_unit_length(self, rm_codebook, latent_dim):
        """Transforms the binary vectors to vectors which lie on surface of unit hypersphere in the following way:
            Transform to +-1: 0 -> -1 and 1 -> 1,
            Rescale length so the corresponding hypersphere has volume 1 (with some care so the transformation works numerically)
        """
        n = latent_dim
        to_hadamard = 2*(rm_codebook-1/2)
        # Normalize with 1/sqrt(n) to get unit vectors
        return to_hadamard/np.sqrt(n)
    
    @torch.no_grad()
    def verify_quality(self):
        # Calculate all inner products
        device = self.codebook.get_device()
        product = torch.matmul(self.codebook.data, self.codebook.data.t())

        # Max and min:
        # Diagonal is always 1, so take this into account
        max = torch.max(product - 2. * torch.eye(product.shape[0], device=device))
        min = torch.min(product)

        # Average:
        # There are num_classes*(num_classes-1) off-diagonal elements
        # Also think about the diagonal
        avg = torch.sum(product - torch.eye(product.shape[0], device=device))/(self.num_classes*(self.num_classes-1))

        # Print
        print(f'''
=============================================
(Min., Avg., Max.) prototype dist.:  ({min:.2f}, {avg:.2f}, {max:.2f})
=============================================
        ''')

    @torch.no_grad()
    def forward(self, y):
        return self.codebook[y]

class RandomCode(nn.Module):
    """
    Implements a random set of hyperspherical prototypes.

    Parameters:
        latent_dim: the latent space dimension
        num_classes: the number of classes in the classification problem. 

    It turns out that a standard normal vector, which is then normalised, 
    is uniform over the hypersphere.
    """
    def __init__(self, latent_dim, num_classes):
        super(RandomCode, self).__init__()
        # Save attributes
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        # Create random (unnormalized) codes of dimension k x n, 
        # where n is the latent dimension and k is the codebook size
        temp_code = torch.randn(num_classes, latent_dim, requires_grad=False)
        # Create normalised codebooks from these codebooks
        self.codebook = nn.Parameter(F.normalize(temp_code, dim=1), requires_grad=False)    

    @torch.no_grad()
    def _eval_code(self):
        # Evaluates codeword length and inter-codeword distances.
        # Calculate and return pairwise correlation matrix (L2-norm distance measure)
        corr_matrix = torch.zeros((self.num_classes,self.num_classes))
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                corr_matrix[i,j] = torch.linalg.norm(self.codebook[i]-self.codebook[j])
        # Calculate and return norms of codewords
        norms = torch.linalg.norm(self.codebook, dim=1)

        return corr_matrix, norms

    @torch.no_grad()
    def verify_quality(self):
        # Get corr. matrices and norms
        matrix, norms = self._eval_code()
        ### Print min and max distances and norms
        # Class code
        distances,_ = torch.sort(torch.flatten(matrix))
        norms, _ = torch.sort(norms)
        print(f'''
=============================================
(Min., Avg., Max.) codeword dist.:  ({distances[self.num_classes]:.2f}, {distances[self.num_classes:].mean():.2f}, {distances[-1]:.2f})
(Min., Avg., Max.) codeword norm:   ({norms[0]:.2f}, {norms.mean():.2f}, {norms[-1]:.2f})
=============================================
        ''')

    @torch.no_grad()
    def forward(self, y):
        # Assume y is a an integer class index
        return self.codebook[y]

class BCHCode(nn.Module):
    """
    Implements hyperspherical prototypes based on Bose-Chaudhuri-Hocquenghem (BCH) codes.
    The implementation is based on the `galois` package.

    Parameters:
        latent_dim: the latent space dimension, required to be 1 smaller than a power of 2.
        num_classes: the number of classes in the classification problem.     
    """
    def __init__(self, latent_dim, num_classes, verbose=False, randomise=False):
        super(BCHCode, self).__init__()
        # Save attributes
        self.verbose = verbose
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        # Check if given latent_dim is 1 smaller than a power of 2
        log2_latent_dim = np.log2(latent_dim+1)
        assert np.isclose(log2_latent_dim % 1, 0), "Expected a latent dimension which is 1 smaller than a power of 2."
        # Get generator matrix
        gen_mat = self.__get_generator_matrix()
        if gen_mat is None: #Could not find a BCH code of length latent_dim
            self.codebook = None
        else: #Found a codebook
            # Enumerate all codewords
            bch_codebook = self.__enumerate_codewords(gen_mat)
            # Rescale to unit length
            euclidian_codebook = self.__rescale_to_unit_length(bch_codebook)
            # Optional: randomise the mapping between prototype and class
            if randomise:
                randperm = torch.randperm(num_classes)
                euclidian_codebook = euclidian_codebook[randperm]
            # Create nn.Parameter from the unit length vectors
            self.codebook = nn.Parameter(torch.tensor(euclidian_codebook, dtype=torch.float), requires_grad=False)
    
    def __get_generator_matrix(self):
        """Generate the smallest BCH code of length self.latent_dim for self.num_classes classes.
        
        Uses the Galois package."""

        k = np.ceil(np.log2(self.num_classes))
        
        code = None
        for attempt in range(25): #max 100 attempts at finding a code
            try:
                code = galois.BCH(int(self.latent_dim), int(k))
            except Exception as e:
                k += 1
            else:
                if self.verbose: print(f"The smallest BCH code found is BCH({int(self.latent_dim)},{int(k)})")
                break

        # Handle successful and unsuccessful attempts
        if code is None: #Fail
            print(f"Failed to find any valid BCH code of length {self.latent_dim} for k between {int(np.ceil(np.log2(self.num_classes)))} and {int(k)}.")
            return None
        else: #Success
            gen_mat = code.G.view(np.ndarray).astype(float)
            # print(f"Generator matrix: {type(gen_mat.view(np.ndarray))}")
        return gen_mat

    def __enumerate_codewords(self, gen_mat):
        """Enumerate all codewords from the generator matrix.
        This is done in the following manner. For all classes k = {0, 1, ..., num_classes},
        the corresponding codeword is given by mod-2 addition of the rows of the generator matrix
        corresponding to 1:s in the binary representation of k.

        We adopt the convention that the LSB is first.
        """
        k = gen_mat.shape[0]
        codebook = np.zeros((self.num_classes, self.latent_dim), dtype=int)
        for class_no in range(self.num_classes):
            # Convert k to an array of length code_dim with the binary representation of k
            # LSB is first
            bin_rep = np.array(list(np.binary_repr(class_no).zfill(k))).astype(int)
            bin_rep = np.flip(bin_rep).reshape(1,-1)
            # Do mod 2 addition of the rows which correspond to the codeword and put in codebook
            ck = np.sum(bin_rep @ gen_mat, axis=0, dtype=int) % 2
            codebook[class_no,:] = ck
        return codebook
    
    def __rescale_to_unit_length(self, codebook):
        """Transforms the binary vectors to vectors which lie on surface of unit hypersphere in the following way:
            Transform to +-1: 0 -> -1 and 1 -> 1,
            Rescale length so the corresponding hypersphere has volume 1 (with some care so the transformation works numerically)
        """
        bpsk_mod = 2*(codebook-1/2)
        # Normalize with 1/sqrt(n) to get unit vectors
        return bpsk_mod/np.sqrt(self.latent_dim)
    
    @torch.no_grad()
    def verify_quality(self):
        # Calculate all inner products
        device = self.codebook.get_device()
        product = torch.matmul(self.codebook.data, self.codebook.data.t())

        # Max and min:
        # Diagonal is always 1, so take this into account
        max = torch.max(product - 2. * torch.eye(product.shape[0], device=device))
        min = torch.min(product)

        # Average:
        # There are num_classes*(num_classes-1) off-diagonal elements
        # Also think about the diagonal
        avg = torch.sum(product - torch.eye(product.shape[0], device=device))/(self.num_classes*(self.num_classes-1))

        # Print
        print(f'''
=============================================
(Min., Avg., Max.) prototype dist.:  ({min:.2f}, {avg:.2f}, {max:.2f})
=============================================
        ''')

    @torch.no_grad()
    def forward(self, y):
        return self.codebook[y]

class MettesCode(nn.Module):
    """
    Implements hyperspherical prototypes based on the paper

    P. Mettes, E. van der Pol, C. Snoek,
    "Hyperspherical Prototype Networks",
    NeurIPS, 2019

    Parameters:
        latent_dim: the latent space dimension.
        num_classes: the number of classes in the classification problem.     
    """
    def __init__(self, latent_dim, num_classes):
        super(MettesCode, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        # Solve the optimisation problem which Mettes 2019 pose
        mettes_codebook = self.__optimise_codebook()
        # Create parameter
        self.codebook = nn.Parameter(mettes_codebook, requires_grad=False)
    
    def __loss_fcn(self, prototypes):
        # Construct inner products (which is dot product since normalised)
        product = torch.matmul(prototypes, prototypes.t())
        # Remove diagnonal from loss
        product -= 2. * torch.eye(product.shape[0])
        # Pick out maximum values for each row 
        max_vals = product.max(dim=1)[0]
        # Loss is mean of the max
        ret = max_vals.mean()
        return ret

    def __optimise_codebook(self, lr=0.1, momentum=0.9, epochs=1000):
        # Solve the optimisation problem posed by Mettes et al.
        prototypes = torch.randn(self.num_classes, self.latent_dim, dtype=torch.float)
        prototypes = nn.Parameter(F.normalize(prototypes, p=2, dim=1))
        optimiser = torch.optim.SGD([prototypes], lr=lr, momentum=momentum)

        for epoch in range(epochs):
            # Take a gradient step
            optimiser.zero_grad()
            loss = self.__loss_fcn(prototypes)
            loss.backward()
            optimiser.step()
            # Re-normalise 
            with torch.no_grad():
                prototypes.data = prototypes.data / torch.linalg.vector_norm(prototypes.data, ord=2, dim=1, keepdim=True)   
        
        return prototypes.clone().detach()
    
    @torch.no_grad()
    def verify_quality(self):
        # Calculate all inner products
        device = self.codebook.get_device()
        product = torch.matmul(self.codebook.data, self.codebook.data.t())

        # Max and min:
        # Diagonal is always 1, so take this into account
        max = torch.max(product - 2. * torch.eye(product.shape[0], device=device))
        min = torch.min(product)

        # Average:
        # There are num_classes*(num_classes-1) off-diagonal elements
        # Also think about the diagonal
        avg = torch.sum(product - torch.eye(product.shape[0], device=device))/(self.num_classes*(self.num_classes-1))

        # Print
        print(f'''
=============================================
(Min., Avg., Max.) prototype dist.:  ({min:.2f}, {avg:.2f}, {max:.2f})
=============================================
        ''')

    @torch.no_grad()
    def forward(self, y):
        return self.codebook[y]

class LogSumExpCode(nn.Module):
    """
    Implements hyperspherical prototypes based a Log-Sum-Exp convex relaxation.

    Parameters:
        latent_dim: the latent space dimension.
        num_classes: the number of classes in the classification problem.     
    """
    def __init__(self, latent_dim, num_classes):
        super(LogSumExpCode, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        # Solve the optimisation problem which Mettes 2019 pose
        lse_codebook = self.__optimise_codebook()
        # Create parameter
        self.codebook = nn.Parameter(lse_codebook, requires_grad=False)
    
    def __loss_fcn(self, prototypes, temp):
        # Construct inner products (which is dot product since normalised)
        product = torch.matmul(prototypes, prototypes.t())
        # Mask away the upper triangular + diagonal part 
        mask = torch.triu(-torch.inf*torch.ones_like(product), diagonal=0)
        masked_product = mask+product #zero out unwanted points
        flattened = torch.flatten(masked_product)
        return 1/temp*torch.logsumexp(temp*flattened-temp, dim=0)+1 #numerically stable calculation of LSE

    def __optimise_codebook(self, lr=0.2, momentum=0.9, epochs=1000):
        # Solve the optimisation problem posed by Mettes et al.
        prototypes = torch.randn(self.num_classes, self.latent_dim, dtype=torch.float)
        prototypes = nn.Parameter(F.normalize(prototypes, p=2, dim=1))
        optimiser = torch.optim.SGD([prototypes], lr=lr, momentum=momentum)

        for epoch in range(epochs):
            # Take a gradient step
            optimiser.zero_grad()
            temperature = 1 + self.num_classes*epoch/epochs #Linear increase in temperature
            loss = self.__loss_fcn(prototypes, temperature)
            loss.backward()
            optimiser.step()
            # Re-normalise 
            with torch.no_grad():
                prototypes.data = prototypes.data / torch.linalg.vector_norm(prototypes.data, ord=2, dim=1, keepdim=True)   
        
        return prototypes.clone().detach()
    
    @torch.no_grad()
    def verify_quality(self):
        # Calculate all inner products
        device = self.codebook.get_device()
        product = torch.matmul(self.codebook.data, self.codebook.data.t())

        # Max and min:
        # Diagonal is always 1, so take this into account
        max = torch.max(product - 2. * torch.eye(product.shape[0], device=device))
        min = torch.min(product)

        # Average:
        # There are num_classes*(num_classes-1) off-diagonal elements
        # Also think about the diagonal
        avg = torch.sum(product - torch.eye(product.shape[0], device=device))/(self.num_classes*(self.num_classes-1))

        # Print
        print(f'''
=============================================
(Min., Avg., Max.) prototype dist.:  ({min:.2f}, {avg:.2f}, {max:.2f})
=============================================
        ''')

    @torch.no_grad()
    def forward(self, y):
        return self.codebook[y]
    
class KasarlaCode(nn.Module):
    """
    Implements hyperspherical prototypes based on the paper

        T. Kasarla, G. J. Burghouts, M. van Spengler, E. van der Pol, R. Cucchiara, and P. Mettes,
        "Maximum Class Separation as Inductive Bias in One Matrix,"
        NeurIPS, 2022

    The core part of the implementation is taken from their GitHub repo, and has only been 
    slightly adapted to fit the code here.

    Parameters:
        num_classes: the number of classes in the classification problem.
    """
    def __init__(self, num_classes):
        super(KasarlaCode, self).__init__()
        self.num_classes = num_classes
        self.latent_dim = num_classes-1
        # Create the Kasarla 2022 codebook
        if num_classes >= 1000:
            print(f"Warning: Increasing recursion limit to handle Kasarla et al.'s construction for {num_classes} classes")
            sys.setrecursionlimit(10000) #for nr_prototypes>=1000
        kasarla_codebook = self.__kasarla_codebook()
        # Create parameter
        self.codebook = nn.Parameter(torch.tensor(kasarla_codebook, dtype=torch.float), requires_grad=False)

    def __V(self, order):
        # Taken from Kasarla et al. 2022
        if order == 1:
            return np.array([[1, -1]])
        else:
            col1 = np.zeros((order, 1))
            col1[0] = 1
            row1 = -1 / order * np.ones((1, order))
            return np.concatenate((col1, np.concatenate((row1, np.sqrt(1 - 1 / (order**2)) * self.__V(order - 1)), axis=0)), axis=1)

    def __kasarla_codebook(self):
        prototypes = self.__V(self.num_classes-1).T
        return prototypes.astype(np.float32)
    
    @torch.no_grad()
    def verify_quality(self):
        # Calculate all inner products
        device = self.codebook.get_device()
        product = torch.matmul(self.codebook.data, self.codebook.data.t())

        # Max and min:
        # Diagonal is always 1, so take this into account
        max = torch.max(product - 2. * torch.eye(product.shape[0], device=device))
        min = torch.min(product)

        # Average:
        # There are num_classes*(num_classes-1) off-diagonal elements
        # Also think about the diagonal
        avg = torch.sum(product - torch.eye(product.shape[0], device=device))/(self.num_classes*(self.num_classes-1))

        # Print
        print(f'''
=============================================
(Min., Avg., Max.) prototype dist.:  ({min:.2f}, {avg:.2f}, {max:.2f})
=============================================
        ''')

    @torch.no_grad()
    def forward(self, y):
        return self.codebook[y]

class PrototypicalLoss(nn.Module):
    """
    Implments the prototypical loss described in 

        T. Kasarla, G. J. Burghouts, M. van Spengler, E. van der Pol, R. Cucchiara, and P. Mettes,
        "Maximum Class Separation as Inductive Bias in One Matrix,"
        NeurIPS, 2023
    """
    def __init__(self):
        super(PrototypicalLoss, self).__init__()

    def forward(self, latent_var, y, prototypes):
        """Computes a cross-entropy-based loss assuming:
            latent_var: batch_size x latent_dim
            y: batch_size
            prototypes: num_classes x latent_dim
        """
        similarity = torch.mm(latent_var, prototypes.t()) #matrix multiplication (b x d) x (K x d)^T -> (b x K)
        out = F.cross_entropy(similarity, y) #cross-entropy loss
        return out
    
    def decode(self, latent_var, prototypes):
        """Performs nearest-neighbour decoding on:
            latent_var: batch_size x latent_dim
            prototypes: num_classes x latent_dim
        """
        with torch.no_grad():
            similarity = torch.mm(latent_var, prototypes.t())
            yhat = similarity.argmax(dim=1) #closest is max inner product
            return yhat

if __name__ == "__main__":
    pass
