import numpy as np
np.triu(np.ones((5,5)),k=0)
def block_diag(matrices):
    n = len(matrices)

    # Create a list of lists to hold the block structure
    blocks = [[None] * n for _ in range(n)]

    for i, matrix in enumerate(matrices):
        blocks[i][i] = matrix

    # Fill the rest with zero matrices of appropriate size
    for i in range(n):
        for j in range(n):
            if blocks[i][j] is None:
                shape = matrices[i].shape[0], matrices[j].shape[1]
                blocks[i][j] = np.zeros(shape)

    # Use np.block to combine the matrices
    return np.block(blocks)

matrices = []
for i in range(32):
  matrices.append(np.triu(np.ones((256,256)),k=0))

pattern = block_diag(matrices)
np.save('attn_mask.npy', arr)
