from itertools import count, islice

import numpy as np

from skimage.draw import rectangle


def n_rooms_square(num_rooms: int, room_size: int, gap_size: int) -> np.ndarray:
    assert float.is_integer(np.sqrt(num_rooms)), f"Argument `num_rooms` should be square, instead got {num_rooms}"
    assert room_size > 1, f"Room size should be larger than 1, got {room_size}"
    assert 0 < gap_size < room_size - 1, f"Argument `gap_size` should be in the range [1, room_size - 2], instead got {gap_size}"

    if room_size % 2 ^ gap_size % 2:
        print("Warning, room_size and gap_size are best kept as uniformly odd or even, not one odd one even.")

    n = int(np.sqrt(num_rooms))  # Get sqrt of num_rooms to get n x n dimensions.
    x = np.zeros([(room_size + 1) * n + 1, (room_size + 1) * n + 1], dtype=np.uint8)
    
    # Draw room-walls with 1s
    walls = np.linspace(0, (room_size + 1) * n, n + 1, dtype=np.int32)
    x[:, walls] = x[walls, :] = 1
    
    # Reopen gaps in walls with 0s
    gaps = np.arange(1 + room_size // 2, (room_size + 1) * n, room_size + 1)  # yields [0, gap_1, gap_2, (n + 1) * room_size - 1][1:-1]
    for gap in gaps:
        lower = gap - (gap_size - 1) // 2
        upper = gap + gap_size // 2 + 1

        if not room_size % 2:  # Shift
            lower -= 1
            upper -= 1

        x[lower:upper,1:-1] = x[1:-1,lower:upper] = 0
    
    return x.astype(np.uint8)
