import itertools


def number_to_binary_list(number):
    binary_str = bin(number)[2:]
    binary_list = [int(digit) for digit in binary_str[::-1]]
    return binary_list


def binary_list_to_number(binary_list):
    binary_str = ''.join(str(digit) for digit in binary_list[::-1])
    decimal_number = int(binary_str, 2)
    return decimal_number


def get_personas_in_group(group_number):
    personas_in_group = []
    binary_list = number_to_binary_list(group_number)
    for i, val in enumerate(binary_list):
        if val:
            personas_in_group.append(i)
    return personas_in_group


def set_of_personas_to_group_code(personas_set):
    assert all(isinstance(item, int) for item in personas_set), "Not all items in the set are integers."
    binary_list_rep = [0] * (max(personas_set) + 1)
    for item in personas_set:
        binary_list_rep[item] = 1
    group_code = binary_list_to_number(binary_list_rep)
    return group_code


def show_info(personas):
    idx_group = set_of_personas_to_group_code(personas)
    binary_representation = number_to_binary_list(idx_group)
    inverse = binary_list_to_number(binary_representation)
    assert idx_group == inverse
    return f"personas: {personas} | group index: {idx_group} | binary representation: {binary_representation} | " \
           f"personas in group: {get_personas_in_group(idx_group)}"


def get_power_group(personas, always_include=[], include_empty_group=False):
    permutations = list(list(itertools.combinations(personas, i)) for i in range(int(not include_empty_group), len(personas) + 1))
    permutations = [set(list(item) + list(always_include)) for sublist in permutations for item in sublist]
    power_group_codes = []
    for perm in permutations:
        power_group_codes.append(set_of_personas_to_group_code(perm))
    return sorted(power_group_codes)
