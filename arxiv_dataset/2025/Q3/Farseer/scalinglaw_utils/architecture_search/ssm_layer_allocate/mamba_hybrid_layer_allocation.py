

class Symbols():
    MAMBA = 'M'
    ATTENTION = '*'
    MLP = '-'
    ALL = 'A'
    VALID = {MAMBA, ATTENTION, MLP, ALL}

def _allocate_auto(total_layers_count: int, target_attention_ratio: float, target_mlp_ratio: float) -> list:
    attention_layers_count: int = round((total_layers_count * target_attention_ratio))
    mamba_layers_count: int = (total_layers_count - attention_layers_count)
    mamba_sections_count: int = (attention_layers_count + 1)
    mamba_section_length: float = (mamba_layers_count / mamba_sections_count)
    layer_type_list = ([Symbols.MAMBA] * total_layers_count)
    x: float = mamba_section_length
    for l in range(total_layers_count):
        if (x < 0.5):
            layer_type_list[l] = Symbols.ATTENTION
            x += mamba_section_length
        else:
            x -= 1
    mlp_layers_count: int = round((total_layers_count * target_mlp_ratio))
    if (mlp_layers_count > 0):
        mamba_layers_count -= mlp_layers_count
        mamba_to_mlp_ratio: float = (mamba_layers_count / mlp_layers_count)
        x: float = mamba_to_mlp_ratio
        for l in range(total_layers_count):
            if (layer_type_list[l] == Symbols.MAMBA):
                if (x < 0.5):
                    layer_type_list[l] = Symbols.MLP
                    x += mamba_to_mlp_ratio
                else:
                    x -= 1
    return layer_type_list

def _allocate_override(total_layers_count: int, override_pattern: str) -> list:
    layer_type_list = list(override_pattern)
    override_pattern_length = len(layer_type_list)
    if (override_pattern_length != total_layers_count):
        raise ValueError(f'The hybrid override pattern is the wrong length: got {override_pattern_length}, expected {total_layers_count}')
    for l in layer_type_list:
        if (l not in Symbols.VALID):
            raise ValueError(f"In hybrid override pattern, '{l}' is not one of {Symbols.VALID}")
    return layer_type_list

def _layer_counts_match(a: list, b: list) -> bool:
    for s in Symbols.VALID:
        if (a.count(s) != b.count(s)):
            return False
    return True

def allocate_layers(total_layers_count: int, target_attention_ratio: float, target_mlp_ratio: float, override_pattern: str=None, if_print=False) -> list:
    assert (total_layers_count > 0)
    assert ((target_attention_ratio >= 0.0) and (target_attention_ratio <= 1.0))
    assert ((target_mlp_ratio >= 0.0) and (target_mlp_ratio <= 1.0))
    assert ((target_attention_ratio + target_mlp_ratio) <= 1.0)
    layer_type_list = _allocate_auto(total_layers_count, target_attention_ratio, target_mlp_ratio)
    if (override_pattern is not None):
        layer_type_list_override = _allocate_override(total_layers_count, override_pattern)
        print('Using hybrid override pattern')
        if (((target_attention_ratio > 0.0) or (target_mlp_ratio > 0.0)) and (not _layer_counts_match(layer_type_list_override, layer_type_list))):
            raise ValueError('The number of each type of layer in the override pattern must match the number in the overridden pattern.')
        if (layer_type_list_override == layer_type_list):
            print('The override pattern matches the overridden pattern')
        else:
            print('Warning: overriding pattern A with pattern B')
            print(f"A: {''.join(layer_type_list)}")
            print(f"B: {''.join(layer_type_list_override)}")
        layer_type_list = layer_type_list_override
    if ((target_attention_ratio > 0.0) or (target_mlp_ratio > 0.0) or (override_pattern is not None)):
        actual_attention_layers_count = layer_type_list.count(Symbols.ATTENTION)
        actual_attention_ratio = (actual_attention_layers_count / total_layers_count)
        actual_mlp_layers_count = layer_type_list.count(Symbols.MLP)
        actual_mlp_ratio = (actual_mlp_layers_count / total_layers_count)
        allocation_string = ''.join(layer_type_list)
        if if_print:
            print(f'Hybrid allocation ({Symbols.MAMBA} is mamba, {Symbols.ATTENTION} is attention, {Symbols.MLP} is mlp):')
            print(allocation_string)
            print(f'{actual_attention_layers_count} attention layers in {total_layers_count} total layers.')
            print(f'Target attention ratio: {target_attention_ratio:.2f}. Actual attention ratio: {actual_attention_ratio:.2f}.')
            print(f'{actual_mlp_layers_count} mlp layers in {total_layers_count} total layers.')
            print(f'Target mlp ratio: {target_mlp_ratio:.2f}. Actual mlp ratio: {actual_mlp_ratio:.2f}.')
    return layer_type_list
if (__name__ == '__main__'):
    test_cases = [(9, 0.0, 0.0, 'MMMMMMMMM')]
    for t in test_cases:
        print('')
        allocate_layers(*t)
