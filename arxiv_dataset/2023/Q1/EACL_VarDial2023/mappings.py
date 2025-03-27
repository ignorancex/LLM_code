label_to_int_mappings_3class = {
    "EN-US": 0,
    "EN-GB": 1,
    "EN": 2,
    "ES-ES": 3,
    "ES-AR": 4,
    "ES": 5,
    "PT-BR": 6,
    "PT-PT": 7,
    "PT": 8,
}

label_to_int_mappings_2class = {
    "EN-US": 0,
    "EN-GB": 1,
    "ES-ES": 2,
    "ES-AR": 3,
    "PT-BR": 4,
    "PT-PT": 5,
}


int_to_label_mappings__3class = {v: k for k, v in label_to_int_mappings_3class.items()}

int_to_label_mappings__2class = {v: k for k, v in label_to_int_mappings_2class.items()}


label_mappings_en_3class = {"EN-US": 0, "EN-GB": 1, "EN": 2}

label_mappings_es_3class = {"ES-ES": 0, "ES-AR": 1, "ES": 2}

label_mappings_pt_3class = {"PT-BR": 0, "PT-PT": 1, "PT": 2}


label_mappings_en_2class = {"EN-US": 0, "EN-GB": 1}

label_mappings_es_2class = {"ES-ES": 0, "ES-AR": 1}

label_mappings_pt_2class = {"PT-BR": 0, "PT-PT": 1}
