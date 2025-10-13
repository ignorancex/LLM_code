import numpy as np

def drf_pred(record):

    wrist_swelling = record.get('wrist_swelling', False)
    visible_deformity = record.get('visible_deformity', False)
    radius_tender = record.get('radius_tender', False)
    palmar_flexion_pain = record.get('palmar_flexion_pain', False)
    supination_pain = record.get('supination_pain', False)
    radioulnar_ballottement_pain = record.get('radioulnar_ballottement_pain', False)

    if wrist_swelling is None:
      wrist_swelling = False
    
    if visible_deformity is None:
      visible_deformity = False
    
    if radius_tender is None:
      radius_tender = False
    
    if palmar_flexion_pain is None:
      palmar_flexion_pain = False
    
    if supination_pain is None:
      supination_pain = False
    
    if radioulnar_ballottement_pain is None:
      radioulnar_ballottement_pain = False

    if wrist_swelling or visible_deformity or radius_tender or palmar_flexion_pain or supination_pain or radioulnar_ballottement_pain:
        return ("High likelihood of distal radial fracture", None)
    else:
        return ("Low likelihood of distal radial fracture", None)

def wf_pred(record):

    age = record.get('age', 30) or 30 # world's median age
    sex = record.get('sex', 'F') or 'F'
    wrist_swelling = record.get('wrist_swelling', False)
    snuffbox_swelling = record.get('snuffbox_swelling', False)
    visible_deformity = record.get('visible_deformity', False)
    radius_tender = record.get('radius_tender', False)
    radial_deviation_pain = record.get('radial_deviation_pain', False)
    thumb_compression_pain = record.get('thumb_compression_pain', False)

    if wrist_swelling is None:
      wrist_swelling = False
    
    if snuffbox_swelling is None:
      snuffbox_swelling = False
    
    if visible_deformity is None:
      visible_deformity = False
    
    if radius_tender is None:
      radius_tender = False
    
    if radial_deviation_pain is None:
      radial_deviation_pain = False
    
    if thumb_compression_pain is None:
      thumb_compression_pain = False
    # Coefficients from the Amsterdam Wrist Rules (for all wrist fractures)
    COEFFICIENTS = {
        "intercept": -3.616,
        "age": 0.0309,
        "sex_male": 0.5862,
        "wrist_swelling": 1.1486,
        "snuffbox_swelling": 0.5757,
        "visible_deformity": 1.7123,
        "radius_tender": 0.7029,
        "radial_deviation_pain": 0.4963,
        "thumb_compression_pain": -0.1793
    }

    CUTOFF_PROBABILITY = 0.21  # 21% cutoff used in the study

    # Compute linear predictor (logit)
    logit = (
        COEFFICIENTS["intercept"] +
        COEFFICIENTS["age"] * age +
        COEFFICIENTS["sex_male"] * (1 if sex.upper() == "M" else 0) +
        COEFFICIENTS["wrist_swelling"] * wrist_swelling +
        COEFFICIENTS["snuffbox_swelling"] * snuffbox_swelling +
        COEFFICIENTS["visible_deformity"] * visible_deformity +
        COEFFICIENTS["radius_tender"] * radius_tender +
        COEFFICIENTS["radial_deviation_pain"] * radial_deviation_pain +
        COEFFICIENTS["thumb_compression_pain"] * thumb_compression_pain
    )

    # Convert logit to probability
    probability = 1 / (1 + np.exp(-logit))

    # Determine fracture likelihood
    return ("High likelihood of wrist fracture", None) if probability >= CUTOFF_PROBABILITY else ("Low likelihood of wrist fracture", None)


def cspine_pred(record):

    age = record.get('age', 30) or 30
    gcs = record.get('gcs', 15) or 15
    hemodynamically_stable = record.get('hemodynamically_stable', True)
    acute_paralysis = record.get('acute_paralysis', False)
    cervical_spine_history = record.get('cervical_spine_history', False)
    dangerous_mechanism = record.get('dangerous_mechanism', False)
    extremity_paresthesias = record.get('extremity_paresthesias', False)
    neck_pain_delayed = record.get('neck_pain_delayed', False)
    ambulatory = record.get('ambulatory', True)
    no_midline_ttp = record.get('no_midline_ttp', True)
    rom_45_degrees = record.get('rom_45_degrees', True)
    simple_mvc = record.get('simple_mvc', True)

    if hemodynamically_stable is None:
      hemodynamically_stable = True
    
    if acute_paralysis is None:
      acute_paralysis = False
    
    if cervical_spine_history is None:
      cervical_spine_history = False
    
    if dangerous_mechanism is None:
      dangerous_mechanism = False
    
    if extremity_paresthesias is None:
      extremity_paresthesias = False
    
    if neck_pain_delayed is None:
      neck_pain_delayed = False
    
    if ambulatory is None:
      ambulatory = True
    
    if no_midline_ttp is None:
      no_midline_ttp = True
    
    if rom_45_degrees is None:
      rom_45_degrees = True
    
    if simple_mvc is None:
      simple_mvc = True

    # High-risk features that mandate radiography
    if age > 65 or extremity_paresthesias or dangerous_mechanism:
        return ("CT recommended", None)

    # Low-risk features which allow safe assessment of range of motion
    if not (neck_pain_delayed or ambulatory or not no_midline_ttp or simple_mvc):
        return ("CT recommended", None)

    # Range of motion check
    if not rom_45_degrees:
        return ("CT recommended", None)

    return ("CT not recommended", None)

def ct_head_pred(record):

    age = record.get('age', 30) or 30
    gcs_initial = record.get('gcs_initial', 15) or 15
    gcs_2hr = record.get('gcs_2hr', 15) or 15
    has_coagulopathy = record.get('has_coagulopathy', False)
    had_seizure = record.get('had_seizure', False)
    vomiting_episodes = record.get('vomiting_episodes', 0)
    suspected_skull_fracture = record.get('suspected_skull_fracture', False)
    signs_of_basal_skull_fracture = record.get('signs_of_basal_skull_fracture', False)
    retrograde_amnesia_duration = record.get('retrograde_amnesia_duration', 0)
    dangerous_mechanism = record.get('dangerous_mechanism', False)
    meets_inclusion = record.get('meets_inclusion', True)

    if has_coagulopathy is None:
      has_coagulopathy = False
    
    if had_seizure is None:
      had_seizure = False
    
    if vomiting_episodes is None:
      vomiting_episodes = 0
    
    if suspected_skull_fracture is None:
      suspected_skull_fracture = False
    
    if signs_of_basal_skull_fracture is None:
      signs_of_basal_skull_fracture = False
    
    if retrograde_amnesia_duration is None:
      retrograde_amnesia_duration = 0
    
    if dangerous_mechanism is None:
      dangerous_mechanism = False
    
    if meets_inclusion is None:
      meets_inclusion = True
    # High-risk criteria that mandate CT scan
    if (age >= 65 or vomiting_episodes > 2 or suspected_skull_fracture or
        signs_of_basal_skull_fracture or gcs_2hr < 15 or retrograde_amnesia_duration > 30 or dangerous_mechanism):
        return ("CT head indicated", None)

    return ("CT head not required", None)

def bcvi_pred(record):

    arterial_hemorrhage = record.get('arterial_hemorrhage', False)
    cervical_bruit_under_50 = record.get('cervical_bruit_under_50', False)
    expanding_hematoma = record.get('expanding_hematoma', False)
    focal_neurologic_deficit = record.get('focal_neurologic_deficit', False)
    neurologic_deficit_incongruous = record.get('neurologic_deficit_incongruous', False)
    stroke_on_imaging = record.get('stroke_on_imaging', False)
    high_energy_mechanism = record.get('high_energy_mechanism', False)
    lefort_fracture = record.get('lefort_fracture', False)
    mandible_fracture = record.get('mandible_fracture', False)
    complex_skull_fracture = record.get('complex_skull_fracture', False)
    base_skull_fracture = record.get('base_skull_fracture', False)
    scalp_degloving = record.get('scalp_degloving', False)
    cervical_spine_injury = record.get('cervical_spine_injury', False)
    severe_tbi = record.get('severe_tbi', False)
    near_hanging = record.get('near_hanging', False)
    clothesline_injury = record.get('clothesline_injury', False)
    thoracic_injury = record.get('thoracic_injury', False)
    upper_rib_fractures = record.get('upper_rib_fractures', False)
    thoracic_vascular_injury = record.get('thoracic_vascular_injury', False)
    cardiac_rupture = record.get('cardiac_rupture', False)

    if arterial_hemorrhage is None:
      arterial_hemorrhage = False
    
    if cervical_bruit_under_50 is None:
      cervical_bruit_under_50 = False
    
    if expanding_hematoma is None:
      expanding_hematoma = False
    
    if focal_neurologic_deficit is None:
      focal_neurologic_deficit = False
    
    if neurologic_deficit_incongruous is None:
      neurologic_deficit_incongruous = False
    
    if stroke_on_imaging is None:
      stroke_on_imaging = False
    
    if high_energy_mechanism is None:
      high_energy_mechanism = False
    
    if lefort_fracture is None:
      lefort_fracture = False
    
    if mandible_fracture is None:
      mandible_fracture = False
    
    if complex_skull_fracture is None:
      complex_skull_fracture = False
    
    if base_skull_fracture is None:
      base_skull_fracture = False
    
    if scalp_degloving is None:
      scalp_degloving = False
    
    if cervical_spine_injury is None:
      cervical_spine_injury = False
    
    if severe_tbi is None:
      severe_tbi = False
    
    if near_hanging is None:
      near_hanging = False
    
    if clothesline_injury is None:
      clothesline_injury = False
    
    if thoracic_injury is None:
      thoracic_injury = False
    
    if upper_rib_fractures is None:
      upper_rib_fractures = False
    
    if thoracic_vascular_injury is None:
      thoracic_vascular_injury = False
    
    if cardiac_rupture is None:
      cardiac_rupture = False
    # Check if any high-risk signs or symptoms are present
    if (arterial_hemorrhage or cervical_bruit_under_50 or expanding_hematoma or
        focal_neurologic_deficit or neurologic_deficit_incongruous or stroke_on_imaging):
        return ("CTA indicated to evaluate for BCVI", None)

    # Check if high-energy mechanism and additional risk factors are present
    if high_energy_mechanism and (lefort_fracture or mandible_fracture or complex_skull_fracture or
                                  base_skull_fracture or scalp_degloving or cervical_spine_injury or
                                  severe_tbi or near_hanging or clothesline_injury or thoracic_injury or
                                  upper_rib_fractures or thoracic_vascular_injury or cardiac_rupture):
        return ("CTA indicated to evaluate for BCVI", None)

    return ("No indication for CTA based on Denver criteria", None)

def bcvi_memphis_pred(record):

    skull_base_carotid = record.get('skull_base_carotid', False)
    skull_base_petrous = record.get('skull_base_petrous', False)
    cervical_spine_fracture = record.get('cervical_spine_fracture', False)
    unexplained_neuro_findings = record.get('unexplained_neuro_findings', False)
    horner_syndrome = record.get('horner_syndrome', False)
    lefort_fracture = record.get('lefort_fracture', False)
    neck_soft_tissue_injury = record.get('neck_soft_tissue_injury', False)

    if skull_base_carotid is None:
      skull_base_carotid = False
    
    if skull_base_petrous is None:
      skull_base_petrous = False
    
    if cervical_spine_fracture is None:
      cervical_spine_fracture = False

    if unexplained_neuro_findings is None:
      unexplained_neuro_findings = False
    
    if horner_syndrome is None:
      horner_syndrome = False
    
    if lefort_fracture is None:
      lefort_fracture = False
    
    if neck_soft_tissue_injury is None:
      neck_soft_tissue_injury = False
    # Check if any criteria are present
    if (skull_base_carotid or skull_base_petrous or cervical_spine_fracture or
        unexplained_neuro_findings or horner_syndrome or lefort_fracture or
        neck_soft_tissue_injury):
        return ("CTA or DSA study indicated to exclude BCVI", None)

    return ("No specific indication for CTA/DSA study", None)

def nexus_cspine_pred(record):

    midline_tenderness = record.get('midline_tenderness', False) # check this feature with Austin
    focal_deficit = record.get('focal_deficit', False)
    altered_alertness = record.get('altered_alertness', False)
    intoxication = record.get('intoxication', False)
    distracting_injury = record.get('distracting_injury', False)

    if midline_tenderness is None:
      midline_tenderness = False
    
    if focal_deficit is None:
      focal_deficit = False
    
    if altered_alertness is None:
      altered_alertness = False
    
    if intoxication is None:
      intoxication = False
    
    if distracting_injury is None:
      distracting_injury = False
    # Check if ALL criteria are negative (low risk)
    low_risk = (
        not midline_tenderness and
        not focal_deficit and
        not altered_alertness and
        not intoxication and
        not distracting_injury
    )

    if low_risk:
        return ("Imaging not necessary", None)

    return ("Imaging recommended", None)

def nexus_chest_ct_pred(record):

    major_injury = record.get('major_injury', False)
    abnormal_cxr = record.get('abnormal_cxr', False)
    distracting_injury = record.get('distracting_injury', False)
    chest_wall_tenderness = record.get('chest_wall_tenderness', False)
    sternum_tenderness = record.get('sternum_tenderness', False)
    thoracic_spine_tenderness = record.get('thoracic_spine_tenderness', False)
    scapula_tenderness = record.get('scapula_tenderness', False)
    rapid_deceleration = record.get('rapid_deceleration', False)

    if major_injury is None:
      major_injury = False
    
    if abnormal_cxr is None:
      abnormal_cxr = False
    
    if distracting_injury is None:
      distracting_injury = False
    
    if chest_wall_tenderness is None:
      chest_wall_tenderness = False
    
    if sternum_tenderness is None:
      sternum_tenderness= False
    
    if thoracic_spine_tenderness is None:
      thoracic_spine_tenderness = False
    
    if scapula_tenderness is None:
      scapula_tenderness = False
    
    if rapid_deceleration is None:
      rapid_deceleration = False

    # Check if any tenderness is present
    any_tenderness = (
        chest_wall_tenderness or
        sternum_tenderness or
        thoracic_spine_tenderness or
        scapula_tenderness
    )

    # Base criteria for both major and minor injury
    ct_indicated = abnormal_cxr or distracting_injury or any_tenderness

    # Additional criterion for minor injury
    if not major_injury:
        ct_indicated = ct_indicated or rapid_deceleration

    if ct_indicated:
        return ("CT indicated", None)

    return ("CT not indicated", None)

def ottawa_foot_pred(record):
    midfoot_pain = record.get('midfoot_pain', False)
    weight_bearing_difficulty = record.get('weight_bearing_difficulty', False)
    navicular_tenderness = record.get('navicular_tenderness', False)
    fifth_metatarsal_tenderness = record.get('fifth_metatarsal_tenderness', False)
    
    if midfoot_pain is None:
      midfoot_pain = False
    
    if weight_bearing_difficulty is None:
      weight_bearing_difficulty = False
    
    if navicular_tenderness is None:
      navicular_tenderness = False
    
    if fifth_metatarsal_tenderness is None:
      fifth_metatarsal_tenderness = False

    # Check midfoot pain (required for X-ray consideration)
    if not midfoot_pain:
        return ("X-ray not required", None)

    # Check additional criteria
    if weight_bearing_difficulty or navicular_tenderness or fifth_metatarsal_tenderness:
        return ("X-ray required", None)

    return ("X-ray not required", None)

def ottawa_ankle_pred(record):

    malleolar_pain = record.get('malleolar_pain', False)
    weight_bearing_difficulty = record.get('weight_bearing_difficulty', False)
    lateral_malleolus_tenderness = record.get('lateral_malleolus_tenderness', False)
    medial_malleolus_tenderness = record.get('medial_malleolus_tenderness', False)


    if malleolar_pain is None:
      malleolar_pain = False
    
    if weight_bearing_difficulty is None:
      weight_bearing_difficulty = False
    
    if lateral_malleolus_tenderness is None:
      lateral_malleolus_tenderness = False
    
    if medial_malleolus_tenderness is None:
      medial_malleolus_tenderness = False

    # Check for malleolar pain (required)
    if not malleolar_pain:
        return ("X-ray not required", None)

    # Check other criteria
    if weight_bearing_difficulty or lateral_malleolus_tenderness or medial_malleolus_tenderness:
        return ("X-ray required", None)

    return ("X-ray not required", None)

def ottawa_knee_pred(record):

    age = record.get('age', 30) or 30
    acute_injury = record.get('acute_injury', True)
    patella_tenderness = record.get('patella_tenderness', False)
    fibular_head_tenderness = record.get('fibular_head_tenderness', False)
    limited_flexion = record.get('limited_flexion', False)
    weight_bearing_difficulty = record.get('weight_bearing_difficulty', False)

    if acute_injury is None:
      acute_injury = True

    if patella_tenderness is None:
      patella_tenderness = False

    if fibular_head_tenderness is None:
      fibular_head_tenderness = False
    
    if limited_flexion is None:
      limited_flexion = False
    
    if weight_bearing_difficulty is None:
      weight_bearing_difficulty = False
      
    if not acute_injury:
        return ("X-ray not required", None)

    # Check Ottawa Knee Rule criteria
    if (age > 55 or patella_tenderness or fibular_head_tenderness or limited_flexion or weight_bearing_difficulty):
        return ("X-ray required", None)

    return ("X-ray not required", None)

def pittsburgh_knee_pred(record):

    age = record.get('age', 30) or 30
    fall_or_trauma = record.get('fall_or_trauma', False)
    cannot_walk_four_steps = record.get('cannot_walk_four_steps', False)

    if fall_or_trauma is None:
      fall_or_trauma = False
    
    if cannot_walk_four_steps is None:
      cannot_walk_four_steps = False

    age_criteria = age < 12 or age > 50
    trauma_with_age = fall_or_trauma and age_criteria

    if trauma_with_age or cannot_walk_four_steps:
        return ("Knee radiograph indicated", None)
    else:
        return ("Knee radiograph not indicated", None)
        
def pecarn_cspine_pred(record):

    gcs = record.get('gcs', 15) or 15
    avpu = record.get('avpu', 'a') or 'a'
    abnormal_breathing = record.get('abnormal_breathing', False)
    abnormal_airway = record.get('abnormal_airway', False)
    abnormal_circulation = record.get('abnormal_circulation', False)
    focal_neuro_deficits = record.get('focal_neuro_deficits', False)
    ams = record.get('ams', False)
    neck_pain = record.get('neck_pain', False)
    neck_tenderness = record.get('neck_tenderness', False)
    head_injury = record.get('head_injury', False)
    torso_injury = record.get('torso_injury', False)

    if abnormal_breathing is None:
      abnormal_breathing = False
    
    if abnormal_airway is None:
      abnormal_airway = False
    
    if abnormal_circulation is None:
      abnormal_circulation = False
    
    if focal_neuro_deficits is None:
      focal_neuro_deficits = False
    
    if ams is None:
      ams = False
    
    if neck_pain is None:
      neck_pain = False
    
    if neck_tenderness is None:
      neck_tenderness = False
    
    if head_injury is None:
      head_injury = False
    
    if torso_injury is None:
      torso_injury = False
    if (3 <= gcs <= 8) or avpu == 'u' or abnormal_breathing or abnormal_airway or abnormal_circulation or focal_neuro_deficits:
        return ('Consider CT', 0.121)
    elif (9 <= gcs <= 14) or avpu in ['v', 'p'] or ams or neck_pain or neck_tenderness or head_injury or torso_injury:
        return ('Consider plain x-ray', 0.028)
    else:
        return ('Consider clinical clearance', 0.002)

def pecarn_tbi_pred(record):

    age = record.get('age', 30) or 30
    gcs = record.get('gcs', 15) or 15
    altered_mental_status = record.get('altered_mental_status', False)
    skull_fracture = record.get('skull_fracture', False)
    scalp_hematoma = record.get('scalp_hematoma', False)
    loc_history = record.get('loc_history', False)
    severe_mechanism = record.get('severe_mechanism', False)
    not_acting_normally = record.get('not_acting_normally', False)
    vomiting_history = record.get('vomiting_history', False)
    severe_headache = record.get('severe_headache', False)

    if altered_mental_status is None:
      altered_mental_status = False
    
    if skull_fracture is None:
      skull_fracture = False
    
    if scalp_hematoma is None:
      scalp_hematoma = False
    
    if loc_history is None:
      loc_history = False
    
    if severe_mechanism is None:
      severe_mechanism = False
    
    if not_acting_normally is None:
      not_acting_normally = False
    
    if vomiting_history is None:
      vomiting_history = False
    
    if severe_headache is None:
      severe_headache = False
    if age < 2:
        if gcs == 14 or altered_mental_status or skull_fracture:
            return ("CT scan", 0.044)
        elif scalp_hematoma or loc_history or severe_mechanism or not_acting_normally:
            return ("Conditional CT Scan", 0.009)
        else:
            return ("Clinical clearance", 0.002)
    else:
        if gcs == 14 or altered_mental_status or skull_fracture:
            return ("CT scan", 0.043)
        elif loc_history or vomiting_history or severe_mechanism or severe_headache:
            return ("Conditional CT Scan", 0.008)
        else:
            return ("Clinical clearance", 0.005)

def pecarn_iai_pred(record):
    gcs = record.get('gcs', 15) or 15  # Assuming default GCS is 15 which is a normal value
    abdominal_wall_trauma = record.get('abdominal_wall_trauma', False)
    seatbelt_sign = record.get('seatbelt_sign', False)
    abdominal_tenderness = record.get('abdominal_tenderness', False)
    thoracic_wall_trauma = record.get('thoracic_wall_trauma', False)
    abdominal_pain = record.get('abdominal_pain', False)
    decreased_breathing_sounds = record.get('decreased_breathing_sounds', False)
    vomiting = record.get('vomiting', False)

    if abdominal_wall_trauma is None:
      abdominal_wall_trauma = False
    
    if seatbelt_sign is None:
      seatbelt_sign = False
    
    if abdominal_tenderness is None:
      abdominal_tenderness = False
    
    if thoracic_wall_trauma is None:
      thoracic_wall_trauma = False
    
    if abdominal_pain is None:
      abdominal_pain = False
    
    if decreased_breathing_sounds is None:
      decreased_breathing_sounds = False
    
    if vomiting is None:
      vomiting = False
    if gcs < 14 or abdominal_wall_trauma or seatbelt_sign:
        return ('CT scan', 0.054)
    elif abdominal_tenderness:
        return ('CT scan', 0.014)
    elif thoracic_wall_trauma or abdominal_pain or decreased_breathing_sounds or vomiting:
        return ('CT scan', 0.007)
    else:
        return ('Clinical clearance', 0.001)

