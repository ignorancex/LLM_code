import numpy as np

def criteria_drf_pred(record):
    trauma = record.get('trauma_relevant', True)

    if trauma is None:
      trauma = True

    if not trauma:
      return "Exclude"
    
    return "Include"

def criteria_wf_pred(record):
    trauma = record.get('trauma_relevant', True)

    if trauma is None:
      trauma = True

    if not trauma:
      return "Exclude"
    
    return "Include"

def criteria_cspine_pred(record):
    
    age = record.get('age', 30) or 30
    gcs = record.get('gcs', 15) or 15
    hemodynamically_stable = record.get('hemodynamically_stable', True)
    acute_paralysis = record.get('acute_paralysis', False)
    cervical_spine_history = record.get('cervical_spine_history', False)
    trauma = record.get('trauma_relevant', True)

    if hemodynamically_stable is None:
      hemodynamically_stable = True
    
    if acute_paralysis is None:
      acute_paralysis = False
    
    if cervical_spine_history is None:
      cervical_spine_history = False
    
    if trauma is None:
      trauma = True

    if not trauma:
      return "Exclude"

    # Exclusion criteria
    if age < 16 or gcs != 15 or not hemodynamically_stable or acute_paralysis or cervical_spine_history:
        return "Exclude"

    return "Include"

def criteria_ct_head_pred(record):

    age = record.get('age', 30) or 30
    gcs_initial = record.get('gcs_initial', 15) or 15
    has_coagulopathy = record.get('has_coagulopathy', False)
    had_seizure = record.get('had_seizure', False)
    meets_inclusion = record.get('meets_inclusion', True)
    trauma = record.get('trauma_relevant', True)

    if has_coagulopathy is None:
      has_coagulopathy = False
    
    if had_seizure is None:
      had_seizure = False
    
    if meets_inclusion is None:
      meets_inclusion = True

    if trauma is None:
      trauma = True

    if not trauma:
      return "Exclude"

    # Exclusion criteria
    if not (meets_inclusion and 13 <= gcs_initial <= 15 and age >= 16 and not has_coagulopathy and not had_seizure):
        return "Exclude"

    return "Include"

def criteria_bcvi_pred(record):
    trauma = record.get('trauma_relevant', True)

    if trauma is None:
      trauma = True

    if not trauma:
      return "Exclude"
    
    return "Include"

def criteria_bcvi_memphis_pred(record):
    trauma = record.get('trauma_relevant', True)

    if trauma is None:
      trauma = True

    if not trauma:
      return "Exclude"
    
    return "Include"

def criteria_nexus_cspine_pred(record):
    trauma = record.get('trauma_relevant', True)
    gcs = record.get('gcs', 15) or 15

    if trauma is None:
      trauma = True

    if (not trauma) or (gcs != 15):
      return "Exclude"
    
    return "Include"

def criteria_nexus_chest_ct_pred(record):
    age = record.get('age', 30) or 30
    gcs = record.get('gcs', 15) or 15
    trauma = record.get('trauma_relevant', True)
    hemodynamically_stable = record.get('hemodynamically_stable', True)
    intubated = record.get('intubated', False)

    if trauma is None:
      trauma = True
    
    if hemodynamically_stable is None:
      hemodynamically_stable = True
    
    if intubated is None:
      intubated = False

    if not trauma:
      return "Exclude"
    
    if gcs != 15 or intubated or not hemodynamically_stable or age < 15:
      return "Exclude"

    return "Include"

def criteria_ottawa_foot_pred(record):

    age = record.get('age', 30) or 30
    blunt_trauma = record.get('blunt_trauma', True)
    acute_injury = record.get('acute_injury', True)
    hindfoot_forefoot = record.get('hindfoot_forefoot', False)

    if blunt_trauma is None:
      blunt_trauma = True
    
    if acute_injury is None:
      acute_injury = True
    
    if hindfoot_forefoot is None:
      hindfoot_forefoot = False

    # Exclusion criteria
    if age < 2 or not blunt_trauma or not acute_injury or hindfoot_forefoot:
        return "Exclude"

    return "Include"

def criteria_ottawa_ankle_pred(record):

    age = record.get('age', 30) or 30
    blunt_trauma = record.get('blunt_trauma', True)
    acute_injury = record.get('acute_injury', True)

    if blunt_trauma is None:
      blunt_trauma = True
    
    if acute_injury is None:
      acute_injury = True

    # Exclusion criteria
    if age < 2 or not blunt_trauma or not acute_injury:
        return "Exclude"

    return "Include"

def criteria_ottawa_knee_pred(record):

    age = record.get('age', 30) or 30
    trauma = record.get('trauma_relevant', True)

    if trauma is None:
      trauma = True

    if age < 2 or not trauma:
      return "Exclude"
    
    return "Include"

def criteria_pittsburgh_knee_pred(record):
    trauma = record.get('trauma_relevant', True)
    acute_injury = record.get('acute_injury', True)
    injury_past_one_week = record.get('injury_past_one_week', True)

    if trauma is None:
      trauma = True
    
    if acute_injury is None:
      acute_injury = True
    
    if injury_past_one_week is None:
      injury_past_one_week = True

    if not trauma or not acute_injury or not injury_past_one_week:
      return "Exclude"
    
    return "Include"

def criteria_pecarn_cspine_pred(record):
    
    age = record.get('age', 30) or 30
    trauma = record.get('trauma_relevant', True)

    if trauma is None:
      trauma = True

    # Exclusion criteria
    if age >= 18 or not trauma:
        return "Exclude"

    return "Include"

def criteria_pecarn_tbi_pred(record):

    age = record.get('age', 30) or 30
    gcs = record.get('gcs', 15) or 15
    trauma = record.get('trauma_relevant', True)

    if trauma is None:
      trauma = True

    # Exclusion criteria
    if age >= 18 or gcs < 14 or not trauma:
        return "Exclude"

    return "Include"

def criteria_pecarn_iai_pred(record):
    
    age = record.get('age', 30) or 30
    trauma = record.get('trauma_relevant', True)
    trauma_within_24_hrs = record.get('trauma_within_24_hrs', True)
    penetrating_trauma = record.get('penetrating_trauma', False)
    is_pregnant = record.get('is_pregnant', False)
    pre_existing_neurologic_disorders = record.get('pre_existing_neurologic_disorders', False)

    if trauma is None:
      trauma = True
    
    if trauma_within_24_hrs is None:
      trauma_within_24_hrs = True
    
    if penetrating_trauma is None:
      penetrating_trauma = False
    
    if is_pregnant is None:
      is_pregnant = False
    
    if pre_existing_neurologic_disorders is None:
      pre_existing_neurologic_disorders = False

    if age >= 18 or not trauma:
      return "Exclude"

    if trauma and not trauma_within_24_hrs:
      return "Exclude"
    
    if penetrating_trauma or is_pregnant or pre_existing_neurologic_disorders:
      return "Exclude"

    return "Include"
