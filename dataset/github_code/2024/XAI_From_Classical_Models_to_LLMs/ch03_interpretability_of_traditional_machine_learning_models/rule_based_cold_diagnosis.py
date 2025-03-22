# Filename: rule_based_cold_diagnosis.py

def diagnose(symptoms):
    """
    Diagnoses a condition based on the provided symptoms using a simple rule-based system.

    Args:
        symptoms (dict): A dictionary where keys are symptom names (e.g., 'fever', 'cough') 
                         and values are booleans indicating whether the symptom is present.

    Returns:
        str: The diagnosis based on the provided symptoms.
    """
    if symptoms.get('fever') and symptoms.get('cough'):
        return "Common Cold"
    elif symptoms.get('fever'):
        return "Fever of unknown origin"
    elif symptoms.get('cough'):
        return "Possible respiratory infection"
    else:
        return "No specific diagnosis"

if __name__ == "__main__":
    # Collect symptoms from the user
    fever = input("Do you have a fever? (yes/no): ").strip().lower() == 'yes'
    cough = input("Do you have a cough? (yes/no): ").strip().lower() == 'yes'

    # Create a dictionary of symptoms
    symptoms = {'fever': fever, 'cough': cough}

    # Get the diagnosis
    diagnosis = diagnose(symptoms)

    # Display the diagnosis
    print(f"Diagnosis: {diagnosis}")
