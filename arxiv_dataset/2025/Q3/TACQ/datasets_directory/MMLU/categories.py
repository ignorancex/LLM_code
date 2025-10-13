subcategories = {
    "abstract_algebra": ["math"],
    "anatomy": ["health"],
    "astronomy": ["physics"],
    "business_ethics": ["business"],
    "clinical_knowledge": ["health"],
    "college_biology": ["biology"],
    "college_chemistry": ["chemistry"],
    "college_computer_science": ["computer science"],
    "college_mathematics": ["math"], # 148
    "college_medicine": ["health"],
    "college_physics": ["physics"],
    "computer_security": ["computer science"],
    "conceptual_physics": ["physics"],
    "econometrics": ["economics"],
    "electrical_engineering": ["engineering"],
    "elementary_mathematics": ["math"], # 388
    "formal_logic": ["philosophy"],
    "global_facts": ["other"],
    "high_school_biology": ["biology"],
    "high_school_chemistry": ["chemistry"],
    "high_school_computer_science": ["computer science"],
    "high_school_european_history": ["history"],
    "high_school_geography": ["geography"],
    "high_school_government_and_politics": ["politics"], # 193
    "high_school_macroeconomics": ["economics"],
    "high_school_mathematics": ["math"], # 270
    "high_school_microeconomics": ["economics"],
    "high_school_physics": ["physics"],
    "high_school_psychology": ["psychology"],
    "high_school_statistics": ["math"], # 217
    "high_school_us_history": ["history"],
    "high_school_world_history": ["history"],
    "human_aging": ["health"],
    "human_sexuality": ["culture"],
    "international_law": ["law"],
    "jurisprudence": ["law"],
    "logical_fallacies": ["philosophy"],
    "machine_learning": ["computer science"],
    "management": ["business"],
    "marketing": ["business"],
    "medical_genetics": ["health"],
    "miscellaneous": ["other"],
    "moral_disputes": ["philosophy"],
    "moral_scenarios": ["philosophy"],
    "nutrition": ["health"],
    "philosophy": ["philosophy"],
    "prehistory": ["history"],
    "professional_accounting": ["other"],
    "professional_law": ["law"],
    "professional_medicine": ["health"],
    "professional_psychology": ["psychology"],
    "public_relations": ["politics"], # 110
    "security_studies": ["politics"], # 245
    "sociology": ["culture"],
    "us_foreign_policy": ["politics"], # 104
    "virology": ["health"],
    "world_religions": ["philosophy"],
}

reasoning_recall_subjects = {
    "reasoning": ["elementary_mathematics", "high_school_mathematics"], # 658
    "recall": ["high_school_government_and_politics", "public_relations", "security_studies", "us_foreign_policy"], # ~600
}

categories = {
    "STEM": ["physics", "chemistry", "biology", "computer science", "math", "engineering"],
    "humanities": ["history", "philosophy", "law"],
    "social sciences": ["politics", "culture", "economics", "geography", "psychology"],
    "other (business, health, misc.)": ["other", "business", "health"],
}

subject_to_category_GPT4o = {
    "abstract_algebra": "STEM",
    "anatomy": "other (business, health, misc.)",
    "astronomy": "STEM",
    "business_ethics": "other (business, health, misc.)",
    "clinical_knowledge": "other (business, health, misc.)",
    "college_biology": "STEM",
    "college_chemistry": "STEM",
    "college_computer_science": "STEM",
    "college_mathematics": "STEM",
    "college_medicine": "other (business, health, misc.)",
    "college_physics": "STEM",
    "computer_security": "STEM",
    "conceptual_physics": "STEM",
    "econometrics": "social sciences",
    "electrical_engineering": "STEM",
    "elementary_mathematics": "STEM",
    "formal_logic": "humanities",
    "global_facts": "other (business, health, misc.)",
    "high_school_biology": "STEM",
    "high_school_chemistry": "STEM",
    "high_school_computer_science": "STEM",
    "high_school_european_history": "humanities",
    "high_school_geography": "social sciences",
    "high_school_government_and_politics": "social sciences",
    "high_school_macroeconomics": "social sciences",
    "high_school_mathematics": "STEM",
    "high_school_microeconomics": "social sciences",
    "high_school_physics": "STEM",
    "high_school_psychology": "social sciences",
    "high_school_statistics": "STEM",
    "high_school_us_history": "humanities",
    "high_school_world_history": "humanities",
    "human_aging": "other (business, health, misc.)",
    "human_sexuality": "social sciences",
    "international_law": "humanities",
    "jurisprudence": "humanities",
    "logical_fallacies": "humanities",
    "machine_learning": "STEM",
    "management": "other (business, health, misc.)",
    "marketing": "other (business, health, misc.)",
    "medical_genetics": "other (business, health, misc.)",
    "miscellaneous": "other (business, health, misc.)",
    "moral_disputes": "humanities",
    "moral_scenarios": "humanities",
    "nutrition": "other (business, health, misc.)",
    "philosophy": "humanities",
    "prehistory": "humanities",
    "professional_accounting": "other (business, health, misc.)",
    "professional_law": "humanities",
    "professional_medicine": "other (business, health, misc.)",
    "professional_psychology": "social sciences",
    "public_relations": "social sciences",
    "security_studies": "social sciences",
    "sociology": "social sciences",
    "us_foreign_policy": "social sciences",
    "virology": "other (business, health, misc.)",
    "world_religions": "humanities",
    "results": "other (business, health, misc.)"
}

categories_to_subjects = {}
categories_to_subjects["STEM"] = ['astronomy', 'college_physics', 'conceptual_physics', 'high_school_physics', 'college_chemistry', 'high_school_chemistry', 'college_biology', 'high_school_biology', 'college_computer_science', 'computer_security', 'high_school_computer_science', 'machine_learning', 'abstract_algebra', 'college_mathematics', 'elementary_mathematics', 'high_school_mathematics', 'high_school_statistics', 'electrical_engineering']
categories_to_subjects["social sciences"] = ['high_school_government_and_politics', 'public_relations', 'security_studies', 'us_foreign_policy', 'human_sexuality', 'sociology', 'econometrics', 'high_school_macroeconomics', 'high_school_microeconomics', 'high_school_geography', 'high_school_psychology', 'professional_psychology']
categories_to_subjects["humanities"] = ['high_school_european_history', 'high_school_us_history', 'high_school_world_history', 'prehistory', 'formal_logic', 'logical_fallacies', 'moral_disputes', 'moral_scenarios', 'philosophy', 'world_religions', 'international_law', 'jurisprudence', 'professional_law']