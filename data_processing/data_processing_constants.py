
MERONYM_RELATIONS = ["has-part", "has-region", "element", "possesses", "material"]
SPATIAL_RELATIONS = ["is-at", "is-inside", "is-outside", "abuts", "between"]
TAXONOMY_RELATIONS = ["subclass-of", "instance-of"]
INCLUDE_RELATIONS = MERONYM_RELATIONS + SPATIAL_RELATIONS + TAXONOMY_RELATIONS 

OPENSTAX_TEXTBOOKS = ["Anatomy_and_Physiology", "Astronomy", "Biology_2e", "Chemistry_2e",
                      "Microbiology", "Psychology", "University_Physics_Volume_1",
                      "University_Physics_Volume_2", "University_Physics_Volume_3"]

EXCLUDE_SECTIONS = [
    "Preface", "Chapter Outline", "Index", "Chapter Outline", "Summary", "Multiple Choice",
    "Fill in the Blank", "short Answer", "Critical Thinking", "References", 
    "Units", "Conversion Factors", "Fundamental Constants", "Astronomical Data",
    "Mathematical Formulas", "The Greek Alphabet", "Chapter 1", "Chapter 2",
    "Chapter 3", "Chapter 4", "Chapter 5", "Chapter 6", "Chapter 7", "Chapter 8"
    "Chapter 9", "Chapter 10", "Chapter 11", "Chapter 12", "Chapter 13", "Chapter 14", 
    "Chapter 15", "Chapter 16", "Chapter 17", "Critical Thinking Questions", 
    "Visual Connection Questions", "Key Terms", "Review Questions", "Glossary",
    "The Periodic Table of Elements", "Measurements and the Metric System"]