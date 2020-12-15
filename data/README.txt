
study_1.csv: this file contains responses for Study 1, both when participants are given the option to be indecisive ("flip a coin"), and when they are not. Each row is a separate query that each participant responded to. Columns are

user_id: integer id for each participant (starting at 0)
group: cf = the participant was given the option to "flip a coin", nocf = the user was only allowed to express a strict preference
decision: which patient is preferred? 1 = left patient, 2 = right patient, 0 = flip a coin
decision_rank: order in which the participant saw the query. 0 is the first query seen by the participant, 1 is the first query, and so on
 
patient features are in the following columns:

left_age: age of the left patient (categorical): 0 = 18 years old, 1 = 32, 2 = 55, 4 = 90 (categorical value 3 is not used)
left_dependents: number of child dependents of the left patient
left_drinking: amount of alcohol consumption for the left patient (categorical): 0 = "not a drinker", 1 = "light drinker", 2 = "heavy drinker"

(same for the right patient)



study_2_indecisive.csv: this file contains responses for Study 2, when participants are given the option to be indecisive ("flip a coin"). Each row is a separate query that each participant responded to. Columns are

user_id: integer id for each participant (starting at 0)
decision: which patient is preferred? 1 = left patient, 2 = right patient, 0 = flip a coin
decision_rank: order in which the participant saw the query. 0 is the first query seen by the participant, 1 is the first query, and so on
 
patient features are in the following columns:

left_1: age of the left patient (years)
left_2: number of alcoholic drinks consumed per day pre-diagnosis by the left patient
left_3: number of child dependents of the left patient

(same for the right patient)



study_2_strict.csv: this file contains responses for Study 2, when participants are not given the option to be indecisive. Each row is a separate query that each participant responded to. Columns are identical to study_2_indecisive.csv, with the following exception:

decision: which patient is preferred? 0 = left patient, 1 = right patient