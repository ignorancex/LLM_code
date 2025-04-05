"""
Functions that create StrategicStudent and Submission objects from the real data for use in experiments.

@author: Noah Burrell <burrelln@umich.edu>
"""

import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np

from classes import StrategicStudent, Submission

def round_grade(raw_grade, maximum):
    """
    Helper function to "coarsen" the grades.

    Parameters
    ----------
    raw_grade : float
        The given grade from the data.
    maximum : float
        The maximum possible grade for a submission in that semester.

    Returns
    -------
    grade : int, [0, 10]
        The coarsened grade.

    """
    
    float_grade = raw_grade * (10.0/maximum)
    
    grade = int(round(float_grade))
    if grade < 0:
        grade = 0
    elif grade > 10:
        grade = 10
        
    return grade

def load17(semester, coarsen_grades=False, drop_TA_grades=False):
    """
    Loads grading data from the file corresponding to the courses in 2017.

    Parameters
    ----------
    semester: str, "Spring" or "Fall"     
    coarsen_grades : bool, optional
        Indicates whether grades from the data should be coarsened into the standard [0, 10] integer range. 
        The default is False, which keeps the grades in the range used in the original data.
    drop_TA_grades : bool, optional
        Indicates whether submissions graded by TAs should be excluded from the list of submissions that is constructed.
        The default is False, meaning that all submissions (with enough peer grades) are included.

    Returns
    -------
    students : list of StrategicStudent objects
        Contains all the students from the course in the given semester.
    submissions : list of Submission objects
        Contains all the submissions from the course in the given semester.

    """
    
    if semester == "Spring":
        array = np.load("336Spring17.npy", allow_pickle=1)
    else: 
        array = np.load("336Fall17.npy", allow_pickle=1)
        
    submission_map = {}
    submission_id_map = {}
    assignment_num_map = {}
    
    count = 0
    dropped_grades = 0
    
    for idx, weeks in enumerate(array):
        for j, week in enumerate(weeks):
            submitted = week[0]
            for submission in submitted:
                assignment_num = submission[0]
                s_id = submission[1]
                
                sub = Submission(s_id, assignment_num)
                sub.true_grade = submission[2]
                sub.TA = bool(submission[3])
                grades = submission[4]
                sub.ta_grades = []
                
                if sub.true_grade == -1:
                    dropped_grades += 1
                    continue
                
                assignment_num_map[s_id] = assignment_num
                
                if drop_TA_grades and sub.TA:
                    continue
                
                for grade in grades:
                    score = grade[0]
                    if grade[3] == 1:
                        sub.ta_grades.append(score)
                    else:
                        count += 1
                        student = grade[2]
                        sub.grades[student] = score
                   
                match = False  
                i = 0
                submissions_list = list(submission_map.values())
                while match == False and i < len(submissions_list):
                    s = submissions_list[i]
                    if s.assignment_number == sub.assignment_number and s.true_grade == sub.true_grade and s.TA == sub.TA and s.grades == sub.grades and len(sub.grades) > 0:
                        match = True
                        count -= len(s.grades)
                        first = s.student_id
                        joint_id = (first, s_id)
                        s.student_id = joint_id
                        submission_map.pop(first)
                        submission_map[joint_id] = s
                        submission_id_map[first] = joint_id
                        submission_id_map[s_id] = joint_id
                    else:
                        i += 1
                if match == False:
                    submission_map[s_id] = sub
                    submission_id_map[s_id] = s_id
                    
    retained_grades = 0 
    
    #print(count)
    
    student_map = {}
    for sub_id, submission in submission_map.items():
        s_id = submission.student_id
        assignment = submission.assignment_number
        for grader, score in submission.grades.items():
                
            if grader not in student_map.keys():
                new_student_obj = StrategicStudent(grader)
                new_student_obj.penalty_tasks = {}
                student_map[grader] = new_student_obj
                
            student_obj = student_map[grader]
            
            if assignment not in student_obj.grades.keys():
                student_obj.grades[assignment] = {}
                
            student_obj.grades[assignment][s_id] = score
            
            retained_grades += 1
    
    for idx, weeks in enumerate(array):
        if idx not in student_map.keys():
            new_student = StrategicStudent(idx)
            student_map[idx] = new_student
        match = student_map[idx]
        for j, week in enumerate(weeks):
            graded = week[1]
            for submission in graded:
                raw_submission_id = submission[1]
                if raw_submission_id not in assignment_num_map.keys() or raw_submission_id not in submission_id_map.keys():
                    continue
                assignment_num = assignment_num_map[raw_submission_id]
                submission_id = submission_id_map[raw_submission_id]
                grade = submission[2]
               
                if assignment_num not in match.grades.keys():
                    match.grades[assignment_num] = {}
                if submission_id not in match.grades[assignment_num].keys():
                    match.grades[assignment_num][submission_id] = grade
                    retained_grades += 1
                if match.grades[assignment_num][submission_id] != grade:
                    print("Matching Error...")
                    print(idx, assignment_num, submission_id, grade)
                else:
                    submission_obj = submission_map[submission_id]
                    if match.id not in submission_obj.grades.keys():
                        submission_obj.grades[match.id] = grade
    
    students = list(student_map.values())
    submissions = list(submission_map.values())
    
    no_updates = False
    
    while not no_updates:
        no_updates = True
        subs_with_enough_grades = []
        students_with_enough_grades = []
        
        for student in students:
            g_dict = student.grades
            to_pop = []
            for assignment in g_dict.keys():
                if len(g_dict[assignment]) < 2:
                    to_pop.append(assignment)
            if len(to_pop) > 0:
                for key in to_pop:
                    graded = g_dict[key]
                    for graded_id in graded.keys():
                        graded_subs = [sub for sub in submissions if sub.assignment_number == key and sub.student_id == graded_id] 
                        graded_sub = graded_subs[0]
                        graded_sub.grades.pop(student.id)
                        dropped_grades += 1
                        retained_grades -= 1
                    g_dict.pop(key)
                no_updates = False
            if len(student.grades) != 0:
                students_with_enough_grades.append(student)
                
        students = students_with_enough_grades[:]
                
        for submission in submissions:
            g_dict = submission.grades 
            if len(g_dict) < 2:
                possible_graders = [stud for stud in students if submission.assignment_number in stud.grades.keys()]
                graders = [g for g in possible_graders if submission.student_id in g.grades[submission.assignment_number].keys()]
                for grader in graders:
                    grade = grader.grades[submission.assignment_number].pop(submission.student_id)
                    if submission.assignment_number not in grader.penalty_tasks.keys():
                        grader.penalty_tasks[submission.assignment_number] = {}
                    grader.penalty_tasks[submission.assignment_number][submission.student_id] = grade
                    dropped_grades += 1
                    retained_grades -= 1
                no_updates = False
            else:
                subs_with_enough_grades.append(submission)    
        
        submissions = subs_with_enough_grades[:]
    
    mismatched = 0
    for submission in submissions:
        
        for grader, grade in submission.grades.items():
            student_matches = [stu for stu in students if stu.id == grader]
            if len(student_matches) != 1:
                mismatched += 1
                print("Sub matching error.")
            else:
                match = student_matches[0]
                if match.grades[submission.assignment_number][submission.student_id] != grade:
                    mismatched += 1
                    print("Sub grade error.")
    #print("Submission Mismatches:", mismatched)
                  
    max_assignment = -1
    mismatched = 0
    for student in students:
        for assignment, grades in student.grades.items():
            if assignment > max_assignment:
                max_assignment = assignment
            for graded, grade in grades.items():
                sub_matches = [s for s in submissions if s.assignment_number == assignment and s.student_id == graded]
                if len(sub_matches) != 1:
                    mismatched += 1
                    print("Student matching error.")
                    print(student.id)
                    print(assignment, grades)
                    print()
                else:
                    match = sub_matches[0]
                    if match.grades[student.id] != grade:
                        mismatched += 1
                        print("Student grade error.")
    #print("Student Mismatches:", mismatched)
    
    for student in students:
        student.included = False
        assignments = set(student.grades.keys())
        
        i1 = assignments.intersection({1, 2, 3, 4})
        i2 = assignments.intersection({5, 6, 7, 8})
        i3 = assignments.intersection({9, 10, 11, 12})
        i4 = assignments.intersection({13, 14, 15, 16})
        
        if len(i1) > 0 and len(i2) > 0 and len(i3) > 0 and len(i4) > 0: 
            student.included = True
    
    '''
    assignment_numbers = [sub.assignment_number for sub in submissions]
    all_nums =  set(assignment_numbers)
    unique_assignments = len(set(assignment_numbers))
    included_students = [stu for stu in students if stu.included]
    
    print()
    print(semester, "2017")
    print("=====================")
    print("Number of Students:", len(students))
    print("Number of Included Students in Evaluation:", len(included_students))
    print("Number of Submissions:", len(submissions))
    print("Number of Assignments:", unique_assignments)
    print("Number of grades dropped:", dropped_grades)
    print("Number of grades retained:", retained_grades)
    print("Assignment Numbers:", str(all_nums))
    '''
    
    if coarsen_grades:
        # Map all the grades into the integer range [0, 10].
        max_val = 100
        for student in students:
            for assignment, grade_dict in student.grades.items():
                for submission_id in grade_dict.keys():
                    grade = grade_dict[submission_id]
                    grade_dict[submission_id] = round_grade(grade, max_val)
            for assignment, grade_dict in student.penalty_tasks.items():
                for submission_id in grade_dict.keys():
                    grade = grade_dict[submission_id]
                    grade_dict[submission_id] = round_grade(grade, max_val)
        for submission in submissions:
            g = submission.true_grade
            submission.true_grade = round_grade(g, max_val)
            for grader in submission.grades.keys():
                grade = submission.grades[grader]
                submission.grades[grader] = round_grade(grade, max_val)
    
    return students, submissions

def load19(semester, coarsen_grades=False, drop_TA_grades=False):
    """
    Loads grading data from the file corresponding to the courses in 2019.

    Parameters
    ----------
    semester: str, "Spring" or "Fall"     
    coarsen_grades : bool, optional
        Indicates whether grades from the data should be coarsened into the standard [0, 10] integer range. 
        The default is False, which keeps the grades in the range used in the original data.
    drop_TA_grades : bool, optional
        Indicates whether submissions graded by TAs should be excluded from the list of submissions that is constructed.
        The default is False, meaning that all submissions (with enough peer grades) are included.

    Returns
    -------
    students : list of StrategicStudent objects
        Contains all the students from the course in the given semester.
    submissions : list of Submission objects
        Contains all the submissions from the course in the given semester.

    """
    
    if semester == "Spring":
        array = np.load("336Spring19.npy", allow_pickle=1)
    else: 
        array = np.load("336Fall19.npy", allow_pickle=1)
        
    submission_map = {}
    submission_id_map = {}
    assignment_num_map = {}
    
    count = 0
    dropped_grades = 0
    
    for idx, weeks in enumerate(array):
        for j, week in enumerate(weeks):
            submitted = week[0]
            for submission in submitted:
                assignment_num = submission[0] + 1
                s_id = submission[1]
                
                sub = Submission(s_id, assignment_num)
                sub.true_grade = submission[2]
                sub.TA = bool(submission[3])
                grades = submission[4]
                sub.ta_grades = []
                
                if sub.true_grade == -1:
                    dropped_grades += 1
                    continue
                
                assignment_num_map[s_id] = assignment_num
                
                if drop_TA_grades and sub.TA:
                    continue
                
                for grade in grades:
                    score = sum(grade[0])
                    if grade[3] == 1:
                        sub.ta_grades.append(score)
                    else:
                        count += 1
                        student = grade[2]
                        sub.grades[student] = score
                   
                match = False  
                i = 0
                submissions_list = list(submission_map.values())
                while match == False and i < len(submissions_list):
                    s = submissions_list[i]
                    if s.assignment_number == sub.assignment_number and s.true_grade == sub.true_grade and s.TA == sub.TA and s.grades == sub.grades and len(sub.grades) > 0:
                        match = True
                        count -= len(s.grades)
                        first = s.student_id
                        joint_id = (first, s_id)
                        s.student_id = joint_id
                        submission_map.pop(first)
                        submission_map[joint_id] = s
                        submission_id_map[first] = joint_id
                        submission_id_map[s_id] = joint_id
                    else:
                        i += 1
                if match == False:
                    submission_map[s_id] = sub
                    submission_id_map[s_id] = s_id
                    
    retained_grades = 0 
    
    #print(count)
    
    student_map = {}
    for sub_id, submission in submission_map.items():
        s_id = submission.student_id
        assignment = submission.assignment_number
        for grader, score in submission.grades.items():
                
            if grader not in student_map.keys():
                new_student_obj = StrategicStudent(grader)
                new_student_obj.penalty_tasks = {}
                student_map[grader] = new_student_obj
                
            student_obj = student_map[grader]
            
            if assignment not in student_obj.grades.keys():
                student_obj.grades[assignment] = {}
                
            student_obj.grades[assignment][s_id] = score
            
            retained_grades += 1
    
    for idx, weeks in enumerate(array):
        if idx not in student_map.keys():
            new_student = StrategicStudent(idx)
            student_map[idx] = new_student
        match = student_map[idx]
        for j, week in enumerate(weeks):
            graded = week[1]
            for submission in graded:
                raw_submission_id = submission[1]
                if raw_submission_id not in assignment_num_map.keys() or raw_submission_id not in submission_id_map.keys():
                    continue
                assignment_num = assignment_num_map[raw_submission_id]
                submission_id = submission_id_map[raw_submission_id]
                grade = sum(submission[2])
               
                if assignment_num not in match.grades.keys():
                    match.grades[assignment_num] = {}
                if submission_id not in match.grades[assignment_num].keys():
                    match.grades[assignment_num][submission_id] = grade
                    retained_grades += 1
                if match.grades[assignment_num][submission_id] != grade:
                    print("Matching Error...")
                    print(idx, assignment_num, submission_id, grade)
                else:
                    submission_obj = submission_map[submission_id]
                    if match.id not in submission_obj.grades.keys():
                        submission_obj.grades[match.id] = grade
    
    students = list(student_map.values())
    submissions = list(submission_map.values())
    
    no_updates = False
    
    while not no_updates:
        no_updates = True
        subs_with_enough_grades = []
        students_with_enough_grades = []
        
        for student in students:
            g_dict = student.grades
            to_pop = []
            for assignment in g_dict.keys():
                if len(g_dict[assignment]) < 2:
                    to_pop.append(assignment)
            if len(to_pop) > 0:
                for key in to_pop:
                    graded = g_dict[key]
                    for graded_id in graded.keys():
                        graded_subs = [sub for sub in submissions if sub.assignment_number == key and sub.student_id == graded_id] 
                        graded_sub = graded_subs[0]
                        graded_sub.grades.pop(student.id)
                        dropped_grades += 1
                        retained_grades -= 1
                    g_dict.pop(key)
                no_updates = False
            if len(student.grades) != 0:
                students_with_enough_grades.append(student)
                
        students = students_with_enough_grades[:]
                
        for submission in submissions:
            g_dict = submission.grades 
            if len(g_dict) < 2:
                possible_graders = [stud for stud in students if submission.assignment_number in stud.grades.keys()]
                graders = [g for g in possible_graders if submission.student_id in g.grades[submission.assignment_number].keys()]
                for grader in graders:
                    grade = grader.grades[submission.assignment_number].pop(submission.student_id)
                    if submission.assignment_number not in grader.penalty_tasks.keys():
                        grader.penalty_tasks[submission.assignment_number] = {}
                    grader.penalty_tasks[submission.assignment_number][submission.student_id] = grade
                    dropped_grades += 1
                    retained_grades -= 1
                no_updates = False
            else:
                subs_with_enough_grades.append(submission)    
        
        submissions = subs_with_enough_grades[:]
    
    mismatched = 0
    for submission in submissions:
        
        for grader, grade in submission.grades.items():
            student_matches = [stu for stu in students if stu.id == grader]
            if len(student_matches) != 1:
                mismatched += 1
                print("Sub matching error.")
            else:
                match = student_matches[0]
                if match.grades[submission.assignment_number][submission.student_id] != grade:
                    mismatched += 1
                    print("Sub grade error.")
    #print("Submission Mismatches:", mismatched)
                  
    max_assignment = -1
    mismatched = 0
    for student in students:
        for assignment, grades in student.grades.items():
            if assignment > max_assignment:
                max_assignment = assignment
            for graded, grade in grades.items():
                sub_matches = [s for s in submissions if s.assignment_number == assignment and s.student_id == graded]
                if len(sub_matches) != 1:
                    mismatched += 1
                    print("Student matching error.")
                    print(student.id)
                    print(assignment, grades)
                    print()
                else:
                    match = sub_matches[0]
                    if match.grades[student.id] != grade:
                        mismatched += 1
                        print("Student grade error.")
    #print("Student Mismatches:", mismatched)
    
    for student in students:
        student.included = False
        assignments = set(student.grades.keys())
        if semester == "Spring":
            i1 = assignments.intersection({1, 2, 3})
            i2 = assignments.intersection({4, 5, 6})
            i3 = assignments.intersection({7, 8, 9})
            i4 = assignments.intersection({10, 11, 13, 14})
        else:
            i1 = assignments.intersection({1, 2, 3})
            i2 = assignments.intersection({4, 5, 6})
            i3 = assignments.intersection({7, 8, 9, 10})
            i4 = assignments.intersection({11, 12, 13, 14})
        if len(i1) > 0 and len(i2) > 0 and len(i3) > 0 and len(i4) > 0: 
            student.included = True
    
    '''
    assignment_numbers = [sub.assignment_number for sub in submissions]
    all_nums =  set(assignment_numbers)
    unique_assignments = len(set(assignment_numbers))
    included_students = [stu for stu in students if stu.included]
    
    print()
    print(semester, "2019")
    print("=====================")
    print("Number of Students:", len(students))
    print("Number of Included Students in Evaluation:", len(included_students))
    print("Number of Submissions:", len(submissions))
    print("Number of Assignments:", unique_assignments)
    print("Number of grades dropped:", dropped_grades)
    print("Number of grades retained:", retained_grades)
    print("Assignment Numbers:", str(all_nums))
    '''
    
    if coarsen_grades:
        # Map all the grades into the integer range [0, 10].
        max_val = 30
        for student in students:
            for assignment, grade_dict in student.grades.items():
                for submission_id in grade_dict.keys():
                    grade = grade_dict[submission_id]
                    grade_dict[submission_id] = round_grade(grade, max_val)
            for assignment, grade_dict in student.penalty_tasks.items():
                for submission_id in grade_dict.keys():
                    grade = grade_dict[submission_id]
                    grade_dict[submission_id] = round_grade(grade, max_val)
        for submission in submissions:
            g = submission.true_grade
            submission.true_grade = round_grade(g, max_val)
            for grader in submission.grades.keys():
                grade = submission.grades[grader]
                submission.grades[grader] = round_grade(grade, max_val)
    
    return students, submissions

def load_all(coarsen_grades=False):
    """
    Loads grading data for all semesters.

    Parameters
    ----------
    coarsen_grades : bool, optional
        Indicates whether grades from the data should be coarsened into the standard [0, 10] integer range. 
        The default is False, which keeps the grades in the range used in the original data.

    Returns
    -------
    s17 : double of lists
        (list of StrategicStudent objects, list of Submission objects).
    f17 : double of lists
        (list of StrategicStudent objects, list of Submission objects).
    s19 : double of lists
        (list of StrategicStudent objects, list of Submission objects).
    f19 : double of lists
        (list of StrategicStudent objects, list of Submission objects).

    """
    
    s17 = load17("Spring", coarsen_grades)
    f17 = load17("Fall", coarsen_grades)
    s19 = load19("Spring", coarsen_grades)
    f19 = load19("Fall", coarsen_grades)
    
    return s17, f17, s19, f19