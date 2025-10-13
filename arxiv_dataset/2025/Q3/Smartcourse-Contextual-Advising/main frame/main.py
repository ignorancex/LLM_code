from course_manager import CourseManager
from utils import write_log, send_enrollment_email, send_grade_email, ask_ai_question
from data_models import Student, Instructor
import textwrap

SECURITY_PASSWORD = "smartcourse12345"


def display_student_menu(manager, username):
    while True:
        print("\nStudent Menu:\n1. Enroll in Course\n2. View My Courses\n3. Drop Course\n4. Ask AI for Advice\n5. Exit")
        choice = input("Your choice: ").strip()

        # 1. Course Selection
        if choice == "1":
            keyword = input("Search keyword (or leave empty for all): ")
            matched = manager.search_courses(keyword)
            if not matched:
                print("No matching courses found.")
                continue
            print("Matched courses:")
            for c in matched:
                print("  ", c)
            course_to_enroll = input("Enter course to enroll: ").strip()
            if course_to_enroll in manager.courses:
                enrolled = manager.get_student_courses(username)
                if course_to_enroll in enrolled:
                    print("You are already enrolled in this course.")
                    continue
                manager.enroll_student(username, course_to_enroll)
                write_log(f"{username} enrolled in {course_to_enroll}")
                send_enrollment_email(
                    username,
                    f"Dear {username},\n\nCongratulations! You have successfully enrolled in the course: {course_to_enroll}.\n\nSincerely,\nSmart Course"
                )
                print(f"Successfully enrolled in {course_to_enroll}")
            else:
                print("Invalid course.")

        # 2. View Enrolled Courses
        elif choice == "2":
            courses = manager.get_student_courses(username)
            if not courses:
                print("(You haven't taken any courses yet.)")
            else:
                for c, g in courses.items():
                    print(f"{c} - Grade: {g if g else 'Not assigned'}")

        # 3. Drop Courses
        elif choice == "3":
            courses = manager.get_student_courses(username)
            if not courses:
                print("You are not enrolled in any courses.")
                continue
            for c in courses:
                print("  ", c)
            course_to_drop = input("Enter course to drop: ").strip()
            if course_to_drop not in courses:
                print("You are not enrolled in this course.")
                continue
            if courses[course_to_drop] is not None:
                print("You cannot drop a course that has been graded.")
                continue
            manager.drop_student_course(username, course_to_drop)
            write_log(f"{username} dropped {course_to_drop}")
            send_enrollment_email(
                username,
                f"Dear {username},\n\nYou have successfully dropped the course: {course_to_drop}.\n\nSincerely,\nSmart Course"
            )
            print("Course dropped.")

        # 4. AI advice
        elif choice == "4":
            question = input("Enter your academic question (e.g., What course should I take next?): ").strip()
            student_courses = manager.get_student_courses(username)
            course_info = "\n".join([f"{c} - {g if g else 'Not assigned'}" for c, g in student_courses.items()])

            student = manager.get_student_by_username(username)
            plan_text = ""
            if student and student.major:
                try:
                    with open(f"{student.major}_plan.txt", "r", encoding="utf-8") as f:
                        plan_text = f.read()
                except FileNotFoundError:
                    print(f"No major plan found for {student.major}.")

            full_prompt = (
                f'I am a student asking the following academic question:\n"{question}"\n\n'
                f"My current course history:\n{course_info}\n\n"
                f"Here is the four-year plan for my major:\n{plan_text}\n\n"
                "Based on my question, my course history, and the plan above, give me a suggestion."
            )

            reply, latency = ask_ai_question(full_prompt)   # ← unpacking tuple
            print(f"\n[AI ADVICE]  (⏱ {latency:.1f}s)")
            for para in reply.strip().split("\n"):
                if para.strip():
                    print(textwrap.fill(para.strip(), width=80))
                else:
                    print()

        # 5. Exit
        elif choice == "5":
            manager.save_enrollments()
            print("Goodbye!")
            break
        else:
            print("Invalid choice.")


def display_instructor_menu(manager):
    while True:
        print("\nInstructor Menu:\n1. View Student Courses\n2. Assign Grade\n3. Exit")
        choice = input("Your choice: ")
        if choice == "1":
            print("Registered Students:")
            # Display all registered students and their enrolled courses
            for student_name in manager.list_all_students():
                print(student_name)
            selected_student = input("Enter student username: ")
            student_courses = manager.get_student_courses(selected_student)
            # If the student has enrolled courses, display them; otherwise, print a notfound message
            if student_courses:
                print(f"Courses enrolled by {selected_student}:")
                for course_name, grade in student_courses.items():
                    print(f"{course_name} - Grade: {grade if grade else 'Not assigned'}")
            else:
                print("No courses found for this student.")

        elif choice == "2":
            print("Registered Students:")
            for student_name in manager.list_all_students():
                print(student_name)
            selected_student = input("Enter student username: ")
            student_courses = manager.get_student_courses(selected_student)
            if not student_courses:
                print("No courses found for this student.")
                continue
            print(f"Courses enrolled by {selected_student}:")
            for course_name, grade in student_courses.items():
                print(f"{course_name} - Grade: {grade if grade else 'Not assigned'}")

            course_name = input("Enter course name to assign grade: ")
            # Instructors can not assign grades to courses that the student is not enrolled in
            if course_name not in student_courses:
                print("Student is not enrolled in this course.")
                continue
            grade = input("Enter grade (A, A-, B+, B, B-, C+, C, D, F): ").strip().upper()
            if grade not in ["A", "A-", "B+", "B", "B-", "C+", "C" "D", "F"]:
                print("Invalid grade. Please enter A, A-, B+, B, B-, C+, C, D, or F.")
                continue
            # Assign the grade to the student for the specified course and send an email notification
            manager.set_student_grade(selected_student, course_name, grade)
            write_log(f"Assigned grade {grade} to {selected_student} for {course_name}")
            send_grade_email(selected_student, f"Dear {selected_student}, \n"
                                           f"\nYou received grade '{grade}' for course {course_name}. "
                                           f"\nPlease take some time to review and confirm your grade to ensure that it aligns with your academic goals. You can do this by logging into your Smart Course account.\n"
                                           f"\nSincerely,\nSmart Course")
            print(f"Grade {grade} assigned to {selected_student} for {course_name}.")

        elif choice == "3":
            manager.save_enrollments() # Make sure to save the grading changes before exiting
            print("Goodbye!")
            break
        else:
            print("Invalid choice.")


def main():
    print("Welcome to SmartCourse!")
    manager = CourseManager()
    while True:
        print("Main Menu:\n1. Login\n2. Register for new account\n3. Exit")
        main_choice = input("Select an option (1/2/3): ").strip()
        if main_choice == "1":
            username = input("Username: ")
            # Ensure the username ends with '@smartcourse.com' to maintain consistency
            if not username.endswith("@smartcourse.com"):
                print("Invalid username. Username must end with '@smartcourse.com'.")
                continue
            password = input("Password: ")
            is_student = manager.is_student_account(username)
            if manager.authenticate_user(username, password, is_student):
                write_log(f"{username} logged in")
                print(f"Welcome, {username}!")
                # Display the appropriate menu based on the user's role
                if is_student:
                    display_student_menu(manager, username)
                else:
                    display_instructor_menu(manager)
                break
            else:
                print("Login failed. Try again.")
        elif main_choice == "2":
            username = input("New username: ")
            # Ensure the username ends with '@smartcourse.com' to maintain consistency
            if not username.endswith("@smartcourse.com"):
                print("Invalid username. Username must end with '@smartcourse.com'.")
                continue
            password = input("Password: ")
            role = input("Role (student/instructor): ").strip().lower()
            if role not in ["student", "instructor"]:
                print("Invalid role. Please enter 'student' or 'instructor'.")
                continue
            # If the role is instructor, ask for the security password
            if role == "instructor":
                security = input("Enter security password to register as instructor: ")
                if security != SECURITY_PASSWORD:
                    print("Security password incorrect. Instructor account not created.")
                    continue
            is_student = role == "student" # Set this boolean variable to determine if the account is for a student or instructor
            # Check if the username already exists in either students or instructors
            if manager.get_student_by_username(username) or any(
                    instructor.username == username for instructor in manager.instructors):
                print("Username already exists. Please choose a different username.")
                continue
            if is_student:
                major = input("Enter your major (cps, acct, math): ").strip().lower()
                if major not in ["cps", "acct", "math"]:
                    print("Sorry, currently this system only support cps, acct and math. Please enter 'cps', 'acct', or 'math'.")
                    continue
                # Append student's username, password and major to the account.txt file if the account is for a student
                manager.students.append(Student(username, password, major))
            else:
                # Append instructor's username and password to the instructors list if the account is for an instructor
                manager.instructors.append(Instructor(username, password))

            manager.create_account(username, password, is_student)
            write_log(f"Created new account: {username}")
            print("Account created. Please log in.")
        elif main_choice == "3":
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

# Ensure the main function is called when the script is run directly
if __name__ == "__main__":
    main()
