import openai
import argparse
import os

from global_config import OPENAI_KEY


# Initialize the OpenAI API client
openai.api_key = OPENAI_KEY


def main(args):
    num_form = args.num_form
    theme = args.theme

    # [CUSTOM] add your own theme here
    if theme == 'business':
        input_to_cont = """\
1. NEW COMPETITIVE PRODUCT
2. FAX TRANSMISSION
3. OLD GOLD - LIGHT BOX 100's PROGRESS REPORT
4. COUPON CODE REGISTRATION FORM
5. THE TOBACCO INSTITUTE FIFTH ANNUAL COLLEGE OF TOBACCO KNOWLEDGE REGISTRATION FORM"""

    elif theme == 'government':
        input_to_cont = """\
1. Request for Leave or Approved Absence
2. Request for Taxpayer Identification Number and Certification
3. Application for a Social Security Card
4. U.S. Passport Application
5. Free Application for Federal Student Aid"""

    elif theme == 'medical':
        input_to_cont = """\
1. Patient Registration Form
2. Medical History Form
3. Informed Consent for Treatment
4. HIPAA Privacy Authorization Form
5. Insurance Information and Verification Form"""

    elif theme == 'education':
        input_to_cont = """\
1. University of California Graduate Admission Application Form
2. New York City Public Elementary School Transfer Request Form
3. National Taiwan University Undergraduate Graduation Application Form
4. University of Texas Official Transcript Request Form
5. Boston High School Parental Consent Form (Field Trip Specific)
6. Stanford University Financial Aid Application Form
7. University of Michigan Semester Course Registration Form
8. Harvard University Student Health Declaration Form
9. National Taiwan Normal University Exchange Student Application Form
10. Fu Jen Catholic University Student Internship Program Application Form"""

    else:
        print("Invalid theme")
        exit()

    # [CUSTOM] add your own theme here
    if theme == 'business':
        theme_input = "a business setting"
    elif theme == 'government':
        theme_input = "government organization in the United States"
    elif theme == 'medical':
        theme_input = "a medical setting"
    elif theme == 'education':
        theme_input = "educational institutions"
    else:
        print("Invalid theme")
        exit()

    prompt = f"""\
Please create {num_form} form titles that might be used in {theme_input}.
The titles should clearly convey the purpose of each form and look realistic, as they would in an actual company.
Directly continue, do not repeat the following:

{input_to_cont}
"""

    response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
        )
    response = response.choices[0].message.content
    print(response)

    # write to txt file
    save_dir = "headers/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, args.file_name)
    with open(save_path, 'w', encoding="utf8") as f:
        f.write(response)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_form", type=int, default=10)
    parser.add_argument("--theme", type=str, default='business')
    parser.add_argument("--file_name", type=str, default='example.txt')
    args = parser.parse_args()

    main(args)