"""
Evaluate SmartCourse AI advice quality under 4 context settings.

Metrics
-------
PlanScore      = (# courses in plan & not taken) / #rec
PersonalScore  = (# courses in plan & (not taken or low grade)) / #rec
Lift           = PersonalScore − PlanScore
Recall         = good(plan) / (plan − taken)
"""

import re, csv, time, difflib, pathlib, random, statistics, requests, json
from course_manager import CourseManager

BOOT_ITER = 10000
LOW_GRADE_THRESHOLD = "B-"
STREAM_MODEL = True

# ---------- CONFIG ----------
TEST_STUDENT  = "user@smartcourse.com"
QUESTION_FILE = "evaluation_questions.txt"
OUT_CSV       = "relevance_scores.csv"
MODEL_NAME    = "llama3.1:8b"
OLLAMA_URL    = "http://127.0.0.1:11434"
# ----------------------------

mgr = CourseManager()
student = mgr.get_student_by_username(TEST_STUDENT)
if not student:
    raise ValueError(f"{TEST_STUDENT} not found.")

plan_file = f"{student.major}_plan.txt"
plan_courses = {l.strip() for l in open(plan_file, encoding="utf-8") if l.strip()}

all_courses = [l.strip() for l in open("course_list.txt", encoding="utf-8") if l.strip()]
all_codes = {c.split(":")[0].strip(): c for c in all_courses}

taken_courses = set(mgr.get_student_courses(TEST_STUDENT).keys())
course_grades = mgr.get_student_courses(TEST_STUDENT)

grade_rank = {g: i for i, g in enumerate(
    ["A", "A-", "B+", "B", "B-", "C+", "C", "D", "F"])}

def grade_low(c: str) -> bool:
    g = course_grades.get(c)
    return g is not None and grade_rank.get(g, 99) >= grade_rank[LOW_GRADE_THRESHOLD]

def ask_ai(prompt: str) -> tuple[str, float]:
    start = time.time()
    payload = {"model": MODEL_NAME, "prompt": prompt, "stream": STREAM_MODEL}
    r = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=600)
    if STREAM_MODEL:
        parts = []
        for line in r.iter_lines():
            if line:
                data = json.loads(line.decode())
                if "response" in data:
                    parts.append(data["response"])
        txt = "".join(parts)
    else:
        txt = r.json().get("response", "")
    return txt.strip(), time.time() - start

def extract_courses(text: str) -> set[str]:
    found = set()
    for code, full in all_codes.items():
        if re.search(rf"\b{re.escape(code)}\b", text, re.I):
            found.add(full)
    if not found:
        for full in all_courses:
            if difflib.SequenceMatcher(None, text.lower(), full.lower()).ratio() > .8:
                found.add(full)
    return found

# Public suffixes to ensure formatting
base_suffix = (
    "\n\nPlease list 5-8 courses strictly in the following format, with the full name on each line, for example:\n"
    "CPS 2232: Data Structure\nMATH 2110: Discrete Structure\n"
)

# Full suffix for full context mode
full_suffix = (
    "\n\nOnly recommend courses that appear in the four-year plan shown above."
    " Do NOT include any course you have already taken."
    " Prioritize courses where you scored below B-."
    + base_suffix
)


noTranscript_suffix = (
    "\n\nOnly recommend courses that appear in the four-year plan shown above."
    " You do NOT have access to the student's transcript;"
    " it’s OK to suggest courses the student may have already taken."
    + base_suffix
)


def build_prompt(mode: str, q: str) -> str:
    history = "\n".join(f"{c} - {g or 'Not assigned'}"
                        for c, g in course_grades.items())
    plan_txt = "\n".join(plan_courses)

    if mode == "full":
        return (
            f'Student question: "{q}"\n'
            f'My course history:\n{history}\n'
            f'My 4-year plan:\n{plan_txt}\n'
            f'Based on ALL information, recommend courses.'
            + full_suffix
        )
    if mode == "noTranscript":
        return (
            f'"{q}"\n'
            f'My 4-year plan:\n{plan_txt}\n'
            'Based on plan only, recommend courses.'
            + noTranscript_suffix
        )
    if mode == "noPlan":
        return (
            f'"{q}"\n'
            f'My course history:\n{history}\n'
            'Based on history only, recommend courses.'
            + base_suffix
        )
    return q + base_suffix

# ---------- evaluation ----------
rows = []
for q in (l.strip() for l in open(QUESTION_FILE, encoding="utf-8") if l.strip()):
    for mode in ("full", "noTranscript", "noPlan", "question"):
        rep, lat = ask_ai(build_prompt(mode, q))
        recs = extract_courses(rep)

        good_plan = {c for c in recs if c in plan_courses and c not in taken_courses}
        good_personal = {
            c for c in recs
            if c in plan_courses and (c not in taken_courses or grade_low(c))
        }

        plan_score = len(good_plan) / len(recs) if recs else 0
        pers_score = len(good_personal) / len(recs) if recs else 0
        lift = pers_score - plan_score
        recall = (len(good_plan) /
                  (len(plan_courses - taken_courses))
                  if plan_courses - taken_courses else 0)

        rows.append([q, mode, len(recs), plan_score, pers_score, lift, recall, f"{lat:.2f}s"])
        print(f"[{mode}] {q[:38]}… Rec:{len(recs)} Plan:{plan_score:.3f} Pers:{pers_score:.3f}")

# Save results to CSV
with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    csv.writer(f).writerow(
        ["Question", "Mode", "#Rec", "PlanScore", "PersonalScore", "Lift", "Recall", "Latency"]
    )
    csv.writer(f).writerows(rows)
print(f"Saved → {OUT_CSV}")

# Aggregate results
grouped = {}
for *_q, m, _r, ps, prs, lf, rc, _ in rows:
    grouped.setdefault(m, []).append((ps, prs, lf, rc))

def ci(vals):
    boots = [statistics.mean(random.choices(vals, k=len(vals))) for _ in range(BOOT_ITER)]
    return statistics.quantiles(boots, n=20)[1:19:17]  # 5th & 95th pctile

print("\n=== Aggregate metrics (95% CI) ===")
for mode, v in grouped.items():
    pl = [x[0] for x in v]; pe = [x[1] for x in v]; li = [x[2] for x in v]; rc = [x[3] for x in v]
    mp, (pl_lo, pl_hi) = statistics.mean(pl), ci(pl)
    ms, (pe_lo, pe_hi) = statistics.mean(pe), ci(pe)
    ml, (li_lo, li_hi) = statistics.mean(li), ci(li)
    mr = statistics.mean(rc)
    print(
        f"{mode:9}  Plan {mp:.3f} CI[{pl_lo:.3f},{pl_hi:.3f}]  "
        f"Personal {ms:.3f} CI[{pe_lo:.3f},{pe_hi:.3f}]  "
        f"Lift {ml:+.3f} CI[{li_lo:+.3f},{li_hi:+.3f}]  Recall {mr:.3f}"
    )
