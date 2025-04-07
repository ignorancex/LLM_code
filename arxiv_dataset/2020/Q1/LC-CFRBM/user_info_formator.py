import csv

def format_sex(sex):
    if sex == 'M':
        return [0]
    else:
        return [1]

def format_occ(occ, mapped):
    map_occ_id1 = {"other" : 0, "academic" : 1, "educator" : 1, "artist" : 2, "clerical" : 3, "admin" : 3, "college" : 4, "grad student" : 4,
        "customer service" : 5, "doctor" : 6, "health care" : 6, "executive" : 7, "managerial" : 7, "farmer" : 8, "homemaker" : 9,
        "K-12 student" : 10, "lawyer" : 11, "programmer" : 12, "retired" : 13, "sales" : 14, "marketing" : 14, "scientist" : 15,
        "self-employed" : 16, "technician" : 17, "engineer" : 17, "tradesman" : 18, "craftsman" : 18, "unemployed" : 19, "writer" : 20}

    map_occ_id2 = {'administrator': 4, 'executive': 3, 'retired': 18, 'lawyer': 6, 'entertainment': 9, 'marketing': 15, 'writer': 2, 
        'none': 16, 'scientist': 8, 'healthcare': 17, 'other': 1, 'student': 5, 'educator': 7, 'technician': 0, 
        'librarian': 11, 'programmer': 10, 'artist': 13, 'salesman': 19, 'doctor': 20, 'homemaker': 12, 'engineer': 14}

    ret = [0 for x in range(21)]

    if not mapped:
        if occ in map_occ_id2:
            ret[map_occ_id2[occ]] = 1
            return ret
    else:
        ret[int(occ)] = 1
        return ret

def format_age(age):
    ret = [0 for x in range(7)]

    age = int(age)

    if age < 18:
        ret[0] = 1
    elif age >= 18 and age <= 24:
        ret[1] = 1
    elif age >= 25 and age <= 34:
        ret[2] = 1
    elif age >= 35 and age <= 44:
        ret[3] = 1
    elif age >= 45 and age <= 49:
        ret[4] = 1
    elif age >= 50 and age <= 55:
        ret[5] = 1
    elif age >= 56:
        ret[6] = 1


    return ret


r = csv.reader(open('../dataset/ml-100k/u.user','rb'), delimiter='|')
w = csv.writer(open('../dataset/ml-100k/u.user.mod','wb'), delimiter='|')

for row in r:
    w.writerow([row[0]] + format_age(row[1]) + format_sex(row[2]) + format_occ(row[3], False))

r = csv.reader(open('../dataset/ml-1m/users.dat','rb'), delimiter='|')
w = csv.writer(open('../dataset/ml-1m/users.dat.mod','wb'), delimiter='|')

for row in r:
    w.writerow([row[0]] + format_age(row[2]) + format_sex(row[1]) + format_occ(row[3], True))
