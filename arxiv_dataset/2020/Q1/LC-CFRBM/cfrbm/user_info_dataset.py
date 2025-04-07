from collections import defaultdict

def load_dataset(train_path, test_path, movie_path, sep, user_based=True):
    all_users_set = set()
    all_movies_set = set()
    all_occs_set = ['administrator','executive','retired','lawyer','entertainment','marketing','writer', 
        'none','scientist','healthcare','other','student','educator','technician', 
        'librarian','programmer','artist','salesman','doctor','homemaker','engineer']
    all_sex_set = ['M','F']
    all_ages_set = ['1','18','25','35','45','50','56']

    with open(train_path, 'rt') as data:
        for i, line in enumerate(data):
            uid, mid, rat, timstamp = line.strip().split(sep)
            if uid not in all_users_set:
                all_users_set.add(uid)
            if mid not in all_movies_set:
                all_movies_set.add(mid)

    tests = defaultdict(list)

    with open(test_path, 'rt') as data:
        for i, line in enumerate(data):
            uid, mid, rat, timstamp = line.strip().split(sep)
            if uid not in all_users_set:
                all_users_set.add(uid)
            if mid not in all_movies_set:
                all_movies_set.add(mid)
            if user_based:
                tests[uid].append((mid, float(rat)))
            else:
                tests[mid].append((uid, float(rat)))


    return list(all_users_set), list(all_movies_set), list(all_occs_set), list(all_sex_set), list(all_ages_set), tests


def load_file(dataset, sep='::', user_based=True):
    profiles = {}
    with open(dataset, 'rt') as data:
        for i, line in enumerate(data):
            uid, mid, rat, timstamp = line.strip().split(sep)
            if user_based:
                profiles[uid].append((mid, float(rat)))
            else:
                profiles[mid].append((uid, float(rat)))
    return profiles


