import csv

r_base = open('../dataset/ml-10M100K/r1.train','rb')
r_test = open('../dataset/ml-10M100K/r1.test','rb')

l_base = []
ratings_base = 0
users = {}
movies = {}

l_test = []
ratings_test = 0

movie_id = 1
user_id = 1

for line in r_base:
    row = line.strip().split('::')
    if row[0] not in users:
        users[row[0]] = user_id
        user_id += 1
    if row[1] not in movies:
        movies[row[1]] = movie_id
        movie_id += 1

    ratings_base += 1
    l_base.append([row[0],row[1],row[2]])


for line in r_test:
    row = line.strip().split('::')
    if row[0] not in users:
        users[row[0]] = user_id
        user_id += 1
    if row[1] not in movies:
        movies[row[1]] = movie_id
        movie_id += 1

    ratings_test += 1
    l_test.append([row[0],row[1],row[2]])

w_base = open('../dataset/ml-10M100K/train_10m1','wb')
w_test = open('../dataset/ml-10M100K/test_10m1','wb')

print len(users.keys()),user_id,len(movies.keys()),movie_id
w_base.write('%%MatrixMarket matrix coordinate real general\n')
w_base.write('% Generated 12-November-2016\n')
w_base.write(str(len(users.keys())) + ' ' + str(len(movies.keys())) + ' ' + str(ratings_base) + '\n')
for l in l_base:
    w_base.write(str(users[l[0]]) + ' ' + str(movies[l[1]]) + ' ' + str(l[2]) + '\n')

w_test.write('%%MatrixMarket matrix coordinate real general\n')
w_test.write('% Generated 12-November-2016\n')
w_test.write(str(len(users.keys())) + ' ' + str(len(movies.keys())) + ' ' + str(ratings_test) + '\n')
for l in l_test:
    w_test.write(str(users[l[0]]) + ' ' + str(movies[l[1]]) + ' ' +str(l[2]) + '\n')