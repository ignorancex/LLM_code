import csv

def format_genre(genre_lst):
    genre_map = {"Action" : 0, "Adventure" : 1, "Animation" : 2, "Children's" : 3, "Children" : 3, "Comedy" : 4, "Crime" : 5, "Documentary" : 6,
        "Drama" : 7, "Fantasy" : 8, "Film-Noir" : 9, "Horror" : 10, "Musical" : 11, "Mystery" : 12, "Romance" : 13, "Sci-Fi" : 14, 
        "Thriller" : 15, "War" : 16, "Western" : 17}



    ret = [0 for x in range(18)]

    for genre in genre_lst:
        if genre in genre_map:
            ret[genre_map[genre]] = 1
    return ret

#1|Toy Story (1995)|01-Jan-1995||http://us.imdb.com/M/title-exact?Toy%20Story%20(1995)|0|0|0|1|1|1|0|0|0|0|0|0|0|0|0|0|0|0|0

"""
r = csv.reader(open('../dataset/ml-10M100K/movies.dat','rb'), delimiter='|')
w = csv.writer(open('../dataset/ml-10M100K/movies.dat.mod','wb'), delimiter='|')

for row in r:
    w.writerow(row[:2] + ['','',''] + format_genre(row[2:]))

"""