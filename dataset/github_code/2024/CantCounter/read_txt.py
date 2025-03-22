
nickname=[]
with open("trump_test.txt","r",encoding="UTF-8") as f:
    a=f.readlines()
for x in a:
    i=len(x)
    x=x[0:i-1]
    nickname.append(x)
print(nickname)