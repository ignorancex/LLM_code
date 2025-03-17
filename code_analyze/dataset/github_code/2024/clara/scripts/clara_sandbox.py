import os

correct = []
for c in os.listdir("E:\\code\\Clara_Data\\Sample\\1A\\OK\\python.3"):
    if ('solution.txt' in c):
        correct.append(c.split('_')[0])
print(correct)