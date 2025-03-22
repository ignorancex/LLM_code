def cap(s):
    ans = 0
    z = len(s)
    i = 0
    while (i < z):
        ans = ans + s[i]
        i = i + 1
    print(ans)