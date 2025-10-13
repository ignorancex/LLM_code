def subset1(array,num):

    if num == 0 or num < 1:
        return None
    elif len(array) == 0:
        return None
    else:
        if abs(array[0] - num )< 0.001 :#        if array[0] == num:
            return [array[0]]
        else:
            with_v = subset1(array[1:],(num - array[0])) 
            if with_v:
                return [array[0]] + with_v
            else:
                return subset1(array[1:],num)
