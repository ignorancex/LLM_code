#kneefx.py
"""
this function finds the knee point
"""
import numpy as np
def find_knee(xxx,yyy):
    
    
    new_x = []
    new_y = []

    for i in range(0,len(xxx),10):
        new_x_i = np.mean(xxx[i:i+10])
        new_y_i = np.mean(yyy[i:i+10])
        new_x.append(new_x_i)
        new_y.append(new_y_i)

    new_x = np.array(new_x)
    new_y = np.array(new_y)

    firstr2 = []
    secondr2 = []
    knee_locs = []
    xi_list = []
    knee_idx_list = []
    fabsdet_list = []
    list_ys = []

    knee_loc = 0

    exclude = 3
   
    for inflection_point in range(exclude,len(new_x)-exclude):
        cod1,cod2,parameters,y_lists = get_cods(inflection_point,new_x,new_y)
        knee_locs.append(new_x[inflection_point])
        firstr2.append(cod1)
        secondr2.append(cod2)
        if len(knee_locs) >= 2:
            pt1 = (knee_locs[-1],firstr2[-1])
            pt2 = (knee_locs[-2],firstr2[-2])
            ptA = (knee_locs[-1],secondr2[-1])
            ptB = (knee_locs[-2],secondr2[-2])
            seg1 = (pt1,pt2)
            seg2 = (ptA,ptB)
            if intersects(seg1,seg2) == True:
                best_inflection_point = inflection_point
                x1, y1 = pt1
                x2, y2 = pt2
                dx1 = x2 - x1
                dy1 = y2 - y1

                xA, yA = ptA
                xB, yB = ptB;
                dx2 = xB - xA
                dy2 = yB - yA;
                DET = (-dx1 * dy2 + dy1 * dx2)

                DETinv = 1.0/DET
                r = DETinv * (-dy2  * (xA-x1) +  dx2 * (yA-y1))
                s = DETinv * (-dy1 * (xA-x1) + dx1 * (yA-y1))

                xi = (x1 + r*dx1 + xA + s*dx2)/2.0
                #print(xi)
                knee_loc = xi
                ys = y_lists
                best_x1 = new_x[:best_inflection_point]
                best_x2 = new_x[best_inflection_point:]
    r2_x = np.array(new_x[exclude:-exclude])
    return knee_loc,best_x1,best_x2,ys[0],ys[1],np.array(firstr2),np.array(secondr2),r2_x
    
def on_segment(p, q, r):

    if r[0] <= max(p[0], q[0]) and r[0] >= min(p[0], q[0]) and r[1] <= max(p[1], q[1]) and r[1] >= min(p[1], q[1]):
        if_on = True
    else:
        if_on = False
        
   
    return if_on

def orientation(p, q, r):

    val = ((q[1] - p[1]) * (r[0] - q[0])) - ((q[0] - p[0]) * (r[1] - q[1]))
    if val == 0 : return 0
    return 1 if val > 0 else -1

def intersects(seg1, seg2):

    p1, q1 = seg1
    p2, q2 = seg2

    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    
    onseg1 = on_segment(p1, q1, p2)
    onseg2 = on_segment(p1, q1, q2)
    onseg3 = on_segment(p2, q2, p1)
    onseg4 = on_segment(p2, q2, q1)
    
    if o1 != o2 and o3 != o4:
        
        return True

    if o1 == 0 and onseg1 : 
        return True

    if o2 == 0 and onseg2 : 
        return True
    if o3 == 0 and onseg3 :
        return True
    if o4 == 0 and onseg4 :
        return True

    return False
def calc_cod(slope, intercept, y_mean, y_predict, y_real):
    residual_sum = 0
    t_sum = 0
    for i in range(len(y_predict)):
        residual_sum = residual_sum + ((y_real[i]-y_predict[i])**2)
        t_sum = t_sum + ((y_real[i]-y_mean)**2)
 
    try:
        cod = 1-(residual_sum/t_sum)
        #print(cod)
    except:
        cod =0
 
    return cod
def get_cods(index,x,y):
    pfit_1 = np.polyfit(x[:index],y[:index],1)
    slope_1 = pfit_1[0]
    intercept_1 = pfit_1[1]
    
    y_predict_1 = x[:index]*slope_1 + intercept_1
    y_mean_1 = np.mean(y)
    y_real_1 = y[:index]
    
    cod_1 = calc_cod(slope_1,intercept_1,y_mean_1,y_predict_1,y_real_1)
    
    pfit_2 = np.polyfit(x[index:],y[index:] ,1)
    slope_2 = pfit_2[0]
    intercept_2 = pfit_2[1]
    
    y_predict_2 = x[index:]*slope_2 + intercept_2
    y_mean_2 = np.mean(y)
    y_real_2 = y[index:]
    
    cod_2 = calc_cod(slope_2,intercept_2,y_mean_2,y_predict_2,y_real_2)
    
    return cod_1,cod_2,[slope_1,intercept_1,slope_2,intercept_2],[y_predict_1,y_predict_2]


