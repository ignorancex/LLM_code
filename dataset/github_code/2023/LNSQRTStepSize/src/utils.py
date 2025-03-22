import torch

import numpy as np
import contextlib
import copy

def check_armijo_conditions(step_size, step_size_old, loss, grad_norm,
                      loss_next, c, beta_b):
    found = 0

    # computing the new break condition
    break_condition = loss_next - \
        (loss - (step_size) * c * grad_norm**2)

    if (break_condition <= 0):
        found = 1

    else:
        # decrease the step-size by a multiplicative factor
        step_size = step_size * beta_b

    return found, step_size, step_size_old

#########################NEW#######################################
def check_armijo_conditions2(step_size, loss, grad_current,d_current, loss_next, c, beta_b):
    found = 0

    temp=0
    for t, m in zip(grad_current,d_current):
        temp+=torch.sum(torch.mul(t,m))
        
    # computing the new break condition
    break_condition = loss_next - (loss + (step_size) * c * temp) 

    if (break_condition <= 0):
        found = 1

    else:
        # decrease the step-size by a multiplicative factor
        step_size = step_size * beta_b

    return found, step_size
########################################################################
def check_armijo_conditions3(counter,step_size, loss, grad_current,d_current, loss_next, c, beta_b, grad_next, sigma, delta, delta1, gammat):
    found = 0
    
    temp1=0
    for t, m in zip(grad_current,d_current):
        temp1+=torch.sum(torch.mul(t,m))
    
    temp2=0
    for t, m in zip(grad_next,d_current):
        temp2+=torch.sum(torch.mul(t,m))
        
    temp3=0
    for t, m in zip(d_current,d_current):
        temp3+=torch.sum(torch.mul(t,m))
        
    if(loss_next <= (loss + (delta* (step_size) * temp1)+(gammat*(delta1)*(step_size ** 2)*(temp3 ** 2)/2.0))):
        found = 1

    if(temp2 >= (sigma*temp1)+(gammat*delta1*(step_size)*(temp3 ** 2))):
        if found == 1:
            found = 3 # both conditions are satisfied
        else:
            found = 2 # only the curvature condition is satisfied

    if (found != 3):
        # step-size might be too large
        if(counter>100):
            step_size = max(step_size * beta_b, 1e-6)
        else:
            step_size = max(step_size * beta_b, 1e-6)

    return found, step_size

########################################################################
def check_FSS_conditions(step_size, step_size_old, loss, temp_norm,
                      loss_next, c, beta_b, MaxPre_loss,eta):
    found = 0

    # computing the new break condition
    #R=loss
    #print("MaxPre_loss")
    #print(MaxPre_loss)
    
    R=(eta*MaxPre_loss)+((1-eta)*loss)
    #print("R")
    #print(R)
    break_condition = loss_next - (R + (step_size) * c * temp_norm)

    if (break_condition <= 0):
        found = 1

    else:
        # decrease the step-size by a multiplicative factor
        step_size = step_size * beta_b

    return found, step_size, step_size_old
#############################################################################
def check_goldstein_conditions(step_size, loss, grad_norm,
                          loss_next,
                          c, beta_b, beta_f, bound_step_size, eta_max):
	found = 0
	if(loss_next <= (loss - (step_size) * c * grad_norm ** 2)):
		found = 1

	if(loss_next >= (loss - (step_size) * (1 - c) * grad_norm ** 2)):
		if found == 1:
			found = 3 # both conditions are satisfied
		else:
			found = 2 # only the curvature condition is satisfied

	if (found == 0):
		raise ValueError('Error')

	elif (found == 1):
			# step-size might be too small
		step_size = step_size * beta_f
		if bound_step_size:
			step_size = min(step_size, eta_max)

	elif (found == 2):
		# step-size might be too large
		step_size = max(step_size * beta_b, 1e-8)

	return {"found":found, "step_size":step_size}

########################################################################
def reset_step(step_size, n_batches_per_epoch, 
               gamma, reset_option=1,
               init_step_size=None, eta_max=None, step=None):
    if reset_option == 0:
        pass

    elif reset_option == 1:
        # try to increase the step-size up to maximum ETA
        step_size = step_size * gamma**(1. / n_batches_per_epoch)
        if eta_max is not None:
            step_size = min(step_size, eta_max)

    elif reset_option == 2:
        step_size = init_step_size

    elif reset_option == 3 and (step % (int(n_batches_per_epoch)) == 1):
        step_size = init_step_size

    return step_size
########################################################################
def FSSreset_step1(s,step_size,counter,params_current,params_prev,grad_current,grad_previous):
    return 1.0
    if(counter==0):
    	return 1.0
    #s=[]
    #for i, j in zip(params_current,params_prev):
    	#s.append(i-j)
    #print("params_current")
    #print(params_current)
    #print("params_prev")
    #print(params_prev)
    s1=0.
    for t, m in zip(s,s):
    	s1+=torch.sum(torch.mul(t,m))
    #print("s")
    #print(s)
    y_current=[]
    for i, j in zip(grad_current,grad_previous):
    	y_current.append(i-j)
    #print("y_current")
    #print(y_current)
    s2=0.
    for t, m in zip(y_current,s):
    	s2+=torch.sum(torch.mul(t,m))
    if(s1==0):
    	print("errrorrrs11111111111111")
    if(s2==0):
    	print("errrorrr")
    else:
    	step_size=s1/s2
    if(step_size<0):
    	step_size=step_size*(-1)
    print("step_size")
    print(step_size)
    return step_size
#######################################################################
def try_sgd_update(params, step_size, params_current, grad_current):
    zipped = zip(params, params_current, grad_current)

    for p_next, p_current, g_current in zipped:
        p_next.data = p_current - step_size * g_current
#######################################################################
def try_sgd_update2(params, step_size, params_current, d_current):
    zipped = zip(params, params_current, d_current)
    
    for p_next, p_current, d_c in zipped:
        p_next.data = p_current - step_size * d_c
#######################################################################
def try_sgd_update3(params, step_size, params_current):
    zipped = zip(params, params_current)
    
    for p_next, p_current in zipped:
        p_next.data = p_current - step_size * p_next.grad
####################################################################
def try_FSSsgd_update(params, step_size, params_current, d_current):
    zipped = zip(params, params_current, d_current)

    for p_next, p_current, d_current in zipped:
        p_next.data = p_current - step_size * d_current
####################################################################
def compute_BetaHZ(counter,grad_current,grad_prev,d_prev):
    Beta=0
    if(counter !=0):
        y_current=[]
        for i, j in zip(grad_current,grad_prev):
            y_current.append(i-j)
        
        beta_temp1=0.     #g(k+1)T*yk
        for t, m in zip(grad_current,y_current):
            beta_temp1+=torch.sum(torch.mul(t,m))
        #print("beta_temp1")
        #print(beta_temp1)
        
        beta_temp2=0.       #dkT*yk
        for t, m in zip(d_prev,y_current):
            beta_temp2+=torch.sum(torch.mul(t,m))
            
        beta_temp3=0.       #||yk||**2
        for t, m in zip(y_current,y_current):
            beta_temp3+=torch.sum(torch.mul(t,m))
            
        beta_temp4=0.     #g(k+1)T*dk
        for t, m in zip(grad_current,d_prev):
            beta_temp4+=torch.sum(torch.mul(t,m))
        
        Beta=(beta_temp1/beta_temp2)-2*((beta_temp3/beta_temp2)*(beta_temp4/beta_temp2))
    return Beta
####################################################################
def compute_BetaWYL(counter,grad_current,grad_prev, grad_prev_norm):
    Beta=0
    if(counter !=0):
        grad_current_norm=compute_grad_norm(grad_current)
        
        beta_temp1=0.     #g(k+1)T*g(k)
        for t, m in zip(grad_current,grad_prev):
            beta_temp1+=torch.sum(torch.mul(t,m))
            
        Beta=((grad_current_norm ** 2)-((grad_current_norm/grad_prev_norm)*(beta_temp1)))/(grad_prev_norm ** 2)
    return Beta
####################################################################
def compute_BetaYWH(counter,grad_current,grad_prev, grad_prev_norm,d_prev):
    Beta=0
    if(counter !=0):
        grad_current_norm=compute_grad_norm(grad_current)
        
        y_current=[]
        for i, j in zip(grad_current,grad_prev):
            y_current.append(i-j)
        
        beta_temp1=0.     #g(k+1)T*g(k)
        for t, m in zip(grad_current,grad_prev):
            beta_temp1+=torch.sum(torch.mul(t,m))
        
        beta_temp2=0.       #dkT*yk
        for t, m in zip(d_prev,y_current):
            beta_temp2+=torch.sum(torch.mul(t,m))
            
        Beta=((grad_current_norm ** 2)-((grad_current_norm/grad_prev_norm)*(beta_temp1)))/(beta_temp2)
    return Beta
####################################################################
def compute_BetaJHJ(counter,grad_current,grad_prev, grad_prev_norm,d_prev):
    Beta=0
    if(counter !=0):
        grad_current_norm=compute_grad_norm(grad_current)
        d_prev_norm=compute_grad_norm(d_prev)
        
        y_current=[]
        for i, j in zip(grad_current,grad_prev):
            y_current.append(i-j)
        
        beta_temp1=0.     #g(k+1)T*g(k)
        for t, m in zip(grad_current,grad_prev):
            beta_temp1+=torch.sum(torch.mul(t,m))
        beta_temp1*=(grad_current_norm/grad_prev_norm)
        
        beta_temp2=0.       #dkT*yk
        for t, m in zip(d_prev,y_current):
            beta_temp2+=torch.sum(torch.mul(t,m))
            
        beta_temp3=0.       #gkT*dk
        for t, m in zip(grad_current,d_prev):
            beta_temp3+=torch.sum(torch.mul(t,m))
        beta_temp3*=(grad_current_norm/d_prev_norm)
        
        beta_temp4=0
        if(beta_temp3>beta_temp1):
             beta_temp4= beta_temp3;
        else:
             beta_temp4= beta_temp1;  
        
        if(beta_temp4<0):
             beta_temp4=0      
            
        Beta=((grad_current_norm ** 2)-(beta_temp4))/(beta_temp2)
    return Beta
####################################################################
def compute_BetaN(counter,grad_current,grad_prev, grad_prev_norm,d_prev):
    Beta=0
    if(counter !=0):
        grad_current_norm=compute_grad_norm(grad_current)
        d_prev_norm=compute_grad_norm(d_prev)
        
        y_current=[]
        for i, j in zip(grad_current,grad_prev):
            y_current.append(i-j)
        
        beta_temp1=0.     #g(k+1)T*g(k)
        for t, m in zip(grad_current,grad_prev):
            beta_temp1+=torch.sum(torch.mul(t,m))
        beta_temp1*=(grad_current_norm/grad_prev_norm)
        if(beta_temp1<0):
             beta_temp1=0 
        
        beta_temp2=0.       #dkT*yk
        for t, m in zip(d_prev,y_current):
            beta_temp2+=torch.sum(torch.mul(t,m))
            
        beta_temp3=0
        if(beta_temp2>(grad_prev_norm ** 2)):
             beta_temp3= beta_temp2
        else:
             beta_temp3= (grad_prev_norm ** 2)  
                    
        Beta=((grad_current_norm ** 2)-(beta_temp1))/(beta_temp3)
    return Beta
####################################################################
def compute_BetaIFR(counter,grad_current,grad_prev, grad_prev_norm,d_prev):
    Beta=0
    if(counter !=0):
        grad_current_norm=compute_grad_norm(grad_current)
          
        beta_temp1=0.     #g(k)T*d(k-1)
        for t, m in zip(grad_current,d_prev):
            beta_temp1+=torch.sum(torch.mul(t,m))
        if(beta_temp1<0):
            beta_temp1*=(-1)
            
        beta_temp2=0.     #g(k-1)T*d(k-1)
        for t, m in zip(grad_prev,d_prev):
            beta_temp2+=torch.sum(torch.mul(t,m))
        beta_temp2*=(-1)
                    
        Beta=(beta_temp1/beta_temp2)*((grad_current_norm ** 2)/(grad_prev_norm **2))
    return Beta
####################################################################
def compute_BetaIDY(counter,grad_current,grad_prev, grad_prev_norm,d_prev):
    Beta=0
    if(counter !=0):
    
        y_current=[]
        for i, j in zip(grad_current,grad_prev):
            y_current.append(i-j)
            
        grad_current_norm=compute_grad_norm(grad_current)
          
        beta_temp1=0.     #g(k)T*d(k-1)
        for t, m in zip(grad_current,d_prev):
            beta_temp1+=torch.sum(torch.mul(t,m))
        if(beta_temp1<0):
            beta_temp1*=(-1)
            
        beta_temp2=0.     #g(k-1)T*d(k-1)
        for t, m in zip(grad_prev,d_prev):
            beta_temp2+=torch.sum(torch.mul(t,m))
        beta_temp2*=(-1)
            
        beta_temp3=0.       #dkT*yk
        for t, m in zip(d_prev,y_current):
            beta_temp3+=torch.sum(torch.mul(t,m))  
                  
        Beta=(beta_temp1/beta_temp2)*((grad_current_norm ** 2)/(beta_temp3))
    return Beta
####################################################################
def compute_Betanew(counter,grad_current,grad_prev, grad_prev_norm,d_prev):
    Beta=0
    if(counter !=0):      
        y_current=[]
        for i, j in zip(grad_current,grad_prev):
            y_current.append(i-j)
            
        grad_current_norm=compute_grad_norm(grad_current)
        d_prev_norm=compute_grad_norm(d_prev)
          
        beta_temp1=0.     #g(k)T*d(k-1)
        for t, m in zip(grad_current,d_prev):
            beta_temp1+=torch.sum(torch.mul(t,m))
        if(beta_temp1<0):
            beta_temp1*=(-1)
            
        beta_temp2=0.       #dkT*yk
        for t, m in zip(d_prev,y_current):
            beta_temp2+=torch.sum(torch.mul(t,m))  
                  
        beta_temp3=grad_current_norm/d_prev_norm
        	
        Beta=((grad_current_norm ** 2)-(beta_temp3*beta_temp1))*(beta_temp2)
    return Beta
####################################################################
def compute_BetaHS(counter,grad_current,grad_prev, grad_prev_norm,d_prev):
    Beta=0
    if(counter !=0):
        y_current=[]
        for i, j in zip(grad_current,grad_prev):
            y_current.append(i-j)
        beta_temp1=0.
        for t, m in zip(grad_current,y_current):
            beta_temp1+=torch.sum(torch.mul(t,m))
        #print("beta_temp1")
        #print(beta_temp1)
        beta_temp2=0.
        for t, m in zip(d_prev,y_current):
            beta_temp2+=torch.sum(torch.mul(t,m))

        if(beta_temp2==0):
            print( "beta_temp2")
            print(beta_temp2) 
        Beta=beta_temp1/beta_temp2
    
    return Beta
####################################################################
def compute_BetaHS2(counter,grad_current,grad_prev, grad_prev_norm,d_prev):
    Beta=0
    cuda0 = torch.device('cuda:0')
    if(counter !=0):
        y_current=[]
        for i, j in zip(grad_current,grad_prev):
            y_current.append(i-j)
        Beta=[]
        for i in range(len(grad_current)):
            Beta+=[torch.tensor(grad_current[i].shape,dtype=torch.double,device=cuda0)]
        for i, (gc,yc,dp) in enumerate(zip(grad_current,y_current,d_prev)):
            Beta[i]=(gc*yc)/(dp*yc)
    return Beta
####################################################################
def compute_BetaFR(counter,grad_current,grad_prev, grad_prev_norm):
    Beta=0
    if(counter !=0):
        grad_current_norm=compute_grad_norm(grad_current)
        Beta=(grad_current_norm ** 2)/(grad_prev_norm ** 2)
    return Beta
####################################################################
def compute_BetaFR2(counter,grad_current,grad_prev, grad_prev_norm):
    Beta=0
    cuda0 = torch.device('cuda:0')
    if(counter !=0):
        grad_current_norm=compute_grad_norm(grad_current)
        Beta=[]
        for i in range(len(grad_current)):
            Beta+=[torch.tensor(grad_current[i].shape,dtype=torch.double,device=cuda0)]
        for i,(gc, gp) in enumerate(zip(grad_current,grad_prev)):
            #Beta[i]=(gc)/(gp)
            Beta[i]=(grad_current_norm ** 2)/(grad_prev_norm **2)
    return Beta
####################################################################
def compute_BetaPRP(counter,grad_current,grad_prev, grad_prev_norm):
    Beta=0
    if(counter !=0):
        y_current=[]
        for i, j in zip(grad_current,grad_prev):
            y_current.append(i-j)
        beta_temp1=0.
        for t, m in zip(grad_current,y_current):
            beta_temp1+=torch.sum(torch.mul(t,m))
        Beta=beta_temp1/(grad_prev_norm ** 2)
    return Beta
####################################################################
def compute_BetaPRP2(counter,grad_current,grad_prev, grad_prev_norm):
    Beta=0
    cuda0 = torch.device('cuda:0')
    if(counter !=0):
        y_current=[]
        for i, j in zip(grad_current,grad_prev):
            y_current.append(i-j)
        Beta=[]
        for i in range(len(grad_current)):
            Beta+=[torch.tensor(grad_current[i].shape,dtype=torch.double,device=cuda0)]
        for i,(gc, yc) in enumerate(zip(grad_current,y_current)):
            Beta[i]=(gc*yc)/(grad_prev_norm ** 2)
    return Beta
####################################################################
def compute_BetaCD(counter,grad_current,grad_prev, d_prev):
    Beta=0
    if(counter !=0):
        grad_current_norm=compute_grad_norm(grad_current)
        beta_temp2=0.
        for t, m in zip(d_prev,grad_prev):
            beta_temp2+=torch.sum(torch.mul(t,m))
        Beta=-1*(grad_current_norm ** 2)/beta_temp2
    return Beta
####################################################################
def compute_BetaCD2(counter,grad_current,grad_prev, d_prev):
    Beta=0
    cuda0 = torch.device('cuda:0')
    if(counter !=0):
        grad_current_norm=compute_grad_norm(grad_current)
        Beta=[]
        for i in range(len(grad_current)):
            Beta+=[torch.tensor(grad_current[i].shape,dtype=torch.double,device=cuda0)]
        for i,(dp,gp) in enumerate(zip(d_prev,grad_prev)):
            Beta[i]=-1*(grad_current_norm ** 2)/(dp*gp)
    return Beta
####################################################################
def compute_BetaDY(counter,grad_current,grad_prev, grad_prev_norm,d_prev):
    Beta=0
    cuda0 = torch.device('cuda:0')
    if(counter !=0):
        y_current=[]
        for i in range(len(grad_current)):
            y_current+=[torch.tensor(grad_current[i].shape,dtype=torch.double,device=cuda0)]
        for i, (j, k) in enumerate(zip(grad_current,grad_prev)):
            y_current[i]=(j-k)
        #print("y_current")
        #print(y_current)
        grad_current_norm=compute_grad_norm(grad_current)
        beta_temp2=0.
        for dt, ym in zip(d_prev,y_current):
            beta_temp2+=torch.sum(dt*ym)
        Beta=(grad_current_norm **2)/beta_temp2
    return Beta
####################################################################
def compute_BetaDY2(counter,grad_current,grad_prev, grad_prev_norm,d_prev):
    Beta=0
    cuda0 = torch.device('cuda:0')
    if(counter !=0):
        y_current=[]
        for i in range(len(grad_current)):
            y_current+=[torch.tensor(grad_current[i].shape,dtype=torch.double,device=cuda0)]
        for i, (j, k) in enumerate(zip(grad_current,grad_prev)):
            y_current[i]=(j-k)
        #print("y_current")
        #print(y_current)
        Beta=[]
        for i in range(len(grad_current)):
            Beta+=[torch.tensor(grad_current[i].shape,dtype=torch.double,device=cuda0)]
        grad_current_norm=compute_grad_norm(grad_current)
        for i , (dt, ym) in enumerate(zip(d_prev,y_current)):
            Beta[i]=(grad_current_norm **2)/(dt*ym)
    return Beta
####################################################################
def compute_BetaLS(counter,grad_current,grad_prev, grad_prev_norm,d_prev):
    Beta=0
    cuda0 = torch.device('cuda:0')
    if(counter !=0):
        y_current=[]
        for i in range(len(grad_current)):
            y_current+=[torch.tensor(grad_current[i].shape,dtype=torch.double,device=cuda0)]
        for i, (j, k) in enumerate(zip(grad_current,grad_prev)):
            y_current[i]=(j-k)
        beta_temp1=0.
        for t, m in zip(grad_current,y_current):
            beta_temp1+=torch.sum(torch.mul(t,m))
        beta_temp2=0.
        for t, m in zip(d_prev,grad_prev):
            beta_temp2+=torch.sum(torch.mul(t,m))
        Beta=-1 * beta_temp1/beta_temp2
    return Beta
####################################################################
def compute_BetaLS2(counter,grad_current,grad_prev, grad_prev_norm,d_prev):
    Beta=0
    cuda0 = torch.device('cuda:0')
    if(counter !=0):
        y_current=[]
        for i in range(len(grad_current)):
            y_current+=[torch.tensor(grad_current[i].shape,dtype=torch.double,device=cuda0)]
        for i, (j, k) in enumerate(zip(grad_current,grad_prev)):
            y_current[i]=(j-k)
        Beta=[]
        for i in range(len(grad_current)):
            Beta+=[torch.tensor(grad_current[i].shape,dtype=torch.double,device=cuda0)]
        for i, (gc,gp,yc,dp) in enumerate(zip(grad_current,grad_prev,y_current,d_prev)):
            Beta[i]=-1*(gc*yc)/(dp*gp)
    return Beta
#####################################################################
def compute_d(counter,grad_current,grad_prev, grad_prev_norm,d_prev,eta0):
    d_current=[]
    if(counter ==0):
        for i in grad_current:
            d_current+=[-1*i]
        eta=eta0
    else:
        y_current=[]
        for i, j in zip(grad_current,grad_prev):
            y_current.append(i-j)
        beta_temp=0.
        for t, m in zip(grad_current,y_current):
            beta_temp+=torch.sum(torch.mul(t,m))
        beta_current=beta_temp/(grad_prev_norm**2)
        for i in range(len(grad_current)):
            d_current+=[(-1*grad_current[i])+(beta_current*d_prev[i])]
        eta=(2+((-1/2)**(counter)))*eta0/3
    return d_current,eta
#####################################################################
def compute_d1(counter,grad_current):
    d_current=[]
    cuda0 = torch.device('cuda:0')
    for i in range(len(grad_current)):
        d_current+=[torch.tensor(grad_current[i].shape,dtype=torch.double,device=cuda0)]
    for i, x in enumerate(grad_current):
            d_current[i]=-1*x
    return d_current
#####################################################################
def compute_d2(counter,grad_current,d_prev, Beta):
    d_current=[]
    cuda0 = torch.device('cuda:0')
    for i in range(len(grad_current)):
        d_current+=[torch.tensor(grad_current[i].shape,dtype=torch.double,device=cuda0)]
    if(counter ==0):
        for i, x in enumerate(grad_current):
            d_current[i]=x
        return d_current
    else:
        zipped = zip(grad_current, d_prev)
        for i, (g,dp) in enumerate(zipped):
            d_current[i]=(g)+(Beta*dp)
    return d_current
#####################################################################
def compute_d3(counter,grad_current,d_prev, Beta):
    d_current=[]
    cuda0 = torch.device('cuda:0')
    for i in range(len(grad_current)):
        d_current+=[torch.tensor(grad_current[i].shape,dtype=torch.double,device=cuda0)]
    if(counter ==0):
        for i, x in enumerate(grad_current):
            d_current[i]=-1*x
    else:
        zipped = zip(grad_current, d_prev,Beta)
        for i, (g,dp,B) in enumerate(zipped):
            d_current[i]=(-1*g)+(B*dp)
    return d_current

#####################################################################
def compute_v(counter,v_prev,old_params,current_params):
    v_current=[]
    cuda0 = torch.device('cuda:0')
    for i in range(len(v_prev)):
        v_current+=[torch.tensor(v_prev[i].shape,dtype=torch.double,device=cuda0)]

    zipped = zip(old_params, current_params,v_prev)
    for i, (a,b,c) in enumerate(zipped):
        v_current[i]=(b-a+c)
    return v_current
#####################################################################
def compute_grad_norm(grad_list):
    grad_norm = 0.
    for g in grad_list:
        if g is None:
            continue
        grad_norm += torch.sum(torch.mul(g, g))
    #grad_norm = torch.sqrt(torch.tensor[grad_norm])
    return grad_norm
#####################################################################    
def compute_param_norm(param_list):
    param_norm = 0.
    print("param_list")
    print(param_list)
    for p in param_list:
        if p is None:
            continue
        param_norm += torch.sum(torch.mul(p, p))
    param_norm = torch.sqrt(param_norm)
    return param_norm
#####################################################################
def get_grad_list(params):
    return [p.grad for p in params]

def get_d_list(params):
    return [((-1)*(p.grad)) for p in params]

@contextlib.contextmanager
def random_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)

@contextlib.contextmanager
def random_seed_torch(seed, device=0):
    cpu_rng_state = torch.get_rng_state()
    if torch.cuda.is_available():
        gpu_rng_state = torch.cuda.get_rng_state(0)

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    try:
        yield
    finally:
        torch.set_rng_state(cpu_rng_state)
        if torch.cuda.is_available():
            torch.cuda.set_rng_state(gpu_rng_state, device)
