import sys
from io import StringIO 
import numpy as np






#DEFINE CLASS FOR CAPTURING PRINT OUTPUT
    #With the code:
    #with Capturing() as output:
    #print(a)
    #you store in output the output of print
    
class Capturing(list):
    '''
    Captures in output the output of print, works as follows:
        with Capturing() as output:
            print(a)
    '''
    
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self
    
    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio    # free up some memory
        sys.stdout = self._stdout




#DEFINE A FUNCTION TO CONVERT FLAOT OUTPUTS OF PRINT 
#INTO TUPLE OF NUMERATOR AND DENOMINATOR 
#   Gives error if prec_temp needs to be increased

def float_to_rat(output_captur,prec):
    '''
    A function that converts to float type the output of Capturing.
    Gives Error if :prec_temp: (auxiliary parameter defined in main)
    needs to be increased

    Parameters
    ----------
    output_captur : String
        Output of Capturing.
    prec : int
        Precision of the float.
    '''
    output_list = list(output_captur)[0]
    float_to_rat_condition_dot = False
    float_to_rat_condition_neg = False
    break_cond = False
    if output_list[0] == '-':
        output_list = output_list.replace('-', '')
        float_to_rat_condition_neg = True
    for i in range(len(output_list)):
        if output_list[i] == '.':
            if len(output_list)-i-1 >= prec:
                output_list = output_list[:i+prec+1]
                float_to_rat_condition_dot = True
                numerator = output_list.replace('.', '')
                denominator = 10**(len(output_list)-i-1)
                return int(numerator) if float_to_rat_condition_neg == False \
                                      else -int(numerator), denominator
            else:
                sys.exit("Increase variable prec_temp at the beginning of the file")
        elif break_cond == False and i == len(output_list)-1 and float_to_rat_condition_dot == False:
            return int(output_list), 1
