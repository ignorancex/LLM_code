import argparse
import itertools

def main():
    parser = argparse.ArgumentParser(description="""Generates (and prints on the terminal) all possible combinatios\n
         between all string elements in piece1 and piece2, attaching to basestr\n
         \n
         Example:\n
            $ python create_cmd_line.py -basestr "isi.exe writeOnRun=1" -piece1 "xR=-0.1" "xR=-0.2" -piece2 "T=0.2" "T=0.3" """)
    parser.add_argument('-basestr' , required=True , nargs=1  , metavar='STR', type=str, default=[''], help='base string into which all combinations of piece1 and piece2 will be attached to')
    parser.add_argument('-piece1'  , required=True , nargs='+', metavar='STR', type=str, default=[''], help='list of pieces to combine with piece2 and piece3')
    parser.add_argument('-piece2'  , required=True , nargs='+', metavar='STR', type=str, default=[''], help='list of pieces to combine with piece1 and piece3')
    parser.add_argument('-piece3'  , required=False, nargs='+', metavar='STR', type=str, default=[''], help='list of pieces to combine with piece1 and piece2')
    parser.add_argument('-one2one' , required=False, action='store_true', default=False, help='if set, matches each of piece1, piece2, piece3, instead of generating all combinations [like the main diagonal of combinations]')

    args = parser.parse_args()

    has_piece3 = (len(args.piece3) > 0) and (len(args.piece3[0]) > 0)

    if args.one2one:
        if has_piece3:
            all_config = [ dict(piece1=a, piece2=b, piece3=c) for a,b,c in zip(args.piece1,args.piece2,args.piece3) ]
        else:
            all_config = [ dict(piece1=a, piece2=b) for a,b in zip(args.piece1,args.piece2) ]
    else:
        param_comb_args = dict(piece1=args.piece1, piece2=args.piece2)
        if has_piece3:
            param_comb_args = dict(piece3=args.piece3,**param_comb_args)
        all_config = get_param_combinations(**param_comb_args)

    print('') # skip one line
    for c in all_config:
        p = [ c['piece1'], c['piece2'] ]
        if has_piece3:
            p.append(c['piece3'])
        fmt_str = '%s' + ' %s'*len(p)
        print(fmt_str%(args.basestr[0],*p))
    print('') # skip one line

def get_param_combinations(**params):
    """
    each argument in param has the form:
    arg1=[arg1_value1,arg1_value2,...]
    arg2=[arg2_value1,arg2_value2,...]
    ...
    argN=[argN_value1,argN_value2,...]

    returns: [ dict_1, dict_2, ... ]
        a list of dict containg all possible combinations of argument values
        
        each dict is:
        dict_1 = { arg1:arg1_value1, arg2:arg2_value1, ... , argN:argN_value1 }
        dict_2 = { arg1:arg1_value2, arg2:arg2_value1, ... , argN:argN_value1 }
        dict_3 = { arg1:arg1_value3, arg2:arg2_value1, ... , argN:argN_value1 }
        ...
        making all the combinations of possible values for all arguments

    WARNING: this can take up a lot of memory
    """
    param_names = list(params.keys())
    param_values = itertools.product(*params.values())
    config = [ { n:v for n,v in zip(param_names,val_combination) } for val_combination in param_values ]
    return config

if __name__ == '__main__':
    main()