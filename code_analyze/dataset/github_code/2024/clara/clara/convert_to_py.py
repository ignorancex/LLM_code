from clara.model import Const, Op, Var
from clara.feedback_python import BINARY_OPS, UNARY_OPS

def convertExp(var, expr):
    fncexprs = []
    e = getExprs(expr)
    if str(var) == '$ret':
        fncexprs.append('return %s' % (e))
    elif e and isinstance(e, list):
        if e[0] == 'AssignElement':
            fncexprs.append('%s = %s' % (var, e[1]))
            fncexprs.append('%s[%s] = %s' % (var, e[2], e[3]))
        elif e[0] == 'ite':
            fncexprs.append('if %s:' % e[1])
            fncexprs.append('   %s = %s' % (var, e[2]))
            fncexprs.append('else:')
            fncexprs.append('   %s = %s' % (var, e[3]))
        elif e[0] == 'GetElem':
            fncexprs.append('%s = %s' % (var, e[1]))
            fncexprs.append('%s = %s%s' % (var, var, e[2]))
        elif e[0] == 'Del':
            fncexprs.append('%s = %s' % (var, e[1]))
            fncexprs.append('del %s%s' % (var, e[2]))
        elif e[0] == 'Del2':
            fncexprs.append('del %s%s' % (e[1], e[2]))
        else:
            fncexprs.append('%s = %s' % (var, e))
    elif str(var) != '$out':
        fncexprs.append('%s = %s' % (var, e))
    else:
        fncexprs.append('print(%s)' % e)
    return fncexprs


def makefunc(model):
    for f in model.getfncs():
        fncexprs = []
        locs = f.locexprs
        for l in locs.keys():
            for (var, expr) in locs[l]:
                e = getExprs(expr)
                if str(var) == '$ret':
                    fncexprs.append('return %s' % (e))
                elif e and isinstance(e, list):
                    if e[0] == 'AssignElement':
                        fncexprs.append('%s = %s' % (var, e[1]))
                        fncexprs.append('%s[%s] = %s' % (var, e[2], e[3]))
                    elif e[0] == 'ite':
                        fncexprs.append('if %s:' % e[1])
                        fncexprs.append('   %s = %s' % (var, e[2]))
                        fncexprs.append('else:')
                        fncexprs.append('   %s = %s' % (var, e[3]))
                    elif e[0] == 'GetElem':
                        fncexprs.append('%s = %s' % (var, e[1]))
                        fncexprs.append('%s = %s%s' % (var, var, e[2]))
                    elif e[0] == 'Del':
                        fncexprs.append('%s = %s' % (var, e[1]))
                        fncexprs.append('del %s%s' % (var, e[2]))
                    elif e[0] == 'Del2':
                        fncexprs.append('del %s%s' % (e[1], e[2]))
                    else:
                        fncexprs.append('%s = %s' % (var, e))
                elif str(var) != '$out':
                    fncexprs.append('%s = %s' % (var, e))
                else:
                    fncexprs.append('print(%s)' % e)
    file = open('c.py', 'w')
    file.write('\n'.join(fncexprs))
    file.close()


def getExprs(exp):
    if isinstance(exp, Const):
        value = exp.value
        if '"' in value:
            value = value.replace("'", '').strip()
        else:
            try:
                value = int(value)
                return value
            except:
                pass
        return value
    if isinstance(exp, Var):
        return exp.name.strip()
    if isinstance(exp, Op):
        name = exp.name
        args = exp.args

        if name == 'ListInit':
            vals = [getExprs(i) for i in args]
            return '[' + ', '.join(vals) + ']'

        elif name == 'SetInit':
            vals = [str(getExprs(i)) for i in args]
            return '{ ' + ', '.join(vals) + ' }'

        elif name == 'DictInit':
            vals = []
            size = len(args)
            for i in range(0, size):
                if i % 2 == 1:
                    continue
                vals += [str(getExprs(args[i])) + ':' + str(getExprs(args[i+1]))]
            return '{ ' + ', '.join(vals) + ' }'

        elif name == 'TupleInit':
            return (getExprs(args[0]), getExprs(args[1]))

        elif name == 'BoundVar':
            return getExprs(args[0])

        elif name == 'StrAppend':
            if Var('$out') in args:
                args = args[1:]
            a = [str(getExprs(i)) for i in args]
            return ', '.join(a)

        elif name == 'AssignElement':
            return ['AssignElement', str(getExprs(args[0])), str(getExprs(args[1])), str(getExprs(args[2]))]

        elif name in BINARY_OPS:
            return '(' + str(getExprs(args[0])) + ' '+ BINARY_OPS[name] + ' ' +str(getExprs(args[1])) + ')'

        elif name in UNARY_OPS:
            return '(' + UNARY_OPS[name] + str(getExprs(args[0])) + ')'

        elif name == 'GetAttr':
            return 'getattr(' + str(args[0]) + ')'

        elif name == 'ite':
            vals = [str(getExprs(i)) for i in args]
            return ['ite'] + vals

        elif name == 'Slice':
            vals = []
            for a in args:
                e = getExprs(a)
                if e != 'None':
                    vals += [str(e)]
            return ':'.join(vals) + ':'if len(vals) == 1 else ''

        elif name == 'GetElement':
            e1 = str(getExprs(args[0]))
            e2 = str(getExprs(args[1]))
            if (not isinstance(args[0], Var)) and len(e2) > 1:
                return ['GetElem', e1, '[' + e2 + ']']
            return e1 + '[' + e2 + ']'

        elif name == 'Delete':
            e1 = str(getExprs(args[0]))
            e2 = str(getExprs(args[1]))
            if (not isinstance(args[0], Var)):
                return ['Del', e1, '[' + e2 + ']']
            return ['Del2' , 'del ' + e1 + '[' + e2 + ']']

        elif name == 'AssAdd':
            e1 = str(getExprs(args[0]))
            e2 = str(getExprs(args[1]))
            return '(' + e1 + '+' + e2 + ')'

        else:
            vals = [str(getExprs(i)) for i in args]
            return name + '(' + ', '.join(vals) + ')'
