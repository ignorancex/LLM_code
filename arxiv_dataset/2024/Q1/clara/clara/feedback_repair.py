'''
Generating (simple, raw) textual feedback from repair
'''


from clara.convert_to_py import convertExp


class RepairFeedback(object):

    def __init__(self, impl, spec, result, cleanstrings=None):
        self.impl = impl
        self.spec = spec
        self.result = result
        self.feedback = []

    def add(self, msg, *args):
        if args:
            msg %= args
        self.feedback.append(msg)

    def genfeedback(self):
        # Iterate all functions
        # fname - function name
        # mapping - one-to-one mapping of variables
        # repairs - list of repairs
        # sm - structural matching betweeb locations of programs
        for fname, (mapping, repairs, sm) in list(self.result.items()):

            # Copy mapping with converting '*' into a 'new_' variable
            nmapping = {k: '$new_%s' % (k,)
                        if v == '*' else v for (k, v) in list(mapping.items())}

            # Go through all repairs
            # loc1 - location from the spec.
            # var1 - variable from the spec.
            # var2 - variable from the impl.
            # cost - cost of the repair
            for rep in repairs:

                loc1 = rep.loc1
                var1 = rep.var1
                var2 = rep.var2
                cost = rep.cost
                expr1 = rep.expr1

                # Get functions and loc2
                fnc1 = self.spec.getfnc(fname)
                fnc2 = self.impl.getfnc(fname)
                loc2 = sm[loc1]

                # Get exprs (spec. and impl.)
                #expr1 = fnc1.getexpr(loc1, var1)
                expr2 = fnc2.getexpr(loc2, var2)

                # Location of the expression
                if expr2.line:
                    # Either line
                    locdesc = 'at line %s' % (expr2.line,)
                else:
                    # Or location description
                    locdesc = fnc2.getlocdesc(loc2)

                # Delete feedback
                if var1 == '-':
                    self.add("Delete '%s := %s' at line %s (cost=%s)",
                             var2, expr2, expr2.line, cost)
                    continue

                # Rewrite expr1 (from spec.) with variables of impl.
                expr1 = expr1.replace_vars(nmapping)

                # '*' means adding a new variable (and also statement)
                if var2 == '*':
                    self.add("Add assignment '%s := %s' %s (cost=%s)",
                             '$new_%s' % (var1,), expr1, locdesc, cost)
                    continue

                if (str(var2) == expr2):
                    self.add("Add assignment '%s := %s' %s (cost=%s)",
                             var2, expr1, locdesc, cost)
                    continue

                # Output original and new (rewriten) expression for var2
                self.add(
                    "Change '%s := %s' to '%s := %s' %s (cost=%s)",
                    var2, expr2, var2, expr1, locdesc, cost)
    
    def genRemovedLocFeedback(self, exps):
        for l in exps:
            (exp, d) = exps[l]
            for v, e in exp:
                if e.line:
                    locdesc = 'at line %s' % (e.line,)
                else:
                    locdesc = d

                allExprs2 = convertExp(v, e)
                if len(allExprs2) > 1:
                    self.add('Delete ')
                    for line in allExprs2:
                        self.add(line)
                    self.add(
                        "%s of the incorrect program (cost=2.0)", locdesc)
                else:
                    ee = allExprs2[0].split(' = ')
                    if (len(ee) == 2 and str(ee[0]) == str(ee[1])):
                        continue
                    else:
                        self.add(
                            "Delete '%s' %s of the incorrect program  (cost=2)",
                            allExprs2[0], locdesc)


    def genConvertedfeedback(self):
        # Iterate all functions
        # fname - function name
        # mapping - one-to-one mapping of variables
        # repairs - list of repairs
        # sm - structural matching betweeb locations of programs
        for fname, (mapping, repairs, sm) in list(self.result.items()):

            nmapping = {k: '$new_%s' % (k,)
                        if v == '*' else v for (k, v) in list(mapping.items())}

            for rep in repairs:
                loc1 = rep.loc1
                var1 = rep.var1
                var2 = rep.var2
                cost = rep.cost
                expr1 = rep.expr1

                fnc1 = self.spec.getfnc(fname)
                fnc2 = self.impl.getfnc(fname)
                loc2 = sm[loc1]

                expr2 = fnc2.getexpr(loc2, var2)

                if expr2.line:
                    locdesc = 'at line %s' % (expr2.line,)
                else:
                    locdesc = fnc2.getlocdesc(loc2)

                if expr1.line:
                    locdesc1 = 'at line %s' % (expr1.line,)
                else:
                    locdesc1 = fnc1.getlocdesc(loc1)

                allExprs2 = convertExp(var2, expr2)

                if var1 == '-':
                    for line in allExprs2:
                        self.add("Delete '%s' around line %s (cost=%s)",
                                 line, expr2.line, cost)
                    continue

                expr1 = expr1.replace_vars(nmapping)

                if var2 == '*':
                    allExprs1 = convertExp('$new_'+var1, expr1)
                    for line in allExprs1:
                        self.add("Add assignment '%s' %s (cost=%s)",
                                 line, locdesc, cost)
                    continue
                allExprs1 = convertExp(var2, expr1)
                if len(allExprs2) > 1 and len(allExprs1) > 1:
                    self.add('Change')
                    for line in allExprs2:
                        self.add(line)
                    self.add('to')
                    for line in allExprs1:
                        self.add(line)
                    self.add(
                        "%s of the incorrect program and %s of the correct program (cost=%s)", locdesc, locdesc1, cost)
                elif len(allExprs2) > 1:
                    self.add('Change')
                    for line in allExprs2:
                        self.add(line)
                    self.add("to '%s' %s of the incorrect program and %s of the correct program (cost=%s)",
                             allExprs1[0], locdesc, locdesc1, cost)
                elif len(allExprs1) > 1:
                    self.add("Change '%s' to ", allExprs2[0])
                    for line in allExprs1:
                        self.add(line)
                    self.add(
                        "%s of the incorrect program and %s of the correct program (cost=%s)", locdesc, locdesc1, cost)
                else:
                    exps = allExprs2[0].split(' = ')
                    if (len(exps) == 2 and str(exps[0]) == str(exps[1])):
                        self.add("Add assignment '%s' %s of the incorrect program and %s of the correct program (cost=%s)",
                                allExprs1[0], locdesc, locdesc1, cost)
                        continue
                    exps = allExprs1[0].split(' = ')
                    if (len(exps) == 2 and str(exps[0]) == str(exps[1])):
                        self.add("Delete assignment '%s' %s of the incorrect program and %s of the correct program (cost=%s)",
                                allExprs2[0], locdesc, locdesc1, cost)
                        continue
                    self.add(
                        "Change '%s' to '%s' %s of the incorrect program and %s of the correct program (cost=%s)",
                        allExprs2[0], allExprs1[0], locdesc, locdesc1, cost)
