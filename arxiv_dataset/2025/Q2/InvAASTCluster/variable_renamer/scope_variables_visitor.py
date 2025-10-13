#!/usr/bin/env python3

#-----------------------------------------------------------------
# pycparser: scope_variables_visitor.py
#
# Using pycparser for injecting a new function in every scope and after each program instruction
# of a C file. These functions receive all the visible initialized variable in that scope/instruction
#
# Eli Bendersky [https://eli.thegreenplace.net/]
# License: BSD
#-----------------------------------------------------------------
from __future__ import print_function
import sys, os
import re
from copy import deepcopy

from shutil import copyfile

# This is not required if you've installed pycparser into
# your site-packages/ with setup.py
sys.path.extend(['.', '..'])

from pycparser import c_parser, c_ast, parse_file, c_generator

#-----------------------------------------------------------------
id_dict = {}
cur_id = 0

#-----------------------------------------------------------------
def node_id (coord):
    global id_dict
    global cur_id
    file = coord.file
    line = coord.line
    column = coord.column
    s = file+str(line)+str(column)
    #print('node_id')
    #print(s)
    if s in id_dict:
        return id_dict[s]
    else:
        cur_id += 1
        return cur_id

#-----------------------------------------------------------------
# A visitor with some state information the variables visible in the current scope
class ScopeVariablesVisitor(c_ast.NodeVisitor):

    def __init__ (self):
        # self.const_code = c_ast.Constant(type='int', value=code)
        self.scope_vars = list([list()])
        self.scope_functions = list()
        self.func_status_id = -1 #because of fakestart function. We need to remove scope_info_-1 and 0
        self.unitialized_vars = list([dict()])


    def get_scope(self):
        # returns the numbers of scope seen until now
        return len(self.scope_vars)-1

    def push_scope(self):
        # copying variables and unitialized variables to new scope.
        self.scope_vars.append(list(self.scope_vars[-1]))
        self.unitialized_vars.append(deepcopy(self.unitialized_vars[-1]))

    def pop_scope(self):
        # retrieving variables and unitialized variables from previous scope.
        self.scope_vars.pop()
        self.unitialized_vars.pop()

    def initialize_variable(self, var_name, s_num):
        # initializes variable declared in scope s_num
        for i in range(s_num,len(self.unitialized_vars)):
            self.scope_vars[i].append(self.unitialized_vars[i][var_name][0])
            del self.unitialized_vars[i][var_name]

    def visit(self, node):
        #node.show()
        return c_ast.NodeVisitor.visit(self, node)

    def get_status_func(self, name="_scope_"):
        vars = []
        params = []
        for v in self.scope_vars[-1]:
            vars.append(c_ast.ID(v[1]))# gets the name of the lvalue v[1]
            # if the type of the parameter is more complex than a simple primitive type
            if isinstance(v[0], c_ast.ArrayDecl) or isinstance(v[0], c_ast.PtrDecl): # or isinstance(v[0], c_ast.Enum):
                ty=deepcopy(v[0])
            elif isinstance(v[0], c_ast.Enum):
                ty = c_ast.TypeDecl(declname=v[1],
                                quals=[],
                                type=v[0])
            else:
                ty = c_ast.TypeDecl(declname=v[1],
                                quals=[],
                                align=[],
                                type=c_ast.IdentifierType([v[0]]))
            newdecl = c_ast.Decl(
                        name=v[1],
                        quals=[],
                        storage=[],
                        funcspec=[],
                        align=[],
                        type=ty,
                        init=None,
                        bitsize=None,
                        coord=None)
            params.append(newdecl)

        func_name=name+str(self.get_scope())+"_"+str(self.func_status_id)
        n_status = c_ast.FuncCall(c_ast.ID(func_name), c_ast.ExprList( vars ))
        self.scope_functions.append(c_ast.FuncDef(decl=c_ast.Decl(
                                    name=func_name,
                                    type=c_ast.FuncDecl(args=c_ast.ExprList(params),type=c_ast.TypeDecl(declname=func_name, quals=[], align=[], type=c_ast.IdentifierType(['void']))),
                                    quals=[],
                                    storage=[],
                                    funcspec=[],
                                    align=[],
                                    init=None,
                                    bitsize=None), param_decls=[], body=c_ast.Compound([])))
        self.func_status_id += 1
        return n_status

    def visit_FileAST(self, node):
        #print('****************** Found FileAST Node *******************')
        n_ext = []
        fakestart_pos = -1 #for the case of our injected function which do not have the fakestart function in their ast
        for e in range(len(node.ext)):
            x = node.ext[e]
            n_ext.append(self.visit(x))
            if isinstance(x, c_ast.FuncDef) and "fakestart" in x.decl.type.type.declname:
                fakestart_pos=e

        n_file_ast = c_ast.FileAST(n_ext[fakestart_pos+1:])
        vars_info_ast = c_ast.FileAST(self.scope_functions[2:])
        return n_file_ast, vars_info_ast

    def visit_Decl(self, node):
        #print('****************** Found Decl Node *******************')
        if not isinstance(node.type, c_ast.TypeDecl):
            # because it can be other type of declaration. Like func declarations.
            node.type = self.visit(node.type)
            if isinstance(node.type, c_ast.Enum):
                # Enum are declared in the var_info.h file!
                return
            else:
                return node
        # node.show()
        type = node.type
        if isinstance(type.type, c_ast.Enum):
             # type = type.type.name
            # node.type = self.visit(node.type)
            type = node.type.type
        else:
            type = type.type.names[0]
        if node.init == None:
            # if the variables is unitialized, we added it to the dict of the current scope
            self.unitialized_vars[-1][node.name] = [[type, node.name], self.get_scope()]
            for v in self.scope_vars[-1]:
                if [type, node.name] == v:
                    self.scope_vars[-1].remove(v)
        else:
            self.scope_vars[-1].append([type, node.name])

        return node

    def visit_ArrayDecl(self, node):
        #print('****************** Found Decl Node *******************')
        # node.show()
        type = node.type
        while isinstance(type, c_ast.ArrayDecl):
            type = type.type
        self.unitialized_vars[-1][type.declname] = [[node, type.declname], self.get_scope()]

        return node

    def visit_Enum(self, node):
        # #print('****************** Found Enum Node *******************')
        # insert each enum on the .h file, after the scope functions of the fakestart function
        self.scope_functions.insert(2, node)
        # return node

    def visit_Assignment(self, node):
        # #print('****************** Found Assignment Node *******************')
        # node.show()
        if isinstance(node.lvalue, c_ast.UnaryOp):
            lval = node.lvalue
            while isinstance(lval, c_ast.UnaryOp):
                lval = lval.expr
            var_name = lval.name
        else:
            var_name = node.lvalue.name
        while not isinstance(var_name, str):
            var_name = var_name.name
        if var_name in self.unitialized_vars[-1] and self.unitialized_vars[-1][var_name]:
            self.initialize_variable(var_name, self.unitialized_vars[-1][var_name][1])

        return node

    def visit_ExprList(self, node):
        #print('****************** Found ExprList Node *******************')
        for e in node.exprs:
            e = self.visit(e)
        return node

    def visit_BinaryOp(self, node):
        #print('****************** Found Binary Operation *******************')
        left = self.visit(node.left)
        right = self.visit(node.right)
        return c_ast.BinaryOp(node.op, left, right, node.coord)

    def visit_TernaryOp(self, node):
        #print('****************** Found IF Node *******************')
        self.push_scope()

        n_status = self.get_status_func('_if_')
        n_cond = self.visit(node.cond)
        n_iftrue = self.visit(node.iftrue)
        n_iftrue = c_ast.Compound([n_status, n_iftrue])
        n_iffalse = self.visit(node.iffalse)
        # if there exists and else statement
        if n_iffalse is not None:
            n_status_2 = self.get_status_func('_else_')
            n_iffalse = c_ast.Compound([n_status_2, n_iffalse])
        #print('****************** New Cond Node *******************')
        n_ternary = c_ast.TernaryOp(n_cond, n_iftrue, n_iffalse, node.coord)

        self.pop_scope()
        return n_ternary

    def visit_FuncDef(self, node):
        #print('****************** Found FuncDef Node *******************')
        self.push_scope()

        decl = node.decl
        param_decls = node.param_decls
        if "main" != node.decl.name and "fakestart" != node.decl.name: #ignore main function
            # if the function has parameters add them to the scope
            if node.decl.type.args:
                for a in node.decl.type.args:
                    if isinstance(a.type, c_ast.ArrayDecl) or isinstance(a.type, c_ast.PtrDecl) or isinstance(a.type.type, c_ast.Enum):
                        type = a.type
                        name = a.name
                    else:
                        type = a.type.type.names[0]
                        name = a.type.declname

                    # if a parameter has the same name as a global variable,
                    # we need to remove the gobal var name from this scope
                    for v in self.scope_vars[-1]:
                        if name == v[1]:
                            self.scope_vars[-1].remove(v)

                    # if there is a global variable not initialized
                    if name in self.unitialized_vars[-1]:
                        del self.unitialized_vars[-1][name]
                    self.scope_vars[-1].append([type, name])

        body = node.body
        coord = node.coord
        n_status = self.get_status_func('_function_')
        n_body_1 = self.visit(body)
        n_body_2 = c_ast.Compound([n_status, n_body_1])
        n_func_def_ast = c_ast.FuncDef(decl, param_decls, n_body_2, coord)

        self.pop_scope()

        return n_func_def_ast

    def visit_FuncCall(self, node):
        #print('****************** Found FuncCall Node *******************')
        # if "scanf" == node.name.name:
        # we can assume if a variable is passed as a parameter then it is used and maybe initialized
        # node.show()
        if node.args: # the function can have zero parameters
            for a in node.args:
                # to get all variables being initialized with scanf
                if isinstance(a, c_ast.Constant):
                    # scanf and printf
                    continue
                elif isinstance(a, c_ast.UnaryOp):
                    # remove this lvalue from the unitialized_vars dict in the current scope
                    # gets the lvalue ID/name
                    # print(node)
                    if isinstance(a.expr, c_ast.ID):
                        var_name=a.expr.name

                    elif isinstance(a.expr, c_ast.ArrayRef):
                        var_name=a.expr.name.name
                    elif isinstance(a.expr, c_ast.StructRef):
                        var_name=a.expr.name.name+a.expr.field.name

                    if var_name in self.unitialized_vars[-1]:
                        self.initialize_variable(var_name, self.unitialized_vars[-1][var_name][1])

                elif isinstance(a, c_ast.BinaryOp) or isinstance(a, c_ast.TernaryOp):
                    a = self.visit(a)

                elif a.name in self.unitialized_vars[-1]:
                    self.initialize_variable(a.name, self.unitialized_vars[-1][a.name][1])

        return node

    def visit_Compound (self, node):
        #print('****************** Found Compound Node *******************')
        self.push_scope()

        block_items = node.block_items
        coord = node.coord
        n_block_items = []
        if block_items is not None:
            for x in block_items:
                n_block_items.append(self.visit(x))

        n_status = self.get_status_func()
        n_block_items.append(n_status)
        n_compound_ast = c_ast.Compound(n_block_items, coord)

        self.pop_scope()
        return n_compound_ast

    def visit_If(self, node):
        #print('****************** Found IF Node *******************')
        self.push_scope()

        if_id = node_id(node.coord)
        n_status = self.get_status_func('_if_')
        n_cond = self.visit(node.cond)
        n_iftrue = self.visit(node.iftrue)
        n_iftrue = c_ast.Compound([n_status, n_iftrue])
        n_iffalse = self.visit(node.iffalse)
        # if there exists and else statement
        if n_iffalse is not None:
            n_status_2 = self.get_status_func('_else_')
            n_iffalse = c_ast.Compound([n_status_2, n_iffalse])
        #print('****************** New Cond Node *******************')
        n_if = c_ast.If(n_cond, n_iftrue, n_iffalse, node.coord)

        self.pop_scope()
        return n_if

    def visit_For(self, node):
        #print('****************** Found For Node *******************')
        self.push_scope()

        for_id = node_id(node.coord)
        n_init = self.visit(node.init)
        n_cond = self.visit(node.cond)
        n_status = self.get_status_func('_loop_')
        n_stmt = self.visit(node.stmt)
        n_next = self.visit(node.next)
        # We dont need to put a scope_info at the end of the for because the compound node already does that
        n_for = c_ast.For(n_init, c_ast.ExprList([n_status, n_cond]), n_next, n_stmt, node.coord)

        self.pop_scope()
        return n_for

    def visit_While(self, node):
        #print('****************** Found While Node *******************')
        self.push_scope()

        while_id = node_id(node.coord)
        n_cond = self.visit(node.cond)
        n_status = self.get_status_func('_loop_')
        n_stmt = self.visit(node.stmt)
        n_while = c_ast.While(c_ast.ExprList([n_status, n_cond]), n_stmt, node.coord)

        self.pop_scope()
        return n_while

    def visit_DoWhile(self, node):
        #print('****************** Found DoWhile Node *******************')
        self.push_scope()

        do_while_id = node_id(node.coord)
        n_cond = self.visit(node.cond)
        n_status = self.get_status_func('_loop_')
        n_stmt = self.visit(node.stmt)
        n_dowhile = c_ast.DoWhile(c_ast.ExprList([n_status, n_cond]), n_stmt, node.coord)

        self.pop_scope()
        return n_dowhile

    def visit_Switch(self, node):
        #print('****************** Found Switch Node *******************')
        self.push_scope()

        switch_id = node_id(node.coord)
        n_status = self.get_status_func('_switch_')
        n_cond = self.visit(node.cond)
        n_stmt = self.visit(node.stmt)
        n_switch = c_ast.Switch(c_ast.ExprList([n_status, n_cond]), n_stmt, node.coord)

        self.pop_scope()
        return n_switch

    def visit_Return(self, node):
        #print('****************** Found Return Node *******************')
        self.push_scope()
        n_status = self.get_status_func('_return_')

        self.pop_scope()
        return c_ast.Compound([ n_status, node ])

    def visit_Break(self, node):
        #print('****************** Found Break Node *******************')
        self.push_scope()

        n_status = self.get_status_func('_break_')
        n_break = c_ast.Compound([ n_status, node ])

        self.pop_scope()
        return n_break

    def visit_Continue(self, node):
        #print('****************** Found Continue Node *******************')
        self.push_scope()

        n_status = self.get_status_func('_continue_')
        n_cont = c_ast.Compound([ n_status, node ])

        self.pop_scope()
        return n_cont

    def visit_Case(self, node):
        #print('****************** Found Case Node *******************')
        self.push_scope()

        stmts = node.stmts
        n_status = self.get_status_func('_case_')
        n_stmts_1 = [ self.visit(x) for x in stmts ]
        n_stmts_2 = c_ast.Compound ([ n_status ] + n_stmts_1, node.coord)

        self.pop_scope()
        return c_ast.Case(node.expr, n_stmts_2, node.coord)

    def visit_Default(self, node):
        #print('****************** Found Default Node *******************')
        self.push_scope()

        n_status = self.get_status_func('_default_')
        n_stmts = self.visit(node.stmts)
        n_stmts_2 = c_ast.Compound ([ n_status ] + n_stmts, node.coord)

        self.pop_scope()
        return c_ast.Default(n_stmts_2, node.coord)


    def generic_visit(self, node):
        #print('******************  Something else ************')
        return node

#-----------------------------------------------------------------
def make_output_dir(input_file, output_dir):
    sincludes = []
    includes = []
    noincludes = []
    with open(input_file, 'r') as reader:
        for line in reader:
            m = re.match('^\s*#\s*include\s*<', line)
            if m:
                sincludes.append(line)
            else:
                m = re.match('^\s*#\s*include', line)
                if m:
                    includes.append(line)
                else:
                    noincludes.append(line)
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    except OSError:
        print("Creation of the directory {0} failed".format(output_dir))

    output_file = output_dir + '/' + os.path.basename(input_file)
    with open(output_file, 'w') as writer:
        writer.writelines(sincludes)
        writer.writelines(includes)
        writer.write('void fakestart() {;}\n')
        writer.writelines(noincludes)

    return output_file, sincludes, includes

#-----------------------------------------------------------------
def instrument_file(input_file, output_dir):
    output_file, sincludes, includes = make_output_dir(input_file, output_dir)#, logfilename, loglibpath)
    ast = parse_file(output_file, use_cpp=True,
            cpp_path='gcc',
            cpp_args=['-E', '-I../utils/fake_libc_include'])

    #print('******************** INPUT FILE: ********************')
    # v = c_ast.NodeVisitor()
    # v.visit(ast)
    # ast.show()
    # exit()
    v = ScopeVariablesVisitor()
    gen = c_generator.CGenerator ()
    n_ast, vars_info_ast = v.visit(ast)
    # n_ast.show()
    # print('******** VARIABLES INFO FILE *******************')
    # vars_info_ast.show()
    str_ast = gen.visit(vars_info_ast)
    # print(str_ast)
    with open(output_dir + '/' + "vars_info.h", 'w') as writer:
        # writer.writelines(sincludes)
        # writer.writelines(includes)
        writer.write(str_ast)

    str_ast = gen.visit(n_ast)
    # print(str_ast)
    # str_ast = remove_fakestart(str_ast)
    with open(output_file, 'w') as writer:
        writer.writelines(["#include \"vars_info.h\"\n"]+sincludes)
        writer.writelines(includes)
        writer.write(str_ast)

#-----------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) >= 2:
        infilename = sys.argv[1]
        outputdir = sys.argv[2]
        instrument_file(infilename, outputdir)
    else:
        print('{0} <input file> <output dir>'.format(sys.argv[0]))
