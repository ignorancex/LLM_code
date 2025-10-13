#!/usr/bin/python
#Title			: var_renaming_visitor.py
#Usage			: python var_renaming_visitor.py -h
#Author			: pmorvalho
#Date			: April 22, 2021
#Description	: Takes an AST find the order per which the variables in the program are being used and renames each variables with its order.
#Notes			:
#Script Version		: 1.0
#Python Version	: 3.8.5
# (C) Copyright 2021 Pedro Orvalho.
#==============================================================================

from scope_variables_visitor import *

#-----------------------------------------------------------------
# A visitor that renames variables depending on the order they are being assigned their first values
class VariableRenamerVisitor(c_ast.NodeVisitor):

    def __init__ (self):
        super().__init__()
        self.renaming = False
        self.vars_maps = list([dict()])
        self.vars_ids = list([dict()])
        self.scopes_history = []
        self.declaring_parameters = False

        self.scope_vars = list([list()])
        self.unitialized_vars = list([dict()])

    def get_scope(self):
        # returns the numbers of scope seen until now
        return len(self.scope_vars)-1

    def push_scope(self):
        # copying variables and unitialized variables to new scope.
        self.scope_vars.append(list(self.scope_vars[-1]))
        self.unitialized_vars.append(deepcopy(self.unitialized_vars[-1]))
        if not self.renaming:
            self.vars_ids.append(deepcopy(self.vars_ids[-1]))
            self.vars_maps.append(deepcopy(self.vars_maps[-1]))
            self.scopes_history.append([])
        else:
            self.vars_maps.append(self.scopes_history[0])
            self.scopes_history.pop(0)

        return len(self.scopes_history)-1

    def pop_scope(self, s):
        # retrieving variables and unitialized variables from previous scope.
        self.scope_vars.pop()
        self.unitialized_vars.pop()
        if not self.renaming:
            self.vars_ids.pop()
            self.scopes_history[s] = self.vars_maps[-1]
        self.vars_maps.pop()


    def initialize_variable(self, var_name, s_num, array=False):
        # initializes variable declared in scope s_num
        type = self.unitialized_vars[s_num][var_name][0][0]
        new_name = self.get_variable_name(type, array=array, s_num=s_num)
        # we need to update the ids for every scope from s_num to the current scope
        for i in range(s_num,len(self.unitialized_vars)):
            self.vars_maps[i][hash(frozenset([var_name, i]))] = new_name
            self.scope_vars[i].append(hash(frozenset(self.unitialized_vars[i][var_name][0])))
            del self.unitialized_vars[i][var_name]

    def get_variable_name(self, type, array=False, s_num=-1):
        type = "array_"+type if array else type
        for i in range(s_num,len(self.unitialized_vars)):
            if type in self.vars_ids[i].keys():
                self.vars_ids[i][type] += 1
            else:
                self.vars_ids[i][type] = 0
        if self.declaring_parameters:
            return '_param_{t}_{id}'.format(t=type,id=self.vars_ids[-1][type])
        return '_{t}_{id}'.format(t=type,id=self.vars_ids[-1][type])

    def visit_FileAST(self, node):
        #print('****************** Found FileAST Node *******************')
        n_ext = []
        fakestart_pos = -1 #for the case of our injected function which do not have the fakestart function in their ast
        for e in range(len(node.ext)):
            x = node.ext[e]
            n_ext.append(self.visit(x))
            if isinstance(x, c_ast.FuncDef) and "fakestart" in x.decl.type.type.declname:
                fakestart_pos=e

        #print('*********************************************************')
        #print('****************** Renaming Variables *******************')
        #print('*********************************************************')
        self.renaming = True
        for e in range(len(n_ext)):
            x = node.ext[e]
            n_ext[e] = self.visit(x)

        n_file_ast = c_ast.FileAST(n_ext[fakestart_pos+1:])
        return n_file_ast

    def visit_Decl(self, node):
        #print('****************** Found Decl Node *******************')
        # print(node)
        if not isinstance(node.type, c_ast.TypeDecl):
            if node.init is not None:
                node.init = self.visit(node.init)
            # because it can be other type of declaration. Like func declarations.
            node.type = self.visit(node.type)
            return node
        # node.show()

        type = node.type
        if isinstance(type.type, c_ast.Enum):
            type = type.type.name
        else:
            type = type.type.names[0]

        if node.init == None:
            # if the variables is unitialized, we added it to the dict of the current scope
            self.unitialized_vars[-1][node.name] = [[type, node.name], self.get_scope()]
            if hash(frozenset([type, node.name])) in self.scope_vars[-1]:
                self.scope_vars[-1].remove(hash(frozenset([type, node.name])))
        else:
            node.init = self.visit(node.init)
            self.scope_vars[-1].append(hash(frozenset([type, node.type.declname])))
            if not self.renaming:
                self.vars_maps[-1][hash(frozenset([node.type.declname, self.get_scope()]))] = self.get_variable_name(type, s_num=self.get_scope())

        if self.renaming and hash(frozenset([node.type.declname, self.get_scope()])) in self.vars_maps[-1].keys():
                            # the variable could be declared but never initialized and/or used
            # print("Renaming: ", node.type.declname, self.vars_maps[-1][hash(frozenset([node.type.declname, self.get_scope()]))])
            node.type.declname = self.vars_maps[-1][hash(frozenset([node.type.declname, self.get_scope()]))]
        return node

    def visit_ArrayDecl(self, node):
        #print('****************** Found Array Decl Node *******************')
        # node.show()
        type = node.type
        if not self.renaming:
            # we need this to find the type of an array of arrays of ....
            while isinstance(type, c_ast.ArrayDecl):
                type = type.type
            self.unitialized_vars[-1][type.declname] = [[type.type.names[0], type.declname], self.get_scope()]

        elif self.renaming:
            # if an array of arrays of ... we need the name of the last type
            if isinstance(node.type, c_ast.ArrayDecl):
                node.type = self.visit(node.type)
                return node

            if hash(frozenset([type.declname, self.get_scope()])) in self.vars_maps[-1].keys():
                node.type.declname = self.vars_maps[-1][hash(frozenset([type.declname, self.get_scope()]))]
        return node

    def visit_PtrDecl(self, node):
        #print('****************** Found Pointer Decl Node *******************')
        # node.show()
        type = node.type
        if not self.renaming:
            # we need this to find the type of a pointer of a pointer...
            while isinstance(type, c_ast.PtrDecl):
                type = type.type
            self.unitialized_vars[-1][type.declname] = [["pointer_"+type.type.names[0], type.declname], self.get_scope()]

        elif self.renaming:
            # if a pointer of a pointer of ... we need the name of the last type
            if isinstance(node.type, c_ast.PtrDecl):
                node.type = self.visit(node.type)
                return node
            if hash(frozenset([type.declname, self.get_scope()])) in self.vars_maps[-1].keys():
                node.type.declname = self.vars_maps[-1][hash(frozenset([type.declname, self.get_scope()]))]
        return node

    def visit_ArrayRef(self, node):
        #print('****************** Found Array Ref Node *******************')
        node.name = self.visit(node.name)
        node.subscript = self.visit(node.subscript)
        return node

    def visit_Assignment(self, node):
        #print('****************** Found Assignment Node *******************')
        node.rvalue = self.visit(node.rvalue)
        if isinstance(node.lvalue, c_ast.UnaryOp):
            var_name = node.lvalue
            while isinstance(var_name, c_ast.UnaryOp):
                var_name = var_name.expr
        elif isinstance(node.lvalue, c_ast.ArrayRef):
            var_name=node.lvalue.name
            while not isinstance(var_name.name, str):
                var_name = var_name.name
            if var_name.name in self.unitialized_vars[-1] and not self.renaming:
                self.initialize_variable(var_name.name, self.unitialized_vars[-1][var_name.name][1], array=True)
        else:
            var_name = node.lvalue


        if var_name.name in self.unitialized_vars[-1] and not self.renaming:
            self.initialize_variable(var_name.name, self.unitialized_vars[-1][var_name.name][1])

        if self.renaming:
            node.lvalue = self.visit(node.lvalue)

        return node

    def visit_ID(self, node):
        #print('****************** Found ID Node *******************')
        if self.renaming:
            scope = self.get_scope()
            k = hash(frozenset([node.name, scope]))
            while k not in self.vars_maps[-1].keys() and scope >= 0:
                # we do this because we are sure the code compiles.
                # So this var is already defined in an upper scope. It is not preety though
                scope -= 1
                k = hash(frozenset([node.name, scope]))
            if scope >= 0:
                node.name = self.vars_maps[-1][k]
        else:
            if node.name in self.unitialized_vars[-1] and not self.renaming:
                self.initialize_variable(node.name, self.unitialized_vars[-1][node.name][1])
        return node

    def visit_ExprList(self, node):
        #print('****************** Found ExprList Node *******************')
        for e in node.exprs:
            e = self.visit(e)
        return node

    def visit_Cast(self, node):
        #print('******************** Found Cast Node *********************')
        node.expr = self.visit(node.expr)
        return node

    # def visit_Struct(self, node):
    #     print('******************** Found Struct Node *********************')
    #
    #     type = "struct"
    #     name = node.name
    #     if not self.renaming and hash(frozenset([name, self.get_scope()])) not in self.vars_maps[-1].keys():
    #         self.vars_maps[-1][hash(frozenset([name, self.get_scope()]))] = self.get_variable_name(type, array=False, s_num=self.get_scope())
    #     if self.renaming:
    #         node.name = self.vars_maps[-1][hash(frozenset([name, self.get_scope()]))]
    #     sn = self.push_scope()
    #     decls = node.decls
    #     coord = node.coord
    #     decls_visited = []
    #
    #     if decls is not None:
    #         for d in decls:
    #             decls_visited.append(self.visit(d))
    #
    #     n_struct_ast = c_ast.Struct(node.name, decls_visited, coord)
    #
    #     self.pop_scope(sn)
    #     return n_struct_ast

    def visit_UnaryOp(self, node):
        #print('****************** Found Unary Operation *******************')
        node.expr = self.visit(node.expr)
        return node

    def visit_BinaryOp(self, node):
        #print('****************** Found Binary Operation *******************')
        left = self.visit(node.left)
        right = self.visit(node.right)
        return c_ast.BinaryOp(node.op, left, right, node.coord)

    def visit_TernaryOp(self, node):
        #print('****************** Found TernaryOp Node *******************')
        # sn = self.push_scope()
        # self.vars_ids.append(deepcopy(self.vars_ids[-1]))

        n_cond = self.visit(node.cond)
        n_iftrue = self.visit(node.iftrue)
        n_iffalse = self.visit(node.iffalse)
        #print('****************** New Cond Node *******************')
        n_ternary = c_ast.TernaryOp(n_cond, n_iftrue, n_iffalse, node.coord)

        # self.pop_scope(sn)
        # self.vars_ids.pop()
        return n_ternary

    def visit_DeclArgs(self, node):
        #print('****************** Found Declaration/Definition Parameters *******************')
        # deals with args in the declarations and definitions of functions
        self.declaring_parameters = True
        if node.args: # the function declaration can have zero parameters
            for a in node.args:
                if a.name is None: # in function declaration when the programmer does not declare the parameters' names
                    continue
                # all arguments are initialized by default
                if isinstance(a.type, c_ast.ArrayDecl):
                    type = a.type
                    # we need this to find the type of an array of arrays of ....
                    while isinstance(type, c_ast.ArrayDecl):
                        type = type.type
                    type = type.type.names[0]
                    name = a.name
                    if not self.renaming and hash(frozenset([name, self.get_scope()])) not in self.vars_maps[-1].keys():
                        self.vars_maps[-1][hash(frozenset([name, self.get_scope()]))] = self.get_variable_name(type, array=True, s_num=self.get_scope())

                # if it is a pointer declaration
                elif isinstance(a.type, c_ast.PtrDecl):
                    type = a.type
                    # we need this to find the type of a pointer of a pointer...
                    while isinstance(type, c_ast.PtrDecl):
                        type = type.type

                    type = "pointer_"+type.type.names[0]
                    name = a.name
                    if not self.renaming and hash(frozenset([name, self.get_scope()])) not in self.vars_maps[-1].keys():
                        self.vars_maps[-1][hash(frozenset([name, self.get_scope()]))] = self.get_variable_name(type, array=False, s_num=self.get_scope())

                elif isinstance(a.type.type, c_ast.Enum):
                    type = a.type
                    type = "enum"
                    name = a.name
                    if not self.renaming and hash(frozenset([name, self.get_scope()])) not in self.vars_maps[-1].keys():
                        self.vars_maps[-1][hash(frozenset([name, self.get_scope()]))] = self.get_variable_name(type, array=False, s_num=self.get_scope())

                else:
                    type = a.type.type.names[0]
                    name = a.type.declname
                    if not self.renaming and hash(frozenset([name, self.get_scope()])) not in self.vars_maps[-1].keys():
                        self.vars_maps[-1][hash(frozenset([name, self.get_scope()]))] = self.get_variable_name(type, s_num=self.get_scope())

                # if a parameter has the same name as a global variable,
                # we need to remove the gobal var name from this scope
                if hash(frozenset([type, name])) not in self.scope_vars[-1]:
                    self.scope_vars[-1].append(hash(frozenset([type, name])))

                # if there is a global variable not initialized
                if name in self.unitialized_vars[-1]:
                    del self.unitialized_vars[-1][name]



            if self.renaming:
                # changes arguments names
                for a in node.args:
                    if a.name is None:
                        continue
                    if isinstance(a.type, c_ast.ArrayDecl) or isinstance(a.type, c_ast.PtrDecl):
                        a.type = self.visit(a.type)
                    else:
                        a.type.declname = self.vars_maps[-1][hash(frozenset([a.type.declname, self.get_scope()]))]

        self.declaring_parameters = False
        return node

    def visit_FuncDecl(self, node):
        #print('****************** Found FuncDecl Node *******************')
        sn = self.push_scope()
        # self.vars_ids.append(deepcopy(self.vars_ids[-1]))
        # we call this separate function because we want to push a scope if we have only a function definition but we dont want to
        # push a new scope in the case of a declaration of a function inside a function definition
        node = self.visit_DeclArgs(node)

        self.pop_scope(sn)
        # self.vars_ids.pop()
        return node


    def visit_FuncDef(self, node):
        #print('****************** Found FuncDef Node *******************')
        sn = self.push_scope()
        # self.vars_ids.append(deepcopy(self.vars_ids[-1]))

        decl = node.decl
        param_decls = node.param_decls
        if "main" != node.decl.name and "fakestart" != node.decl.name: #ignore main function
            # if the function has parameters add them to the scope
            decl = self.visit_DeclArgs(decl.type)

        body = node.body
        coord = node.coord
        n_body_1 = self.visit(body)
        n_func_def_ast = c_ast.FuncDef(decl, param_decls, n_body_1, coord)

        self.pop_scope(sn)
        # self.vars_ids.pop()
        return n_func_def_ast

    def visit_FuncCall(self, node):
        #print('****************** Found FuncCall Node *******************')
        if node.args: # the function can have zero parameters
            for a in node.args:
                # to get all variables being initialized with scanf
                if isinstance(a, c_ast.Constant):
                    # scanf and printf
                    continue
                elif isinstance(a, c_ast.UnaryOp):
                    # remove this lvalue from the unitialized_vars dict in the current scope
                    # gets the lvalue ID/name
                    if isinstance(a.expr, c_ast.ID):
                        var_name=a.expr.name
                        if var_name in self.unitialized_vars[-1] and not self.renaming:
                            self.initialize_variable(var_name, self.unitialized_vars[-1][var_name][1])

                    elif isinstance(a.expr, c_ast.ArrayRef):
                        var_name=a.expr.name.name
                        if var_name in self.unitialized_vars[-1] and not self.renaming:
                            self.initialize_variable(var_name, self.unitialized_vars[-1][var_name][1], array=True)

                    elif isinstance(a.expr, c_ast.StructRef):
                        var_name=a.expr.name.name+a.expr.field.name
                        if var_name in self.unitialized_vars[-1] and not self.renaming:
                            self.initialize_variable(var_name, self.unitialized_vars[-1][var_name][1], array=False)

                elif isinstance(a, c_ast.BinaryOp) or isinstance(a, c_ast.TernaryOp):
                    a = self.visit(a)
                elif a.name in self.unitialized_vars[-1] and not self.renaming:
                    self.initialize_variable(a.name, self.unitialized_vars[-1][a.name][1], array=False)

                if self.renaming:
                    a = self.visit(a)

        return node

    def visit_Compound (self, node):
        #print('****************** Found Compound Node *******************')
        sn = self.push_scope()
        block_items = node.block_items
        coord = node.coord
        n_block_items = []

        if block_items is not None:
            for x in block_items:
                n_block_items.append(self.visit(x))

        n_compound_ast = c_ast.Compound(n_block_items, coord)

        self.pop_scope(sn)
        return n_compound_ast

    def visit_If(self, node):
        #print('****************** Found IF Node *******************')
        sn = self.push_scope()

        if_id = node_id(node.coord)
        n_cond = self.visit(node.cond)
        n_iftrue = self.visit(node.iftrue)
        n_iffalse = self.visit(node.iffalse)
        #print('****************** New Cond Node *******************')
        n_if = c_ast.If(n_cond, n_iftrue, n_iffalse, node.coord)

        self.pop_scope(sn)
        return n_if

    def visit_For(self, node):
        #print('****************** Found For Node *******************')
        sn = self.push_scope()

        for_id = node_id(node.coord)
        n_init = self.visit(node.init)
        n_cond = self.visit(node.cond)
        n_stmt = self.visit(node.stmt)
        n_next = self.visit(node.next)
        # We dont need to put a scope_info at the end of the for because the compound node already does that
        n_for = c_ast.For(n_init, n_cond, n_next, n_stmt, node.coord)

        self.pop_scope(sn)
        return n_for

    def visit_While(self, node):
        #print('****************** Found While Node *******************')
        sn = self.push_scope()

        while_id = node_id(node.coord)
        n_cond = self.visit(node.cond)
        n_stmt = self.visit(node.stmt)
        n_while = c_ast.While(n_cond, n_stmt, node.coord)

        self.pop_scope(sn)
        return n_while

    def visit_DoWhile(self, node):
        #print('****************** Found DoWhile Node *******************')
        sn = self.push_scope()

        do_while_id = node_id(node.coord)
        n_cond = self.visit(node.cond)
        n_stmt = self.visit(node.stmt)
        n_dowhile = c_ast.DoWhile(n_cond, n_stmt, node.coord)

        self.pop_scope(sn)
        return n_dowhile

    def visit_Switch(self, node):
        #print('****************** Found Switch Node *******************')
        sn = self.push_scope()

        switch_id = node_id(node.coord)
        n_cond = self.visit(node.cond)
        n_stmt = self.visit(node.stmt)
        n_switch = c_ast.Switch(n_cond, n_stmt, node.coord)

        self.pop_scope(sn)
        return n_switch

    def visit_Return(self, node):
        #print('****************** Found Return Node *******************')
        node.expr = self.visit(node.expr)
        return node

    def visit_Break(self, node):
        #print('****************** Found Break Node *******************')
        return node

    def visit_Continue(self, node):
        #print('****************** Found Continue Node *******************')
        return node

    def visit_Case(self, node):
        #print('****************** Found Case Node *******************')
        sn = self.push_scope()

        stmts = node.stmts
        n_stmts_1 = [ self.visit(x) for x in stmts ]
        n_stmts_2 = c_ast.Compound (n_stmts_1, node.coord)

        self.pop_scope(sn)
        return c_ast.Case(node.expr, n_stmts_2, node.coord)

    def visit_Default(self, node):
        #print('****************** Found Default Node *******************')
        sn = self.push_scope()
        node.stmts = self.visit(node.stmts)
        self.pop_scope(sn)
        return node


    def generic_visit(self, node):
        #print('******************  Something else ************')
        return node

def instrument_file(input_file, output_dir):
    output_file, sincludes, includes = make_output_dir(input_file, output_dir)#, logfilename, loglibpath)
    ast = parse_file(output_file, use_cpp=True,
            cpp_path='gcc',
            cpp_args=['-E', '-I../utils/fake_libc_include'])

    # #print('******************** INPUT FILE: ********************')
    v = ScopeVariablesVisitor()
    renamer = VariableRenamerVisitor()
    gen = c_generator.CGenerator ()
    n_ast, vars_info_ast = v.visit(ast)
    # n_ast.show()

    n_ast = renamer.visit(n_ast)
    renamer = VariableRenamerVisitor()
    # vars_info_ast.show()
    # exit()
    vars_info_ast = renamer.visit(vars_info_ast)
    # n_ast.show()
    # exit()
    # #print('******** VARIABLES INFO FILE *******************')
    str_ast = gen.visit(vars_info_ast)
    # print(str_ast)
    with open(output_dir + '/' + "vars_info.h", 'w') as writer:
        # writer.writelines(sincludes)
        # writer.writelines(includes)
        writer.write(str_ast)

    str_ast = gen.visit(n_ast)
    # print(str_ast)
    with open(output_file, 'w') as writer:
        writer.writelines(["#include \"vars_info.h\"\n"]+sincludes)
        writer.writelines(includes)
        writer.write("\n"+str_ast)

#-----------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) >= 2:
        infilename = sys.argv[1]
        outputdir = sys.argv[2]
        instrument_file(infilename, outputdir)
    else:
        print('{0} <input file> <output dir>'.format(sys.argv[0]))
