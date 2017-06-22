"""Compile a subset of the Python AST to x64-64 assembler."""

import argparse
import ast
import sys


class Assembler:
    def __init__(self, output_file=sys.stdout):
        self.output_file = output_file

    def instr(self, opcode, *args):
        print('\t{}\t{}'.format(opcode, ', '.join(str(a) for a in args)),
              file=self.output_file)

    def label(self, name):
        print('{}:'.format(name), file=self.output_file)

    def directive(self, line):
        print(line, file=self.output_file)

    def comment(self, text):
        print('# {}'.format(text), file=self.output_file)


class LocalsVisitor(ast.NodeVisitor):
    def __init__(self):
        self.local_names = []

    def add(self, name):
        if name not in self.local_names:
            self.local_names.append(name)

    def visit_Assign(self, node):
        assert len(node.targets) == 1, 'can only assign one variable at a time'
        self.visit(node.value)
        self.add(node.targets[0].id)

    def visit_For(self, node):
        self.add(node.target.id)
        for statement in node.body:
            self.visit(statement)


class Compiler:
    def __init__(self, assembler=None):
        if assembler is None:
            assembler = Assembler()
        self.asm = assembler
        self.func = None

    def compile(self, node):
        self.header()
        self.visit(node)
        self.footer()

    def visit(self, node):
        name = node.__class__.__name__
        visit_func = getattr(self, 'visit_' + name, None)
        assert visit_func is not None, '{} not supported'.format(name)
        visit_func(node)

    def header(self):
        self.asm.directive('.section __TEXT, __text')
        self.asm.comment('')

    def footer(self):
        self.compile_putc()

    def compile_putc(self):
        self.asm.label('putc')
        self.compile_enter()
        self.asm.instr('movl', '$0x2000004', '%eax')    # write
        self.asm.instr('movl', '$1', '%edi')            # stdout
        self.asm.instr('movq', '%rbp', '%rsi')          # address
        self.asm.instr('addq', '$16', '%rsi')
        self.asm.instr('movq', '$1', '%rdx')            # length
        self.asm.instr('syscall')
        self.compile_return()

    def visit_Module(self, node):
        for statement in node.body:
            self.visit(statement)

    def visit_FunctionDef(self, node):
        assert self.func is None, 'nested function definitions not supported'
        assert node.args.vararg is None, '*args not supported'
        assert not node.args.kwonlyargs, 'keyword-only args not supported'
        assert not node.args.kwarg, 'keyword args not supported'
        self.func = node.name
        self.label_num = 1
        self.locals = {a.arg: i for i, a in enumerate(node.args.args)}

        locals_visitor = LocalsVisitor()
        locals_visitor.visit(node)
        for name in locals_visitor.local_names:
            if name not in self.locals:
                self.locals[name] = len(self.locals) + 1
        self.break_labels = []

        self.asm.comment('{} locals: {}'.format(self.func, self.locals))

        if node.name == 'main':
            self.asm.directive('.globl _main')
            self.asm.label('_main')
        else:
            self.asm.label(node.name)

        self.num_extra_locals = len(self.locals) - len(node.args.args)
        self.compile_enter(self.num_extra_locals)
        for statement in node.body:
            self.visit(statement)

        if not isinstance(node.body[-1], ast.Return):
            if self.func == 'main':
                self.compile_exit(0)
            else:
                self.compile_return(self.num_extra_locals)

        self.asm.comment('')
        self.func = None

    def compile_enter(self, num_extra_locals=0):
        for i in range(num_extra_locals):
            self.asm.instr('pushq', '$0')
        self.asm.instr('pushq', '%rbp')
        self.asm.instr('movq', '%rsp', '%rbp')

    def compile_return(self, num_extra_locals=0):
        self.asm.instr('popq', '%rbp')
        if num_extra_locals > 0:
            self.asm.instr('leaq', '{}(%rsp),%rsp'.format(num_extra_locals * 8))
        self.asm.instr('ret')

    def compile_exit(self, return_code):
        if return_code is None:
            self.asm.instr('popq', '%rdi')
        else:
            self.asm.instr('movl', '${}'.format(return_code), '%edi')
        self.asm.instr('movl', '$0x2000001', '%eax')
        self.asm.instr('syscall')

    def visit_Return(self, node):
        if node.value:
            self.visit(node.value)
        if self.func == 'main':
            self.compile_exit(None if node.value else 0)
        else:
            if node.value:
                self.asm.instr('popq', '%rax')
            self.compile_return(self.num_extra_locals)

    def visit_Num(self, node):
        self.asm.instr('pushq', '${}'.format(node.n))

    def visit_Str(self, node):
        assert len(node.s) == 1, 'only supports str of length 1'
        self.asm.instr('pushq', '${}'.format(ord(node.s)))

    def local_offset(self, name):
        index = self.locals[name]
        return (len(self.locals) - index) * 8 + 8

    def visit_Name(self, node):
        self.asm.instr('pushq', '{}(%rbp)'.format(self.local_offset(node.id)))

    def visit_Assign(self, node):
        assert len(node.targets) == 1, 'can only assign one variable at a time'
        self.visit(node.value)
        self.asm.instr('popq', '{}(%rbp)'.format(self.local_offset(node.targets[0].id)))

    def visit_AugAssign(self, node):
        self.visit(node.target)
        self.visit(node.value)
        self.visit(node.op)
        self.asm.instr('popq', '{}(%rbp)'.format(self.local_offset(node.target.id)))

    def simple_binop(self, op):
        self.asm.instr('popq', '%rdx')
        self.asm.instr('popq', '%rax')
        self.asm.instr(op, '%rdx', '%rax')
        self.asm.instr('pushq', '%rax')

    def visit_Mult(self, node):
        self.asm.instr('popq', '%rdx')
        self.asm.instr('popq', '%rax')
        self.asm.instr('imulq', '%rdx')
        self.asm.instr('pushq', '%rax')

    def visit_Add(self, node):
        self.simple_binop('addq')

    def visit_Sub(self, node):
        self.simple_binop('subq')

    def visit_BinOp(self, node):
        self.visit(node.left)
        self.visit(node.right)
        self.visit(node.op)

    def visit_UnaryOp(self, node):
        assert isinstance(node.op, ast.USub), 'only unary minus is supported'
        self.visit(ast.Num(n=0))
        self.visit(node.operand)
        self.visit(ast.Sub())

    def visit_Expr(self, node):
        self.visit(node.value)
        self.asm.instr('popq', '%rax')

    def visit_And(self, node):
        self.simple_binop('and')

    def visit_BitAnd(self, node):
        self.simple_binop('and')

    def visit_Or(self, node):
        self.simple_binop('or')

    def visit_BitOr(self, node):
        self.simple_binop('or')

    def visit_BitXor(self, node):
        self.simple_binop('xor')

    def visit_BoolOp(self, node):
        self.visit(node.values[0])
        for value in node.values[1:]:
            self.visit(value)
            self.visit(node.op)

    def visit_Call(self, node):
        assert not node.keywords, 'keyword args not supported'
        for arg in node.args:
            self.visit(arg)
        self.asm.instr('call', node.func.id)
        if node.args:
            self.asm.instr('addq', '${}'.format(8 * len(node.args)), '%rsp')
        self.asm.instr('pushq', '%rax')

    def label(self, slug):
        label = '{}_{}_{}'.format(self.func, self.label_num, slug)
        self.label_num += 1
        return label

    def visit_Compare(self, node):
        assert len(node.ops) == 1, 'only single comparisons supported'
        self.visit(node.left)
        self.visit(node.comparators[0])
        self.visit(node.ops[0])

    def compile_comparison(self, jump_not, slug):
        self.asm.instr('popq', '%rdx')
        self.asm.instr('popq', '%rax')
        self.asm.instr('cmpq', '%rdx', '%rax')
        self.asm.instr('movq', '$0', '%rax')
        label = self.label(slug)
        self.asm.instr(jump_not, label)
        self.asm.instr('incq', '%rax')
        self.asm.label(label)
        self.asm.instr('pushq', '%rax')

    def visit_Lt(self, node):
        self.compile_comparison('jnl', 'less')

    def visit_LtE(self, node):
        self.compile_comparison('jnle', 'less_or_equal')

    def visit_Gt(self, node):
        self.compile_comparison('jng', 'greater')

    def visit_GtE(self, node):
        self.compile_comparison('jnge', 'greater_or_equal')

    def visit_Eq(self, node):
        self.compile_comparison('jne', 'equal')

    def visit_NotEq(self, node):
        self.compile_comparison('je', 'not_equal')

    def visit_If(self, node):
        self.visit(node.test)
        self.asm.instr('popq', '%rax')
        self.asm.instr('cmpq', '$0', '%rax')
        label_else = self.label('else')
        label_end = self.label('end')
        self.asm.instr('jz', label_else)
        for statement in node.body:
            self.visit(statement)
        if node.orelse:
            self.asm.instr('jmp', label_end)
        self.asm.label(label_else)
        for statement in node.orelse:
            self.visit(statement)
        if node.orelse:
            self.asm.label(label_end)

    def visit_While(self, node):
        while_label = self.label('while')
        break_label = self.label('break')
        self.break_labels.append(break_label)
        self.asm.label(while_label)
        self.visit(node.test)
        self.asm.instr('popq', '%rax')
        self.asm.instr('cmpq', '$0', '%rax')
        self.asm.instr('jz', break_label)
        for statement in node.body:
            self.visit(statement)
        self.asm.instr('jmp', while_label)
        self.asm.label(break_label)
        self.break_labels.pop()

    def visit_Break(self, node):
        self.asm.instr('jmp', self.break_labels[-1])

    def visit_Pass(self, node):
        pass

    def visit_For(self, node):
        # Turn for+range loop into a while loop:
        #   i = start
        #   while i < stop:
        #       body
        #       i = i + step
        assert isinstance(node.iter, ast.Call) and node.iter.func.id == 'range', \
            'for can only be used with range()'
        range_args = node.iter.args
        if len(range_args) == 1:
            start = ast.Num(n=0)
            stop = range_args[0]
            step = ast.Num(n=1)
        elif len(range_args) == 2:
            start, stop = range_args
            step = ast.Num(n=1)
        else:
            start, stop, step = range_args
            if (isinstance(step, ast.UnaryOp) and isinstance(step.op, ast.USub) and
                    isinstance(step.operand, ast.Num)):
                # Handle negative step
                step = ast.Num(n=-step.operand.n)
            assert isinstance(step, ast.Num) and step.n != 0, \
                'range() step must be a nonzero integer constant'
        self.visit(ast.Assign(targets=[node.target], value=start))
        test = ast.Compare(
            left=node.target,
            ops=[ast.Lt() if step.n > 0 else ast.Gt()],
            comparators=[stop],
        )
        incr = ast.Assign(
            targets=[node.target],
            value=ast.BinOp(left=node.target, op=ast.Add(), right=step),
        )
        self.visit(ast.While(test=test, body=node.body + [incr]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', help='filename to compile')
    args = parser.parse_args()

    with open(args.filename) as f:
        source = f.read()
    node = ast.parse(source, filename=args.filename)
    compiler = Compiler()
    compiler.compile(node)
