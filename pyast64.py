"""Compile a subset of the Python AST to x64-64 assembler.

Read more about it here: http://benhoyt.com/writings/pyast64/

Released under a permissive MIT license (see LICENSE.txt).
"""

import argparse
import ast
import sys


class Assembler:
    """The Assembler takes care of outputting instructions, labels, etc.,
    as well as a simple peephole optimization to combine sequences of pushes
    and pops.
    """

    def __init__(self, output_file=sys.stdout, peephole=True):
        self.output_file = output_file
        self.peephole = peephole
        # Current batch of instructions, flushed on label and end of function
        self.batch = []

    def flush(self):
        if self.peephole:
            self.optimize_pushes_pops()
        for opcode, args in self.batch:
            print('\t{}\t{}'.format(opcode, ', '.join(str(a) for a in args)),
                  file=self.output_file)
        self.batch = []

    def optimize_pushes_pops(self):
        """This finds runs of push(es) followed by pop(s) and combines
        them into simpler, faster mov instructions. For example:

        pushq   8(%rbp)
        pushq   $100
        popq    %rdx
        popq    %rax

        Will be turned into:

        movq    $100, %rdx
        movq    8(%rbp), %rax
        """
        state = 'default'
        optimized = []
        pushes = 0
        pops = 0

        # This nested function combines a sequence of pushes and pops
        def combine():
            mid = len(optimized) - pops
            num = min(pushes, pops)
            moves = []
            for i in range(num):
                pop_arg = optimized[mid + i][1][0]
                push_arg = optimized[mid - i - 1][1][0]
                if push_arg != pop_arg:
                    moves.append(('movq', [push_arg, pop_arg]))
            optimized[mid - num:mid + num] = moves

        # This loop actually finds the sequences
        for opcode, args in self.batch:
            if state == 'default':
                if opcode == 'pushq':
                    state = 'push'
                    pushes += 1
                else:
                    pushes = 0
                    pops = 0
                optimized.append((opcode, args))
            elif state == 'push':
                if opcode == 'pushq':
                    pushes += 1
                elif opcode == 'popq':
                    state = 'pop'
                    pops += 1
                else:
                    state = 'default'
                    pushes = 0
                    pops = 0
                optimized.append((opcode, args))
            elif state == 'pop':
                if opcode == 'popq':
                    pops += 1
                elif opcode == 'pushq':
                    combine()
                    state = 'push'
                    pushes = 1
                    pops = 0
                else:
                    combine()
                    state = 'default'
                    pushes = 0
                    pops = 0
                optimized.append((opcode, args))
            else:
                assert False, 'bad state: {}'.format(state)
        if state == 'pop':
            combine()
        self.batch = optimized

    def instr(self, opcode, *args):
        self.batch.append((opcode, args))

    def label(self, name):
        self.flush()
        print('{}:'.format(name), file=self.output_file)

    def directive(self, line):
        self.flush()
        print(line, file=self.output_file)

    def comment(self, text):
        self.flush()
        print('# {}'.format(text), file=self.output_file)


class LocalsVisitor(ast.NodeVisitor):
    """Recursively visit a FunctionDef node to find all the locals
    (so we can allocate the right amount of stack space for them).
    """

    def __init__(self):
        self.local_names = []
        self.global_names = []
        self.function_calls = []

    def add(self, name):
        if name not in self.local_names and name not in self.global_names:
            self.local_names.append(name)

    def visit_Global(self, node):
        self.global_names.extend(node.names)

    def visit_Assign(self, node):
        assert len(node.targets) == 1, \
            'can only assign one variable at a time'
        self.visit(node.value)
        target = node.targets[0]
        if isinstance(target, ast.Subscript):
            self.add(target.value.id)
        else:
            self.add(target.id)

    def visit_For(self, node):
        self.add(node.target.id)
        for statement in node.body:
            self.visit(statement)

    def visit_Call(self, node):
        self.function_calls.append(node.func.id)


class Compiler:
    """The main Python AST -> x86-64 compiler."""

    def __init__(self, assembler=None, peephole=True):
        if assembler is None:
            assembler = Assembler(peephole=peephole)
        self.asm = assembler
        self.func = None

    def compile(self, node):
        self.header()
        self.visit(node)
        self.footer()

    def visit(self, node):
        # We could have subclassed ast.NodeVisitor, but it's better to fail
        # hard on AST nodes we don't support
        name = node.__class__.__name__
        visit_func = getattr(self, 'visit_' + name, None)
        assert visit_func is not None, '{} not supported - node {}'.format(
                name, ast.dump(node))
        visit_func(node)

    def header(self):
        self.asm.directive('.section __TEXT, __text')
        self.asm.comment('')

    def footer(self):
        self.compile_putc()
        self.asm.flush()

    def compile_putc(self):
        # Insert this into every program so it can call putc() for output
        self.asm.label('putc')
        self.compile_enter()
        self.asm.instr('movl', '$0x2000004', '%eax')    # write
        self.asm.instr('movl', '$1', '%edi')            # stdout
        self.asm.instr('movq', '%rbp', '%rsi')          # address
        self.asm.instr('addq', '$16', '%rsi')
        self.asm.instr('movq', '$1', '%rdx')            # length
        self.asm.instr('syscall')
        self.compile_return(has_arrays=False)

    def visit_Module(self, node):
        for statement in node.body:
            self.visit(statement)

    def visit_FunctionDef(self, node):
        assert self.func is None, 'nested functions not supported'
        assert node.args.vararg is None, '*args not supported'
        assert not node.args.kwonlyargs, 'keyword-only args not supported'
        assert not node.args.kwarg, 'keyword args not supported'

        self.func = node.name
        self.label_num = 1
        self.locals = {a.arg: i for i, a in enumerate(node.args.args)}

        # Find names of additional locals assigned in this function
        locals_visitor = LocalsVisitor()
        locals_visitor.visit(node)
        for name in locals_visitor.local_names:
            if name not in self.locals:
                self.locals[name] = len(self.locals) + 1
        if 'array' in locals_visitor.function_calls:
            self.locals['_array_size'] = len(self.locals) + 1
        self.globals = set(locals_visitor.global_names)
        self.break_labels = []

        # Function label and header
        if node.name == 'main':
            self.asm.directive('.globl _main')
            self.asm.label('_main')
        else:
            self.asm.label(node.name)
        self.num_extra_locals = len(self.locals) - len(node.args.args)
        self.compile_enter(self.num_extra_locals)

        # Now compile all the statements in the function body
        for statement in node.body:
            self.visit(statement)

        if not isinstance(node.body[-1], ast.Return):
            # Function didn't have explicit return at the end,
            # compile return now (or exit for "main")
            if self.func == 'main':
                self.compile_exit(0)
            else:
                self.compile_return(self.num_extra_locals)

        self.asm.comment('')
        self.func = None

    def compile_enter(self, num_extra_locals=0):
        # Make space for extra locals (in addition to the arguments)
        for i in range(num_extra_locals):
            self.asm.instr('pushq', '$0')
        # Use rbp for a stack frame pointer
        self.asm.instr('pushq', '%rbp')
        self.asm.instr('movq', '%rsp', '%rbp')

    def compile_return(self, num_extra_locals=0, has_arrays=None):
        if has_arrays is None:
            has_arrays = '_array_size' in self.locals
        if has_arrays:
            offset = self.local_offset('_array_size')
            self.asm.instr('movq', '{}(%rbp)'.format(offset), '%rbx')
            self.asm.instr('addq', '%rbx', '%rsp')
        self.asm.instr('popq', '%rbp')
        if num_extra_locals > 0:
            self.asm.instr('leaq', '{}(%rsp),%rsp'.format(
                    num_extra_locals * 8))
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
            # Returning from main, exit with that return code
            self.compile_exit(None if node.value else 0)
        else:
            if node.value:
                self.asm.instr('popq', '%rax')
            self.compile_return(self.num_extra_locals)

    def visit_Num(self, node):
        self.asm.instr('pushq', '${}'.format(node.n))

    def local_offset(self, name):
        index = self.locals[name]
        return (len(self.locals) - index) * 8 + 8

    def visit_Name(self, node):
        # Only supports locals, not globals
        offset = self.local_offset(node.id)
        self.asm.instr('pushq', '{}(%rbp)'.format(offset))

    def visit_Assign(self, node):
        # Only supports assignment of (a single) local variable
        assert len(node.targets) == 1, \
            'can only assign one variable at a time'
        self.visit(node.value)
        target = node.targets[0]
        if isinstance(target, ast.Subscript):
            # array[offset] = value
            self.visit(target.slice.value)
            self.asm.instr('popq', '%rax')
            self.asm.instr('popq', '%rbx')
            local_offset = self.local_offset(target.value.id)
            self.asm.instr('movq', '{}(%rbp)'.format(local_offset), '%rdx')
            self.asm.instr('movq', '%rbx', '(%rdx,%rax,8)')
        else:
            # variable = value
            offset = self.local_offset(node.targets[0].id)
            self.asm.instr('popq', '{}(%rbp)'.format(offset))

    def visit_AugAssign(self, node):
        # Handles "n += 1" and the like
        self.visit(node.target)
        self.visit(node.value)
        self.visit(node.op)
        offset = self.local_offset(node.target.id)
        self.asm.instr('popq', '{}(%rbp)'.format(offset))

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

    def compile_divide(self, push_reg):
        self.asm.instr('popq', '%rbx')
        self.asm.instr('popq', '%rax')
        self.asm.instr('cqo')
        self.asm.instr('idiv', '%rbx')
        self.asm.instr('pushq', push_reg)

    def visit_Mod(self, node):
        self.compile_divide('%rdx')

    def visit_FloorDiv(self, node):
        self.compile_divide('%rax')

    def visit_Add(self, node):
        self.simple_binop('addq')

    def visit_Sub(self, node):
        self.simple_binop('subq')

    def visit_BinOp(self, node):
        self.visit(node.left)
        self.visit(node.right)
        self.visit(node.op)

    def visit_UnaryOp(self, node):
        assert isinstance(node.op, ast.USub), \
            'only unary minus is supported, not {}'.format(node.op.__class__.__name__)
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

    def builtin_array(self, args):
        assert len(args) == 1, 'array(len) expected 1 arg, not {}'.format(len(args))
        self.visit(args[0])
        # Allocate array on stack, add size to _array_size, push address
        self.asm.instr('popq', '%rax')
        self.asm.instr('shlq', '$3', '%rax')  # len*8 to get size in bytes
        offset = self.local_offset('_array_size')
        self.asm.instr('addq', '%rax', '{}(%rbp)'.format(offset))
        self.asm.instr('subq', '%rax', '%rsp')
        self.asm.instr('movq', '%rsp', '%rax')
        self.asm.instr('pushq', '%rax')

    def visit_Call(self, node):
        assert not node.keywords, 'keyword args not supported'
        builtin = getattr(self, 'builtin_{}'.format(node.func.id), None)
        if builtin is not None:
            builtin(node.args)
        else:
            for arg in node.args:
                self.visit(arg)
            self.asm.instr('call', node.func.id)
            if node.args:
                # Caller cleans up the arguments from the stack
                self.asm.instr('addq', '${}'.format(8 * len(node.args)), '%rsp')
            # Return value is in rax, so push it on the stack now
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
        # Handles if, elif, and else
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
        # Handles while and break (also used for "for" -- see below)
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
        assert isinstance(node.iter, ast.Call) and \
            node.iter.func.id == 'range', \
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
            if (isinstance(step, ast.UnaryOp) and
                    isinstance(step.op, ast.USub) and
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

    def visit_Global(self, node):
        # Global names are already collected by LocalsVisitor
        pass

    def visit_Subscript(self, node):
        self.visit(node.slice.value)
        self.asm.instr('popq', '%rax')
        local_offset = self.local_offset(node.value.id)
        self.asm.instr('movq', '{}(%rbp)'.format(local_offset), '%rdx')
        self.asm.instr('pushq', '(%rdx,%rax,8)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', help='filename to compile')
    parser.add_argument('-n', '--no-peephole', action='store_true',
                        help='enable peephole assembler optimizer')
    args = parser.parse_args()

    with open(args.filename) as f:
        source = f.read()
    node = ast.parse(source, filename=args.filename)
    compiler = Compiler(peephole=not args.no_peephole)
    compiler.compile(node)
