#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
import re
import os
import numpy as np
import numpy.distutils.__config__
import theano as th
import theano.tensor as T

__all__ = ["DAG"]


class Node(object):

    """
    superclass for all types of nodes.
    """

    def __init__(self):
        self.children = []

    def add_child(self, node, factor=1):
        self.children.append((factor, node))

    def value(self):
        raise Exception("child node has to implement suitable val(...) method")


class Operator(Node):

    def __init__(self, op):
        super(Operator, self).__init__()
        self.op = op

    def value(self):
        op = self.op
        if op == '+':
            return sum(f * n.value() for f, n in self.children)
        elif op == '*':
            from operator import mul
            return reduce(mul, (f * n.value() for f, n in self.children))
        elif op == 'square':
            assert len(self.children) == 1
            f, n = self.children[0]
            return (f * n.value()) ** 2.
        elif op == '^':
            assert len(self.children) == 2
            (fl, l), (fr, r) = self.children
            return (fl * l.value()) ** (fr * r.value())
        elif op == '/':
            assert len(self.children) == 2
            (fl, l), (fr, r) = self.children
            return (fl * l.value()) / (fr * r.value())
        elif op == 'sin':
            assert len(self.children) == 1
            f, n = self.children[0]
            return f * T.sin(n.value())
        elif op == 'exp':
            assert len(self.children) == 1
            f, n = self.children[0]
            return f * T.exp(n.value())
        else:
            raise Exception("Unkonwn Op: '%s'" % op)

    def __str__(self):
        return "op[%s]" % self.op


class Variable(Node):

    def __init__(self):
        super(Variable, self).__init__()
        self.var = T.scalar()
        self.bound = [-np.Inf, np.Inf]  # default

    def update(self, **attrs):
        self.__dict__.update(attrs)

    def set_name(self, name):
        self.var.name = name

    def value(self):
        return self.var

    def __str__(self):
        return "Variable[%s]" % self.var


class Constant(Node):

    def __init__(self, v):
        super(Constant, self).__init__()
        self.v = v

    def value(self):
        return self.v

    def __str__(self):
        return "Constant[%s]" % self.v


def parse_bound(b):
    """
    helper function to translate "[xxxx,yyyy]" into a tuple for an interval
    """
    def parse_number(n):
        if n == "I":
            return np.Inf
        elif n == "-I":
            return -np.Inf
        else:
            return float(n)
    return map(parse_number, parse_bound.bnd.match(b).groups())
parse_bound.bnd = re.compile(r"\[([^,]+),([^\]]+)\]")


def parse_attributes(tokens):
    """
    each node has a list of possible attributes.
    they are like [ "b [1,1]", "d 0.5", ...]
    """
    attrs = {}
    for token in tokens:
        op = token[0]
        data = token[2:]
        if op == 'd':
            # "d 0.4,22.2,...
            assert "d" not in attrs
            attrs["data"] = map(float, data.split(","))

        elif op == 'b':
            # bounds like "[1,1]"
            attrs["bound"] = parse_bound(data)
        else:
            raise Exception("unknown op: '%s' in parse_attributes" % what)

    return attrs


class DAG(object):

    """
    Python representation of a COCONUT Dag.
    """
    class Objective(object):

        def __init__(self, expression, name=None):
            self.minimize = True
            self.expression = expression
            self.name = name

    class Constraint(object):

        def __init__(self, expression, bounds=None, name=None):
            self.expression = expression
            self.name = name
            self.bounds = bounds if bounds is not None else (-np.inf, np.inf)

    def __init__(self):
        self.variables = []
        self.objective = None  # TODO or several ones?
        self.constraints = []

    def __str__(self):
        return "DAG@%s" % id(self)

    @staticmethod
    def parse(fn, dag_ser):
        from collections import OrderedDict, defaultdict
        #cls = re.compile(r"^<([^>]+)>")
        nodes = {}
        edges = OrderedDict()  # edgeN : operator | constant
        globs = []  # instructions: V 2, M 0 min, c 0 9, ... ?
        variables = defaultdict(Variable)
        objective = None
        constr_nodes = []

        for line in dag_ser.splitlines():
            i = line.find("> ")  # first split at the end of the node id
            token0 = line[1:i]
            tokens = line[i + 2:].split(": ")
            # print "tokens:", tokens
            #token0 = cls.match(token0)
            # if token0 is None or len(token0.groups()) != 1:
            #    raise Exception("token_0: '%s' unknown" % tokens[0])
            #token0 = token0.groups()[0]

            if token0 == "N":
                globs.append(tokens[1:])
                data = tokens[0].split(" ")
                t = data.pop(0)
                if t == 'V':
                    # ignored!
                    pass

                elif t == "M":
                    objective = int(data[0]), data[1]

                elif t == "N":
                    # N <number> 'name'
                    variables[int(data[0])].set_name(data[1][1:-1])

                elif t == "O":
                    # names are ignored
                    print "ignored:", tokens

                elif t == 'c':
                    constr_nodes.append(int(data[1]))

                elif t == 'C':
                    # constraint name, ignored
                    pass

                else:
                    raise Exception("unknown node type '%s'" % t)

            elif token0 == "E":
                src, targ, val = tokens.pop().split(" ")
                k = int(src), int(targ)
                edges[k] = float(val)

            else:  # token0 must be a number
                n = int(token0)
                assert n not in nodes

                op = tokens.pop()
                attrs = parse_attributes(tokens)
                # print "op:    ", op
                # print "tokens:", tokens
                if op[0] == 'V':
                    idx = int(op[2:])  # index in "V <idx>"
                    var = variables[idx]
                    var.update(**attrs)
                    nodes[n] = var

                elif op[0] == 'C':
                    nodes[n] = Constant(float(op[2:]))

                elif op in ['+', '*', '^', '^I', '/', 'sin', 'cos', 'exp', 'sinh', 'cosh', 'log', 'gauss', 'poly']:
                    nodes[n] = Operator(op)

                elif op == '2':
                    nodes[n] = Operator("square")

                else:
                    raise Exception("Unknown operator '%s'" % op)

        print "globs:"
        print globs
        print "objective:", objective
        # print var_nodes

        # for k, v in edges.iteritems():
        #    print k, v

        # start processing
        # x is the vector of variables, ordering is important!
        x = [variables[i] for i in range(len(variables))]

        bounds = [None] * len(variables)
        for idx, var in variables.iteritems():
            bounds[idx] = var.bound
        bounds = np.array(bounds)
        print "bounds:\n", bounds

        for (src, targ), val in edges.iteritems():
            # print src, targ, val
            nodes[src].add_child(nodes[targ], val)

        obj = nodes[objective[0]].value()
        constr = [nodes[idx].value() for idx in constr_nodes]
        exprs = [obj] + constr
        f = th.function(inputs=[_.var for _ in x],
                        outputs=exprs)
        print "Objective:\n", th.printing.pp(obj)
        for i, c in enumerate(constr):
            print "Constraint %d:\n", th.printing.pp(c)
        #args = 2. * np.random.rand(len(variables), 1000) - 1.
        # print np.apply_along_axis(lambda x : f(*x), 0, args)

        print "10 random evaluations:"
        for _ in range(10):
            arg = 2. * np.random.rand(len(variables)) - 1.
            print "f(%s) = %s" % (arg, f(*arg))

        #outfile = os.path.expanduser("%s.png" % os.path.splitext(fn)[0])
        #th.printing.pydotprint(f, outfile=outfile)

        return DAG()  # END parse()

if __name__ == "__main__":
    import sys
    from glob import glob
    for arg in sys.argv[1:]:
        for fn in glob(arg):
            print fn
            print DAG.parse(fn, open(fn, "r").read())
