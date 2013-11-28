#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
import re
import os
from collections import OrderedDict, defaultdict
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

    def get_input(self, i):
        f, v = self.children[i]
        return f * v.value() if f != 1 else v.value()

    def update(self, **attrs):
        self.__dict__.update(attrs)

    def get_all_input(self):
        p = lambda (f, v): f * v.value() if f != 1 else v.value()
        return map(p, self.children)

    def value(self):
        raise Exception("child node has to implement suitable val(...) method")


class Operator(Node):

    def __init__(self, op):
        super(Operator, self).__init__()
        self.op = op

    def value(self):
        op = self.op
        if op == '+':
            return sum(self.get_all_input())
        elif op == '*':
            from operator import mul
            return reduce(mul, self.get_all_input())
        elif op == 'square':
            assert len(self.children) == 1
            return self.get_input(0) ** 2.
        elif op == '^':
            assert len(self.children) == 2
            b = self.get_input(0)
            e = self.get_input(1)
            return b ** e
        elif op == '/':
            assert len(self.children) == 2
            n = self.get_input(0)
            d = self.get_input(0)
            return n / d
        elif op == 'sin':
            assert len(self.children) == 1
            return T.sin(self.get_input(0))
        elif op == 'exp':
            assert len(self.children) == 1
            return T.exp(self.get_input(0))
        else:
            raise Exception("Unkonwn Op: '%s'" % op)

    def __str__(self):
        return "op[%s]" % self.op


class Variable(Node):

    def __init__(self):
        super(Variable, self).__init__()
        self.var = T.scalar()
        self.bound = [-np.Inf, np.Inf]  # default

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

# END Classes in a Node

# START DAG specific classes


class Objective(object):

    def __init__(self, name=None):
        self.minimize = True
        self.name = name
        self.expression = None
        self.add = None
        self.mult = None
        self.node = None

    def set_node(self, node):
        """
        the node in the dag
        """
        self.node = node

    def value(self):
        # obj = (mult * f(x)) + add
        v = self.node.value()
        if self.mult is not None:
            v *= self.mult
        if self.add is not None:
            v += self.add
        return v

    def __str__(self):
        return "Objective{%s, add=%s, mult=%s, node=%s}" \
            % ("min" if self.minimize else "max", self.add, self.mult, self.node)


class Constraint(object):

    def __init__(self, expression, name=None):
        self.expression = expression
        self.name = name
        self.bound = [-np.Inf, np.Inf]  # default

# END DAG specific classes


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
    helper function
    each node has a list of possible attributes.
    they are like [ "b [1,1]", "d 0.5", ...]
     - b: bounds and they are parsed
     - d: data and a list of parsed floats
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
    A Python representation of a (simplified) COCONUT Dag based on "theano".
    """

    def __init__(self):

        self.variables = defaultdict(Variable)
        self.objective = Objective()
        self.constraints = []

    def __str__(self):
        return "DAG@%s" % id(self)

    @staticmethod
    def parse(fn, dag_ser):
        #cls = re.compile(r"^<([^>]+)>")
        nodes = {}
        edges = OrderedDict()  # edgeN : operator | constant

        # instructions in "N" nodes: V 2, M 0 min, c 0 9, ... ?
        globs = []  # just for debugging

        dag = DAG()
        variables =
        objective = dag.objective
        constr_nodes = []

        for line in dag_ser.splitlines():
            i = line.find("> ")  # first split at the end of the node id
            token0 = line[1:i]
            tokens = line[i + 2:].split(": ")

            if token0 == "N":
                globs.append(tokens[1:])
                attrs = parse_attributes(tokens[1:])
                print "ATTRS:", attrs
                data = tokens[0].split()
                t = data.pop(0)
                if t == 'V':
                    # ignored!
                    pass

                elif t == "M":
                    objective.set_node(nodes[int(data[0])])
                    assert data[1] in ["min", "max"]
                    objective.minimize = data[1] == "min"

                elif t == "N":
                    # N <number> 'name'
                    variables[int(data[0])].set_name(data[1][1:-1])

                elif t == "O":
                    # objective: O 'oname' : d obj_add [, obj_mult]
                    # print "N objective:", tokens
                    # print "N objective:", attrs
                    if "data" in attrs:
                        d = attrs["data"]
                        if len(d) >= 1:
                            objective.add = float(d[0])
                        if len(d) == 2:
                            objective.mult = float(d[1])

                elif t == 'c':
                    constr_nodes.append(int(data[1]))

                elif t == 'C':
                    # constraint name, ignored
                    pass

                else:
                    raise Exception("unknown node type '%s'" % t)

            elif token0 == "E":
                src, targ, val = tokens.pop().split()
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

                elif op in ['+', '*', '^', '^I', '/', 'sin', 'cos', 'exp']:
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

        obj = objective.value()
        constr = [nodes[idx].value() for idx in constr_nodes]
        exprs = [obj] + constr
        print exprs
        f = th.function(inputs=[_.var for _ in x],  outputs=exprs)
        print "Objective:\n", th.printing.pp(obj)
        for i, c in enumerate(constr):
            print "Constraint %d:\n" % i, th.printing.pp(c)
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
