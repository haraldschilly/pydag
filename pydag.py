#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
import re
import os
from collections import OrderedDict, defaultdict

import logging
logger = logging.getLogger("pyDAG")
logger.addHandler(logging.StreamHandler())

import numpy as np
import numpy.distutils.__config__
import theano as th
import theano.tensor as T

__all__ = ["DAG"]


class Node(object):

    """
    superclass for all types of nodes.

    the children are ordered (important for non-commutative operations)
    and the `get_*` methods return the values of the weighted children.

    the `value` method needs to be overwritten and is crucial for
    evaluating the entire graph.
    """

    def __init__(self):
        self.children = []

    def add_child(self, node, factor=1.):
        self.children.append((factor, node))

    def get_input(self, i):
        f, v = self.children[i]
        return f * v.value() if f != 1. else v.value()

    def get_all_input(self):
        p = lambda (f, v): f * v.value() if f != 1. else v.value()
        return map(p, self.children)

    def update(self, **attrs):
        self.__dict__.update(attrs)

    def value(self):
        raise Exception("child node has to implement suitable val(...) method")


class Operator(Node):

    def __init__(self, op):
        super(Operator, self).__init__()
        self.op = op
        # maybe updated via an attribute with a float-array
        self.data = None

    def add(self):
        nbch = len(self.children)
        assert nbch >= 1
        if nbch == 2:
            s = self.get_input(0) + self.get_input(1)
        else:
            # s = sum(self.get_all_input()) # <- that's probably worse
            s = T.stack(*self.get_all_input()).sum()
        if self.data is not None:
            s += self.data[0]
        return s

    def mult(self):
        nbch = len(self.children)
        assert nbch >= 1
        if nbch == 2:
            m = self.get_input(0) * self.get_input(1)
        else:
            m = T.stack(*self.get_all_input()).prod()
        if self.data is not None:
            m *= self.data[0]
        return m

    def square(self):
        assert len(self.children) == 1
        v = self.get_input(0)
        if self.data is not None:
            v += self.data[0]
        return v ** 2.

    def power(self):
        assert len(self.children) == 2
        basis = self.get_input(0)
        exponent = self.get_input(1)
        return T.pow(basis, exponent)

    def division(self):
        """
        Note: This maps to the `true_div` operation.
        """
        assert len(self.children) == 2
        n = self.get_input(0)
        d = self.get_input(1)
        return T.true_div(n, d)

    def sqrt(self):
        assert len(self.children) == 1
        v = self.get_input(0)
        if self.data is not None:
            v += self.data[0]
        return T.sqrt(v)

    def min_op(self):
        logger.warning("min op is untested")
        vals = self.get_all_input()
        if self.data is not None:
            vals.append(self.data[0])
        return vals.min()

    def max_op(self):
        logger.warning("max op is untested")
        vals = self.get_all_input()
        if self.data is not None:
            vals.append(self.data[0])
        return vals.max()

    def sin(self):
        assert len(self.children) == 1
        return T.sin(self.get_input(0))

    def cos(self):
        assert len(self.children) == 1
        return T.cos(self.get_input(0))

    def exp(self):
        assert len(self.children) == 1
        return T.exp(self.get_input(0))

    # this is a bit awkward, but useful in the parser function to know
    # which operations are actually defined.
    op_functions = {
        '+': add,
        '*': mult,
        '2': square,
        'sqrt': sqrt,
        '^': power,
        '/': division,
        'min': min_op,
        'max': max_op,
        'sin': sin,
        'cos': cos,
        'exp': exp
    }

    def value(self):
        return self.op_functions[self.op](self)

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
        # necessary, e.g., if the obj fctn is constant
        self.v = T.as_tensor_variable(v)

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

    def __init__(self, node_id):
        self.node_id = node_id  # will be replaced in set_expression later!

    def set_expression(self, ex):
        self.expression = ex

    @property
    def bound(self):
        return self.expression.bound

    def value(self):
        return self.expression.value()

    def set_name(self, name):
        self.name = name

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

        self.objective = Objective()
        self.variables = defaultdict(Variable)
        self.constraints = {}

    def __str__(self):
        return "DAG@%s" % id(self)

    @staticmethod
    def parse(fn, dag_ser):
        #cls = re.compile(r"^<([^>]+)>")
        nodes = {}
        edges = OrderedDict()  # edgeN : operator | constant
        constr_mapping = {}  # [(constr_idx <-> node_id), ...]

        # instructions in "N" nodes: V 2, M 0 min, c 0 9, ... ?
        globs = []  # just for debugging

        dag = DAG()

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
                    dag.objective.set_node(nodes[int(data[0])])
                    assert data[1] in ["min", "max"]
                    dag.objective.minimize = data[1] == "min"

                elif t == "N":
                    # N <number> 'name'
                    dag.variables[int(data[0])].set_name(data[1][1:-1])

                elif t == "O":
                    # objective: O 'oname' : d obj_add [, obj_mult]
                    # print "N objective:", tokens
                    # print "N objective:", attrs
                    logger.info("O oname '%s' ignored" % data[0])
                    if "data" in attrs:
                        d = attrs["data"]
                        if len(d) >= 1:
                            dag.objective.add = float(d[0])
                        if len(d) == 2:
                            dag.objective.mult = float(d[1])

                elif t == 'c':
                    idx, node_id = map(int, data)
                    assert idx not in constr_mapping
                    # problem: node is not defined so far!
                    dag.constraints[idx] = Constraint(node_id)

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

                ops = tokens.pop().split()
                # print "op:    ", op
                # print "tokens:", tokens
                if ops[0] == 'V':
                    idx = int(ops[1])  # index in "V <idx>"
                    nodes[n] = dag.variables[idx]

                elif ops[0] == 'C':
                    nodes[n] = Constant(float(ops[1]))

                elif ops[0] in Operator.op_functions.keys():
                    nodes[n] = Operator(ops[0])

                else:
                    raise Exception("Unknown operator '%s'" % ops[0])

                # finally, update attributes like "data" in the node
                attrs = parse_attributes(tokens)
                nodes[n].update(**attrs)

        print "globs:"
        print globs

        # start processing
        # x is the vector of variables, ordering is important!
        # the following translates the dictionary into an ordered vector.
        dag.variables = [dag.variables[i] for i in range(len(dag.variables))]

        bounds = [None] * len(dag.variables)
        for idx, var in enumerate(dag.variables):
            bounds[idx] = var.bound
        bounds = np.array(bounds)
        print "bounds:\n", bounds

        for (src, targ), val in edges.iteritems():
            # print src, targ, val
            nodes[src].add_child(nodes[targ], val)

        obj = dag.objective.value()

        # print "DEBUG Constraints"
        # for idx in constr_nodes:
        #    print nodes[idx]
        #    print [ n.children for (f, n) in nodes[idx].children]

        # that should be true, right?
        assert sorted(dag.constraints.keys()) == range(len(dag.constraints))

        # set the constraints of the dag
        for c in dag.constraints.values():
            c.set_expression(nodes[c.node_id])

        dag.constraints = [dag.constraints[i]
                           for i in range(len(dag.constraints))]
        exprs = [obj] + [c.value() for c in dag.constraints]
        print
        print "exprs = ", exprs

        f = th.function(inputs=[_.var for _ in dag.variables],
                        outputs=exprs)

        print
        print "Objective:", dag.objective
        print th.printing.pp(obj)
        for i, c in enumerate(dag.constraints):
            print
            print "Constraint %d:" % i
            print "  bound:", c.bound
            print "   expr:", th.printing.pp(c.value())

        # print
        # print dag.constraints

        print
        print "10 random evaluations:"
        for _ in range(10):
            arg = 2. * np.random.rand(len(dag.variables)) - 1.
            print "f(%s) = %s" % (arg, f(*arg))

        #outfile = os.path.expanduser("%s.png" % os.path.splitext(os.path.basename(fn))[0])
        #th.printing.pydotprint(f, outfile=outfile)

        print
        print "Debug Printing of f"
        print th.printing.debugprint(f)

        return dag  # END parse()

if __name__ == "__main__":
    import sys
    from glob import glob
    for arg in sys.argv[1:]:
        for fn in glob(arg):
            print fn
            print DAG.parse(fn, open(fn, "r").read())
