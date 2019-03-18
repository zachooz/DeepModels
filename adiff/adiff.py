def reverse_ad_add(x, y):
    return Node(
        x.val + y.val,
        [x, y],
        lambda x, y: (1, 1)
    )


def reverse_ad_mul(x, y):
    return Node(
        x.val * y.val,
        [x, y],
        lambda x, y: (y, x)
    )


def reverse_ad_sub(x, y):
    return Node(
        x.val - y.val,
        [x, y],
        lambda x, y: (1, -1)
    )


def reverse_ad_div(x, y):
    return Node(
        x.val / y.val,
        [x, y],
        lambda x, y: (1/y, -x/(y*y))
    )


# Reverse mode AD graph node
class Node:

    # derivative_func: takes self as argument, returns the values
    #    of the input adjoints
    def __init__(self, val, input_nodes=[], derivative_func=None):
        self.val = val
        self.deriv = 0
        self.input_nodes = input_nodes
        self.derivative_func = derivative_func
        # Keeps track of how many times nodes are used as inputs
        self.fanout = 0

    # Initiate a backward pass starting from this node
    def backward(self):
        self.deriv = 1
        self.compute_fanout()
        self.backprop()

    # Compute the out degree of each node reachable from self
    def compute_fanout(self):
        self.fanout += 1
        if self.fanout == 1:
            for node in self.input_nodes:
                node.compute_fanout()

    # Propagate backward along the graph
    def backprop(self):
        if self.derivative_func is not None:
            self.fanout -= 1
            # Wait until this node has accumulated adjoint updates from all of its
            #    uses, then propagate back to the inputs
            if self.fanout == 0:
                input_vals = map(lambda node: node.val, self.input_nodes)
                input_adjoints = self.derivative_func(*input_vals)
                for i in range(len(self.input_nodes)):
                    self.input_nodes[i].deriv += input_adjoints[i] * self.deriv
                    self.input_nodes[i].backprop()

    @staticmethod
    def lift(x):
        if isinstance(x, Node):
            return x
        else:
            return Node(x)

    # Math operators. Again, second arg can be number or Node
    def __add__(self, other):
        other = Node.lift(other)
        return reverse_ad_add(self, other)

    def __sub__(self, other):
        other = Node.lift(other)
        return reverse_ad_sub(self, other)

    def __mul__(self, other):
        other = Node.lift(other)
        return reverse_ad_mul(self, other)

    def __truediv__(self, other):
        other = Node.lift(other)
        return reverse_ad_div(self, other)


def test(x_val, y_val):
    x = Node(x_val)
    y = Node(y_val)
    out1 = x * y + x / y
    out1.backward()
    assert x.deriv == (y_val + 1 / y_val)
    assert y.deriv == (x_val - x_val / y_val ** 2)

    x = Node(x_val)
    y = Node(y_val)
    out2 = x * x * y - x * y * y
    out2.backward()
    assert x.deriv == (2 * x_val * y_val - y_val ** 2)
    assert y.deriv == (x_val ** 2 - 2 * x_val * y_val)


xy_vals = [(1, 1), (0.5, 1.1), (13.4, 0.2)]

for x_val, y_val in xy_vals:
    test(x_val, y_val)
