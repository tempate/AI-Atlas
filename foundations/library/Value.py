import math


class Value:
    def __init__(self, data, parents=[]):
        self.data = data
        self.grad = 0
        self.parents = parents
        self._backward = lambda: None

    def __repr__(self):
        return f"Value={self.data:.3f}"

    def __neg__(self):
        return self * Value(-1)

    def __add__(self, other):
        value = Value(self.data + other.data, [self, other])

        def _backward():
            self.grad += value.grad
            other.grad += value.grad
        value._backward = _backward

        return value

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        value = Value(self.data * other.data, [self, other])

        def _backward():
            self.grad += other.data * value.grad
            other.grad += self.data * value.grad
        value._backward = _backward

        return value

    def __truediv__(self, other):
        return self * other**(-1)

    def __pow__(self, scalar):
        value = Value(self.data ** scalar, [self])

        def _backward():
            self.grad += scalar * self.data ** (scalar - 1) * value.grad
        value._backward = _backward

        return value

    def tanh(self):
        t = math.tanh(self.data)
        value = Value(t, [self])

        def _backward():
            self.grad += (1 - t**2) * value.grad
        value._backward = _backward

        return value

    def relu(self):
        value = Value(0 if self.data < 0 else self.data, [self])

        def _backward():
            self.grad += (self.data < 0) * value.grad
        value._backward = _backward

        return value

    def backward(self):
        nodes = []
        visited = set()

        def build(node):
            if node not in visited:
                visited.add(node)
                for parent in node.parents:
                    build(parent)
                nodes.append(node)
        build(self)

        self.grad = 1.0
        for node in reversed(nodes):
            node._backward()
