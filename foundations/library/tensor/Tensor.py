import numpy as np


class Tensor:
    def __init__(self, data, parents=None):
        self.data = np.array(data, dtype=float)
        self.grad = np.zeros_like(self.data)
        self._backward = lambda: None

        # WARNING: we cannot set the default parents to []
        # since lists are mutable objects.
        self.parents = parents if parents is not None else []

    def __repr__(self):
        body = np.array2string(self.data, precision=3)
        return f"Tensor=({body})"

    def __add__(self, other):
        tensor = Tensor(self.data + other.data, [self, other])

        def _backward():
            self.grad += self._reduce_to(
                tensor.grad, self.grad.shape)
            other.grad += self._reduce_to(
                tensor.grad, other.grad.shape)
        tensor._backward = _backward

        return tensor

    def __radd__(self, list):
        return Tensor(list) + self

    def __neg__(self):
        return self * Tensor([-1])

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, list):
        return Tensor(list) - self

    def __mul__(self, other):
        tensor = Tensor(self.data * other.data, [self, other])

        def _backward():
            self.grad += self._reduce_to(
                other.data * tensor.grad, self.grad.shape)
            other.grad += self._reduce_to(
                self.data * tensor.grad, other.grad.shape)
        tensor._backward = _backward

        return tensor

    def __truediv__(self, other):
        return self * other**(-1)

    def __pow__(self, scalar):
        tensor = Tensor(self.data ** scalar, [self])

        def _backward():
            self.grad += scalar * self.data ** (scalar - 1) * tensor.grad
        tensor._backward = _backward

        return tensor

    def __matmul__(self, other):
        tensor = Tensor(self.data @ other.data, [self, other])

        def _backward():
            self.grad += tensor.grad @ other.data.T
            other.grad += self.data.T @ tensor.grad
        tensor._backward = _backward

        return tensor

    def mean(self):
        tensor = Tensor(self.data.mean(), [self])

        def _backward():
            self.grad += tensor.grad / self.data.size
        tensor._backward = _backward

        return tensor

    def tanh(self):
        t = np.tanh(self.data)
        tensor = Tensor(t, [self])

        def _backward():
            self.grad += (1 - t**2) * tensor.grad
        tensor._backward = _backward

        return tensor

    def relu(self):
        tensor = Tensor(np.maximum(0, self.data), [self])

        def _backward():
            self.grad += (self.data > 0) * tensor.grad
        tensor._backward = _backward

        return tensor

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

        self.grad = np.ones_like(self.data)
        for node in reversed(nodes):
            node._backward()

    @staticmethod
    def _reduce_to(grad, target_shape):
        extra = grad.ndim - len(target_shape)
        grad = np.sum(grad, axis=tuple(range(extra)))
        for axes, size in enumerate(target_shape):
            if size == 1:
                grad = np.sum(grad, axis=axes, keepdims=True)
        return grad
