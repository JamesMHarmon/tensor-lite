from typing import List

def grad(node):
    nodes = topo_sort(node)
    node.grad = 1.0

    for node in nodes:
        backpropGrads = node._backward(node.grad)
        for parent, backpropGrad in zip(node.parents(), backpropGrads):
            parent.grad += backpropGrad

def topo_sort(node) -> List:
    visited = set()
    nodes = []

    def addParentNodes(node):
        for parentNode in node.parents():
            addParentNodes(parentNode)

        if node not in visited:
            nodes.append(node)
            visited.add(node)

    addParentNodes(node)

    nodes.reverse()

    return nodes
