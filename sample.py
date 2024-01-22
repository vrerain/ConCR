import os
import json
import re
import random
from tqdm import tqdm

random.seed(123456)

class ContextTree():
    def __init__(self, value, idx=-1):
        self.value = value
        self.idx=idx
        self.children = []
        self.parent = None
        self.visit = False

    def add(self, child):
        self.children.append(child)
    
    def setParent(self, parent):
        self.parent = parent

def createTree(funcs, levels):
    curLevel = -1
    root = ContextTree("Functions")
    curNode = root
    nodes = []
    for i in range(len(funcs)):
        f = funcs[i]
        level = levels[i]
        if (level == -1):
            break
        if level > curLevel and level - curLevel == 1:
            child = ContextTree(f, len(nodes))
            child.setParent(curNode)
            curNode.add(child)
            curNode = child
            curLevel = level
            nodes.append(child)
            continue
        if level == curLevel:
            child = ContextTree(f, len(nodes))
            child.setParent(curNode.parent)
            curNode.parent.add(child)
            curNode = child
            nodes.append(child)
            continue
        if level < curLevel:
            nums = curLevel - level
            child = ContextTree(f, len(nodes))
            p = curNode.parent
            for i in range(nums):
                p = p.parent
            child.setParent(p)
            p.add(child)
            curLevel = level
            curNode = child
            nodes.append(child)
            continue
    return root, nodes


def findNeighbor(root, visited):
    childrens=[]
    if(len(root.children) > 0):
        for c in root.children:
            if (visited[c.idx] == 0):
                childrens.append(c)
        return childrens
    else:
        tempNode = root.parent
        if (len(tempNode.children) > 0):
            for c in tempNode.children:
                if (visited[c.idx] == 0):
                    childrens.append(c)
            return childrens
    return childrens

def context_walking(root, needLen, visited, nodes):
    tempNode = root
    ids=0
    while(ids < needLen):
        childrens = findNeighbor(tempNode, visited)
        if (len(childrens) > 0):
            choiceNode = childrens[random.choice(list(range(len(childrens))))]
            visited[choiceNode.idx] = 1
            ids += 1
            tempNode=choiceNode
        else:
            for i in range(len(visited)):
                if (visited[i] == 0 and (visited[nodes[i].parent.idx] == 1 or nodes[i].parent.idx == -1)):
                    visited[i] = 1
                    ids += 1
                    tempNode = nodes[i]
                    break
    return visited


def sample(funcs, levels, sample_count):
    root, nodes = createTree(funcs, levels)
    func_len = len(nodes)
    routes = []
    for i in range(sample_count):
        walking_length = random.randint(0, func_len)
        route = context_walking(root, walking_length, [0] * func_len, nodes)
        routes.append(route + [-1] * (len(funcs) - len(route)))
    return routes