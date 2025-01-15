import random
from pysc2.agents import base_agent
from pysc2.lib import actions, features

class Node:
    def run(self, obs):
        raise NotImplementedError

class Selector(Node):
    def __init__(self, children):
        self.children = children

    def run(self, obs):
        for child in self.children:
            if child.run(obs):
                return True
        return False
    
class Sequence(Node):
    def __init_(self, children):
        self.children = children
    
    def run(self, obs):
        for child in self.children:
            if not child.run(obs):
                return False
        return True

class Leaf(Node):
    def __init__(self, action):
        self.action = action

    def run(self, obs):
        return self.action(obs)
    
