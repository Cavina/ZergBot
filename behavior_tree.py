class Node:
    "Base Class"
    def run(self):
        raise NotImplementedError("Must implement the run method by a subclass")
    
class Selector(Node):
    def __init__(self, children):
        self.children = children
        self.current_child = 0
        self.status = "FAILURE"

    def run(self, obs):
        while self.current_child < len(self.children):
            action = self.children[self.current_child].run(obs)
            if action:
                self.current_child = 0
                self.status = "SUCCESS"
                return action
            self.current_child += 1
        self.current_child = 0
        return "FAILURE"
    

class Sequence(Node):
    def __init__(self, children):
        self.children = children
        self.current_child = 0

    def run(self, obs):
        while self.current_child < len(self.children):
            result = self.children[self.current_child].run(obs)
            if result == "FAILURE":
                self.current_child = 0
            elif result == "RUNNING":
                return "RUNNING"
            self.current_child += 1
        self.current_child = 0
        return "SUCCESS"
    
class ActionNode(Node):
    def __init__(self, name, action):
        self.name = name
        self.action = action

    def run(self, obs):
        action = self.action(obs)
        if action:
            return action
        return "FAILURE"
