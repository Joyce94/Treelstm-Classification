

class Forest:
    def __init__(self, trees):
        self.step = 0
        self.trees = trees
        self.node_list = self.trees_to_nodes(trees)
        self.max_level = self.get_max_level()
        self.max_step = self.get_max_step()

    def clean_state(self):
        for node in self.node_list:
            node.state = None
            node.f = None
            node.loss = None
            node.out = None

    def trees_to_nodes(self,trees):
        node_list = []
        for idx,tree in enumerate(trees):
            tree.forest_ix = len(node_list)
            tree.mark = idx
            node_list.append(tree)
            self.add_forest_ix(tree,node_list,idx)
        return node_list

    def add_forest_ix(self,tree,node_list,idx):
        for child in tree.children:
            child.forest_ix = len(node_list)
            child.mark = idx
            node_list.append(child)
            self.add_forest_ix(child,node_list,idx)

    def get_max_level(self):
        return max([n.level for n in self.node_list])

    def get_max_step(self):
        return max([n.step for n in self.node_list])

