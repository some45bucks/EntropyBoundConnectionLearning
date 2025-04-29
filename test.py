import numpy as np

class Env:
    def __init__(self):
        self.input_size = -1
        self.output_size = -1

    def step(self, action):
        pass

    def reset(self):
        pass

class Node:
    def __init__(self, node_type: str, node_id: int, threshold: float = 0.0, reset: float = 0.0, bias: float = 0.0, weight: float = 0.0):
        self.node_type = node_type
        self.node_id = node_id
        self.input_edges = []
        self.output_edges = []
        self.activation = 0.0

        self.threshold = threshold
        self.reset = reset
        self.bias = bias
        self.weight = weight
    
    @property
    def average_input_probability(self):
        if len(self.input_edges) == 0:
            return .5
        return np.mean([edge.probability for edge in self.input_edges])
    
    @property
    def average_output_probability(self):
        if len(self.output_edges) == 0:
            return .5
        return np.mean([edge.probability for edge in self.output_edges])
    
    def add_input_edge(self, edge):
        self.input_edges.append(edge)

    def add_output_edge(self, edge):
        self.output_edges.append(edge)

    def forward(self):
        x = np.sum([edge.forward_output() for edge in self.input_edges])
        self.activation += self.weight * x + self.bias
        if self.activation >= self.threshold:
            self.activation = self.reset
            x = 1.0
        else:
            x = 0.0
        
        for edge in self.output_edges:
            edge.forward_input(x)
        return x


class Edge:
    def __init__(self, from_node: Node, to_node: Node, probability: float, weight: float):
        self.weight = weight
        self.probability = probability
        self.from_node = from_node
        self.to_node = to_node
        self.active = 1
        self.next = 0

    @property
    def entropy(self):
        return -(self.probability * np.log(self.probability) + (1 - self.probability) * np.log(1 - self.probability))
    
    def set_active(self):
        self.active = int(np.random.rand() < self.probability)

    def forward_output(self):
        x = self.next
        return x*self.weight*self.active
    
    def forward_input(self, x):
        self.next = x

class Network:
    def __init__(self, input_size: int, output_size: int):
        self.input_size = input_size
        self.output_size = output_size
        self.nodes = []
        self.edges = []
        self.edge_exists = {}

        for i in range(input_size):
            node = Node("input", i)
            self.nodes.append(node)
        for i in range(output_size):
            node = Node("output", i + input_size)
            self.nodes.append(node)

    @property
    def total_entropy(self):
        return np.sum([edge.entropy for edge in self.edges])
    
    def new_node_expected_entropy(self):
        node_entropy = 0
        for node in self.nodes:

            node_entropy += -node.average_input_probability * np.log(node.average_input_probability) - (1 - node.average_input_probability) * np.log(1 - node.average_input_probability)
            node_entropy += -node.average_output_probability * np.log(node.average_output_probability) - (1 - node.average_output_probability) * np.log(1 - node.average_output_probability)

    def fill_network_entropy(self, entropy_level: float):
        while self.total_entropy < entropy_level:
            entropy_diff = entropy_level - self.total_entropy
            new_node_expected_entropy = self.new_node_expected_entropy()
            if new_node_expected_entropy <= entropy_diff:
                new_node = Node("hidden", len(self.nodes), np.random.rand(-1, 1), np.random.rand(-1, 1), np.random.rand(-1, 1), np.random.rand(-1, 1))
                new_id = len(self.nodes)
                self.add_node(new_node)
                for i in range(len(self.nodes)):
                    if np.random.rand() < self.nodes[i].average_output_probability:
                        self.add_edge(i, new_id, np.random.rand(-1,1), np.random.rand(-1,1))
                    
                    if np.random.rand() < self.nodes[i].average_input_probability:
                        self.add_edge(new_id, i, np.random.rand(-1,1), np.random.rand(-1,1))
            else:
                all_input_probs = np.argsort([node.average_input_probability for node in self.nodes])
                all_output_probs = np.argsort([node.average_output_probability for node in self.nodes])

                _break = False
                for i in range(len(all_input_probs)):
                    for j in range(len(all_output_probs)):
                        max_output_prob_id = all_input_probs[i]
                        max_input_prob_id = all_input_probs[j]

                        if not (max_output_prob_id, max_input_prob_id) in self.edge_exists:
                            max_input_prob = all_input_probs[max_input_prob_id]
                            max_output_prob = all_output_probs[max_output_prob_id]

                            new_edge_prob = max_output_prob * max_input_prob
                            new_edge_weight = np.random.rand(-1, 1)
                            self.add_edge(max_output_prob_id, max_input_prob_id, new_edge_prob, new_edge_weight)
                            _break = True
                            break
                    if _break:
                        break

    def randomize_edges(self):
        for edge in self.edges:
            edge.set_active()

    def add_node(self, node: Node):
        self.nodes.append(node)

    def add_edge(self, n1: int, n2: int, probability: float, weight: float):
        edge = Edge(self.nodes[n1], self.nodes[n2], probability, weight)
        self.edges.append(edge)
        self.nodes[n1].add_output_edge(edge)
        self.nodes[n2].add_input_edge(edge)
        self.edge_exists[(n1, n2)] = True

    def forward(self, x):
        for i in range(self.input_size):
            for edge in self.nodes[i].output_edges:
                edge.forward_input(x[i])
        
        for node in self.nodes:
            node.forward()

        y = np.zeros(self.output_size)

        for i in range(self.input_size):
            y[i] = self.nodes[i].forward()

        return y
    
    def update(self, reward):
        pass

    def render(self):
        pass
          

def main(entropy_level: float, env: Env, num_env_steps: int, batch_size: int, step_ratio: int, seed: int):
    np.random.seed(seed)

    # create network
    network = Network(env.input_size, env.output_size)
    network.fill_network_entropy(entropy_level)
    
    num_batches = num_env_steps // batch_size

    state = env.reset()
    for batch in range(num_batches):
        # randomize edges
        network.fill_network_entropy(entropy_level)
        network.randomize_edges()
        for step in range(batch_size):
            # get action from network
            for sub_step in range(step_ratio):
                action = network.forward(state)

            state, reward = env.step(action)

        network.update(reward)


if __name__ == "__main__":
    # parse arguments
    import argparse
    parser = argparse.ArgumentParser(description="Run the network.")
    parser.add_argument("--entropy_level", type=float, default=100, help="Entropy level for the network. (correlated with number of unknown edges)")
    parser.add_argument("--num_env_steps", type=int, default=1000, help="Number of environment steps.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--step_ratio", type=int, default=10, help="Step ratio for the network.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()
    entropy_level = args.entropy_level
    num_env_steps = args.num_env_steps
    batch_size = args.batch_size
    step_ratio = args.step_ratio
    seed = args.seed
    env = Env()

    main(entropy_level, env, num_env_steps, batch_size, step_ratio, seed)