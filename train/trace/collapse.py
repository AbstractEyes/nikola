from collections import defaultdict


class CollapseTrace:
    """
    Tracks per-node symbolic collapse data for resonant optimization.

    Includes:
    - Collapse occurrence
    - Alignment drift (phase deviation)
    - Entropy state (optional)
    """

    def __init__(self):
        self.collapse_map = defaultdict(bool)          # node_id → did collapse
        self.alignment_map = defaultdict(float)        # node_id → alignment delta
        self.entropy_map = defaultdict(float)          # node_id → uncertainty
        self._nodes = set()

    def register_node(self, node_id):
        self._nodes.add(node_id)

    def log_collapse(self, node_id: int, collapsed: bool):
        self.collapse_map[node_id] = collapsed
        self._nodes.add(node_id)

    def log_alignment_delta(self, node_id: int, delta: float):
        self.alignment_map[node_id] = delta
        self._nodes.add(node_id)

    def log_entropy(self, node_id: int, entropy: float):
        self.entropy_map[node_id] = entropy
        self._nodes.add(node_id)

    def get_collapsed_nodes(self):
        return [nid for nid in self._nodes if self.collapse_map[nid]]

    def get_node_alignment(self, node_id):
        return self.alignment_map.get(node_id, 0.0)

    def get_node_entropy(self, node_id):
        return self.entropy_map.get(node_id, 0.0)

    def export_collapse_dict(self) -> dict:
        """Returns: node_id → did_collapse (bool)"""
        return dict(self.collapse_map)

    def export_alignment_dict(self) -> dict:
        """Returns: node_id → alignment delta (float)"""
        return dict(self.alignment_map)

    def items(self):
        """
        Generator for all tracked node states.
        Yields: node_id, collapse, alignment_delta, entropy
        """
        for node_id in self._nodes:
            yield (
                node_id,
                self.collapse_map[node_id],
                self.alignment_map.get(node_id, 0.0),
                self.entropy_map.get(node_id, 0.0)
            )
