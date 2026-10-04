"""
qai.core.federated -- Federated Learning privacy-preserving aggregation protocols.
"""
import numpy as np
from typing import List, Any, Dict

class FederatedServer:
    """Central Federated Learning aggregator server."""
    def __init__(self):
        self.global_weights = None

    def aggregate(self, client_weights: List[np.ndarray]) -> np.ndarray:
        """Federated Averaging (FedAvg) protocol across client model weights."""
        if not client_weights:
            raise ValueError("client_weights cannot be empty")
        self.global_weights = np.mean(client_weights, axis=0)
        return self.global_weights

def federated_averaging(client_weights: List[np.ndarray]) -> np.ndarray:
    """Functional FedAvg aggregation protocol."""
    server = FederatedServer()
    return server.aggregate(client_weights)
