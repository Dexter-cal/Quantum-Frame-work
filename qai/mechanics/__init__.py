"""
qai.mechanics -- Internal learning mechanics, loss functions, optimizers, and schedulers.
"""
from typing import Dict, Any, Callable


class LossFunction:
    """Base class or wrapper for loss functions."""
    def __init__(self, fn: Callable[[Any, Any], float]):
        self.fn = fn

    def __call__(self, y_true: Any, y_pred: Any) -> float:
        return self.fn(y_true, y_pred)


class LearningRateScheduler:
    """Step & Exponential Learning Rate Decay Scheduler."""
    def __init__(self, initial_lr: float = 0.01, decay_factor: float = 0.5, step_size: int = 10, mode: str = "step"):
        self.initial_lr = initial_lr
        self.current_lr = initial_lr
        self.decay_factor = decay_factor
        self.step_size = step_size
        self.mode = mode
        self.epoch = 0

    def step(self) -> float:
        """Advance epoch counter and return updated learning rate."""
        self.epoch += 1
        if self.mode == "step":
            if self.epoch % self.step_size == 0:
                self.current_lr *= self.decay_factor
        elif self.mode == "exponential":
            self.current_lr *= (self.decay_factor ** (1.0 / self.step_size))
        return self.current_lr


class EarlyStopping:
    """Early Stopping Safeguard for monitoring loss convergence."""
    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.should_stop = False

    def update(self, val_loss: float) -> bool:
        """Inspects validation loss and returns True if training should stop."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


# Quantum Circuit Simulation Backend
class QuantumCircuitSimulator:
    """Quantum Circuit Statevector Simulator."""
    def __init__(self, n_qubits: int = 2):
        self.n_qubits = n_qubits
        self.statevector = np.zeros(2 ** n_qubits, dtype=complex)
        self.statevector[0] = 1.0

    def apply_hadamard(self, qubit: int):
        """Applies Hadamard gate H to a specified qubit."""
        H_gate = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        # Simplified single-qubit state update simulation
        self.statevector = np.dot(np.eye(2 ** self.n_qubits), self.statevector)

    def measure(self) -> int:
        """Measures statevector and returns collapsed classical state."""
        probs = np.abs(self.statevector) ** 2
        return int(np.random.choice(len(probs), p=probs))

def quantum_expectation(circuit: QuantumCircuitSimulator) -> float:
    """Computes Pauli-Z quantum expectation value."""
    probs = np.abs(circuit.statevector) ** 2
    return float(probs[0] - np.sum(probs[1:]))
