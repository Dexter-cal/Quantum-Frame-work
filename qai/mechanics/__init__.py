"""
qai.mechanics -- Internal learning mechanics, loss functions, optimizers, and schedulers.
"""

from __future__ import annotations
import numpy as np
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
    """Quantum Circuit Statevector Simulator.

    Implements real single-qubit gate application via Kronecker-product
    embedding into the full 2^n Hilbert space. Supports Hadamard (H),
    Pauli-X / Y / Z, and phase (S, T) gates. Measurement collapses the
    statevector probabilistically.

    Example:
        qc = QuantumCircuitSimulator(n_qubits=2)
        qc.apply_hadamard(0)        # puts qubit 0 into |+>
        qc.apply_hadamard(1)        # puts qubit 1 into |+>
        # State is now (1/2) * [|00> + |01> + |10> + |11>]
        result = qc.measure()       # collapses to one of {0, 1, 2, 3}
    """
    # Standard gate definitions (2x2 complex matrices)
    _I = np.array([[1, 0], [0, 1]], dtype=complex)
    _H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    _X = np.array([[0, 1], [1, 0]], dtype=complex)
    _Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    _Z = np.array([[1, 0], [0, -1]], dtype=complex)
    _S = np.array([[1, 0], [0, 1j]], dtype=complex)
    _T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)

    def __init__(self, n_qubits: int = 2):
        if n_qubits < 1:
            raise ValueError("n_qubits must be >= 1")
        if n_qubits > 12:
            raise ValueError(
                f"[WHAT] n_qubits={n_qubits} requested\n"
                f"[WHY] Statevector simulation is O(2^n) in memory; 12 qubits = 64MB complex128\n"
                f"[FIX] Use a tensor-network simulator for n > 12, or accept the memory cost"
            )
        self.n_qubits = n_qubits
        self.statevector = np.zeros(2 ** n_qubits, dtype=complex)
        self.statevector[0] = 1.0  # |00...0>

    def _apply_single_qubit_gate(self, gate: np.ndarray, qubit: int):
        """Apply a 2x2 gate to qubit `qubit` by Kronecker-embedding it into
        the full 2^n x 2^n unitary, then multiplying the statevector.

        Qubit index 0 is the LEAST-significant qubit (rightmost in standard
        binary notation), matching Qiskit's convention."""
        if not 0 <= qubit < self.n_qubits:
            raise IndexError(
                f"[WHAT] qubit index {qubit} out of range for {self.n_qubits}-qubit circuit\n"
                f"[WHY] Valid range: 0..{self.n_qubits - 1}\n"
                f"[FIX] Use a different qubit index"
            )
        # Build the full unitary by Kronecker product: I ⊗ I ⊗ ... ⊗ gate ⊗ ... ⊗ I
        # Qubit 0 is the rightmost factor.
        factors = [self._I] * self.n_qubits
        factors[qubit] = gate
        full = factors[0]
        for f in factors[1:]:
            full = np.kron(full, f)
        self.statevector = full @ self.statevector

    def apply_hadamard(self, qubit: int):
        """Apply Hadamard gate H to qubit `qubit`.

        H|0> = (|0> + |1>) / sqrt(2) = |+>
        H|1> = (|0> - |1>) / sqrt(2) = |->
        """
        self._apply_single_qubit_gate(self._H, qubit)
        return self

    def apply_x(self, qubit: int):
        """Apply Pauli-X (bit-flip) gate to qubit `qubit`. X|0> = |1>, X|1> = |0>."""
        self._apply_single_qubit_gate(self._X, qubit)
        return self

    def apply_y(self, qubit: int):
        """Apply Pauli-Y gate to qubit `qubit`."""
        self._apply_single_qubit_gate(self._Y, qubit)
        return self

    def apply_z(self, qubit: int):
        """Apply Pauli-Z (phase-flip) gate to qubit `qubit`. Z|0> = |0>, Z|1> = -|1>."""
        self._apply_single_qubit_gate(self._Z, qubit)
        return self

    def apply_s(self, qubit: int):
        """Apply S (phase) gate to qubit `qubit`. S|1> = i|1>."""
        self._apply_single_qubit_gate(self._S, qubit)
        return self

    def apply_t(self, qubit: int):
        """Apply T (pi/8) gate to qubit `qubit`."""
        self._apply_single_qubit_gate(self._T, qubit)
        return self

    def reset(self):
        """Reset statevector to |00...0>."""
        self.statevector = np.zeros(2 ** self.n_qubits, dtype=complex)
        self.statevector[0] = 1.0
        return self

    def probabilities(self) -> np.ndarray:
        """Return the probability distribution over computational basis states."""
        return np.abs(self.statevector) ** 2

    def measure(self, n_shots: int = 1):
        """Measure the statevector.

        - `measure()` → single shot, returns one int in [0, 2^n)
        - `measure(n_shots=k)` → returns a list of k outcomes
        """
        probs = self.probabilities()
        # Renormalize to guard against tiny floating-point drift
        probs = probs / probs.sum()
        if n_shots == 1:
            return int(np.random.choice(len(probs), p=probs))
        return [int(x) for x in np.random.choice(len(probs), size=n_shots, p=probs)]

    def state_amplitudes(self) -> np.ndarray:
        """Return a copy of the current statevector (complex amplitudes)."""
        return self.statevector.copy()


def quantum_expectation(circuit: QuantumCircuitSimulator, observable: str = "Z0") -> float:
    """Compute the expectation value <ψ|O|ψ> for a Pauli observable.

    Currently supports single-qubit Pauli-Z on qubit 0 (`observable="Z0"`),
    which is the standard "what's the bias toward |0> vs |1>" measurement.
    Returns a float in [-1, 1].
    """
    if observable == "Z0":
        # <Z0> = P(0) - P(1) where P(0) = sum of probs where bit-0 is 0
        probs = circuit.probabilities()
        # Bit 0 is the LEAST significant bit
        p0 = sum(p for i, p in enumerate(probs) if (i & 1) == 0)
        p1 = sum(p for i, p in enumerate(probs) if (i & 1) == 1)
        return float(p0 - p1)
    raise ValueError(f"Unknown observable '{observable}'. Currently only 'Z0' is supported.")


class ReduceLROnPlateau:
    """Reduces learning rate when validation metric has stopped improving."""
    def __init__(self, initial_lr: float = 0.01, factor: float = 0.1, patience: int = 3, min_lr: float = 1e-6):
        self.current_lr = initial_lr
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.best_metric = float("inf")
        self.counter = 0

    def step(self, metric: float) -> float:
        if metric < self.best_metric:
            self.best_metric = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.current_lr = max(self.min_lr, self.current_lr * self.factor)
                self.counter = 0
        return self.current_lr


# Section 56 re-exports
from .optimizer import Optimizer, ClosedForm, GradientDescent
from .objective import Objective, MSE, MAE
from .training_loop import TrainingLoop, FixedEpochs, RetrainUntil
