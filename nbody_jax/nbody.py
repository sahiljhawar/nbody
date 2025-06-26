from functools import partial
from typing import Iterable
import jax.numpy as jnp
from jax import jit
import numpy as np


class NBodySimulation:
    """
    JAX-based N-body simulation that maintains the same API as the C++ version.
    """

    def __init__(
        self,
        masses: Iterable[float],
        positions: Iterable[Iterable[float]],
        velocities: Iterable[Iterable[float]],
        G: float = 1.0,
    ) -> None:
        """
        Initialize the N-body simulation.

        Parameters
        ----------
        masses : array_like
            Masses of the bodies.
        positions : array_like
            Initial positions of the bodies (shape: (N, 3)).
        velocities : array_like
            Initial velocities of the bodies (shape: (N, 3)).
        G : float, optional
            Gravitational constant. Default is 1.0.
        """
        self.G = G
        self.masses = jnp.array(masses)
        self.positions = jnp.array(positions)
        self.velocities = jnp.array(velocities)
        self.n_bodies = len(masses)

        # Validate input dimensions
        if self.positions.shape != (self.n_bodies, 3):
            raise ValueError(
                f"Positions shape {self.positions.shape} doesn't match expected ({self.n_bodies}, 3)"
            )
        if self.velocities.shape != (self.n_bodies, 3):
            raise ValueError(
                f"Velocities shape {self.velocities.shape} doesn't match expected ({self.n_bodies}, 3)"
            )

        self.history = [self.positions.copy()]
        self.time_steps = []

        self._compute_forces_jit = jit(self._compute_forces_impl)
        self._rk4_step_jit = jit(self._rk4_step_impl)

        print(f"{self.n_bodies} bodies initialized with G = {self.G}")

    @partial(jit, static_argnums=(0,))
    def _compute_forces_impl(self, positions, masses, G):
        delta = positions[None, :, :] - positions[:, None, :]  # (N, N, 3)
        dist_sq = jnp.sum(delta**2, axis=-1) + 1e-9
        dist = jnp.sqrt(dist_sq)
        force_magnitudes = G * masses[:, None] * masses[None, :] / dist_sq
        force_vectors = force_magnitudes[..., None] * delta / dist[..., None]
        net_forces = jnp.sum(
            jnp.where(jnp.eye(len(masses), dtype=bool)[..., None], 0.0, force_vectors),
            axis=1,
        )
        return net_forces

    @partial(jit, static_argnums=(0,))
    def _rk4_step_impl(self, positions, velocities, masses, G, dt):
        forces1 = self._compute_forces_impl(positions, masses, G)
        acc1 = forces1 / masses[:, None]
        dpos1 = velocities
        dvel1 = acc1

        pos2 = positions + 0.5 * dt * dpos1
        vel2 = velocities + 0.5 * dt * dvel1
        forces2 = self._compute_forces_impl(pos2, masses, G)
        acc2 = forces2 / masses[:, None]
        dpos2 = vel2
        dvel2 = acc2

        pos3 = positions + 0.5 * dt * dpos2
        vel3 = velocities + 0.5 * dt * dvel2
        forces3 = self._compute_forces_impl(pos3, masses, G)
        acc3 = forces3 / masses[:, None]
        dpos3 = vel3
        dvel3 = acc3

        pos4 = positions + dt * dpos3
        vel4 = velocities + dt * dvel3
        forces4 = self._compute_forces_impl(pos4, masses, G)
        acc4 = forces4 / masses[:, None]
        dpos4 = vel4
        dvel4 = acc4

        new_positions = positions + dt / 6.0 * (dpos1 + 2 * dpos2 + 2 * dpos3 + dpos4)
        new_velocities = velocities + dt / 6.0 * (dvel1 + 2 * dvel2 + 2 * dvel3 + dvel4)

        return new_positions, new_velocities

    def compute_forces(self):
        """
        Compute current gravitational forces.

        Returns
        -------
        np.ndarray
            Forces array of shape (n_bodies, 3).
        """
        forces_jax = self._compute_forces_jit(self.positions, self.masses, self.G)
        return np.array(forces_jax)

    def step_rk4(self, dt: float) -> None:
        """
        Perform one RK4 integration step.

        Parameters
        ----------
        dt : float
            Time step size.
        """
        self.positions, self.velocities = self._rk4_step_jit(
            self.positions, self.velocities, self.masses, self.G, dt
        )
        self.history.append(self.positions.copy())

    def total_energy(self) -> float:
        """
        Calculate total energy (kinetic + potential) of the system.

        Returns
        -------
        float
            Total energy of the system.
        """
        v_squared = jnp.sum(self.velocities**2, axis=1)
        ke = 0.5 * jnp.sum(self.masses * v_squared)

        r_vectors = self.positions[:, None, :] - self.positions[None, :, :]
        r_distances = jnp.sqrt(jnp.sum(r_vectors**2, axis=-1))

        # Avoid self-interaction and division by zero
        r_distances = jnp.where(r_distances < 1e-10, jnp.inf, r_distances)

        # Upper triangular mask to avoid double counting
        i, j = jnp.triu_indices(self.n_bodies, k=1)
        pe_pairs = -self.G * self.masses[i] * self.masses[j] / r_distances[i, j]
        pe = jnp.sum(pe_pairs)

        return float(ke + pe)

    def simulate(self, t_end: float, dt: float) -> None:
        """
        Run the complete simulation.

        Parameters
        ----------
        t_end : float
            End time of the simulation.
        dt : float
            Time step size.
        """
        t = 0.0
        self.time_steps = [t]

        while t + dt <= t_end:
            self.step_rk4(dt)
            t += dt
            self.time_steps.append(t)

    def get_history(self):
        """
        Get trajectory history in the format expected by the C++ interface.

        Returns
        -------
        list
            List of trajectory frames, each containing positions for all bodies.
        """
        return [[pos.tolist() for pos in frame] for frame in self.history]

    def get_positions(self):
        """
        Get current positions in the format expected by the C++ interface.

        Returns
        -------
        list
            Current positions as nested list.
        """
        return self.positions.tolist()

    def get_velocities(self):
        """
        Get current velocities in the format expected by the C++ interface.

        Returns
        -------
        list
            Current velocities as nested list.
        """
        return self.velocities.tolist()

    def get_masses(self):
        """
        Get masses in the format expected by the C++ interface.

        Returns
        -------
        list
            Masses as list.
        """
        return self.masses.tolist()

    def get_time_steps(self):
        """
        Get time steps in the format expected by the C++ interface.

        Returns
        -------
        list
            Time steps as list.
        """
        return self.time_steps

    def get_n_bodies(self) -> int:
        """Get number of bodies."""
        return self.n_bodies

    def get_G(self) -> float:
        """Get gravitational constant."""
        return self.G
