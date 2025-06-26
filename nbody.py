import os
import sys
import shutil
from pathlib import Path
import time
from typing import Iterable
import warnings
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from p_tqdm import p_map

# plt.rcParams["text.usetex"] = True # Quite slow, but enables LaTeX rendering

matplotlib.use("Agg")

red = "#F42969"
orange = "orange"
blue = "#22ADFC"
purple = "#4635CE"
green = "#4CAF50"
yellow = "#FFD700"
pink = "#FF69B4"
teal = "#008080"
gold = "#FFD700"
gray = "#808080"
brown = "#8B4513"
lavender = "#E6E6FA"
cyan = "#00FFFF"
deep_blue = "#00008B"
tomato = "#D5120D"
COLOR_ARRAY = [blue, orange, red, teal, deep_blue, brown, tomato]


BACKEND = os.getenv("NBODY_BACKEND", None)

if BACKEND is None:
    warnings.warn("`NBODY_BACKEND` environment variable not set, defaulting to `cpp`.")
    BACKEND = "cpp"

if BACKEND == "cpp":
    try:
        sys.path.append("build/")
        import nbody_cpp as nbody
        print("Using C++ backend")
    except ImportError:
        print("C++ backend not available, falling back to JAX")
        BACKEND = "jax"

if BACKEND == "jax":
    try:
        import nbody_jax as nbody
        print("Using JAX backend")
    except ImportError:
        raise ImportError("Neither C++ nor JAX backend is available")


class NBodyWrapper:
    """
    Python wrapper for the C++ N-body simulation with additional functionality.

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

    Attributes
    ----------
    sim : nbody.NBodySimulation
        The underlying C++ or JAX simulation object.
    """

    def __init__(
        self,
        masses: Iterable[float],
        positions: Iterable[Iterable[float]],
        velocities: Iterable[Iterable[float]],
        G: float = 1.0,
    ) -> None:
        self.sim = nbody.NBodySimulation(masses, positions, velocities, G)

    def simulate(self, t_end: float, dt: float, verbose: bool = True) -> None:
        """
        Run the N-body simulation.

        Parameters
        ----------
        t_end : float
            End time of the simulation.
        dt : float
            Time step size.
        verbose : bool, optional
            If True, prints the elapsed simulation time. Default is True.
        """
        if verbose:
            print(f"Initial total energy: {self.sim.total_energy():.3f}")
        start_time = time.time()
        self.sim.simulate(t_end, dt)
        end_time = time.time()
        if verbose:
            print(f"Simulation completed in {end_time - start_time:.3f} seconds")
            print(f"Final total energy: {self.sim.total_energy():.3f}")

    def history(self) -> np.ndarray:
        """
        Trajectory history of all bodies over the simulation.

        Returns
        -------
        np.ndarray
            Array of shape (num_steps, num_bodies, 3) containing the positions at each time step.
        """
        return np.array(self.sim.get_history())

    def positions(self) -> np.ndarray:
        """
        Current positions of the bodies.

        Returns
        -------
        np.ndarray
            Array of shape (num_bodies, 3) containing the current positions.
        """
        return np.array(self.sim.get_positions())

    def velocities(self) -> np.ndarray:
        """
        Current velocities of the bodies.

        Returns
        -------
        np.ndarray
            Array of shape (num_bodies, 3) containing the current velocities.
        """
        return np.array(self.sim.get_velocities())

    def masses(self) -> np.ndarray:
        """
        Masses of the bodies.

        Returns
        -------
        np.ndarray
            Array of shape (num_bodies,) containing the masses.
        """
        return np.array(self.sim.get_masses())

    def total_energy(self) -> float:
        """
        Total energy of the system (kinetic + potential).

        Returns
        -------
        float
            Total energy of the system.
        """
        return self.sim.total_energy()

    def time_steps(self) -> np.ndarray:
        """
        Get the total time steps in the simulation.

        Returns
        -------
        np.ndarray
            Array of time steps taken during the simulation.

        """
        return np.array(self.sim.get_time_steps())


def plot_trajectories(sim_wrapper, title="N-Body Trajectories", save_fig=None):
    """Plot full trajectories of all bodies in the simulation"""
    history = sim_wrapper.history()
    masses = sim_wrapper.masses()

    plt.figure(figsize=(12, 10))

    mass_sizes = 50 + 200 * (masses / np.max(masses))

    for i, _ in enumerate(masses):
        x_traj = history[:, i, 0]
        y_traj = history[:, i, 1]
        color = COLOR_ARRAY[i % len(COLOR_ARRAY)]

        # full trajectory
        plt.plot(
            x_traj,
            y_traj,
            color=color,
            linewidth=1.5,
            alpha=0.7,
            label=f"Body {i + 1} (m={masses[i]:.1f})",
        )

        # starting position
        plt.scatter(
            x_traj[0],
            y_traj[0],
            color=color,
            s=mass_sizes[i],
            marker="o",
            edgecolor="black",
            alpha=0.8,
            linewidth=2,
        )

        # current position
        plt.scatter(
            x_traj[-1],
            y_traj[-1],
            color=color,
            s=mass_sizes[i] * 1.5,
            marker="*",
            edgecolor="black",
            alpha=1.0,
            linewidth=2,
        )

    plt.xlabel("X Position", fontsize=12)
    plt.ylabel("Y Position", fontsize=12)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.legend(loc="best", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.axis("equal")

    if save_fig:
        plt.savefig(save_fig, dpi=300, bbox_inches="tight")


def create_frame(args):
    frame, history, masses, time_steps, mass_sizes, outdir, xlim, ylim = args
    fig, ax = plt.subplots(figsize=(10, 10))
    if xlim is None:
        ax.set_xlim(np.min(history[:, :, 0]) * 1.1, np.max(history[:, :, 0]) * 1.1)
    else:
        ax.set_xlim(xlim)
    if ylim is None:
        ax.set_ylim(np.min(history[:, :, 1]) * 1.1, np.max(history[:, :, 1]) * 1.1)
    else:
        ax.set_ylim(ylim)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_ylabel("Y Position", fontsize=12)
    ax.set_xlabel("X Position", fontsize=12)

    trail = min(1000, frame)
    start_frame = max(0, frame - trail)

    for i in range(len(masses)):
        color = COLOR_ARRAY[i % len(COLOR_ARRAY)]
        ax.plot(
            history[start_frame : frame + 1, i, 0],
            history[start_frame : frame + 1, i, 1],
            color=color,
            linewidth=1,
            alpha=0.6,
        )
        ax.scatter(
            history[frame, i, 0],
            history[frame, i, 1],
            s=mass_sizes[i],
            color=color,
            edgecolor="black",
            linewidth=2,
            zorder=10,
        )

    ax.set_title(f"dt = {time_steps[frame]:.4f}", fontsize=14)
    filepath = os.path.join(outdir, f"frame_{frame:04d}.png")
    fig.savefig(filepath, bbox_inches="tight")
    plt.close(fig)
    return filepath


def create_animation(sim_wrapper, xlim, ylim, interval=50, save_animation=None):
    history = sim_wrapper.history()
    masses = sim_wrapper.masses()
    time_steps = sim_wrapper.time_steps()
    mass_sizes = 50 + 200 * (masses / np.max(masses))

    outdir = "_frames_tmp"
    os.makedirs(outdir, exist_ok=True)
    frames = min(1000, len(history))

    args = [
        (frame, history, masses, time_steps, mass_sizes, outdir, xlim, ylim)
        for frame in range(frames)
    ]

    print("Generating frames...")

    p_map(create_frame, args)

    print("Creating animation...")
    images = [
        imageio.imread(os.path.join(outdir, f"frame_{i:04d}.png"))
        for i in range(frames)
    ]
    imageio.mimsave(save_animation, images, fps=1000 // interval)
    print(f"Animation saved to {save_animation}")

    shutil.rmtree(outdir, ignore_errors=True)


def binary_system():
    masses = [70, 50]
    positions = [
        [1, 0, 0],
        [0, 0, 0],
    ]

    G = 1.0
    v_orbit = np.sqrt(G * 100 / 2)
    velocities = [
        [0, v_orbit / 2, 0],
        [0, -v_orbit / 2, 0],
    ]

    t_end = 5.0
    dt = 0.01
    sim = NBodyWrapper(masses, positions, velocities, 1)
    sim.simulate(t_end, dt, verbose=True)

    plot_trajectories(
        sim,
        title="Binary System Trajectories",
        save_fig=f"images/binary_system_trajectories_{BACKEND}.png",
    )
    xlim = None
    ylim = None
    create_animation(
        sim,
        xlim,
        ylim,
        50,
        save_animation=f"images/binary_system_animation_{BACKEND}.mp4",
    )


def nbody_system():
    masses = [10, 10, 10, 10, 10]

    positions = [
        [0.0, 0.0, 0.0],
        [1.0, -1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [-1.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
    ]

    G = 1.0
    v = np.sqrt(G * masses[0] / 1.0)

    velocities = [
        [0.0, 0.0, 0.0],
        [0.0, v, 0.0],
        [0.0, -v, 0.0],
        [-v, 0.0, 0.0],
        [v, 0.0, 0.0],
    ]

    t_end = 5.0
    dt = 0.01
    sim = NBodyWrapper(masses, positions, velocities, 1)
    sim.simulate(t_end, dt, verbose=True)

    plot_trajectories(
        sim,
        title=f"{len(masses)}-Body System Trajectories",
        save_fig=f"images/{len(masses)}_body_system_trajectories_{BACKEND}.png",
    )

    xlim = [-20, 20]
    ylim = [-20, 20]

    create_animation(
        sim,
        xlim,
        ylim,
        50,
        save_animation=f"images/{len(masses)}_body_system_animation_{BACKEND}.mp4",
    )


def main():
    Path("images").mkdir(exist_ok=True)

    print("=== N-Body Simulation ===\n")
    print("\n=== Binary System Simulation ===\n")
    binary_system()
    print()

    print("\n=== N-Body System Simulation ===\n")
    nbody_system()


if __name__ == "__main__":
    main()
