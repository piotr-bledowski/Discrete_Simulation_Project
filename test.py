import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parameters for the simulation
GRID_SIZE = 50        # Size of the grid (GRID_SIZE x GRID_SIZE)
TIME_STEP = 0.5       # Time step for the simulation
VISCOSITY = 0.1       # Viscosity of the fluid
FORCE_STRENGTH = 5.0  # Strength of the external force

# Initialize velocity and pressure fields
u = np.zeros((GRID_SIZE, GRID_SIZE))  # Velocity in x-direction
v = np.zeros((GRID_SIZE, GRID_SIZE))  # Velocity in y-direction
pressure = np.zeros((GRID_SIZE, GRID_SIZE))  # Pressure field

# Function to apply an external force


def apply_force(u, v, location, strength):
    x, y = location
    u[x, y] += strength[0]
    v[x, y] += strength[1]

# Function to compute divergence of the velocity field


def compute_divergence(u, v):
    div = np.zeros_like(u)
    div[1:-1, 1:-1] = (u[2:, 1:-1] - u[:-2, 1:-1] +
                       v[1:-1, 2:] - v[1:-1, :-2]) / 2.0
    return div

# Function to solve the Poisson equation for pressure


def solve_pressure(pressure, div, iterations=50):
    for _ in range(iterations):
        pressure[1:-1, 1:-1] = (div[1:-1, 1:-1] +
                                pressure[2:, 1:-1] + pressure[:-2, 1:-1] +
                                pressure[1:-1, 2:] + pressure[1:-1, :-2]) / 4.0

# Function to update velocity field based on pressure gradient and viscosity


def update_velocity(u, v, pressure):
    u[1:-1, 1:-1] -= (pressure[2:, 1:-1] - pressure[:-2, 1:-1]) / 2.0
    v[1:-1, 1:-1] -= (pressure[1:-1, 2:] - pressure[1:-1, :-2]) / 2.0

    # Add viscosity
    u[1:-1, 1:-1] += VISCOSITY * \
        (u[2:, 1:-1] + u[:-2, 1:-1] + u[1:-1, 2:] +
         u[1:-1, :-2] - 4 * u[1:-1, 1:-1])
    v[1:-1, 1:-1] += VISCOSITY * \
        (v[2:, 1:-1] + v[:-2, 1:-1] + v[1:-1, 2:] +
         v[1:-1, :-2] - 4 * v[1:-1, 1:-1])


# Visualization setup
fig, ax = plt.subplots()
x, y = np.meshgrid(np.arange(GRID_SIZE), np.arange(GRID_SIZE))
quiver = ax.quiver(x, y, u, v, scale=1, scale_units='xy')

# Update function for animation


def update(frame):
    global u, v, pressure

    # Apply an external force at a fixed location
    apply_force(u, v, location=(GRID_SIZE // 2, GRID_SIZE // 2),
                strength=(FORCE_STRENGTH, 0))

    # Compute divergence of velocity
    div = compute_divergence(u, v)

    # Solve for pressure
    solve_pressure(pressure, div)

    # Update velocity based on pressure gradient and viscosity
    update_velocity(u, v, pressure)

    # Update the quiver plot
    quiver.set_UVC(u, v)
    return quiver,


# Create animation
ani = FuncAnimation(fig, update, frames=200,
                    interval=TIME_STEP * 1000, blit=True)
plt.show()
