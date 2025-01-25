import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Grid and simulation parameters
N = 64  # Grid size
dt = 0.1  # Time step
diff = 0.001  # Diffusion rate
visc = 0.001  # Viscosity
grid = np.zeros((N, N, 3))  # Initialize density and velocity grids

# Initialize source term, interaction variables, and force timers
density_source = np.zeros((N, N))
velocity_source = np.zeros((N, N, 2))  # Force vectors for velocity (u, v)
force_timer = np.zeros((N, N), dtype=int)  # Timer to track force duration
force_duration = 10  # Frames for which force is applied after a click

# Boundary conditions
def set_boundary_conditions(x, index_=None):
    if index_ is None or index_ == 0:  # Density
        x[0, :] = x[1, :]      # Top wall
        x[-1, :] = x[-2, :]    # Bottom wall
        x[:, 0] = x[:, 1]      # Left wall
        x[:, -1] = x[:, -2]    # Right wall
    else:  # Velocity
        x[0, :] = 0
        x[-1, :] = 0
        x[:, 0] = 0
        x[:, -1] = 0
    return x

# Add source
def add_source(D0, S, dt):
    D0 += dt * S
    return D0

# Diffusion using Gauss-Seidel iteration
def diffuse_gauss_seidel(D0, diff, dt, iterations=20):
    D = np.copy(D0)
    a = dt * diff * N * N
    for _ in range(iterations):
        for i in range(1, N-1):
            for j in range(1, N-1):
                D[i, j] = (D0[i, j] + a * (D[i-1, j] + D[i+1, j] +
                           D[i, j-1] + D[i, j+1])) / (1 + 4 * a)
        D = set_boundary_conditions(D)  # Apply after each iteration
    return D

# Advection
def advect(D0, index_, dt):
    u = D0[:, :, 1]
    v = D0[:, :, 2]
    dt0 = dt * N
    for i in range(1, N-1):
        for j in range(1, N-1):
            x = i - dt0 * u[i, j]
            y = j - dt0 * v[i, j]
            x = min(N-1.5, max(0.5, x))
            y = min(N-1.5, max(0.5, y))
            i0, j0 = int(x), int(y)
            i1, j1 = i0 + 1, j0 + 1
            s1, t1 = x - i0, y - j0
            s0, t0 = 1 - s1, 1 - t1
            D0[i, j, index_] = (s0 * (t0 * D0[i0, j0, index_] + t1 * D0[i0, j1, index_]) +
                                s1 * (t0 * D0[i1, j0, index_] + t1 * D0[i1, j1, index_]))
    D0 = set_boundary_conditions(D0, index_)
    return D0

# Projection
def project(D0, iterations=20):
    h = 1 / N
    u = D0[:, :, 1]
    v = D0[:, :, 2]
    div = np.zeros((N, N))
    p = np.zeros((N, N))

    for i in range(1, N-1):
        for j in range(1, N-1):
            div[i, j] = -0.5 * h * \
                (u[i + 1, j] - u[i - 1, j] + v[i, j + 1] - v[i, j - 1])

    div = set_boundary_conditions(div)
    p = set_boundary_conditions(p)

    for _ in range(iterations):
        for i in range(1, N-1):
            for j in range(1, N-1):
                p[i, j] = (div[i, j] + p[i - 1, j] + p[i + 1, j] +
                           p[i, j - 1] + p[i, j + 1]) / 4
        p = set_boundary_conditions(p)

    for i in range(1, N-1):
        for j in range(1, N-1):
            u[i, j] -= 0.5 * (p[i + 1, j] - p[i - 1, j]) / h
            v[i, j] -= 0.5 * (p[i, j + 1] - p[i, j - 1]) / h

    D0[:, :, 1] = set_boundary_conditions(u, 1)
    D0[:, :, 2] = set_boundary_conditions(v, 2)
    return D0

# Click handler for user interaction
def on_click(event):
    if event.xdata is not None and event.ydata is not None:
        x, y = int(event.xdata), int(event.ydata)
        # Add density and velocity sources at the clicked position
        density_source[y, x] += 50
        velocity_source[y, x, 0] += (np.random.rand() - 0.5) * 100  # Random force x
        velocity_source[y, x, 1] += (np.random.rand() - 0.5) * 100  # Random force y
        force_timer[y, x] = force_duration  # Set timer for this force

# Plotting setup
fig, ax = plt.subplots()
im = ax.imshow(grid[:, :, 0], animated=True)
plt.colorbar(im)

# Update function for animation
def update(frame):
    global grid, density_source, velocity_source, force_timer

    # Apply sources where the timer is active
    active_forces = force_timer > 0
    grid[:, :, 0] = add_source(grid[:, :, 0], density_source * active_forces, dt)
    grid[:, :, 1] = add_source(grid[:, :, 1], velocity_source[:, :, 0] * active_forces, dt)
    grid[:, :, 2] = add_source(grid[:, :, 2], velocity_source[:, :, 1] * active_forces, dt)

    # Decrease force timers
    force_timer[force_timer > 0] -= 1

    # Diffuse and advect
    grid[:, :, 0] = diffuse_gauss_seidel(grid[:, :, 0], diff, dt)
    grid[:, :, 1] = diffuse_gauss_seidel(grid[:, :, 1], visc, dt)
    grid[:, :, 2] = diffuse_gauss_seidel(grid[:, :, 2], visc, dt)
    grid = advect(grid, 0, dt)
    grid = advect(grid, 1, dt)
    grid = advect(grid, 2, dt)

    # Project to enforce incompressibility
    grid = project(grid)

    # Update the visualization
    im.set_array(grid[:, :, 0])
    return [im]

# Connect click handler and create animation
fig.canvas.mpl_connect('button_press_event', on_click)
anim = FuncAnimation(fig, update, frames=None, interval=50, blit=True)
plt.show()
