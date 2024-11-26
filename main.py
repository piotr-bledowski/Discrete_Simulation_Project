import numpy as np

N = 64 # Grid size
dt = 0.1 # Time step
diff = 0.0001 # Diffusion rate
visc = 0.0001 # Viscosity (Diffusion for velocity)
grid = np.ones((N, N, 3)) # Initialize density and velocity grids

