import numpy as np

N = 64 # Grid size
dt = 0.1 # Time step
diff = 0.0001 # Diffusion rate
visc = 0.0001 # Viscosity (Diffusion for velocity)
grid = np.ones((N, N, 3)) # Initialize density and velocity grids


def set_boundary_conditions(x):
    '''
    Boundary conditions are necessary for the simulation's accuracy and stability.
    Here we assume a simple box boundary.
    '''
    x[0,:] = x[1,:]
    x[-1,:] = x[-2,:]
    x[:,0] = x[:,1]
    x[:,-1] = x [:,-2]
    return x


def add_source(D0: np.array, S: np.array, dt: float):
    ''' Add s o u r ce term t o t h e d e n s i t y or velocity grid.
    D0 - initial matrix
    S - matrix of sources with shape the same as the initial
    matrix (sources can be applied by e.g. mouse click)
    '''
    D0 += dt * S
    return D0


def diffuse_gauss_seidel(D0, diff, dt, iterations=20):
    D = np.copy(D0)
    a = dt * diff * N * N
    for _ in range(iterations):
        for i in range(1, N-1):
            for j in range(1, N-1):
                D[i, j] = (D0[i, j] + a * (D[i-1, j] + D[i+1, j] + D[i, j-1] + D[i, j+1])) / (1 + 4 * a)
    set_boundary_conditions(D)
    return D

def advect(D0, index_, dt):
    #N = D0.shape[0]
    u = D0[:, :, 1]
    v = D0[:, :, 2]
    dt0 = dt * N
    for i in range(1, N-1):
        for j in range(1, N-1):
            x = i - dt0 * u[i, j]
            y = j - dt0 * v[i, j]
            # Clamping coordinates
            x = min(N-1.5, max(0.5, x))
            y = min(N-1.5, max(0.5, y))
            i0, j0 = int(x), int(y)
            i1, j1 = i0 + 1, j0 + 1
            s1, t1 = x - i0, y - j0
            s0, t0 = 1 - s1, 1 - t1
            D0[i, j, index_] = (s0 * (t0 * D0[i0, j0, index_] + t1 * D0[i0, j1, index_]) + s1 * (t0 * D0[i1, j0, index_] + t1 * D0[i1, j1, index_]))
    set_boundary_conditions(D0[:, :, index_])
    return D0

def project(D0, iterations=20):
    h = 1 / N
    #N = D0.shape[0]
    u = D0[:, :, 1]
    v = D0[:, :, 2]
    div = np.zeros((D0.shape[:2]))
    p = np.zeros((D0.shape[:2]))

    for i in range(1, N-1):
        for j in range(1, N-1):
            div[i, j] = -0.5 * h * (u[i + 1, j] - u[i - 1, j] + v[i, j + 1] - v[i, j - 1])
    
    set_boundary_conditions(div)
    set_boundary_conditions(p)
    
    for _ in range(iterations):
        for i in range(1, N-1):
            for j in range(1, N-1):
                p[i, j] = (div[i, j] + p[i - 1, j] + p[i + 1, j] + p[i, j - 1] + p[i, j + 1]) / 4
    
        set_boundary_conditions(p)
    
    for i in range(1, N-1):
        for j in range(1, N-1):
            u[i, j] -= 0.5 * (p[i + 1, j] - p[i - 1, j]) / h
            v[i, j] -= 0.5 * (p[i, j + 1] - p[i, j - 1]) / h
    
    set_boundary_conditions(u)
    set_boundary_conditions(v)
    return D0


density_source = np.zeros(grid.shape[:2])
density_source[0, 0] = 1

while True:
    # update density
    grid[:, :, 0] = add_source(grid[:, :, 0], density_source, dt)
    grid[:, :, 0] = diffuse_gauss_seidel(grid[:, :, 0], diff, dt)
    grid = advect(grid, 0, dt)
    # update velocity
    grid[:, :, 1] = add_source(grid[:, :, 1], density_source, dt)
    grid[:, :, 2] = add_source(grid[:, :, 2], density_source, dt)
    grid[:, :, 1] = diffuse_gauss_seidel(grid[:, :, 1], visc, dt)
    grid[:, :, 2] = diffuse_gauss_seidel(grid[:, :, 2], visc, dt)
    grid = project(grid)
    grid = advect(grid, 1, dt)
    grid = advect(grid, 2, dt)
    grid = project(grid)
