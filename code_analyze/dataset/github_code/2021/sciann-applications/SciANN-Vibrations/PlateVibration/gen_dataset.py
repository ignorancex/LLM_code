import numpy as np 


def gen_grid(nx=5, ny=5, nt=10, Lx=1.0, Ly=1.0, T=1.0):

    # domain grids
    x_grid, y_grid, t_grid = np.meshgrid(
        np.linspace(0, Lx, nx)[1:-1],
        np.linspace(0, Ly, ny)[1:-1],
        np.linspace(0,  T, nt)[1:],
        indexing='ij'
    )
    x_grid, y_grid, t_grid = [x.reshape(-1,1) for x in [x_grid, y_grid, t_grid]]

    # init grid
    x_init, y_init, t_init = np.meshgrid(
        np.linspace(0, Lx, (nx-2)*int(np.sqrt(nt))),
        np.linspace(0, Ly, (ny-2)*int(np.sqrt(nt))),
        [0],
        indexing='ij'
    )
    x_init, y_init, t_init = [x.reshape(-1,1) for x in [x_init, y_init, t_init]]
    
    # bc X grid 
    x_Xbc, y_Xbc, t_Xbc = np.meshgrid(
        np.linspace(0, Lx, nx*int((ny-2)/4)),
        np.linspace(0, Ly, 2),
        np.linspace(0,  T, nt)[1:],
        indexing='ij'
    )
    x_Xbc, y_Xbc, t_Xbc = [x.reshape(-1,1) for x in [x_Xbc, y_Xbc, t_Xbc]]

    # bc Y grid 
    x_Ybc, y_Ybc, t_Ybc = np.meshgrid(
        np.linspace(0, Lx, 2),
        np.linspace(0, Ly, ny*int((nx-2)/4)),
        np.linspace(0,  T, nt)[1:],
        indexing='ij'
    )
    x_Ybc, y_Ybc, t_Ybc = [x.reshape(-1,1) for x in [x_Ybc, y_Ybc, t_Ybc]]

    x_bc, y_bc, t_bc = [np.concatenate([x,y],axis=0) for x,y in zip([x_Xbc, y_Xbc, t_Xbc], [x_Ybc, y_Ybc, t_Ybc])]

    # x_grid 
    x_grid = np.concatenate([x_grid, x_init, x_Xbc, x_Ybc], axis=0)
    y_grid = np.concatenate([y_grid, y_init, y_Xbc, y_Ybc], axis=0)
    t_grid = np.concatenate([t_grid, t_init, t_Xbc, t_Ybc], axis=0)

    # test grid
    x_test, y_test, t_test = np.meshgrid(
        np.linspace(0, Lx, 3*nx),
        np.linspace(0, Ly, 3*ny),
        np.linspace(0,  T, 3*nt),
        indexing='ij'
    )

    # init ids 
    t0_ids = np.where(t_grid.flatten() == 0.)[0]
    bc_ids = np.where(
        np.logical_or(
            np.logical_or(x_grid.flatten() == 0., x_grid.flatten() == Lx),
            np.logical_or(y_grid.flatten() == 0., y_grid.flatten() == Ly)
        )
    )[0]
    dom_ids = np.where(
        np.logical_and(
            t_grid.flatten() > 0., 
            np.logical_and(
                np.logical_and(x_grid.flatten() > 0., x_grid.flatten() < Lx),
                np.logical_and(y_grid.flatten() > 0., y_grid.flatten() < Ly),
            )
        )
    )[0]
    
    return {
        'x': x_grid,
        'y': y_grid,
        't': t_grid,
        't0_ids': t0_ids,
        'bc_ids': bc_ids,
        'dom_ids': dom_ids,
        'x_test': x_test,
        'y_test': y_test,
        't_test': t_test
    }



if __name__ == "__main__":
    g = gen_grid(nx=5, ny=25, nt=100, Lx=1, Ly=5, T=10)

