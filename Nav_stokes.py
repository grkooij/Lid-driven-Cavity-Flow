import numpy as np
import matplotlib.pyplot as plt
import yt
from IPython.display import clear_output
import itertools
import time
import yt
from yt.units import meter

class Box:
    #Initialise an object that keeps track of simulation
    #specific variables
    def __init__(self, variables):
        self.Nx = variables[0]
        self.Ny = variables[1]
        self.dt = variables[2]
        self.t_end = variables[3]
        self.re = variables[4]
        self.lx = variables[5]
        self.ly = variables[6]
        self.N_savesteps = variables[7]
        self.U_top = variables[8]
        self.U_bot = variables[9]
        self.V_left = variables[10]
        self.V_right = variables[11]
        

def initialise_grid(box):
    #Initialises the computational grids using the
    #box object

    Nx = box.Nx
    Ny = box.Ny
    
    sf = np.zeros((Nx,Ny), dtype='float64')
    vort = np.zeros((Nx,Ny), dtype='float64')
    w = np.zeros((Nx,Ny), dtype='float64')
    
    return w, vort, sf


def stream_func(box, sf, vort, w):
    
    """
    For the successive over-relaxation procedure we can't use
    Numpy indexing which would make this a factor ~10 faster.
    This is because the vectorised Numpy operations do not update
    the array in between calculations, which is required here
    since it uses old and new values of the streamfunction.
    This function is by far the most time-consuming. Specifying a
    lenient max-error helps with computation time but comes at the
    cost of less accurate results.
    Loops are very inefficient in Python - nested loops even more so.
    A better alternative would be C or C++, which can be a factor 100
    faster.
    """
    
    beta = 1.5
    max_error = 0.1
    iters = 50
    dx = box.lx/(box.Nx-1)
    
    for its in range(iters):
        w[:,:]= sf[:,:]

        for i,j in itertools.product(range(1,box.Nx-1), range(1,box.Ny-1)):
            sf[i,j] = beta/4.*(sf[i+1,j] + sf[i-1,j] + sf[i,j+1] + sf[i,j-1] + dx*dx*vort[i,j]) + \
                (1. - beta)*sf[i,j]

        if np.sum(np.abs(w - sf)) < max_error:
            break

    return sf

def set_vort_bounds(box, sf, vort):
    """
    Set the boundary conditions of the vorticity.
    The corners are always set to zero, as that is how the
    grid is initialised.
    """
    
    dx = box.lx/(box.Nx-1)
    Nx = box.Nx
    Ny = box.Ny
        
    vort[1:Nx-1, Ny-1] = -2.*sf[1:Nx-1, Ny-2]/dx/dx - box.U_top*2./dx   #Top wall
    vort[1:Nx-1, 0   ] = -2.*sf[1:Nx-1, 1   ]/dx/dx + box.U_bot*2./dx   #Bottom wall
    vort[0   , 1:Ny-1] = -2.*sf[1   , 1:Ny-1]/dx/dx - box.V_left*2./dx  #Left wall 
    vort[Nx-1, 1:Ny-1] = -2.*sf[Nx-2, 1:Ny-1]/dx/dx + box.V_right*2./dx #Right wall

    return vort

def vort_rhs(box, w, vort, sf):
    U = 1.
    dx = box.lx/(box.Nx-1)
    visc = U/box.re
        
    #A more comprehensible but very slow loop
    
    """for i,j in itertools.product(range(1,box.Nx-1), range(1,box.Ny-1)):

        w[i,j] = -((sf[i,j+1] - sf[i,j-1])*(vort[i+1,j] - vort[i-1,j])            - \
                 (sf[i+1,j] - sf[i-1,j])*(vort[i,j+1] - vort[i,j-1]))/4./dx/dx + \
                 visc*(vort[i+1,j] + vort[i-1,j] + vort[i,j+1] + vort[i,j-1]   - \
                 4.*vort[i,j])/dx/dx"""
                    
    
        
    #Much faster (factor 10 ish) with numpy indexing in C
    w[1:-1,1:-1] = -((sf[1:-1,2:] - sf[1:-1,0:-2])*(vort[2:,1:-1] - vort[0:-2,1:-1])            - \
            (sf[2:,1:-1] - sf[0:-2,1:-1])*(vort[1:-1,2:] - vort[1:-1,0:-2]))/4./dx/dx + \
            visc*(vort[2:,1:-1] + vort[0:-2,1:-1] + vort[1:-1,2:] + vort[1:-1,0:-2]   - \
            4.*vort[1:-1,1:-1])/dx/dx


    return w

def pressure(box, sf):
    
    #Create a pressure grid
    p = np.zeros((box.Nx, box.Ny), dtype = 'float64')
    w = np.zeros((box.Nx, box.Ny), dtype = 'float64')
    s = np.zeros((box.Nx, box.Ny), dtype = 'float64')
    
    dx = box.lx/(box.Nx-1)
    dy = box.ly/(box.Nx-1)
    
    #Discretised Poisson
    """for i,j in itertools.product(range(1,box.Nx-1), range(1,box.Ny-1)):
            p[i,j] = (p[i-1,j] + p[i+1, j] + p[i,j+1] + p[i, j-1])/4. - \
            ((sf[i+1,j] - 2*sf[i,j] + sf[i-1,j])*(sf[i,j+1] - 2*sf[i,j] + sf[i,j-1])/dx/dx/dy/dy - \
            ((sf[i+1,j] - sf[i-1,j])*(sf[i,j+1] - sf[i,j-1])/dx/dy/4.)**2)/2."""
    beta = 1.2
    max_error = 0.1
    iters = 50
    dx = box.lx/(box.Nx-1)
    
    for its in range(iters):
        w[:,:]= p[:,:]        
        for i,j in itertools.product(range(1,box.Nx-1), range(1,box.Ny-1)):
            
            s[i,j] = 2*(sf[i+1,j] - 2*sf[i,j] + sf[i-1,j])*(sf[i,j+1] - 2*sf[i,j] + sf[i,j-1])/dx/dx - \
            (sf[i+1,j+1] - sf[i+1,j-1] - sf[i-1,j+1] + sf[i-1,j-1])**2/dx/dx
            
            p[i,j] = beta*(p[i-1,j] + p[i+1, j] + p[i,j+1] + p[i, j-1] - s[i,j])/4. + (1.-beta)*p[i,j]
             

        if np.sum(np.abs(w - p)) < max_error:
            
            break
            
    
    """p[1:-1, 1:-1] = p[0:-2, 1:-1]/4. + p[2:, 1:-1]/4. + p[1:-1, 2:]/4. + p[1:-1, 0:-2]/4. + \
            ((sf[2:,1:-1] - 2*sf[1:-1,1:-1] + sf[0:-2,1:-1])*(sf[1:-1,2:] - 2*sf[1:-1,1:-1] + sf[1:-1,0:-2])/dx/dx/dy/dy - \
            ((sf[2:,1:-1] - sf[0:-2,1:-1])*(sf[1:-1,2:] - sf[1:-1,0:-2])/dx/dy/4.)**2)/2.
       """
    
    return p

def velocity(box, streamfunction):
    
    dx = box.lx/(box.Nx-1)
    dy = box.ly/(box.Nx-1)
    
    u = np.zeros((box.Nx, box.Ny), dtype = 'float64')
    v = np.zeros((box.Nx, box.Ny), dtype = 'float64')
    
    #Simple NumPy indexing to speed up the calculation
    u[1:-1,1:-1] = (streamfunction[1:-1,2:] - streamfunction[1:-1,1:-1])/dy
    v[1:-1,1:-1] = (streamfunction[2:,1:-1] - streamfunction[1:-1,1:-1])/dx
    
    #Filling the ghost zones of the velocity, not strictly necessary
    #but looks better. In general, hydro codes do not return ghost
    #zone values but for this assignment it makes sense to return them
    #so that it can be checked that the boundaries are set correctly.
    #I include a minus for left and right due to the definition of the
    #stream function.
    u[1:box.Nx-1,box.Ny-1] = box.U_top
    u[1:box.Nx-1,0]        = box.U_bot
    v[box.Nx-1,1:box.Ny-1] = -box.V_right
    v[0,1:box.Ny-1]        = -box.V_left
    
    #return [-np.gradient(streamfunction, axis=0), np.gradient(streamfunction, axis=1)]
    return [u, -v]

def main(box):
    N_savevars = 5 #Always constant
    
    #Create a grid to save data
    results = np.zeros([N_savevars, box.N_savesteps, box.Nx, box.Ny], dtype= 'float64')
    
    #Keeps track of when to save the data
    when_to_save = np.linspace(1, box.t_end/box.dt, box.N_savesteps, dtype='int') - 1
    
    #Initialise the computational grids
    w, vort , sf = initialise_grid(box)
    
    #Time loop 
    t = 0.
    
    count_global = 0
    count_save = 0
    while t <= box.t_end:
        

        sf = stream_func(box, sf, vort, w)
        vort = set_vort_bounds(box, sf, vort)
        w = vort_rhs(box, w, vort, sf)
       
        #Update the vorticity
        vort[:,:] += box.dt*w[:,:]
        
        t += box.dt
                
        if count_global in when_to_save:
            
            #Save results in the solution matrix.
            #Note that this can become RAM expensive for very large boxes.
            #But, for the purposes of this assignment being 2D with small grids
            #this is completely fine.
            #Should I wish to expand the code to support very large grids,
            #it is better to save the data to disk at every savestep.
            #Just to show that the implementation is simple and straightforward
            #it would look like this:
            
            # sf = np.reshape(sf, box.Nx*box.Ny)
            # vort = np.reshape(vort, box.Nx*box.Ny)
            # np.savetxt("data{:04d}.dat".format(count_save), np.c_[sf, vort])
            
            #Which is the typical way that hydrodynamics codes save data to disk
            #e.g. PLUTO, Athena or AREPO, though of course in other data formats (.dbl, .hdf5
            #being the most common). 
            
            results[0, count_save, :] = sf
            results[1, count_save, :] = vort
                    
            #Print approximate progress
            clear_output(wait=True)
            print("Calculation at {} %".format(count_save*100/box.N_savesteps))
            count_save += 1
                        
        count_global += 1  
    
    clear_output(wait=True)
    print("Calculation finished.")
    print("Starting post-processing of the velocity and pressure...")
    for i in range(box.N_savesteps):
        
        sf = results[0, i, :]
        ux, uy = velocity(box, sf)
        p = pressure(box, sf)
        
        results[2, i, :] = ux
        results[3, i, :] = uy
        results[4, i, :] = p
        
    print("Done")
    return results
        
def save_data(box, results, name):
    #Save the data in single arrays
    print("Saving data to disk...")
    Nsavesteps = 100
    
    sf = np.reshape(results[0], box.N_savesteps*box.Nx*box.Ny)
    vort = np.reshape(results[1], box.N_savesteps*box.Nx*box.Ny)
    ux = np.reshape(results[2], box.N_savesteps*box.Nx*box.Ny)
    uy = np.reshape(results[3], box.N_savesteps*box.Nx*box.Ny)
    p = np.reshape(results[4], box.N_savesteps*box.Nx*box.Ny)
    
    np.savetxt("{}.dat".format(name), np.c_[sf, vort, ux, uy, p])
    print("Done.")
    
def make_yt_plot(box, data, var, var_string, title, log, savename):
    #This routine creates plots with python-yt

    var = var[:,:, np.newaxis]

    bbox = np.array([[0., 100.], [0., 100.], [0, 100.]])
    ds = yt.load_uniform_grid(data, var.shape, bbox=bbox)
    
    width = box.lx*meter
    
    slc = yt.SlicePlot(ds, "z", var_string, width=width)
    
    slc.set_log(var_string, log)
    slc.annotate_title(title)
    slc.save(savename)
    
    
    
def velocity_tracer(box, vels):
    v = np.zeros(box.N_savesteps, dtype = 'float64')
    
    #Loop over the data, saving the value of the velocity
    #for the grid cell in the middle
    
    for ind, grid2d in enumerate(vels):
        v[ind] = grid2d[int(np.floor(box.Nx/2)), int(np.floor(box.Ny/2))]
    
    return v

def problem_E():
    nx = ny = 200
    dt = 0.0001
    tend = 20.
    re = 1000
    Lx = Ly = 1
    N_savesteps = 100
    U_top = 1.
    U_bot = V_left = V_right = 0.
    variables = [nx,ny,dt,tend,re,Lx,Ly,N_savesteps,U_top,U_bot,V_left,V_right]
    box = Box(variables)

    start = time.time()
    results = main(box)

    end = time.time()
    print("Time taken: {:3.1f} minutes".format((end - start)/60))

    #Save data
    save_data(box, results, "problem_e")

    #Unpack results
    sf = results[0]
    vort = results[1]
    ux = results[2]
    uy = results[3]
    p = results[4]
    
    #Free RAM
    del results
    
    
    print("Plotting some results with Python-yt...")
    
    data = dict(Streamfunction = (sf[-1][:,:,np.newaxis], "m**2/s"), Vorticity = (vort[-1][:,:,np.newaxis], "s**-1"), \
                u = (ux[-1][:,:,np.newaxis], "m/s"), v = (uy[-1][:,:,np.newaxis], "m/s"), \
                Pressure = (p[-1][:,:,np.newaxis], "Pa"), Velocity = (np.sqrt(ux[-1]**2 + uy[-1]**2)[:,:,np.newaxis], \
                                                                     "m/s"))
    
    title_sf = "Plot of the stream function at t = {} s".format(tend)
    title_vort = "Plot of the Vorticity at t = {} s".format(tend)
    title_p = "Plot of the pressure at t = {} s".format(tend)
    title_vel = "Plot of the velocity magnitude at t = {} s".format(tend)
    title_velu = "Plot of the horizontal velocity at t = {} s".format(tend)
    title_velv = "Plot of the vertical velocity at t = {} s".format(tend)
    
    make_yt_plot(box, data, sf[-1], "Streamfunction", title_sf, False, "sf_e.pdf")
    make_yt_plot(box, data, vort[-1],  "Vorticity", title_vort, True, "vort_e.pdf") 
    make_yt_plot(box, data, p[-1],  "Pressure", title_p, False, "pres_e.pdf") 
    make_yt_plot(box, data, np.sqrt(ux[-1]**2 + uy[-1]**2),  "Velocity", title_vel, False, "vmag_e.pdf")
    make_yt_plot(box, data, ux[-1],  "u", title_velu, False, "velu_e.pdf")
    make_yt_plot(box, data, uy[-1],  "v", title_velv, False, "velv_e.pdf")
    
    #Placing a velocity tracer in the middle of the grid
    v_tracery = velocity_tracer(box, uy)
    v_tracerx = velocity_tracer(box, ux)
    plt.plot(np.linspace(0,tend, 100), v_tracery)
    plt.savefig('traceryE.pdf')
    plt.clf()

    plt.plot(np.linspace(0,tend, 100), v_tracerx)
    plt.savefig('tracerxE.pdf')
    plt.clf()
problem_E()