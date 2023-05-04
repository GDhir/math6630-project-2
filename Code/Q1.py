import numpy as np
import matplotlib.pyplot as plt
import scipy

from numpy import linalg as LA

def calcA( N ):

    totalmodes = 2*N + 1

    A = np.zeros( (totalmodes, totalmodes), dtype=complex )

    for i in range( totalmodes ):
        k = i - N
        print(k)
        A[ i, i ] = -2*( np.pi*np.pi )*k*k

        if i + 1 < totalmodes:

            A[ i, i + 1 ] = np.pi*( k + 1 )

        if i - 1 > 0:

            A[ i, i - 1 ] = -np.pi*( k - 1 )
            

    # eigen_val, _ = np.linalg.eig(A)
    eigen_val,_ = np.linalg.eig(A)
    # x = [ele.real for ele in eigen_val]
    # # extract imaginary part
    # y = [ele.imag for ele in eigen_val]

    # plt.figure()
    # plt.scatter( x, y )
    # # plt.xlim( -2, 2 )
    # plt.xlabel( "$Re( \lambda )$" )
    # plt.ylabel( "$Im( \lambda )$" )
    # plt.savefig( "eigenplot_u.png" )

    maxval = np.max( np.abs( eigen_val ) )

    dtrk = 2.78/maxval
    dt = 200/maxval

    print(dt)
    print(dtrk)
    print(dt/dtrk)

    return dt, dtrk, A


def plotdt():

    Nvals = np.array( [10, 20, 40, 80, 160, 320] )
    dt = np.zeros( len(Nvals) )

    dtxx = np.zeros( len(Nvals) )

    dtx = np.zeros( len(Nvals) )

    invNvals = np.array( [1/N**2 for N in Nvals] )
    

    for idx, N in enumerate( Nvals ):
        totalmodes = 2*N + 1

        A = np.zeros( (totalmodes, totalmodes) )
        Ax = np.zeros( (totalmodes, totalmodes) )
        Axx = np.zeros( (totalmodes, totalmodes) )

        for i in range( totalmodes ):
            k = i - N
            
            A[ i, i ] = -2*( np.pi**2 )*k*k
            Axx[ i, i ] = -2*( np.pi**2 )*k*k

            if i + 1 < totalmodes:

                A[ i, i + 1 ] = np.pi*( k + 1 )
                Ax[ i, i + 1 ] = np.pi*( k + 1 )

            if i - 1 > 0:

                A[ i, i - 1 ] = -np.pi*( k - 1 )
                Ax[ i, i - 1 ] = -np.pi*( k - 1 )
                

        # eigen_val, _ = np.linalg.eig(A)
        eigen_val,_ = np.linalg.eig(A)

        maxval = np.max( np.abs( eigen_val ) )

        dt[idx] = 2/maxval

        eigen_val,_ = np.linalg.eig(Ax)

        maxval = np.max( np.abs( eigen_val ) )

        dtx[idx] = 2/maxval

        eigen_val,_ = np.linalg.eig(Axx)

        maxval = np.max( np.abs( eigen_val ) )

        dtxx[idx] = 2/maxval

    slope_intercept = np.polyfit( np.log2(Nvals), np.log2(dt), 1 )

    slope_interceptx = np.polyfit( np.log2(Nvals), np.log2(dtx), 1 )

    slope_interceptxx = np.polyfit( np.log2(Nvals), np.log2(dtxx), 1 )

    print( slope_intercept )
    print( slope_interceptx )
    print( slope_interceptxx )

    oddNvals = 2*Nvals + 1

    plt.loglog( oddNvals, dt, "-o", label = "$\Delta t$" )
    plt.loglog( oddNvals, invNvals, "-x", label = "$1/N^2$" )
    plt.xlabel( "N" )
    plt.ylabel( "$\Delta t$" )
    plt.legend()
    plt.savefig( "dtvariation.png" )

    plt.figure()
    plt.loglog( Nvals, dtx, '-o', label="$\Delta t$ for $-sin(2 \pi x)u_x$" )
    plt.loglog( Nvals, dtxx, '-x', label="$\Delta t$ for $u_{xx}/2$")
    plt.xlabel( "N" )
    plt.ylabel( "$\Delta t$" )
    plt.legend()
    plt.savefig( "dtComponentwise.png" )

    return A

    # print(eigen_val)

    # print(maxval)

def crankNicolson(N, dt, A):

    # dt, _, A = calcA( N )

    totalmodes = 2*N + 1

    t = 0

    I = np.eye( totalmodes, dtype=complex )
    # print(I)

    G = np.linalg.inv( I - A*dt/2 )
    F = I + A*dt/2
    K = np.dot( G, F )

    # totaltimevals = [0.25, 0.27, 0.5, 0.6]
    totaltimevals = [1]
    # plt.figure()

    M = 2*N + 1
    # dx = 1/M
    # x = np.linspace( 0, 1 - dx, M )

    u = np.zeros( M, dtype=complex )

    for totaltime in totaltimevals: 

        uhat = np.zeros( totalmodes, dtype=complex )
        uhat[ N + 1 ] = 1/( 2j )
        uhat[ N - 1 ] = -1/( 2j )

        while t <= totaltime:

            uhat =  np.dot( K, uhat )

            # print( uhat )

            t += dt

        # u = np.zeros( M, dtype=complex )

        for j in range( M ):

            xj = j/M

            for mode in range( totalmodes ):

                k = mode - N

                u[ j ] += uhat[mode]*np.exp( 2*np.pi*k*xj*1j, dtype=complex )

        # plt.plot( x, u.real, label="t=" + str(round(t, 2)) )

    # plt.ylabel( "u(x, t) (Solution Value)" )
    # plt.xlabel( "X" )
    # plt.legend()
    # plt.savefig( "uplot_different_times.png" )

    return u
    # plt.show()

def RK4(N, dt, A):

    # _, dt, A = calcA( N )

    totalmodes = 2*N + 1

    t = 0

    # totaltimevals = [0.25, 0.27, 0.5, 0.6]
    totaltimevals = [1]
    # plt.figure()

    M = 2*N + 1
    # dx = 1/M
    # x = np.linspace( 0, 1 - dx, M )
    u = np.zeros( M, dtype=complex )

    for totaltime in totaltimevals: 

        uhat = np.zeros( totalmodes, dtype=complex )
        uhat[ N + 1 ] = 1/( 2j )
        uhat[ N - 1 ] = -1/( 2j )

        while t <= totaltime:

            alpha1 = dt*np.dot( A, uhat )
            alpha2 = uhat + alpha1/2
            alpha2 = dt*np.dot( A, alpha2 )

            alpha3 = uhat + alpha2/2
            alpha3 = dt*np.dot( A, alpha3 )

            alpha4 = uhat + alpha3
            alpha4 = dt*np.dot( A, alpha4 )    

            uhat = uhat + (alpha1 + 2*alpha2 + 2*alpha3 + alpha4)/6        

            # print( uhat )

            t += dt

        # u = np.zeros( M, dtype=complex )

        for j in range( M ):

            xj = j/M

            for mode in range( totalmodes ):

                k = mode - N

                u[ j ] += uhat[mode]*np.exp( 2*np.pi*k*xj*1j, dtype=complex )

        # plt.plot( x, u.real, label="t=" + str(round(t, 2)) )

    # plt.ylabel( "u(x, t) (Solution Value)" )
    # plt.xlabel( "X" )
    # plt.legend()
    # plt.savefig( "uplot_different_times_RK4.png" )
    
    return u
    # plt.show()

    

if __name__ == "__main__":
    N = 20
    # plotdt()
    dt, dtrk, A = calcA(N)

    import timeit

    start = timeit.default_timer()
    ucn = crankNicolson(N, dt, A)
    stop = timeit.default_timer()

    timecn = stop - start
    print('Time Crank Nicolson: ', stop - start)  

    start = timeit.default_timer()
    urk4 = RK4(N, dtrk, A)
    stop = timeit.default_timer()

    timerk4 = stop - start
    print('Time RK4: ', stop - start)  

    print("timecompare = ", timerk4/timecn)

    M = 2*N + 1
    dx = 1/M
    x = np.linspace( 0, 1 - dx, M )

    plt.figure()
    plt.plot( x, ucn, "-o", label="Crank Nicolson" )
    plt.plot( x, urk4, "-x", label="RK4" )
    plt.ylabel( "u(x, t) (Solution Value)" )
    plt.xlabel( "X" )
    plt.legend()

    print( np.sqrt( np.sum( np.abs( ucn - urk4 )**2 )*dx ) )

    plt.savefig( "Q1SolutionComparisont=" + str(round(dt, 3) ) + ".png" )
