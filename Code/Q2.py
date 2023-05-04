import numpy as np
import matplotlib.pyplot as plt
import scipy

from numpy import linalg as LA

def computeD3(N):

    totalmodes = 2*N + 1

    M = 2*N + 1

    A = np.zeros( ( M, totalmodes ), dtype=complex )
    C = np.zeros( ( M, totalmodes ), dtype=complex )

    dx = 1/M

    for i in range(M):
        x = -10 + i*20/M
        for j in range(totalmodes):

            k = j - N

            A[ i, j ] = np.exp( 2*np.pi*k*x*1j/20, dtype=complex )/np.sqrt(20)

    for i in range(totalmodes):

        k = i - N
        ck = 2*np.pi*k*1j/20
        C[ i, i ] = ck**3

    B = 20*np.transpose( A.conj() )/M
    B[ 0, : ] = B[ 0, : ]/2
    B[ -1, : ] = B[ -1, : ]/2

    # for validation
    # B1 = np.linalg.inv( A )
    # B1[ 0, : ] = B1[ 0, : ]/2
    # B1[ -1, : ] = B1[ -1, : ]/2

    D3 = np.dot( A, np.dot( C, B ) )

    # for validation
    # D31 = np.dot( A, np.dot( C, B1 ) )
    # I = np.dot( B, A )
    # print(I)

    # for i in range(M):
    #     for j in range(M):

    #         if np.abs( I[i, j] ) > 1e-6:
    #             print( i, j, I[i, j] )

    # u = np.zeros( M, dtype=complex )

    # for i in range(M):

    #     x = -10 + i*20/M
    #     # print(x)

    #     u[i] = -2/( np.cosh( x )**2 )

    # print(u)

    # for validation
    # uxxx = np.dot( D3, u )
    # uxxx1 = np.dot( D31, u )
    # print( uxxx ) 
    # print( uxxx1 )
    # print( np.max( np.abs( uxxx - uxxx1 ) ) )

    return D3

def computeD1(N):

    totalmodes = 2*N + 1

    M = 2*N + 1

    A = np.zeros( ( M, totalmodes ), dtype=complex )
    C = np.zeros( ( M, totalmodes ), dtype=complex )

    dx = 1/M

    for i in range(M):
        x = -10 + i*20/M
        for j in range(totalmodes):

            k = j - N

            A[ i, j ] = np.exp( 2*np.pi*k*x*1j/20, dtype=complex )/np.sqrt(20)

    for i in range(totalmodes):

        k = i - N
        ck = 2*np.pi*k*1j/20
        C[ i, i ] = ck

    B = 20*np.transpose( A.conj() )/M
    B[ 0, : ] = B[ 0, : ]/2
    B[ -1, : ] = B[ -1, : ]/2

    # for validation
    # B1 = np.linalg.inv( A )
    # B1[ 0, : ] = B1[ 0, : ]/2
    # B1[ -1, : ] = B1[ -1, : ]/2

    D1 = np.dot( A, np.dot( C, B ) )   

    return D1

def applyN( uvals, N, D1 ):

    Du = np.dot( D1, uvals )
    uDu = np.multiply( uvals, Du )

    return 6*uDu

def applyS1( fvals, N, dt, D1 ):
    
    Nf = applyN( fvals, N, D1 )

    fintermediate = fvals + dt*Nf/2

    Nfintermediate = applyN( fintermediate, N, D1 )

    fnew = fvals + dt*Nfintermediate

    return fnew

def applyS2( fvals, N, dt, D3 ):

    M = 2*N + 1
    I = np.eye( M )
    

    # G = np.linalg.inv( I + D3*dt/2 )
    G = I + D3*dt/2
    F = I - D3*dt/2

    b = np.dot( F, fvals )

    fnew = np.linalg.solve( G, b )
    # H = np.dot( G, F )

    # fnew = np.dot( H, fvals )

    return fnew

def applyStrangSplitting( uvals, N, dt, D1, D3 ):

    w1 = applyS2( uvals, N, dt/2, D3 )
    w2 = applyS1( w1, N, dt, D1 )

    unew = applyS2( w2, N, dt/2, D3 )

    return unew

def runSim( N, fignameprefix ):

    M = 2*N + 1
    uvals = np.zeros( M, dtype=complex )

    xvals = np.zeros( M )

    for i in range(M):
        
        x = -10 + i*20/M
        xvals[i] = x

        uvals[i] = -2/( np.cosh( x )**2 )

    plt.figure()
    plt.plot( xvals, np.real(uvals), label = "Initial u (t = 0)" )

    D1 = computeD1( N )
    D3 = computeD3( N )

    # D3 = computeD3( N )

    eigen_val,_ = np.linalg.eig(D1)

    # eigen_val3,_ = np.linalg.eig(D3)

    # dt3 = 1/np.max( np.abs( np.imag( eigen_val3 ) ) )

    # print( dt3 )
    dt = 0

    if( N >= 40 ):
        dt = 0.1/np.max( np.abs( np.imag( eigen_val ) ) )
    else:
        dt = 0.04/np.max( np.abs( np.imag( eigen_val ) ) )
    print( dt )

    t = 0

    while t <= 1:

        uvals = applyStrangSplitting( uvals, N, dt, D1, D3 )
        t += dt

    # print(uvals)
    
    
    plt.plot( xvals, np.real(uvals), label = "t = " + str( round( t, 2) ) )
    plt.xlabel( "X" )
    plt.ylabel( "u" )
    plt.legend()
    plt.savefig( fignameprefix + "t = " + str( round( t, 2) ) + ".png" )
    # plt.show()

    return uvals

def testhadamard():

    a = np.array( [1, 2, 3, 4] )
    b = np.array( [3, 4, 5, 6] )

    print( a*b )
    print( np.multiply(a, b) )


def testconjugate():

    A = np.zeros( (2, 2), dtype=complex )
    A[0, 0] = 1 + 2j
    A[0, 1] = 1 + 3j
    A[1, 0] = 1 - 4j
    A[1, 1] = 3 - 5j

    B = A.conj()
    C = np.transpose( A.conj() )
    print(A)
    print(B)
    print(C)

if __name__ == "__main__":

    Nvals = np.array( [64] )
    dt = np.zeros( len( Nvals ) )
    errorvals = np.zeros( len( Nvals ) )
    dt3 = np.zeros( len( Nvals ) )

    uvalsall = []
    uref = []

    Nref = Nvals[0]

    for idx, N in enumerate( Nvals ):

        # D1 = computeD1( N )
        # D3 = computeD3( N )

        # eigen_val,_ = np.linalg.eig(D1)

        # eigen_val3,_ = np.linalg.eig(D3)
        # print(eigen_val3)

        # dt3[ idx ] = 1/np.max( np.abs( eigen_val3 ) )

        # dt[ idx ] = 0.2/np.max( np.abs( np.imag( eigen_val ) ) )

        fignameprefix = "solutionPlot_N=" + str(N) 

        uvals = runSim(N, fignameprefix )
        print(N)
        uvalsall.append( uvals )

        if idx == 0:
            uref.append( uvals )

    # from scipy.interpolate import CubicSpline

    # Mref = 2*Nref + 1
    # xvalsref = np.zeros( Mref )

    # for i in range(Mref):
        
    #     x = -10 + i*20/Mref
    #     xvalsref[i] = x

    # spl = CubicSpline(xvalsref, uref[0])

    # for idx, N in enumerate( Nvals ):

    #     M = 2*N + 1
    #     currerror = 0
    #     ucurr = uvalsall[idx]

    #     for i in range(M):
        
    #         x = -10 + i*20/M
    #         currerror += np.abs( ucurr[i] - spl( x ) )**2

    #     errorvals[idx] = np.sqrt( currerror/M )

    # slope_intercept = np.polyfit( np.log2(Nvals), np.log2(errorvals), 1 )
    # print( slope_intercept )

    # plt.figure()
    # plt.loglog( Nvals[1:], errorvals[1:], "-o" )
    # plt.xlabel( "N" )
    # plt.ylabel( "$L_2$ Norm of the Error" )
    # plt.savefig( "errorlogplot.png" )
    # np.savetxt('test.txt', errorvals, delimiter=',')

    # plt.figure()
    # plt.semilogx( Nvals[1:], errorvals[1:], "-o" )
    # plt.xlabel( "N" )
    # plt.ylabel( "$L_2$ Norm of the Error" )
    # plt.savefig( "errorsemilogplot.png" )

    # plt.figure()
    # plt.plot( Nvals[1:], errorvals[1:], "-o" )
    # plt.xlabel( "N" )
    # plt.ylabel( "$L_2$ Norm of the Error" )
    # plt.savefig( "error_nologplot.png" )

    # Nvalsinv = [ 1/N**3 for N in Nvals ]

    # plt.figure()
    # plt.loglog( Nvals, Nvalsinv, "-o", label = "$1/N^3$"  )
    # # plt.loglog( Nvals, dt, label= "Timestep based on eigenvalues of $u_x$" )
    # plt.loglog( Nvals, dt3, "-x", label= "Timestep based on eigenvalues of $u_{xxx}$" )
    # plt.xlabel( "N" )
    # plt.ylabel( "$\Delta t$" )
    # plt.legend()
    # plt.savefig( "Q2Timestep_uxxx.png" )