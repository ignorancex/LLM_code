def sp2xyz(s,lightsource,xbar,ybar,zbar)
    #  Copyright EPFL-LSP
    #  Derived from version 2005 (R.D Hersch / F. Crï¿½tï¿½)
    #  {r-spectrum, Illuminant} --> XYZ
    #  The input spectrum s can be a vector or a n x 36 spectra matrix.
    #  With sa n x 36 input matrix, a n x 3 output matrix is produced with one XYZ 
    #   value for each spectrum row of the input matrix, EP, RDH, 1.3.03
    k = 100 / (lightsource * ybar.T);
    # Element wise multiplication of each row by the illuminant
    Vs = s * kron(ones(size(s,1), 1), lightsource);
    # Producing final XYZ value by dot products between _bar predefined
    # values & previosly computed Vs
    XYZ = k * Vs * [xbar.T ybar.T zbar.T];
    return XYZ