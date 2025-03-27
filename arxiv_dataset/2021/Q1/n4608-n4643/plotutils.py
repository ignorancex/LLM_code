# Plotting-related code

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Ellipse
from skimage.measure import profile_line



# Nicer-looking logarithmic axis labeling

def niceLogFunc( x_value, pos ):
	return ('{{:.{:1d}f}}'.format(int(np.maximum(-np.log10(x_value),0)))).format(x_value)	
NiceLogFormatter = ticker.FuncFormatter(niceLogFunc)

def MakeNiceLogAxes( whichAxis="xy", axisObj=None ):
	"""
	Makes one or more axes of a figure display tick labels using non-scientific
	notation (e.g., "0.01" instead of "10^{-2}")
	"""
	
	if axisObj is None:
		ax = plt.gca()
	else:
		ax = axisObj
	if whichAxis in ["x", "xy"]:
		ax.xaxis.set_major_formatter(NiceLogFormatter)
	if whichAxis in ["y", "xy"]:
		ax.yaxis.set_major_formatter(NiceLogFormatter)

def SetAxesObjTickLabelSize( axesObj, fontsize ):
    axesObj.tick_params(axis='both', which='both', labelsize=fontsize)



def add_colorbar( mappable, loc="right", size="5%", pad=0.05, label_pad=2,
                    tick_label_size=10 ):
    """
    Function which adds a colorbar to a "mappable" object (e.g., the result of
    calling plt.imshow).

    Example:
        img = plt.imshow(somedata, ...)
        add_colorbar(img, ...)

    Parameters
    ----------
    mappable : instance of object implementing "mappable" interface
        E.g., instance of Image, ContourSet, etc. -- basically any Artist subclass that
        inherits from the ScalarMappable mixin
        https://matplotlib.org/api/cm_api.html

    loc : str, optional
        location for colorbar -- one of "right", "left", "top", "bottom"

    size : str, optional
        relative size for colorbar as fraction of main plot, as a percentage (e.g. "2%")

    pad : float, optional
        padding between colorbar and main plot

    label_pad : str, optional
        padding between colorbar and its tick labels

    tick_lable_size : float, optional
        font size for tick labels

    Returns
    -------
    cbar : instance of matplotlib.colorbar.Colorbar
        The generated colorbar
    """
    if loc in ["top", "bottom"]:
        orient = "horizontal"
        if loc == "top":
            tickPos = 'top'
        else:
            tickPos = 'bottom'
    else:
        orient = "vertical"
        if loc == "left":
            tickPos = 'left'
        else:
            tickPos = 'right'
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cbar_axes = divider.append_axes(loc, size=size, pad=pad)
    cbar = fig.colorbar(mappable, cax=cbar_axes, orientation=orient)

    # fiddle with tick label locations
    if loc in ["left", "right"]:
        cbar_axis = cbar_axes.yaxis
    else:
        cbar_axis = cbar_axes.xaxis
    cbar_axis.set_ticks_position(tickPos)
    cbar_axis.set_label_position(tickPos)
    cbar.ax.tick_params(labelsize=tick_label_size, pad=label_pad)
    return cbar


def ExtractCenteredSubimage( imData, xc, yc, pix, width=None, height=None, verbose=False ):
    """Extracts and returns a subimage centered at xc,yc, along with 
    corresponding x and y position vectors. If width is None, then the full
    image is returned, along with the corresponding pixel vectors.
    
    Parameters
    ----------
    imData : 2D ndarray of int or float
        the input image
    
    xc, yc : float
        image pixel location (1-based coords) to center extraction around
    
    pix : float
        pixel scale (e.g., arcsec/pix)
    
    width : int or None, optional
        width of subimage to extract, in pixels
        if None, then the entire image is returned (along with xPos, yPos)
    
    height : int or None, optional
        height of subimage to extract, in pixels
        if None, then height = width

    verbose : bool, optional
    
    Returns
    -------
    (imdata_ext, xPos, yPos) : tuple of (2D ndarray, 1D ndarray, 1D ndarray; all of float)
        imData_ext = extracted subimage centered on xc,yc (or entire image if
            width = None)
        xPos = array of pixel coordinate values for x-axis, relative to xc
            e.g., [-1.0, 0.0, 1.0] for 3x3 image centered at 1,1 with pix=1
                  [-0.2, 0.0, 0.2] for 3x3 image centered at 1,1 with pix=0.2
        xPos = array of pixel coordinate values for y-axis, relative to yc
    """

    ySize, xSize = imData.shape
    xPos = pix*(np.arange(1.0, xSize + 1.0) - xc)
    yPos = pix*(np.arange(1.0, ySize + 1.0) - yc)

    if width is not None:
        if height is None:
            height = width
        halfwidth = int(0.5*width)
        x1 = int(xc - halfwidth - 1)
        if (x1 < 0):
            x1 = 0
        x2 = int(xc + halfwidth)
        if (x2 > xSize):
            x2 = -1
        halfheight = int(0.5*height)
        y1 = int(yc - halfheight - 1)
        if (y1 < 0):
            y1 = 0
        y2 = int(yc + halfheight)
        if (y2 > ySize):
            y2 = -1
        xPos = xPos[x1:x2]
        yPos = yPos[y1:y2]
        imdata_ext = imData[y1:y2,x1:x2]
        if verbose:
            print("   pu.ExtractCenteredSubimage: extracting imData[y1:y2,x1:x2] = imData[%d:%d,%d:%d]" % (y1,y2,x1,x2))
    else:
        imdata_ext = imData
    
    return (imdata_ext, xPos, yPos)


def nicecont( imageData, xc=0, yc=0, width=None, height=None, levels=None, pix=1.0,
            axisLabel="pixels", title=None, imageExt=0, log=False, offset=0.0, axesObj=None,
            labelSize=12, labelpad=5, printAxisLabels="both", noErase=False, extraLevels=None,
            color='k', extraColor='r', linewidth=0.5, linestyle='-', secondScale=None,
            secondLabel='kpc', verbose=False ):
    """
    Function which contour-plots an image.
    
    Parameters
    ----------
    imageData : 2D ndarray or str
        2D Numpy array OR FITS image filename (image data is assumed to be
        in 0th header-data unit, unless imageExt is set to something else)
    
    xc, yc : int
        optional center for axes (e.g., center of galaxy) -- by default, these
        are assumed to be IRAF-style 1-based coordinates!
    
    width, height : int
        width and height of subimage (centered on xc,yc) to be plotted;
        if height=None, then a square subimage of size width x width will be extracted
        
    levels : sequence (tuple, list, or Numpy array) of float or None, optional
        contour intensity levels to be plotted (if log=True, then these should be 
        log10 of the original values)
    
    pix : float, optional
        pixel scale (e.g., arcsec/pix or kpc/pix, for axis labeling)
    
    axisLabel : str, optional
        label for x and y axes
    
    title : str, optional
        title for plot
    
    imageExt = int or str, optional
        specification of a different header-data unit in input FITS image 
        (if imageData points to a file)
    
    log : bool, optional
        if True, convert image data to log10(data)

    offset : float, optional
        additive offset to be applied to data (*after* taking log10, if requested)

    axesObj : instance of matplotlib.axes.Axes, optional
        Axes instance to receive the plotting commands

    labelSize : float, optional
        font sizes of x- and y-axis labels
    
    labelpad : float, optional
        shifts position of axis label relative to axis [default=5]

    printAxisLabels : str, optional
        ["both" or "xy", "x", "y"] -- specifies which, if any, of the
        x- or y-axis labels to print

    extraLevels = a list of one or more contour intensity levels to overplot in a
        different color

    color = color for the contours
    
    extraColor = color for contours specified by extraLeveles

    noErase = set this equal to True to draw the contours into an existing plot
        window without erase things first (only used if axesObj is None)

    linewidth = float
    
    linestyle = one of 'solid', 'dashed', 'dashdot', 'dotted'
    
    secondScale = float
        If set, then a second axis scale is drawn (e.g., for pc or kpc)
        value = conversion from original scale (e.g., kpc/arcsec)
    
    secondLabel = str [default = 'kpc']
        label for axis with second scale

    Example:
        >>> nicecont("image.fits", xc=202.4, yc=500.72, levels=np.arange(1.0, 20.0, 0.5))
    """

    # handle case of user supplying a FITS filename
    if type(imageData) == str:
        hdulist = fits.open(imageData)
        imData = hdulist[imageExt].data
    else:
        imData = imageData

    if log is True:
        imData = np.log10(imData)
    imData = imData + offset

    # determine xPos,yPos and extract centered subimage, if requested
    (imData, xPos, yPos) = ExtractCenteredSubimage(imData, xc, yc, pix, width, height,
                                                    verbose=verbose)

    if axesObj is None:
        if noErase is False:
            plt.clf()
        if levels is not None:
            plt.contour(xPos, yPos, imData, levels, colors=color, linewidths=linewidth,
                        linestyles=linestyle)
        else:
            plt.contour(xPos, yPos, imData, colors=color, linewidths=linewidth,
                        linestyles=linestyle)
        if extraLevels is not None:
            plt.contour(xPos, yPos, imData, extraLevels, colors=extraColor, linewidths=1.0,
                        linestyles=linestyle)
        plt.gca().set_aspect('equal')
        if axisLabel is not None:
            if printAxisLabels in ["both", "xy", "x"]:
                plt.xlabel(axisLabel, fontsize=labelSize)
            if printAxisLabels in ["both", "xy", "y"]:
                plt.ylabel(axisLabel, fontsize=labelSize)
        if title is not None:
            plt.title(title)

        if secondScale is not None:
            yrange_orig = np.array(plt.ylim())
            yrange_second = yrange_orig * secondScale
            topy = plt.twinx()
            topy.tick_params(axis='y', length=10)
            topy.tick_params(axis='y', length=5, which="minor")
            topy.set_ylim(yrange_second[0], yrange_second[1])
            plt.ylabel(secondLabel, fontsize=labelSize)
            plt.show()

    else:    # user supplied a matplotlib.axes.Axes object to receive the plotting commands
        if levels is not None:
            axesObj.contour(xPos, yPos, imData, levels, colors=color, linewidths=linewidth,
                            linestyles=linestyle)
        else:
            axesObj.contour(xPos, yPos, imData, colors=color, linewidths=linewidth,
                            linestyles=linestyle)
        if extraLevels is not None:
            plt.contour(xPos, yPos, imData, extraLevels, colors=extraColor, linewidths=0.75,
                        linestyles=linestyle)
        axesObj.set_aspect('equal')
        if axisLabel is not None:
            if printAxisLabels in ["both", "xy", "x"]:
                axesObj.set_xlabel(axisLabel, fontsize=labelSize, labelpad=labelpad)
            if printAxisLabels in ["both", "xy", "y"]:
                axesObj.set_ylabel(axisLabel, fontsize=labelSize, labelpad=labelpad)
        if title is not None:
            axesObj.set_title(title)

        if secondScale is not None:
            xrange_orig = np.array(plt.xlim())
            xrange_second = xrange_orig * secondScale
            topx = plt.twiny()
            topx.tick_params(length=10)
            topx.tick_params(length=5, which="minor")
            topx.set_xlim(xrange_second[0], xrange_second[1])
            plt.xlabel(secondLabel, fontsize=labelSize)
            plt.show()



def ExtractCenteredSubimage( imData, xc, yc, pix, width=None, height=None, verbose=False ):
    """Extracts and returns a subimage centered at xc,yc, along with 
    corresponding x and y position vectors. If width is None, then the full
    image is returned, along with the corresponding pixel vectors.
    
    Parameters
    ----------
    imData : 2D ndarray of int or float
        the input image
    
    xc, yc : float
        image pixel location (1-based coords) to center extraction around
    
    pix : float
        pixel scale (e.g., arcsec/pix)
    
    width : int or None, optional
        width of subimage to extract, in pixels
        if None, then the entire image is returned (along with xPos, yPos)
    
    height : int or None, optional
        height of subimage to extract, in pixels
        if None, then height = width

    verbose : bool, optional
    
    Returns
    -------
    (imdata_ext, xPos, yPos) : tuple of (2D ndarray, 1D ndarray, 1D ndarray; all of float)
        imData_ext = extracted subimage centered on xc,yc (or entire image if
            width = None)
        xPos = array of pixel coordinate values for x-axis, relative to xc
            e.g., [-1.0, 0.0, 1.0] for 3x3 image centered at 1,1 with pix=1
                  [-0.2, 0.0, 0.2] for 3x3 image centered at 1,1 with pix=0.2
        xPos = array of pixel coordinate values for y-axis, relative to yc
    """

    ySize, xSize = imData.shape
    xPos = pix*(np.arange(1.0, xSize + 1.0) - xc)
    yPos = pix*(np.arange(1.0, ySize + 1.0) - yc)

    if width is not None:
        if height is None:
            height = width
        halfwidth = int(0.5*width)
        x1 = int(xc - halfwidth - 1)
        if (x1 < 0):
            x1 = 0
        x2 = int(xc + halfwidth)
        if (x2 > xSize):
            x2 = -1
        halfheight = int(0.5*height)
        y1 = int(yc - halfheight - 1)
        if (y1 < 0):
            y1 = 0
        y2 = int(yc + halfheight)
        if (y2 > ySize):
            y2 = -1
        xPos = xPos[x1:x2]
        yPos = yPos[y1:y2]
        imdata_ext = imData[y1:y2,x1:x2]
        if verbose:
            print("   pu.ExtractCenteredSubimage: extracting imData[y1:y2,x1:x2] = imData[%d:%d,%d:%d]" % (y1,y2,x1,x2))
    else:
        imdata_ext = imData
    
    return (imdata_ext, xPos, yPos)


def PlotImage( imageData, xc=0, yc=0, width=None, height=None, zrange=None, cmap="jet",
            pix=1.0, axisLabel="pixels", title=None, imageExt=0, log=False, axesObj=None,
            labelSize=12, tickLabelSize=11, printAxisLabels="both", colorbar=True, colorbarLoc="right", 
            colorbarLabel=None, cbarLabelSize=11, cbarTickLabelSize=10, noErase=False ):
    """Function which plots an image, along with axis tick marks and labels and
    (optionally) a colorbar.
    
    Parameters
    ----------
    imageData = 2D Numpy array OR FITS image filename (image data is assumed to be
            in 0th header-data unit, unless imageExt is set to something else)
    
    xc, yc : float, optional
        center for axes (e.g., center of galaxy) in 1-based pixel coords
    
    width, height: int, optional
        width and height in pixels of subimage (centered on xc,yc) to be displayed; 
        if height=None, then a square subimage of size width x width will be extracted
    
    zrange : 2-element sequence of float, optional
        two-element list/tuple/array containing lower and upper limits of data
            values (values below/above these limits will be clipped to the limits)
    
    cmap: str, optional
        specification of colormap which maps pixel values to on-screen colors;
        default is to use matplotlib's "jet" colormap)
    
    pix: float, optional
        arcsec/pixel scale of image (for axis labeling)
    
    axisLabel: str, optional
        label for x and y axes
    
    title: str, optional
        title for plot
    
    imageExt : int or str, optional
        specification of a header-data unit within the input FITS file 
        (if imageData points to a file)
    
    log : bool, optional
        if True, then convert image data to log10(data)

    axesObj : matplotlib.axes.Axes instance, optional
        Axes instance to receive plotting commands
    
    labelSize : float, optional
        sizes of x- and y-axis labels

    printAxisLabels = ["both" or "xy", "x", "y"] -- specifies which, if any, of the
                x- or y-axis labels to print
    
    colorbar : bool, optional
        if True (default), then a colorbar is drawn next to the image
    
    colorbarLoc : str or None, optional
        if not None, then this specifies the colorbar location:
        one of ["top", "bottom", "left", "right"]
        
    colorbarLabel : str or None, optional
        if not None, then this is the label for the colorbar
    
    cbarLabelSize : float, optional
        font size for colorbar label
    
    cbarTickLabelSize : float, optional
        font size for colorbar tick labels
    
    noErase : bool, optional
        If true *and* axesObj is None (standard plotting to separate figure), then
        cf() is *not* called first. If axesObj is not None, then this is ignored
        (we assume user wants to draw image on top of pre-existing stuff)

    Returns
    -------
    axesImg : instance of matplotlib.image.AxesImage
    
    
    Example:
        >>> PlotImage("image.fits", xc=202.4, yc=500.72))
    """

    # handle case of user supplying a FITS filename
    if type(imageData) == str:
        hdulist = fits.open(imageData)
        imData = hdulist[imageExt].data
    else:
        imData = imageData
    if log is True:
        imData = np.log10(imData)

    # determine xPos,yPos and extract centered subimage, if requested
    (imData, xPos, yPos) = ExtractCenteredSubimage(imData, xc, yc, pix, width, height, verbose=False)

    # define x-axis and y-axis ranges for labeling purposes
    xtent = np.array([xPos[0], xPos[-1], yPos[0], yPos[-1]])

    zmin = zmax = None
    if zrange is not None:
        if log is True:
            zmin = math.log10(zrange[0])
            zmax = math.log10(zrange[1])
        else:
            zmin = zrange[0]
            zmax = zrange[1]

    if cbarLabelSize is None:
        cbarLabelSize = labelSize

    if axesObj is None:
        if noErase is False:
            plt.clf()
        axesImg = plt.imshow(imData, interpolation="nearest", origin="lower", extent=xtent,
                            vmin=zmin, vmax=zmax, aspect="equal", cmap=cmap)
        if tickLabelSize is not None:
            # axesImg is an AxesImage object, so we have to query its ax() method to
            # get the proper Axes object
            axesObj = axesImg.axes
            SetAxesObjTickLabelSize(axesObj, tickLabelSize)
        if axisLabel is not None:
            if printAxisLabels in ["both", "xy", "x"]:
                print("hi there!")
                plt.xlabel(axisLabel, fontsize=labelSize)
            if printAxisLabels in ["both", "xy", "y"]:
                plt.ylabel(axisLabel, fontsize=labelSize)
        if title is not None:
            plt.title(title)
        if colorbar is True:
            ax = plt.gca()
            cbar = add_colorbar(axesImg, loc=colorbarLoc, size="5%", pad=0.05, label_pad=2,
                                tick_label_size=cbarTickLabelSize)
            cbar.solids.set_edgecolor("face")  # Remove gaps in PDF http://stackoverflow.com/a/15021541
            if colorbarLabel is not None:
                cbar.set_label(colorbarLabel, fontsize=cbarLabelSize)
            plt.sca(ax)  # Activate main plot before returning

    else:
        axesImg = axesObj.imshow(imData, interpolation="nearest", origin="lower", extent=xtent,
                                vmin=zmin, vmax=zmax, aspect="equal", cmap=cmap)
        if tickLabelSize is not None:
            SetAxesObjTickLabelSize(axesObj, tickLabelSize)
        if axisLabel is not None:
            if printAxisLabels in ["both", "xy", "x"]:
                axesObj.set_xlabel(axisLabel, fontsize=labelSize)
            if printAxisLabels in ["both", "xy", "y"]:
                axesObj.set_ylabel(axisLabel, fontsize=labelSize)
        if title is not None:
            axesObj.set_title(title)
        if colorbar is True:
            cbar = add_colorbar(axesImg, loc=colorbarLoc, size="5%", pad=0.05, label_pad=2,
                                tick_label_size=cbarTickLabelSize)
            cbar.solids.set_edgecolor("face")  # Remove gaps in PDF http://stackoverflow.com/a/15021541
            if colorbarLabel is not None:
                cbar.set_label(colorbarLabel, fontsize=cbarLabelSize)
            plt.sca(axesObj)  # Activate main plot before returning

    return axesImg



def DrawPALine( PA, radius, fmt='g-', color=None, linewidth=1.0, xc=0.0, yc=0.0,
                addDots=False, dot_ms=6, alpha=1.0, axesObj=None ):
    """Given a pre-existing plot, draws a line passing through the central
    coordinates (by default, center = 0,0 in data coordinates) at PA = PA
    relative to *+y axis*, with radius = radius.

        PA = position angle CCW from +y axis
        radius = radial length of line (data units)
        fmt = matplotlib format string for line
        color = optional color specification
        linewidth = matplotlib linewidth specification
        xc, yc = coordinates for center of line (data units)
        addDots = if True, small circles are drawn at either end of the line
        dot_ms = markersize value for dots (if addDots is True)
        axesObj = optional matplotlib.axes object, specifying which axes gets
            the ellipse drawn into it
    """

    if (PA < 0) or (PA > 180):
        print("PA must lie between 0 and 180 degrees!")
        return None
    PA_x = -PA
    dx = radius * math.sin(math.radians(PA_x))
    dy = radius * math.cos(math.radians(PA_x))
    vertical = False
    if (dx == 0.0):
        vertical = True
    else:
        slope = dy/dx
    xx = [xc + dx, xc - dx]
    yy = [yc + dy, yc - dy]

    if axesObj is None:
        ax = plt.gca()
    else:
        ax = axesObj
    
    if color is None:
        color = fmt[0]
    linestyle = fmt[1]
    if vertical:
        axvline(xc, color=color, linestyle=linestyle, linewidth=linewidth, alpha=alpha)
    else:
        ax.axline((xc,yc), slope=slope, color=color, linestyle=linestyle, linewidth=linewidth, alpha=alpha)
    if addDots is True:
        ax.plot(xx,yy, color + "o", ms=dot_ms, alpha=alpha)


def DrawEllipse( PA, a, ell, edgecolor='g', linestyle='-', linewidth=1.0, fillColor=None,
                alpha=1.0, xc=0.0, yc=0.0, axesObj=None ):
    """Given a pre-existing plot, draws an ellipse with semi-major axis a and
    ellipticity ell, with major axis at PA = PA relative to +y-axis,
    centered on coordinates (xc,yc) (by default, = 0,0 in data coordinates).

        PA = position angle CCW from +y axis
        a = semi-major axis of ellipse (data units)
        ell = ellipticity (1 - b/a) of ellipse
        edgecolor = color for ellipse outline
        linestyle, linewidth = matplotlib specification for ellipse outline
        fillColor = if not None, then the ellipse is filled using the
            specified color
        fmt = matplotlib format string for line
        color = optional color specification
        linewidth = matplotlib linewidth specification
        xc, yc = coordinates for center of line (data units)
        xc, yc = coordinates for center of ellipse (data units)
        axesObj = optional matplotlib.axes object, specifying which axes gets
            the ellipse drawn into it
    """

    if axesObj is None:
        ax = plt.gca()
    else:
        ax = axesObj
    b = (1 - ell)*a
    if fillColor is None:
        faceColor = 'None'
    else:
        faceColor = fillColoir
    ellPatch = Ellipse((xc,yc), 2*b, 2*a, angle=PA, facecolor=faceColor, edgecolor=edgecolor,
                        linestyle=linestyle, linewidth=linewidth, alpha=alpha)
    ax.add_patch(ellPatch)



def ExtractProfile( imdata, x0,y0, x1,y1, width=1 ):
    """
    This uses skimage.measure.profile_line to extract a profile from pixel
    coordinate (x0,y0) to pixel coordinate (x1,y1)
    
    This function uses IRAF coordinates (1-based, x = column number)
    
    Parameters
    ----------
    imdata : 2D ndarray of float
        image data array

    x0 : int or float
        x-coordinate of start position (1-based)

    y0 : int or float
        y-coordinate of start position (1-based)

    x1 : int or float
        x-coordinate of end position (1-based)

    y1 : int or float
        y-coordinate of end position (1-based)
    
    width : int, optional
        width of profile (perpendicular to profile) in pixels

    Returns
    -------
    (rr, ii) : tuple of 1D ndarray of float
        rr = radius vector (r = 0 at start position)
        ii = intensity vector
    """

    # switch x,y to numpy's y,x and switch to 0-based counting
    try:
        ii = profile_line(imdata, (y0 - 1, x0 - 1), (y1 - 1, x1 - 1), linewidth=width, 
                        reduce_func=np.nanmean)
    except RuntimeWarning:
        # we don't care whether all the values in a bin were NaN
        pass
    npts = len(ii)
    rr = np.linspace(0, npts - 1, npts)
    return rr,ii


def GetProfileAtAngle( imdata, xc,yc, angle, radius, width=1 ):
    """
    Returns a 1D profile cut through an image at specified angle, extending to
    specified radius.
    Note: this is designed to imitate pvect, so angles are measured CCW from +x axis!
    
    This function uses IRAF coordinates (1-based, x = column number)

    Parameters
    ----------
    imdata : 2D ndarray of float
        image data array
    
    xc : int or float
        x-coordinate of center to extract profile from (IRAF ordering, 1-based)
    
    yc : int or float
        y-coordinate of center to extract profile from (IRAF ordering, 1-based)
    
    angle : float
        angle measured CCW from +x axis, in degrees
    
    radius : int
        length of profile, in pixels
    
    width : int, optional
        width of profile (perpendicular to profile) in pixels
    
    Returns
    -------
    rr,ii : tuple of 1D ndarray of float
        rr = array of radius values (= 0 at (xc,yc))
        ii = data pixel values along profile [= Nan if all pixels for that bin
                were masked]
    """
    angle_rad = math.radians(angle)
    x_end = xc + math.cos(angle_rad) * radius
    y_end = yc + math.sin(angle_rad) * radius
    x_start = xc - math.cos(angle_rad) * radius
    y_start = yc - math.sin(angle_rad) * radius
    rr,ii = ExtractProfile(imdata, x_start,y_start, x_end,y_end, width=width)
    rr = rr - radius
    return rr, ii
