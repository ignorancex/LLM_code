/* 
	General raster-image program.
	Postscript version.

		::: Initialization :::
scan_begin()
	initializes scanning.

scan_end()
	writes raster to file.

		::: Coloring routines :::

scan_set(z,c)
	color pixel at location z with color c.

scan(color)
	creates an image where each point in the window
	is painted color(z) (which should be in 0-255).

Remark:  if a coloring routine is called before
scan_begin(), then scan_begin() is executed anyway.
If the routine is scan itself, then scan_end() is
called at the end.  Thus a single call to scan()
creates a raster image file.

		::: Parameters :::

scan_window(c,r) 
	sets the complex center and radius of the window
	(default (0,0), r=2).

scan_rgb(i,r,g,b)
	sets color i in the rgb color map

scan_hls(i,h,l,s)
	sets color i using hsb coordinates

scan_dim(i,j)
	dimension picture to ixj (default= full screen, 1152x900).
	If the dimensions are not square, the indicated
square window is included and the scan is expanded to
cover the rest of the image.

scan_pix_radius(r)
	double *r;
	Returns the radius of a single pixel for
	the current window and dimensions.

*/

#include <stdlib.h>
#include <stdio.h>
#include "cx.h"
#include "k3.h"

/* Postscript page size and origin */
#define PAGEX 540.0
#define PAGEY 720.0
#define CORNX 36
#define CORNY 36
#define LINE 72		/* Image line length */

/* Default dimensions */
#define CENTER {0.0,0.0}
#define READY 0

/* Color maps */
#define NONE 0
#define GRAY 1
#define COLOR 200

static int cmap=NONE;
static int ready=0;
static int idim=IDIM, jdim=JDIM;
static complex center=CENTER;
static double radius=RADIUS;
complex delta, dim, start;
char *picture;
int red[256], green[256], blue[256];

void scan_dim(i,j)
	int i,j;
{
	idim=i;
	jdim=j;
}

void scan_window(c,r)
	complex c;
	double r;
{
	center = c;
	radius = r;
}

void scan_begin()
{
	int i;

	if(idim > jdim)
	{	dim.y = radius;
		dim.x = (dim.y*idim)/jdim;
	} else
	{	dim.x = radius;
		dim.y = (dim.x*jdim)/idim;
	}
	start.x = center.x - dim.x;
	start.y = center.y + dim.y;
	delta.x =  2*dim.x/idim;
	delta.y = -2*dim.y/jdim;
	picture = malloc(idim*jdim);
	for(i=0; i<idim*jdim; i++) picture[i]=0;
	ready=1;
	if(cmap != NONE) return;
	cmap = GRAY;	/* Default color map */
	for(i=0; i<256; i++) red[i] = i;
}

void scan_pix_radius(r)
	double *r;
{
	*r = (idim > jdim) ? radius/jdim : radius/idim;
}

void scan_set(z,c)
	complex z;
	int c;
{
	int i, j;

	if(!ready) scan_begin();
	i = (z.x-start.x)/delta.x;
	j = (z.y-start.y)/delta.y;
	if(i>=0 && i < idim &&
	   j>=0 && j < jdim) picture[i+idim*j]=c;
}

void scan_gray(i,gray)
	int i;
	double gray;
{
	red[i] = 255*gray;
	cmap = GRAY;
}

void scan_rgb(i,r,g,b)
	int i;
	int r, g, b;
{
	red[i] = r; green[i] = g; blue[i] = b;
	cmap = COLOR;
}

void scan_hls(i,h,l,s)
	int i;
	double h, l, s;
{
	int r, g, b;

	hlsrgb(h,l,s,&r,&g,&b);
	red[i] = r; green[i] = g; blue[i] = b;
	cmap = COLOR;
}
	
void scan(color)
	int (*color)();
{

	int i,j;
	complex z;

	if(!ready) scan_begin();
	z.x = start.x;
	for (i=0; i<idim; i++)
	{
		z.y = start.y; 
		for (j=0; j<jdim; j++)
		{
			picture[i+idim*j]=color(z);
			z.y += delta.y;
		}
		z.x += delta.x;
	}
}

void scan_end(argc,argv)
	int argc;
	char *argv[];
{
	int i;
	int cornx, corny, marginx, marginy, sx, sy;
	double tx, ty, t;

	printf("%%!\n"); 
	printf("%%%% Dynamics Scan Version 1.1\n");
	printf("%%%% Scan window: %.5lf %.5lf %.5lf %.5lf\n",
	center.x-radius,center.y-radius,
	center.x+radius,center.y+radius);
	printf("%%%% Produced by command line:\n");
	printf("%%%% ");
	for(i=0; i<argc; i++)
		printf("%s ",argv[i]);
	printf("\n%%%%\n"); 

	sx = idim;
	sy = jdim;

	cornx = CORNX;
	corny = CORNY;

	printf("%%%%BoundingBox: %d %d %d %d\n",
		cornx, corny, cornx+sx, corny+sy);
	printf("%d %d translate\n",cornx,corny);
	printf("%d %d scale\n",sx,sy);
	if(cmap == GRAY) ps_image();
	if(cmap == COLOR) ps_colorimage();
	printf("\n");
	printf("showpage\n");
	ready=0;
}

void ps_image()
{
	int i, j;
	unsigned char k;

	printf("/linebuf %d string def\n",
		idim);
	printf("%d %d 8\n",idim,jdim);
	printf("[%d 0 0 -%d 0 %d]\n",
		idim, jdim, jdim);
	printf("{currentfile linebuf readhexstring pop}\n");
	printf("image\n");
	for (j=0; j<jdim; j++)
	for (i=0; i<idim; i++)
	{	k = picture[i+idim*j];
		printhex(red[k]);
	}
}

void ps_colorimage()
{
	int i, j;
	unsigned char k;

	printf("/linebuf %d string def\n",
		3*idim);
	printf("%d %d 8\n",idim,jdim);
	printf("[%d 0 0 -%d 0 %d]\n",
		idim, jdim, jdim);
	printf("{currentfile linebuf readhexstring pop}\n");
	printf("false 3 colorimage\n");
	for (j=0; j<jdim; j++)
	for (i=0; i<idim; i++)
	{	k = picture[i+idim*j];
		printhex(red[k]);
		printhex(green[k]);
		printhex(blue[k]);
	}
}

void printhex(s)
	int s;
{	
	static int chars=0;

	s = s & 0xff;
	if(s < 16) printf("0%1x",s);
	else printf("%2x",s);
	chars += 2;
	if(LINE <= chars) {printf("\n"); chars=0;}
}

/* Convert hue/lightness/saturation to rgb */
/* Domain is [0,360] x [0,100] x [0,100]. */
/* Range is [0,255] x [0,255] x [0,255]. */

void hlsrgb(h, l, s, r, g, b)
	double h, l, s;
	int *r, *g, *b;
{
	double M, m;
	double floor(), bound(), gun();

	h = h - floor(h/360) * 360;
	l = bound(0.0, l/100, 1.0);
	s = bound(0.0, s/100, 1.0);
	M = l + s*(l > 0.5 ? 1.0 - l : l);
	m = 2*l - M;
	*r = 255*gun(m, M, h);
	*g = 255*gun(m, M, h-120);
	*b = 255*gun(m, M, h-240);
}

double
gun(m, M, h)
double m, M, h;
{
	if(h < 0)
		h += 360;
	if(h < 60)
		return(m + h*(M-m)/60);
	if(h < 180)
		return(M);
	if(h < 240)
		return(m + (240-h)*(M-m)/60);
	return(m);
}

double
bound(low, x, high)
double low, x, high;
{
	if(x < low)
		return(low);
	if(x > high)
		return(high);
	return(x);
}
