"""
This code is meant to be run within CASA with execfile('prepare_data.py')
In the middle, one has to run transform_to_K.py outside CASA

This code does a cut in region, so that we do not have so much nan values
Then does the moments in appropiate ranges.
"""
# cut image to leave out empty space

imagenamebase = "B5-NOEMA+30m-HC3N-10-9" # 
fitsimage = imagenamebase + ".fits"
cutimage = imagenamebase + "_cut.image"
cutfits = imagenamebase + "_cut.fits"
cutregion = "box [ [314pix, 322pix], [706pix, 702pix] ]"
kelvinfits = imagenamebase + "_cut_K.fits"

if not os.path.exists(cutfits):
  imsubimage(imagename=fitsimage, outfile=cutimage, region=cutregion)
  exportfits(imagename=cutimage, fitsimage=cutfits, velocity=True, dropdeg=True)
  os.system("rm -r " + cutimage)


if os.path.exists(kelvinfits):
# calculate moments in Kelvin ONLY IF THE FILE IN K exists, run transform_to_K_km_s.py to do so
  mom0image = imagenamebase + "_cut_K.mom0"
  mom1image = imagenamebase + "_cut_K.mom1"
  mom2image = imagenamebase + "_cut_K.mom2"
  mom8image = imagenamebase + "_cut_K.mom8"
  momimages = [mom0image, mom1image, mom2image, mom8image]
  moments = [0,1,2,8]
  channels = "19~28" #9.3 to 11.2 km/s for HC3N 10-9
  #channels = "16~24" #9.2 to 11.2 km/s for HC3N 8-7
  #channels = "17~24" # 9.2 to 11.0 for H2CO

  for moment, momimage in zip(moments, momimages):
    if not os.path.exists(momimage+".fits"):
      immoments(imagename=kelvinfits, moments=moment, chans=channels, outfile=momimage)
      exportfits(imagename=momimage, fitsimage=momimage+".fits", velocity=True, dropdeg=True)
      os.system("rm -r " + momimage)

else:
  print('Please run file transform_to_K.py before proceeding.')
