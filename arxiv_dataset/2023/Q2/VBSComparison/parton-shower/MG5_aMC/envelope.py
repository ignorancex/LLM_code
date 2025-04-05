#!/usr/bin/python

## import modules
import os
import yoda
import glob
####################

## declare modes
# generate yodas
# generate rivet-mkhtml command
####################

import  optparse, sys, math

parser = optparse.OptionParser(usage=__doc__)
parser.add_option('-o', '--output', default='-', dest='OUTPUT_FILE')
parser.add_option('-c', '--central', default='-', dest='CENTRAL_FILE')
opts, fileargs = parser.parse_args()
            
####################

## Put the incoming objects into a dict from each path to a list of histos and scalings

filenames = [opts.CENTRAL_FILE]
for fa in fileargs:
  filenames += [fa]

# check for at least one input file
if len(filenames) < 1:
  print "need at least one input file"
  exit(1)

## generate all variations
if True:
  # from yoda-manipulate:
  scatters_all = {}
  scatters_pdf = {}
  scatters_scale = {}
  scatters_central = {}
  for filename in filenames:
    # print filename
    for p, ao in yoda.readYODA(filename).iteritems():
      if isinstance(ao, yoda.Scatter2D):
        scatter = ao
      elif isinstance(ao, yoda.Histo1D):
        scatter = yoda.Scatter2D(ao.path, ao.title)
        for bin in ao.bins:
          scatter.addPoint(yoda.Point2D(bin.xMid, bin.height, 0.5*bin.xWidth, bin.heightErr))
      else:
        print "cannot treat ", ao
        continue
      path = ao.path.split()
      scatter.path = path[0]
      if len(path) <= 1: #central
        scatters_central.setdefault(path[0], []).append(scatter)
        scatters_all.setdefault(path[0], []).append(scatter)
        scatters_pdf.setdefault(path[0], []).append(scatter)
        scatters_scale.setdefault(path[0], []).append(scatter)
      else:
        if path[1] == "PDF=":
          scatters_all.setdefault(path[0], []).append(scatter)
          scatters_pdf.setdefault(path[0], []).append(scatter)
          #if path[2] == "260000":
          #  scatters_central.setdefault(path[0], []).append(scatter)
        elif path[1] == "dyn=":
          scatters_all.setdefault(path[0], []).append(scatter)
          scatters_scale.setdefault(path[0], []).append(scatter)
        elif path[1] == "0":
          pass # ignore
        elif path[1] == "central":
          pass # ignore
        else:
          print "Unknown path object: ", ao.path

  # write out
  scatters_outall = {}
  scatters_outpdf = {}
  scatters_outscale = {}

  ## pdf
  for p, scatters in scatters_pdf.iteritems():
    for pcentral, scattercentral in scatters_central.iteritems():
      if pcentral == p:
        scatters_outpdf[scattercentral[0].path] = scattercentral[0].clone()
    for i, point in enumerate(scatters_outpdf[scatters[0].path].points):
      upper = float('-inf')
      lower = float('inf')
      sumy = 0
      sumysq = 0
      for scatter in scatters:
        val = scatter.points[i].y;
## NNPDF error is the standard deviation, not the envelope
#        if val > upper:
#          upper = val
#        if val < lower:
#          lower = val
        sumy += val
        sumysq += val**2
      n = len(scatters)
      err = math.sqrt(sumysq/n - (sumy/n)**2)
      point.yErrs = (err,err)
  # write envelopes
  outname = opts.OUTPUT_FILE.replace(".yoda","_pdf.yoda")
  yoda.write(scatters_outpdf.values(), outname)
#  ## pdf + stat
#  for p, scatters in scatters_pdf.iteritems():
#    for pcentral, scattercentral in scatters_central.iteritems():
#      if pcentral == p:
#        scatters_outpdf[scattercentral[0].path] = scattercentral[0].clone()
#    for i, point in enumerate(scatters_outpdf[scatters[0].path].points):
#      upper = float('-inf')
#      lower = float('inf')
#      for scatter in scatters:
#        val = scatter.points[i].yMax;
#        if val > upper:
#          upper = val
#        val = scatter.points[i].yMin;
#        if val < lower:
#          lower = val
#        point.yErrs = (point.y-lower, upper-point.y)
#  # write envelopes
#  outname = opts.OUTPUT_FILE.replace(".yoda","_pdfstat.yoda")
#  yoda.write(scatters_outpdf.values(), outname)
  ## scale
  for p, scatters in scatters_scale.iteritems():
    for pcentral, scattercentral in scatters_central.iteritems():
      if pcentral == p:
        scatters_outscale[scattercentral[0].path] = scattercentral[0].clone()
    for i, point in enumerate(scatters_outscale[scatters[0].path].points):
      upper = float('-inf')
      lower = float('inf')
      for scatter in scatters:
        val = scatter.points[i].y;
        if val > upper:
          upper = val
        if val < lower:
          lower = val
        point.yErrs = (point.y-lower, upper-point.y)
  outname = opts.OUTPUT_FILE.replace(".yoda","_scale.yoda")
  yoda.write(scatters_outscale.values(), outname)
#  ## scale + stat
#  for p, scatters in scatters_scale.iteritems():
#    for pcentral, scattercentral in scatters_central.iteritems():
#      if pcentral == p:
#        scatters_outscale[scattercentral[0].path] = scattercentral[0].clone()
#    for i, point in enumerate(scatters_outscale[scatters[0].path].points):
#      upper = float('-inf')
#      lower = float('inf')
#      for scatter in scatters:
#        val = scatter.points[i].yMax;
#        if val > upper:
#          upper = val
#        val = scatter.points[i].yMin;
#        if val < lower:
#          lower = val
#        point.yErrs = (point.y-lower, upper-point.y)
#  outname = opts.OUTPUT_FILE.replace(".yoda","_scalestat.yoda")
#  yoda.write(scatters_outscale.values(), outname)

  ## all -- linear combination of both
  scatters_outall = scatters_outscale
  for p, scatters in scatters_outall.iteritems():
    for ppdf, scatterpdf in scatters_outpdf.iteritems():
      if ppdf == p:
        break
    for i, point in enumerate(scatters_outall[p].points):
      lowers = scatters.points[i].yErrs[0]
      uppers = scatters.points[i].yErrs[1]
      lowerp = scatterpdf.points[i].yErrs[0]
      upperp = scatterpdf.points[i].yErrs[1]
      point.yErrs = (lowers+lowerp, uppers+upperp)
  # write envelopes
  outname = opts.OUTPUT_FILE.replace(".yoda","_all.yoda")
  yoda.write(scatters_outall.values(), outname)

