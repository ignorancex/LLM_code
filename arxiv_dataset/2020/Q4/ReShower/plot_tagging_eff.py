#!/usr/bin/env python

import os
import sys
import glob
from ROOT import *

gROOT.LoadMacro("AtlasStyle.C")
gROOT.LoadMacro("AtlasUtils.C")
SetAtlasStyle()

gStyle.SetOptStat(0)
gROOT.SetBatch(1)

pid = 6
if len(sys.argv) > 1:
    pid = int(sys.argv[1])

if pid == 6:
    hname = "ljet_topTag"
elif pid == 25:
    hname = "ljet_higgsTag"
else:
    print "ERROR: unknown pid", pid
    exit(1)

files = glob.glob("histograms/histograms.pid_%s.*.root" % pid)

points = []
for filename in files:
    f = TFile.Open(filename)
    h = f.Get(hname)
    pt = float(filename.split('.')[2].split('_')[1])
    eff = h.GetBinContent(2)
    unc = h.GetBinError(2)
    points += [(pt, eff, unc)]
    f.Close()
points.sort(key=lambda point: point[0])
print points

g = TGraphErrors()
ipoint = 0
for point in points:
    pt = point[0]
    eff = point[1]
    unc = point[2]
    g.SetPoint(ipoint, pt, eff)
    g.SetPointError(ipoint, 0., unc)
    ipoint += 1

c = TCanvas("C", "C", 1200, 800)
frame = TH1F(
    "frame", ";Large-R jet p_{T} [GeV]", 20, 200., 1200.)
frame.Draw()

frame.GetYaxis().SetTitle("Tagging efficiency")
frame.GetYaxis().SetTitleOffset(1.0)

gPad.SetTopMargin(0.05)
gPad.SetLeftMargin(0.10)
gPad.SetTopMargin(0.10)
g.Draw("lep")

t = TLatex()
t.SetNDC()
if pid == 6:
    t.DrawLatex(0.15, 0.80, "t#rightarrow bq#bar{q}\'")
    t.DrawLatex(0.15, 0.75, "|m-m_{top}| < 30 GeV, #tau_{32}<0.6")
elif pid == 25:
    t.DrawLatex(0.15, 0.80, "H#rightarrow b#bar{b}")
    t.DrawLatex(0.15, 0.75, "|m-m_{h}| < 30 GeV, #tau_{21}<0.5")

c.SaveAs("plots/eff_%i.png" % pid)
