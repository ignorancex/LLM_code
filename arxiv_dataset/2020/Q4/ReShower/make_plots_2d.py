#!/usr/bin/env python

import os
import sys

from ROOT import *

gROOT.LoadMacro("AtlasStyle.C")
gROOT.LoadMacro("AtlasUtils.C")
SetAtlasStyle()

gStyle.SetOptStat(0)
gROOT.SetBatch(1)


def DrawCutLines(m, tau):
    l = TLine()
    l.SetLineWidth(2)
    l.SetLineColor(kRed)
    l.SetLineStyle(kSolid)

    # mass window
    l.DrawLine(tau[0], m[0], tau[1], m[0])
    l.DrawLine(tau[0], m[1], tau[1], m[1])

    # substructure variable
    l.DrawLine(tau[0], m[0], tau[0], m[1])
    l.DrawLine(tau[1], m[0], tau[1], m[1])


f_higgs_low = TFile.Open("histograms/histograms.pid_25.pt_250.root")
f_higgs_mid = TFile.Open("histograms/histograms.pid_25.pt_400.root")
f_higgs_hi = TFile.Open("histograms/histograms.pid_25.pt_800.root")

f_top_low = TFile.Open("histograms/histograms.pid_6.pt_350.root")
f_top_mid = TFile.Open("histograms/histograms.pid_6.pt_500.root")
f_top_hi = TFile.Open("histograms/histograms.pid_6.pt_800.root")

obs = "ljet_m_vs_tau21"
h_higgs = {
    'low': f_higgs_low.Get(obs).Clone("%s_low" % obs),
    'mid': f_higgs_mid.Get(obs).Clone("%s_mid" % obs),
    'hi': f_higgs_hi.Get(obs).Clone("%s_hi" % obs)
}
# for h in h_higgs.values():
#    h.GetYaxis().SetTitle("Jet mass [GeV]")

obs = "ljet_m_vs_tau32"
h_top = {
    'low': f_top_low.Get(obs).Clone("%s_low" % obs),
    'mid': f_top_mid.Get(obs).Clone("%s_mid" % obs),
    'hi': f_top_hi.Get(obs).Clone("%s_hi" % obs)
}

c = TCanvas("C", "C", 600, 600)

gStyle.SetPalette(kGreyScale)
TColor.InvertPalette()

txt = TLatex()
txt.SetTextFont(42)
txt.SetNDC()

gPad.SetLeftMargin(0.15)
gPad.SetRightMargin(0.05)
h_higgs['low'].GetYaxis().SetTitleOffset(1.4)
h_higgs['low'].DrawNormalized("col")
DrawCutLines(m=(95, 155), tau=(0., 0.5))
txt.DrawLatex(0.7, 0.85, "H#rightarrow b#bar{b}")
txt.DrawLatex(0.7, 0.80, "Low p_{T}")
gPad.RedrawAxis()
c.SaveAs("plots/higgs_m_vs_tau21_low.png")

h_higgs['mid'].GetYaxis().SetTitleOffset(1.4)
h_higgs['mid'].DrawNormalized("col")
DrawCutLines(m=(95, 155), tau=(0., 0.5))
txt.DrawLatex(0.7, 0.85, "H#rightarrow b#bar{b}")
txt.DrawLatex(0.7, 0.80, "Mid p_{T}")
gPad.RedrawAxis()
c.SaveAs("plots/higgs_m_vs_tau21_mid.png")

h_higgs['hi'].GetYaxis().SetTitleOffset(1.4)
h_higgs['hi'].DrawNormalized("col")
DrawCutLines(m=(95, 155), tau=(0., 0.5))
txt.DrawLatex(0.7, 0.85, "H#rightarrow b#bar{b}")
txt.DrawLatex(0.7, 0.80, "High p_{T}")
gPad.RedrawAxis()
c.SaveAs("plots/higgs_m_vs_tau21_hi.png")


# Top quark

h_top['low'].GetYaxis().SetTitleOffset(1.4)
h_top['low'].DrawNormalized("col")
DrawCutLines(m=(142.5, 202.5), tau=(0., 0.6))
txt.DrawLatex(0.7, 0.85, "t#rightarrow bq#bar{q}\'")
txt.DrawLatex(0.7, 0.80, "Low p_{T}")
gPad.RedrawAxis()
c.SaveAs("plots/top_m_vs_tau32_low.png")

h_top['mid'].GetYaxis().SetTitleOffset(1.4)
h_top['mid'].DrawNormalized("col")
DrawCutLines(m=(142.5, 202.5), tau=(0., 0.6))
txt.DrawLatex(0.7, 0.85, "t#rightarrow bq#bar{q}\'")
txt.DrawLatex(0.7, 0.80, "Mid p_{T}")
gPad.RedrawAxis()
c.SaveAs("plots/top_m_vs_tau32_mid.png")

h_top['hi'].GetYaxis().SetTitleOffset(1.4)
h_top['hi'].DrawNormalized("col")
DrawCutLines(m=(142.5, 202.5), tau=(0., 0.6))
txt.DrawLatex(0.7, 0.85, "t#rightarrow bq#bar{q}\'")
txt.DrawLatex(0.7, 0.80, "High p_{T}")
gPad.RedrawAxis()
c.SaveAs("plots/top_m_vs_tau32_hi.png")
