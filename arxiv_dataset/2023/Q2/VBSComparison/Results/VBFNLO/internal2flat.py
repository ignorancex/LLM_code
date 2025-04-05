#! /usr/bin/python

dataconversion = {
    "jetsinclusive" : "hist.njetincl.dat",
    "jetsexclusive" : "hist.njetexcl.dat",
    "pT_j1"         : "hist.ptj1.dat",
    "y_j1"          : "hist.yj1.dat",
    "pT_j2"         : "hist.ptj2.dat",
    "y_j2"          : "hist.yj2.dat",
    "pT_j3"         : "hist.ptj3.dat",
    "y_j3"          : "hist.yj3.dat",
    "m_ll"          : "hist.mll.dat",
    "m_jj"          : "hist.mjj.dat",
    "Deltay_jj"     : "hist.yjj.dat",
    "z_e"           : "hist.ze.dat",
    "z_mu"          : "hist.zmu.dat",
    "z_j3"          : "hist.zj3.dat"
    }

fout = open("LHC.dat",'w')

cs = 0
cserr = 0

for key in dataconversion:
  fout.write("""
# BEGIN HISTO1D /MC_VBSCAN_MCComparison/{0}
Path=/MC_VBSCAN_MCComparison/{0}
ScaledBy=1
Title=
XLabel=
YLabel=
# xlow   xhigh   val     errminus        errplus
""".format(key))
  fin = open(dataconversion[key],'r')
  for l in fin:
    entries = l.split()
    fout.write("{0}   {1}   {2}   {3}   {3}\n".format(entries[0],entries[1],entries[6],entries[7]))
    if key == "jetsinclusive" and float(entries[0]) == 1.5:
      cs = float(entries[6])
      cserr = float(entries[7])
  fout.write("# END HISTO1D\n")
  fin.close()

fout.write("""
# BEGIN HISTO1D /MC_VBSCAN_MCComparison/crosssection
Path=/MC_VBSCAN_MCComparison/crosssection
ScaledBy=1
Title=
XLabel=
YLabel=
# xlow   xhigh   val     errminus        errplus
0.000000e+00    1.000000e+06   {0}   {1}   {1}
# END HISTO1D
""".format(str(cs*1e-6),str(cserr*1e-6)))

fout.close()



