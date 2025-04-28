from astropy.io import ascii
data_xray = ascii.read('table3_Xray_YSO.dat', format='fixed_width', delimiter=' ')


f_out = open('NGC1333_XRay_YSO.reg', 'w')#'wb')
f_out.write("# Region file format: DS9 version 4.1\n")
f_out.write("global color=green dashlist=8 3 width=1 font=\"helvetica 10 normal roman\" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\n")
f_out.write("fk5\n")
for i in range(len(data_xray['Src'])):
    # Src H MM SS.SS  dg mm ss.ss Cl  Flu  er
    ra_i = 15 * (data_xray['H'][i] + (data_xray['MM'][i] + 
                 data_xray['SS.SS'][i] / 60.) /60.)
    dec_i =  data_xray['dg'][i] + (data_xray['mm'][i] + 
                 data_xray['ss.ss'][i] / 60.) /60.
    src_i = data_xray['Src'][i]
    text_out = "circle({0}, {1}, 0.0027778) # color='red' ".format(ra_i, dec_i)
    text_out = text_out + " width=2 text={" + "{0}".format(src_i) + "}\n"
    f_out.write(text_out)
f_out.close()


#  18 3 28 43.26 +31 17 33.11 I    23   9
# # Region file format: DS9 version 4.1
# global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1
# fk5
# circle(52.3039167,31.3040417,10.550") # color=red width=2 text={source aa}

