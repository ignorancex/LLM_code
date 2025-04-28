import readfile

for c in ('red', 'blue'):
    data = readfile.table('centrals_mandelbaum_2015_%s.csv' %c)
    lo = readfile.table('centrals_mandelbaum_2015_%s_lo.csv' %c)
    hi = readfile.table('centrals_mandelbaum_2015_%s_hi.csv' %c)
    x = data[0]
    y = data[1]
    ylo = data[1] - lo[1]
    yhi = hi[1] - data[1]
    output = 'centrals_mandelbaum2015_%s.txt' %c
    out = open(output, 'w')
    print >>out, '# Mstar  Mtot  Mtot_lo  Mtot_hi'
    for i in zip(x, y, ylo, yhi):
        print >>out, '{0:.2e}  {1:.2e}  {2:.2e}  {3:.2e}'.format(*i)
    out.close()