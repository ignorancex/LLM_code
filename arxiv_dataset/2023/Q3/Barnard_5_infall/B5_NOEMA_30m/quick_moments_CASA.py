imagename = 'B5-NOEMA+30m-H3CN-10-9.image'
outfile = 'B5-NOEMA+30m-H3CN-10-9'
moments = [0,1,2]
rms = 0.02284
chans_int = '19~28'

for moment in moments:
	immoments(imagename, outfile=outfile+'.mom'+str(moment), chans=chans_int, moments=moment)
	exportfits(imagename=outfile+'.mom'+str(moment), fitsimage=outfile+'_mom'+str(moment)+'.fits', dropdeg=True)
