import os
import shutil
from pathlib import Path

try:
    import numpy as np
except ImportError:
    print('Failed to import numpy')
    traceback.format_exc()
try:
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
except ImportError:
    print('Failed to import matplotlib')
    traceback.format_exc()

#from agginfo import *
from aggscatpy.agginfo import get_sizelist
from aggscatpy.agginfo import grs_voleq
from aggscatpy.agginfo import check_particle_type
from aggscatpy.agginfo import check_particle_size


def particle_rendering(partype,size,ireal,amon=None,xcamera=3.5,rotx=0.0,roty=0.0,rotz=0.0,\
        particle_color='rgb<0.15,0.15,0.15>',\
        background=False,bg_color='White',\
        reference=True,ref_length=None,ref_dist=1.7,ref_posang=240.0,ref_color=None,\
        ref_fontsize=1.0,fn=None,path=None):

    """
    Perform rendering of a particle shape via POVRAY

    Parameters
    ----------
    partype          : str
                       Particle type
    size             : str
                       Dust particle size
    amon             : str 
                       Monomer radius (necessary if you call aggregate files).
    ireal            : str 
                       Realization number
    xcamera          : float (optional)
                       Distance from the origin to the camera along the x-axis \
                               in units of the characteristic radius (for aggregates) \
                               or the volume-equivalent radius (for irregular grains).
    rotx, roty, rotz : float (optional)
                       Angle of rotation about the x, y, z axes, respectively (in degrees). 
    particle_color   : str (optional)
                       Particle color
    background       : bool (optional)
                       A background color is set when True
    bgcolor          : str (optional)
                       Background color 
    reference        : bool (optinal)
                       A reference bar is shown when True
    ref_length       : float (optional)
                       A physical length of reference bar in units of microns
    ref_dist         : float (optional)
                       A distance from the origin to the center of the reference bar in the image plane\
                               in units of the characteristic radius (for aggregates) \
                               or the volume-equivalent radius (for irregular grains).
    ref_posang       : float (optional)
                       A position angle of the reference bar center measured from the z axis \
                               in units of degrees
    ref_color        : str (optional)
                       A color of the reference bar and text
    ref_fontsize     : float (optional)
                       A fontsize relative to the default font size 
    fn               : str (optional)
                       A filename used for a generated png image
    path             : str (optional)
                       A path to which a generated image will be saved.

    """

    # check particle type
    check_particle_type(partype)

    # check particle size
    check_particle_size(partype,amon,size)
    
    # check realization index and if we have a povray script
    if partype=='grs':
        if Path('./aggregate.pov').exists()==False:
            msg="Povray script irregular.pov is not found. "\
                "Please place this file, which is provided in aggscatvir/python/povray, to the current directory"
            raise ValueError(msg)

        if ireal not in ['1','2','3','4','5','6','7','8','9','10']:
            msg='Incorrect realization index'
            raise ValueError(msg)
    else:
        if Path('./aggregate.pov').exists()==False:
            msg="Povray script irregular.pov is not found. "\
                "Please place this file, which is provided in aggscatvir/python/povray, to the current directory"
            raise ValueError(msg)

        if ireal not in ['1','2','3','4']:
            msg='Incorrect realization index'
            raise ValueError(msg)

    # reference bar color 
    if reference:
        if ref_color==None:
            ref_color='rgb<0,0,0>'
    else:
        ref_color='rgbt<0,0,0,1>'

    # make background transparent
    if background==False:
        bg_color='rgbt<0,0,0,1>' # for transparent

    if partype=='grs':

        Vrender=np.zeros(11)

        # original rendering volume of each grs grain 
        # in units of the volume before surface distortion applied.
        # [DO NOT CHANGE THESE VALUES]
        Vrender[0]=-9999    # this index not used. 
        Vrender[1]=1.06969
        Vrender[2]=1.10489
        Vrender[3]=1.04890
        Vrender[4]=1.01614
        Vrender[5]=1.10923
        Vrender[6]=1.18158
        Vrender[7]=1.09252
        Vrender[8]=1.02141
        Vrender[9]=1.07988
        Vrender[10]=1.02732

        # get volume equivalent radius
        av=grs_voleq(size)
        
        # length normalization
        anorm = av

        # scale the particle size
        av_scaled = av / Vrender[int(ireal)]**(1.0/3.0)

        #
        dustmodel='grs'+ireal.rjust(3,'0')
        
        #shutil.copy('../particles/irregular/'+dustmodel+'.inc', './coord.dat')
        filename=Path(__file__).resolve().parents[2].joinpath('particles/irregular/'+dustmodel+'.inc')
        shutil.copy(filename,'./coord.dat')

        #
        xc=0.0
        yc=0.0
        zc=0.0

        # determine reference bar position
        refy = ref_dist * anorm * np.sin(ref_posang*np.pi/180.0)
        refz = ref_dist * anorm * np.cos(ref_posang*np.pi/180.0)
        
        # reference bar length
        if ref_length==None:
            ref_length=av
        
        refbar_text='"{:6.4f} um"'.format(ref_length)

        # camera position
        dc = xcamera * anorm 

        # reference bar thickness
        bar_thickness = 0.05 * (anorm/5.0) 
        fontsize = 0.5 * (dc/8.0) * ref_fontsize

        # reference text position
        ref_text_yoffset = 0.167 * (dc/8.0) 
        
        # generate parameter file for povray rendering
        output='povparam.inc'
        with open(output,'w') as fout:
            fout.write('#declare camera_distance = %13.6e;\n'%(dc))
            fout.write('#declare ref_length = %13.6e;\n'%(ref_length))
            fout.write('#declare ref_barwidth = %13.6e;\n'%(bar_thickness))
            fout.write('#declare av_scaled = %13.6e;\n'%(av_scaled))
            fout.write('#declare xc = %13.6e;\n'%(xc))
            fout.write('#declare yc = %13.6e;\n'%(yc))
            fout.write('#declare zc = %13.6e;\n'%(zc))
            fout.write('#declare refy = %13.6e;\n'%(refy))
            fout.write('#declare refz = %13.6e;\n'%(refz))
            fout.write('#declare refclr = %-s;\n'%(ref_color))
            fout.write('#declare reftext = %-s;\n'%(refbar_text))
            fout.write('#declare reftext_yoff = %13.6e;\n'%(ref_text_yoffset))
            fout.write('#declare fontsize = %13.6e;\n'%(fontsize))
            fout.write('#declare rotate_x = %13.6e;\n'%(rotx))
            fout.write('#declare rotate_y = %13.6e;\n'%(roty))
            fout.write('#declare rotate_z = %13.6e;\n'%(rotz))
            fout.write('#declare parcolor = %-s;\n'%(particle_color))
            fout.write('#declare bgcolor = %-s;\n'%(bg_color))

        #
        if fn==None:
            output_file_name='grs_rv'+size+'um_'+ireal
        else:
            output_file_name=fn

        if path==None:
            output_file_name='./'+output_file_name
        else:
            output_file_name=path+output_file_name
        print('writing ... ',output_file_name+'.png')

        # run povray
        if background:
            os.system('povray irregular.pov +H1200 +W1200 +A0.1\
                    Output_File_Name='+output_file_name)
        else:
            os.system('povray irregular.pov +H1200 +W1200 +A0.1\
                    +FN +UA Output_File_Name='+output_file_name)

    else:

        
        dustmodel=partype+'_'+size+'_'+ireal+'_'+amon
        #filename='../particles/aggregate/'+dustmodel+'.pos'
        filename=Path(__file__).resolve().parents[2].joinpath('particles/aggregate/'+dustmodel+'.pos') 

        # reading header part
        header=''
        with open(filename,'r') as f:
            dum=f.readline()
            while (dum.strip()[0]=='#'):
                if 'Characteristic rad' in dum:
                    char=float(dum.split()[-1])
                if 'Number of monomers' in dum:
                    nnn=int(dum.split()[-1])
                header = header + dum
                dum = f.readline()

        # load monomer position 
        am,x,y,z=np.loadtxt(filename,unpack=True,comments='#',usecols=[0,1,2,3])

        # projected center of aggregate in the image plane
        xc = 0.5*(min(x)+max(x))
        yc = 0.5*(min(y)+max(y))
        zc = 0.5*(min(z)+max(z))

        # determine reference bar position
        refy = ref_dist * char * np.sin(ref_posang*np.pi/180.0)
        refz = ref_dist * char * np.cos(ref_posang*np.pi/180.0)

        # reference bar length
        if ref_length==None:
            ref_length=char
        
        refbar_text='"{:6.4f} um"'.format(ref_length)

        # camera position
        dc = xcamera * char

        # set floor value for the camera distance
        #dc = max(15.0*am[0],dc)

        # reference bar thickness
        #bar_thickness = 0.05 * (char/5.0) 
        bar_thickness = 0.03 * (dc/10.0) #(char/5.0) 
        fontsize = 0.5 * (dc/8.0) * ref_fontsize

        # reference text position
        ref_text_yoffset = 0.167 * (dc/8.0)

        # generate parameter file for povray rendering
        output='povparam.inc'
        with open(output,'w') as fout:
            fout.write('#declare amon = %13.6e;\n'%(am[0]))
            fout.write('#declare camera_distance = %13.6e;\n'%(dc))
            fout.write('#declare ref_length = %13.6e;\n'%(ref_length))
            fout.write('#declare ref_barwidth = %13.6e;\n'%(bar_thickness))
            fout.write('#declare xc = %13.6e;\n'%(xc))
            fout.write('#declare yc = %13.6e;\n'%(yc))
            fout.write('#declare zc = %13.6e;\n'%(zc))
            fout.write('#declare refy = %13.6e;\n'%(refy))
            fout.write('#declare refz = %13.6e;\n'%(refz))
            fout.write('#declare refclr = %-s;\n'%(ref_color))
            fout.write('#declare reftext = %-s;\n'%(refbar_text))
            fout.write('#declare reftext_yoff = %13.6e;\n'%(ref_text_yoffset))
            fout.write('#declare fontsize = %13.6e;\n'%(fontsize))
            fout.write('#declare rotate_x = %13.6e;\n'%(rotx))
            fout.write('#declare rotate_y = %13.6e;\n'%(roty))
            fout.write('#declare rotate_z = %13.6e;\n'%(rotz))
            fout.write('#declare parcolor = %-s;\n'%(particle_color))
            fout.write('#declare bgcolor = %-s;\n'%(bg_color))

        # generate particle coordinate file
        output='coord.dat'
        with open(output,'w') as fout:
            for i in range(nnn):
                fout.write('<%13.6e,%13.6e,%13.6e>,\n'%(x[i],y[i],z[i]))

        #
        if fn==None:
            output_file_name=dustmodel
        else:
            output_file_name=fn

        if path==None:
            output_file_name='./'+output_file_name
        else:
            output_file_name=path+output_file_name
        print('writing ... ',output_file_name+'.png')

        # run povray
        if background:
            os.system('povray aggregate.pov +H1200 +W1200 +A0.1\
                    Output_File_Name='+output_file_name)
        else:
            os.system('povray aggregate.pov +H1200 +W1200 +A0.1\
                    +FN +UA Output_File_Name='+output_file_name)


def view_particle(partype,size,ireal=None,amon=None):
    """
    Displaly the pre-rendering particle images

    Parameters
    ----------
    partype          : str
                       Particle type
    size             : str
                       Dust particle size
    ireal            : str (optional)
                       Realization number. If not specified, all realizations are displayed.
    amon             : str 
                       Monomer radius (necessary if you call aggregate files).
    """

    # check particle type
    check_particle_type(partype)

    # check particle size
    check_particle_size(partype,amon,size)

    if partype=='grs':
        if ireal==None:
            plt.figure(figsize=(15,20))
            for jjj in range(1,11): # running from 1 ... 10
                filename='particles/images/'+partype+'_rv'+size+'um_'+str(jjj)+'.png'
                filename=Path(__file__).resolve().parents[2].joinpath(filename)

                ax = plt.subplot(4,3,jjj)
                ax.axis("off")
                ax.text(0.1, 0.9, 'Realization #'+str(jjj),fontsize=18,transform = ax.transAxes)

                try:
                    img=mpimg.imread(filename)
                    imgplot=ax.imshow(img)
                except FileNotFoundError:
                    print('Error: No such a file. Fail to open ',filename)

        else:
            plt.figure(figsize=(15,20))
            filename='particles/images/'+partype+'_rv'+size+'um_'+ireal+'.png'
            filename=Path(__file__).resolve().parents[2].joinpath(filename)
            
            ax = plt.subplot(111)
            ax.axis("off")
            ax.text(0.1, 0.9, 'Realization #'+ireal,fontsize=25,transform = ax.transAxes)

            try:
                img=mpimg.imread(filename)
                imgplot=ax.imshow(img)
            except FileNotFoundError:
                print('Error: No such a file. Fail to open ',filename)

    else:
        plt.figure(figsize=(10,10))
        if ireal==None:

            for jjj in range(1,5): # running from 1 ... 4
                filename='particles/images/'+partype+'_'+size+'_'+str(jjj)+'_'+amon+'.png'
                filename=Path(__file__).resolve().parents[2].joinpath(filename)
                
                ax = plt.subplot(2,2,jjj)
                ax.axis("off")
                ax.text(0.1, 0.9, 'Realization #'+str(jjj),fontsize=18,transform = ax.transAxes)

                try:
                    img=mpimg.imread(filename)
                    imgplot=ax.imshow(img)
                except FileNotFoundError:
                    print('Error: No such a file. Fail to open ',filename)
        
        else:
            filename='particles/images/'+partype+'_'+size+'_'+ireal+'_'+amon+'.png'
            filename=Path(__file__).resolve().parents[2].joinpath(filename)
            
            ax = plt.subplot(111)
            
            try:
                img=mpimg.imread(filename)
                ax.text(0.1, 0.9, 'Realization #'+ireal,fontsize=25,transform = ax.transAxes)
                ax.axis("off")
                imgplot=ax.imshow(img)
            except FileNotFoundError:
                print('Error: No such a file. Fail to open ',filename)

