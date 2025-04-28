from matplotlib import patches, lines
from matplotlib import pyplot as plt

import rebound

import parse_log
import find_primary

class Sim:
    
    def __init__(self,name):
        self.archive = name + '.bin'
        self.log = name + '.log'

        (CEs, colls, ejecs) = parse_log(self.log)
        self.CEs = CEs
        self.colls = colls
        self.ejecs = ejecs

        self.sa = rebound.SimulationArchive(self.archive)
        
        self.t0 = min([s.t for s in self.sa if len(s.particles) > 3])
        self.tend = self.sa[-1].t
        self.post_moons = np.where([s.t >= self.t0 for s in self.sa])[0]
        self.post_moons = [int(i) for i in self.post_moons] # have to do int casts explicitly
        self.t = self.sa[self.post_moons[0]].t
        self.t = [self.sa[i].t for i in self.post_moons]
        self.p1moons = []
        self.p2moons = []
        self.stmoons = []
        self.ejmoons = []
        self.p1mset = [] # these are not actually sets as I want to preserve order
        self.p2mset = []
        self.stmset = []
        self.ejmset = []
        self.mhost = [[] for m in name_moons_flat]

        for i in self.post_moons:
            p1m = []
            p2m = []
            stm = []
            ejm = []
            for j,m in enumerate(name_moons_flat):
                try:
                    pl = find_primary(m,name_pl,self.sa[i])
                except:
                    pl = None
                if pl == name_pl[0]:
                    p1m.append(m)
                    if not m in self.p1mset:
                        self.p1mset.append(m)
                if pl == name_pl[1]:
                    p2m.append(m)
                    if not m in self.p2mset:
                        self.p2mset.append(m)
                if pl == name_star:
                    stm.append(m)
                    if not m in self.stmset:
                        self.stmset.append(m)
                if pl == 'Galaxy':
                    ejm.append(m)
                    if not m in self.ejmset:
                        self.ejmset.append(m)
                        
                self.mhost[j].append(pl)


            self.p1moons.append(p1m)
            self.p2moons.append(p2m)
            self.stmoons.append(stm)
            self.ejmoons.append(ejm)
        
        self.p1nm = [len(n) for n in self.p1moons]
        self.p2nm = [len(n) for n in self.p2moons]
        self.Nmoonmax = [len(self.p1mset),len(self.p2mset)]
        self.stnm = [len(n) for n in self.stmoons]
        self.Nmoonstar = len(self.stmset)
        self.ejnm = [len(n) for n in self.ejmoons]
        self.Nmoonej = len(self.ejmset)
        self.mhost = np.array(self.mhost)
#        for i in range(len(self.mhost[0])-1):
#            if (self.mhost[:,i+1] != self.mhost[:,i]).any():
#                print(self.t[i],self.mhost[:,i+1])
        
        return
    
    def make_timeline(self):
        
        CE_col = 'red'
        col = {name_moons_flat[0]:'sienna',
               name_moons_flat[1]:'olivedrab',
               name_moons_flat[2]:'blue',
               name_moons_flat[3]:'deeppink',
               name_moons_flat[4]:'darksalmon',
               name_moons_flat[5]:'lightgreen',
               name_moons_flat[6]:'dodgerblue',
               name_moons_flat[7]:'pink',
               'Planet1':'black',
               'Planet2':'gainsboro',
               name_star:'yellow'}
        
        margin = 0.025 #figure coordinates
        axscale = 2
        moonwidth = (1 - (8*margin)) / (axscale+self.Nmoonmax[0]+self.Nmoonmax[1]+self.Nmoonstar+self.Nmoonej)
        axwidth = axscale*moonwidth
        
        xsize = 7
        ysize = 4
        fig = plt.figure(figsize=(xsize,ysize))
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        
        ytracker = margin
        galbox = patches.Rectangle((margin,ytracker),1-2*margin,1-2*margin,edgecolor='k',fill=None)
        ax.add_patch(galbox)
        
        ytracker += (margin + self.Nmoonej*moonwidth + axwidth)
        starbox = patches.Rectangle((2*margin,ytracker),1-4*margin,3*margin+
                                    (self.Nmoonmax[0]+self.Nmoonmax[1]+self.Nmoonstar)*moonwidth,
                                    edgecolor='k',fill=None)
        ytracker += (margin + self.Nmoonstar*moonwidth)
        ax.add_patch(starbox)

        plbox = []
        Npl = 2
        for i in range(Npl):
            plbox.append(patches.Rectangle((3*margin,ytracker),1-6*margin,self.Nmoonmax[i-1]*moonwidth,
                                           edgecolor='k',fill=None))
            ytracker += (margin + self.Nmoonmax[i-1]*moonwidth)
            ax.add_patch(plbox[i])
        
        circsize = 0.025
        starcirc = patches.Ellipse((starbox.get_x(),starbox.get_y()+0.5*starbox.get_height()),circsize,
                                   circsize*xsize/ysize,
                                   facecolor=col[name_star],edgecolor='k')
        ax.add_patch(starcirc)
        p1circ = patches.Ellipse((plbox[1].get_x(),plbox[1].get_y()+0.5*plbox[1].get_height()),circsize,
                                  circsize*xsize/ysize,
                                  facecolor=col['Planet1'],edgecolor='k')
        ax.add_patch(p1circ)
        p2circ = patches.Ellipse((plbox[0].get_x(),plbox[0].get_y()+0.5*plbox[0].get_height()),circsize,
                                  circsize*xsize/ysize,
                                  facecolor=col['Planet2'],edgecolor='k')
        ax.add_patch(p2circ)
            
        xstart = 4*margin
        xend = 1-4*margin
        xaxis = lines.Line2D([xstart,xend],[margin+0.5*axwidth,margin+0.5*axwidth],c='k')
        ax.add_line(xaxis)
        
        # add CE events
        CElines = []
        for CE in self.CEs:
            trel = (CE.t-self.t0)/(self.tend-self.t0)
            CElines.append(lines.Line2D([xstart+(xend-xstart)*trel,xstart+(xend-xstart)*trel],
                                        [plbox[0].get_y()+plbox[0].get_height(),plbox[1].get_y()],c=CE_col,
                                        zorder=99))
            ax.add_line(CElines[-1])
        
        #set up slots for moon timelines
        pl1slots = {}
        count = 0
        for p in self.p1mset:
            pl1slots[p] = plbox[1].get_y() + plbox[1].get_height() - count*moonwidth - margin
            count += 1
        count = 0
        pl2slots = {} 
        for p in self.p2mset:
            pl2slots[p] = plbox[0].get_y() + plbox[0].get_height() - count*moonwidth - margin
            count += 1
        stslots = {}
        count = 0
        for p in self.stmset:
            stslots[p] = plbox[0].get_y()-(count+0.5)*moonwidth
            count += 1
        ejslots = {}
        count = 0
        for p in self.ejmset:
            ejslots[p] = starbox.get_y()-(count+0.5)*moonwidth
            count+=1

        moon_y = [[] for i in name_moons_flat]
        
        # add the moon timelines
        trel = (np.array(self.t)-self.t0)/(self.tend-self.t0)
        for j,m in enumerate(name_moons_flat):
            for i in range(len(self.t)):
                if self.mhost[j][i] == 'Planet1':
                    moon_y[j].append(pl1slots[m])
                if self.mhost[j][i] == 'Planet2':
                    moon_y[j].append(pl2slots[m])
                if self.mhost[j][i] == name_star:
                    moon_y[j].append(stslots[m])
                if self.mhost[j][i] is None:
                    moon_y[j].append(np.nan)
                if self.mhost[j][i] == 'Galaxy':
                    moon_y[j].append(ejslots[m])
            line = lines.Line2D(xstart+(xend-xstart)*trel,moon_y[j],c=col[m])
            ax.add_line(line)
            
        # add collision events
        for c in self.colls:
            if 'Planet' in c.names[0] and 'Planet' in c.names[1]:
                w = xstart - plbox[0].get_x() + (c.t-self.t0)/(self.tend-self.t0)*(xend-xstart)
                if '1' in c.names[0]:
                    plbox[1].set_width(w)
                if '2' in c.names[0]:
                    plbox[0].set_width(w)
                continue
            
            try:
                yind = name_moons_flat.index(c.names[0])
                n1 = c.names[0]
                n2 = c.names[1]
            except:
                print('planet lost?',c.names)
            else:
                try:
                    xind = min(np.where([i is None for i in self.mhost[yind]])[0][1:])
                except:
                    yind = name_moons_flat.index(c.names[1]) #just in case you removed the wrong body
                    xind = min(np.where([i is None for i in self.mhost[yind]])[0][1:])
                    n1 = c.names[1]
                    n2 = c.names[0]
                xy = ((c.t-self.t0)/(self.tend-self.t0) * (xend-xstart) + xstart, moon_y[yind][xind-1])
                ax.add_patch(patches.Ellipse(xy,circsize,circsize*xsize/ysize,lw=3,
                                             fc=col[n2],ec=col[n1]))
                
        return