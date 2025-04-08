import math
import module.input.element as element
from numpy import *

class Sphere:
    def __init__(self):
        self.interval = 0.0
        self.spherePoints = {}
    
    def rotate(self,R):
        new = []
        a = math.pi/9
        for p in self.spherePoints[R]:
            new.append([p[0],math.cos(a)*p[1]-math.sin(a)*p[2],math.sin(a)*p[1]+math.cos(a)*p[2]])
        self.spherePoints[R] = new
        
    def angle(self, Range, theta): return  2*math.pi/round(2*math.pi*Range*math.sin(theta)/self.interval)    
    def cirNo(self, Range): return int(round(math.pi*Range/self.interval))
    def cirTheta(self,pNo): return math.pi/pNo
    
    def circle(self, Range, dpi, theta, chk=1):
        pi = 0
        cosT = math.cos(theta); sinT = math.sin(theta)
        newRadi = Range*sinT; z = Range*cosT
        for _ in range(int(round(2*math.pi/dpi))):
            cosP = math.cos(pi); sinP = math.sin(pi)           
            x = newRadi*cosP; y = newRadi*sinP
            self.spherePoints[Range].append([x,z,y])
            if chk:
                self.spherePoints[Range].append([x,-z,y])
            pi += dpi
            
    def draw(self, interval = 1.0, shell = 1.4):
        self.interval = interval
        for radi in element.SFED_radi.values():
            Range = radi+shell
            self.spherePoints[Range] = []
            Rpi = math.pi/2 #angle 90
            circleNo = self.cirNo(Range)
            dtheta = self.cirTheta(circleNo)
            if circleNo%2:
                for i in range(int(circleNo/2)):
                    theta = Rpi - (1 + i)*dtheta
                    dpi = self.angle(Range,theta)
                    self.circle(Range, dpi, theta)
                self.circle(Range, dtheta, math.pi/2,0)
            else:
                for i in range(int(circleNo/2)):
                    theta = Rpi - (i + 0.5)*dtheta
                    dpi = self.angle(Range,theta)
                    self.circle(Range, dpi, theta)
            if radi>1.4:
                self.rotate(Range)
        return self.spherePoints
    
class Surface:
    def __init__(self):
        self.shell = 0.0
        self.sfed = 0.0
        self.p20 = 0
        self.volume = 0.0
        self.dipole = 0.0
        self.GI = 0.0
        self.surfacePoints = []
        self.spherePoints = []
        self.pointValue = []        #point value(sun(terms))
        self.grid = []
        self.minmax = []
        self.maxpgrp = []
        self.minpgrp = []
        self.maxcenter = []
        self.mincenter = []
        self.gIndex = []
        self.maxgp = [-999.99,-999.99,-999.99]
        self.mingp = [999.99,999.99,999.99]
        #self.ADatomL = []
    
    def minmaxchk(self, mol) :
        min = [1000,1000,1000]
        max = [-1000,-1000,-1000]
        for xyz in mol.coors:
            if xyz[0] > max[0] : max[0] = xyz[0]
            if xyz[0] < min[0] : min[0] = xyz[0]
            if xyz[1] > max[1] : max[1] = xyz[1]
            if xyz[1] < min[1] : min[1] = xyz[1]
            if xyz[2] > max[2] : max[2] = xyz[2]
            if xyz[2] < min[2] : min[2] = xyz[2]
        for i in range(3) :
            self.minmax.append([min[i],max[i]])
            
    def set_atom_range(self, mol):
        self.minmaxchk(mol)
        for i in range(mol.atomN):
            if mol.vdw_types[i] in element.SFED_radi:
                mol.atom_range[i] = element.SFED_radi[mol.vdw_types[i]]+self.shell
            else:
                return 0
            '''
            try:
                mol.atom_range[i] = element.SFED_radi[mol.vdw_types[i]]+self.shell
            except:
                #print (mol.rdMol.GetProp('ID'),i,mol.vdw_types[i],'radii not exist')
                return 0
            '''
            if mol.atom_range[i] >self.GI: self.GI = mol.atom_range[i]
        return 1

    def createGrid(self):
        #grid size
        GSX=int((self.minmax[0][1] - self.minmax[0][0])/self.GI)+3
        GSY=int((self.minmax[1][1] - self.minmax[1][0])/self.GI)+3
        GSZ=int((self.minmax[2][1] - self.minmax[2][0])/self.GI)+3
        for x in range(GSX):
            self.grid.append([])
            for y in range(GSY):
                self.grid[x].append([])
                for z in range(GSZ):
                    self.grid[x][y].append([])

    def allocation(self, mol):
        for i in range(mol.atomN):
            xyz = mol.coors[i]
            gridx = int((xyz[0] - self.minmax[0][0])/self.GI)+1
            gridy = int((xyz[1] - self.minmax[1][0])/self.GI)+1
            gridz = int((xyz[2] - self.minmax[2][0])/self.GI)+1
            self.gIndex.append([gridx,gridy,gridz])
            self.grid[gridx][gridy][gridz].append(i)

    def adjacency(self,mol):
        for i in range(mol.atomN):
            OL = []
            for x in range(self.gIndex[i][0]-1,self.gIndex[i][0]+2):
                for y in range(self.gIndex[i][1]-1,self.gIndex[i][1]+2):
                    for z in range(self.gIndex[i][2]-1, self.gIndex[i][2]+2):
                        OL.extend(self.grid[x][y][z])
            OL.sort()
            for j in OL[OL.index(i)+1:]:
                self.overlap(mol, i, j)
     
    def overlap(self, mol, i, j):
        xyz1 = mol.coors[i]; xyz2 = mol.coors[j]
        range1 = mol.atom_range[i]; range2 = mol.atom_range[j]
        dx = xyz2[0]-xyz1[0]; dy = xyz2[1]-xyz1[1]; dz = xyz2[2]-xyz1[2]
        dis3 = (dx**2+dy**2+dz**2)**0.5
        if dis3 < range1 + range2:
            dis2 = (dx**2+dy**2)**0.5
            #T: theta, 1: a1, 2: a2, P: pi
            T1 = math.acos(dz/dis3);  T2 = math.pi - T1
            V1 = (dis3**2+range1**2-range2**2)/(2*dis3);    V2 = dis3-V1
            if dis2 != 0:
                if dy<0:
                    P1 = math.pi*2 - math.acos(dx/dis2)
                    P2 = P1 + math.pi
                else:
                    P1 = math.acos(dx/dis2)
                    P2 = P1 + math.pi 
            else:
                P1 = 0
                P2 = math.pi
            self.removeOverlap(self.spherePoints[i], [T1, P1, V1])
            self.removeOverlap(self.spherePoints[j], [T2, P2, V2])

    def removeOverlap(self,points,info):
        overlapnum = 0
        a = math.sin(info[1]); b = math.cos(info[1])
        cos = math.cos(info[0]); sin = math.sin(info[0])
        z_rotation = [b*sin, a*sin, cos]
        for i in range(len(points)):
            zValue = points[i-overlapnum][0]*z_rotation[0] + points[i-overlapnum][1]*z_rotation[1] + points[i-overlapnum][2]*z_rotation[2]
            if zValue > info[2]:
                points.pop(i-overlapnum)
                overlapnum += 1
    
    def Hrotate(self, mol, i, points):
        j = mol.rdMol.GetAtomWithIdx(i).GetNeighbors()[0].GetIdx()
        xyz1 = mol.coors[i]; xyz2 = mol.coors[j]

        dx = xyz2[0]-xyz1[0]; dy = xyz2[1]-xyz1[1]; dz = xyz2[2]-xyz1[2]
        dis3 = (dx**2+dy**2+dz**2)**0.5;dis2 = (dx**2+dy**2)**0.5
        theta = math.acos(dz/dis3)
        v = (dis3**2+mol.atom_range[i]**2-mol.atom_range[j]**2)/(2*dis3)
        if dis2 != 0:
            if dy<0:
                pi = math.pi*2 - math.acos(dx/dis2)
            else:
                pi = math.acos(dx/dis2) 
        else:
            pi = 0
        P = []
        a = math.cos(pi+math.pi/2); b = math.sin(pi+math.pi/2)
        cos = math.cos(-theta); sin = math.sin(-theta)
        rMtx = [[a**2*(1-cos)+cos, a*b*(1-cos), b*sin], [a*b*(1-cos), b**2*(1-cos)+cos,-a*sin], [-b*sin, a*sin, cos]]
        for p in points:
            if p[2]<v:
                P.append([rMtx[0][0]*p[0]+rMtx[1][0]*p[1]+rMtx[2][0]*p[2],rMtx[0][1]*p[0]+rMtx[1][1]*p[1]+rMtx[2][1]*p[2],rMtx[0][2]*p[0]+rMtx[1][2]*p[1]+rMtx[2][2]*p[2]])
        return P
    
    def molecule(self, mol, sPoints, shell = 1.4):
        import copy
        self.shell = shell
        if self.set_atom_range(mol) == 0:
            return 0
        for i in range(mol.atomN):
            if mol.rdMol.GetAtomWithIdx(i).GetSymbol() == 'H':
                self.spherePoints.append(self.Hrotate(mol, i, copy.deepcopy(sPoints[mol.atom_range[i]])))
            else:
                self.spherePoints.append(copy.deepcopy(sPoints[mol.atom_range[i]]))
    
        self.createGrid()
        self.allocation(mol)
        self.adjacency(mol)
        self.move(mol)
        #self.calASA(mol,sPoints)
        #self.PDBout(mol)
        #self.atomPDBout(mol)
        return self.surfacePoints
    
    def move(self,mol):
        for i in range(mol.atomN):
            xyz = mol.coors[i]
            for point in self.spherePoints[i]:
                self.surfacePoints.append([point[0] + xyz[0], point[1] + xyz[1], point[2] + xyz[2]])
        return 1

    def PDBout(self,mol):
        pdbfile = open('%s.pdb'%mol.rdMol.GetProp('_Name'),'w')
        for i in self.surfacePoints:
            pdbfile.write("HETATM00000 %-4s HOH H0000    %8.3f%8.3f%8.3f\n"%('X', i[0], i[1], i[2]))
        pdbfile.write('TER\nEND\n')
        pdbfile.close()
        exit()
        return 1
        