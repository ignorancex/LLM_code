from sys import argv, exit
import matplotlib.pyplot as plt
import math
import numpy as np
from scipy import stats
import csv


# def readCsv(fileName):
# 	X = []
# 	Y = []
# 	THETA = []
# 	LBL = []

# 	with open(fileName, 'rt') as f:
# 		A = csv.reader(f)

# 		for idx, line in enumerate(A):
# 			if(idx == 0):
# 				continue
# 			else:
# 				X.append(float(line[1]))
# 				Y.append(float(line[2]))
# 				THETA.append(float(line[3]))
# 				LBL.append(float(line[4]))

# 	return (X, Y, THETA, LBL)


def read(fileName):
	f = open(fileName, 'r')
	A = f.readlines()
	f.close()

	X = []
	Y = []
	THETA = []
	LBL = []

	for line in A:
		(x, y, theta, lbl) = line.split(' ')
		X.append(float(x))
		Y.append(float(y))
		THETA.append(float(theta))
		LBL.append(int(lbl.rstrip('\n')))

	return (X, Y, THETA, LBL)


def blueFix(st, end, X, Y, LBL, Node_meta):
	mid = int(st + (end - st)/2)
	# print("Blue nodes: ", end-st)
	xMid = 0; yMid =0; fill = True

	for i in range(st, end-9):
		if(fill == True):
			X1 = [X[j] for j in range(i, i+8)]; Y1 = [Y[j] for j in range(i, i+8)] 
			X2 = [X[j] for j in range(i+8, i+16)]; Y2 = [Y[j] for j in range(i+8, i+16)]

			(m1, c1, _, _, _) = stats.linregress(X1, Y1)
			(m2, c2, _, _, _) = stats.linregress(X2, Y2)

			dm1 = math.degrees(math.atan(m1)); dm2 = math.degrees(math.atan(m2))
			delTheta = dm1 - dm2
			# print(delTheta, dm1, dm2)
			
			if((delTheta > 70 and delTheta < 110) or (delTheta > -110 and delTheta < -70)):
				xMid = X2[0]; yMid = Y2[0]
				# Node_meta.append((X[st], Y[st], xMid, yMid, LBL[mid]))
				# Node_meta.append((xMid, yMid, X[end], Y[end], LBL[mid]))
				Node_meta.append((X[st], Y[st], xMid, yMid, LBL[mid], st, i+8))
				Node_meta.append((xMid, yMid, X[end], Y[end], LBL[mid], i+8, end))
				fill = False

			# x1s = X1[0]; x1e = X1[-1] 
			# y1s = m1*x1s + c1; y1e = m1*x1e + c1
			# x2s = X2[0]; x2e = X2[-1]
			# y2s = m2*x2s + c2; y2e = m2*x2e + c2

			# ax = plt.subplot(1,1,1)
			# ax.plot([x1s, x1e], [y1s, y1e], 'r-')
			# ax.plot([x2s, x2e], [y2s, y2e], 'g-')
			# ax.plot(X[st:end], Y[st:end], 'bo')
			# ax.plot(X1, Y1, 'ro')
			# ax.plot(X2, Y2, 'go')
			# plt.show()

	if(fill == True):
		# Node_meta.append((X[st], Y[st], X[end], Y[end], LBL[mid]))
		Node_meta.append((X[st], Y[st], X[end], Y[end], LBL[mid], st, end))

	# ax = plt.subplot(1,1,1)
	# ax.plot(X[st:end], Y[st:end], 'bo')
	# ax.plot(xMid, yMid, 'ro')

	# plt.show()
	# print("Inside blueFix--------")



def meta(X, Y, LBL):
	Node_meta = []
	st = end = 0

	for i in range(1, len(LBL)):
		if LBL[i] == LBL[i-1]:
			end = i
			continue

		mid = int(st + (end - st)/2)
		
		if (LBL[mid] == 1):
			# Hack : Ending long corridor
			if((end-st) > 75):
				steps = np.linspace(st, end, 4).astype(int)
				blueFix(st, steps[1], X, Y, LBL, Node_meta)
				blueFix(steps[1], steps[2], X, Y, LBL, Node_meta)
				blueFix(steps[2], end, X, Y, LBL, Node_meta)
			else:
				blueFix(st, end, X, Y, LBL, Node_meta)
		else:
			# Node_meta.append((X[st], Y[st], X[end], Y[end], LBL[mid]))
			Node_meta.append((X[st], Y[st], X[end], Y[end], LBL[mid], st, end))
	
		st = end + 1
		end = st
	return Node_meta


def drawTheta(Node_meta, thetas):
	ax = plt.subplot(1,1,1)

	i = 0
	for line in Node_meta:
		
		lbl = line[4]
		x = [line[0], line[2]]
		y = [line[1], line[3]]

		x2 = math.cos(thetas[i]) + x[0]
		y2 = math.sin(thetas[i]) + y[0]
		plt.plot([x[0], x2], [y[0], y2], 'm->')

		if lbl == 0:
			ax.plot(x, y, 'ro')
			ax.plot(x, y, 'r-')

		elif lbl == 1:
			ax.plot(x, y, 'bo')
			ax.plot(x, y, 'b-')

		elif lbl == 2:
			ax.plot(x, y, 'go')
			ax.plot(x, y, 'g-')

		elif lbl == 3:
			ax.plot(x, y, 'yo')
			ax.plot(x, y, 'y-')

		i = i+1

	plt.show()


def draw(X, Y, LBL):
	X0 = []; Y0 = []; X1 = []; Y1 = []; X2 = []; Y2 =[]; X3 = []; Y3 = [];
	
	for i in range(len(LBL)):
		if LBL[i] == 0:
			X0.append(X[i])
			Y0.append(Y[i])

		elif LBL[i] == 1:
			X1.append(X[i])
			Y1.append(Y[i])

		elif LBL[i] == 2:
			X2.append(X[i])
			Y2.append(Y[i])

		elif LBL[i] == 3:
			X3.append(X[i])
			Y3.append(Y[i])

	fig = plt.figure()
	ax = plt.subplot(111)

	ax.plot(X0, Y0, 'ro', label='Rackspace')
	ax.plot(X1, Y1, 'bo', label='Corridor')
	ax.plot(X2, Y2, 'go', label='Trisection')
	ax.plot(X3, Y3, 'yo', label='Intersection')
	
	plt.show()


def drawManh(Nodes):
	ax = plt.subplot(1,1,1)

	for line in Nodes:
		lbl = line[4]
		x = [line[0], line[2]]
		y = [line[1], line[3]]

		if lbl == 0:
			ax.plot(x, y, 'ro')
			ax.plot(x, y, 'r-')

		elif lbl == 1:
			ax.plot(x, y, 'bo')
			ax.plot(x, y, 'b-')

		elif lbl == 2:
			ax.plot(x, y, 'go')
			ax.plot(x, y, 'g-')

		elif lbl == 3:
			ax.plot(x, y, 'yo')
			ax.plot(x, y, 'y-')

	plt.show()


def drawMeta(Node_meta):
	ax = plt.subplot(1,1,1)

	i = 0
	for line in Node_meta:
		
		lbl = line[4]
		x = [line[0], line[2]]
		y = [line[1], line[3]]

		if lbl == 0:
			ax.plot(x, y, 'ro')
			ax.plot(x, y, 'r-')

		elif lbl == 1:
			ax.plot(x, y, 'bo')
			ax.plot(x, y, 'b-')

		elif lbl == 2:
			ax.plot(x, y, 'go')
			ax.plot(x, y, 'g-')

		elif lbl == 3:
			ax.plot(x, y, 'yo')
			ax.plot(x, y, 'y-')

		i = i+1

	plt.show()


def outRemove(Node_meta):
	i = 0; Nodes = []
	# print(len(Node_meta[i]))
	while (i < len(Node_meta)):
		line = Node_meta[i]
		lbl = line[4]
		x = [line[0], line[2]]
		y = [line[1], line[3]]

		leng = ((x[0]-x[1])**2 + (y[0]-y[1])**2)**(0.5)
		if(leng < 0.1):
			pass

		else:
			Nodes.append(line)
		
		i = i+1

	return Nodes


def calcTheta(x1, x2, y1, y2):
	if(x2 == x1):
		if(y2 > y1):
			theta = math.pi/2
		else:
			theta = 3*math.pi/2
	else:
		theta = math.atan((y2-y1)/(x2-x1))

	if(x2-x1 < 0):
		theta += math.pi

	return theta


def getPositve(ang):
	if(ang < 0):
		ang += 360
	return ang


def manh(Node_meta, thetas):
	Nodes = []; accTheta = 90; Thetas = []
	line = Node_meta[0]
	x = [line[0], line[2]]; y = [line[1], line[3]]
	leng = ((x[0]-x[1])**2 + (y[0]-y[1])**2)**(0.5)

	# print("Total theta: ", accTheta, "Length: ", leng)
	for i in range(1, len(Node_meta)):
		line = Node_meta[i]
		x = [line[0], line[2]]; y = [line[1], line[3]]
		leng = ((x[0]-x[1])**2 + (y[0]-y[1])**2)**(0.5)

		curAng = getPositve(math.degrees(thetas[i])); prevAng = getPositve(math.degrees(thetas[i-1]))
		delTheta = curAng - prevAng

		binTheta = 0
		
		if(line[4] == 1):
			if((delTheta > 0 and delTheta < 45) or (delTheta > 315 and delTheta < 360) or (delTheta < 0 and delTheta > -45) or (delTheta < -315 and delTheta > -360)):
				binTheta = 0
			elif((delTheta > 55 and delTheta < 135) or (delTheta < -225 and delTheta > -315)):
				binTheta = 90
			elif((delTheta > 135 and delTheta < 225) or (delTheta < -135 and delTheta > -225)):
				binTheta = 180
			elif((delTheta > 225 and delTheta < 315) or (delTheta < -45 and delTheta > -135)):
				binTheta = 270
		else:
			if((delTheta > 0 and delTheta < 45) or (delTheta > 315 and delTheta < 360) or (delTheta < 0 and delTheta > -45) or (delTheta < -315 and delTheta > -360)):
				binTheta = 0
			elif((delTheta > 45 and delTheta < 135) or (delTheta < -225 and delTheta > -315)):
				binTheta = 90
			elif((delTheta > 135 and delTheta < 225) or (delTheta < -135 and delTheta > -225)):
				binTheta = 180
			elif((delTheta > 225 and delTheta < 315) or (delTheta < -45 and delTheta > -135)):
				binTheta = 270
		
		accTheta += binTheta
		Nodes.append((leng, accTheta, line[4], line[5], line[6]))
		# print("Delta theta: ", delTheta, "Binned to: ", binTheta, "Total theta: ", accTheta, "Label: ", line[4], "Cur. Angle: ", curAng, "Pre. Angle: ", prevAng)

	return Nodes


def extManh(Nodes_manh):
	Nodes = []; i = 0

	l1 = 0; b1 = 0; l2 = 0; b2 = 0
	for line in Nodes_manh:
		mag = line[0]; theta = line[1]; lbl = line[2]; stPose = line[3]; endPose = line[4]
		
		if((theta - Nodes_manh[i-1][1] == 180) and (i != 0)):
			# l1 = l2 - 0.2
			# b1 = b2 + 0.2
			l1 = l2
			b1 = b2
			l2 = l1 + mag*math.cos(math.radians(theta))
			b2 = b1 + mag*math.sin(math.radians(theta))
			Nodes.append((l1, b1, l2, b2, lbl, stPose, endPose))			
		else:
			l1 = l2
			b1 = b2
			l2 = l1 + mag*math.cos(math.radians(theta))
			b2 = b1 + mag*math.sin(math.radians(theta))

			# if((lbl == 0) and (abs(b1-b2) < 0.25)):
			# 	continue
			Nodes.append((l1, b1, l2, b2, lbl, stPose, endPose))
			
		i = i+1

	NodesN = []; i = 0
	l1 = 0; b1 = 0; l2 = 0; b2 = 0;
	for line in Nodes_manh:
		mag = line[0]; theta = line[1]; lbl = line[2]; stPose = line[3]; endPose = line[4]
		
		if((theta - Nodes_manh[i-1][1] == 180) and (i != 0)):
			l1 = l2 - 0.2
			b1 = b2 + 0.2
			# l1 = l2
			# b1 = b2
			l2 = l1 + mag*math.cos(math.radians(theta))
			b2 = b1 + mag*math.sin(math.radians(theta))
			NodesN.append((l1, b1, l2, b2, lbl, stPose, endPose))			
		else:
			l1 = l2
			b1 = b2
			l2 = l1 + mag*math.cos(math.radians(theta))
			b2 = b1 + mag*math.sin(math.radians(theta))

			# if((lbl == 0) and (abs(b1-b2) < 0.25)):
			# 	continue
			NodesN.append((l1, b1, l2, b2, lbl, stPose, endPose))
			
		i = i+1

	return Nodes, NodesN


def writeMlp(Nodes, dense=True):
	# Saving format: Rightward = b and Downwards = l

	poses = open("mlp_in.txt", 'w')
	densePoses = open("mlp_in_dense.txt", 'w')
	for line in Nodes:
		if(dense == True):
			info = str(-line[1])+" "+str(line[0])+" "+ str(-line[3])+" "+ str(line[2])+" "+ str(line[4])
			
			infoDense = str(line[5])+" "+str(line[6])
			densePoses.write(infoDense)
			densePoses.write("\n")
		else:
			info = str(-line[1])+" "+str(line[0])+" "+ str(-line[3])+" "+ str(line[2])+" "+ str(line[4])
		
		poses.write(info)
		poses.write("\n")

	poses.close()
	densePoses.close()


def getConstr(Nodes_manh, Nodes):
	poses = []
	
	for i in range(len(Nodes_manh)):
		theta = Nodes_manh[i][1]
		l1 = Nodes[i][0]; b1 = Nodes[i][1]; l2 = Nodes[i][2]; b2 = Nodes[i][3]
		stId = Nodes[i][5]; endId = Nodes[i][6]

		poses.append((l1, b1, math.radians(theta), stId+1))
		poses.append((l2, b2, math.radians(theta), endId-1))

	return poses


# def drawConstr(poses):
# 	poses = np.asarray(poses)
	
# 	ax = plt.subplot(111)

# 	for i in range(len(poses)):
# 		x = poses[i, 0]; y = poses[i, 1]; theta = poses[i, 2]

# 		x2 = math.cos(theta) + x
# 		y2 = math.sin(theta) + y
# 		plt.plot([x, x2], [y, y2], 'm->')

# 	ax.plot(poses[:, 0], poses[:, 1], 'ro')
# 	plt.plot(poses[:, 0], poses[:, 1], 'k-')

# 	plt.show()


def getIcpId(poses, LBL):
	fill = np.zeros((len(poses), len(poses)))
	icp = []

	for i in range(len(poses)):
		minDist = 10000; idi = -1; idj = -1; denseIdi = -1; denseIdj = -1
		thres = 1.9

		line = poses[i]
		x1 = line[0]; y1 = line[1]; denseId1 = line[3]; theta1 = line[2]; lbl1 = LBL[denseId1]
		
		if(lbl1 == 1):
			for j in range(len(poses)):
				lbl2 = LBL[poses[j][3]]; theta2 = poses[j][2]

				a = np.array([math.cos(theta1), math.sin(theta1)]).reshape(2, 1)
				b = np.array([math.cos(theta2), math.sin(theta2)]).reshape(2, 1)
				# print(np.dot(a.T, b))

				if((fill[i, j] == 1) or (fill[j, i] == 1) or (i == j) or (lbl2 != 1) or (np.dot(a.T, b) != 1)):
					continue
				else:
					line = poses[j]
					x2 = line[0]; y2 = line[1]; denseId2 = line[3]; theta2 = line[2]; lbl2 = LBL[denseId2]
					
					dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)
				
					if(dist < minDist):
						minDist = dist; idi = i; idj = j; denseIdi = denseId1; denseIdj = denseId2

					fill[i, j] = 1; fill[j, i] = 1

			if(minDist < thres):
				# print(minDist, idi, idj)
				icp.append((denseIdi, denseIdj))
	return icp


def writeIcp(icp):
	poses = open("icp.txt", 'w')

	for i in range(len(icp)):
		info = str(3100+icp[i][0]) + " " + str(3100+icp[i][1]) + "\n"
		poses.write(info)

	poses.close()


def start(fileName):
	(X, Y, THETA, LBL) = read(fileName)
	
	X = X[3100: 6000]; Y = Y[3100: 6000]; LBL = LBL[3100: 6000]; THETA = THETA[3100: 6000]
	# X = X[3100: -1]; Y = Y[3100: -1]; LBL = LBL[3100: -1]
	
	print(len(X))
	# draw(X, Y, LBL)

	Node_meta = meta(X, Y, LBL)
	Node_meta = outRemove(Node_meta)
	# drawMeta(Node_meta)

	Nodes = []

	thetas = []
	for line in Node_meta:
		lbl = line[4]
		x = [line[0], line[2]]
		y = [line[1], line[3]]

		theta = calcTheta(x[0], x[1], y[0], y[1])
		thetas.append(theta)
	# drawTheta(Node_meta, thetas)
	
	Nodes_manh = manh(Node_meta, thetas)
	Nodes = extManh(Nodes_manh)
	drawManh(Nodes)

	# writeMlp(Nodes, dense=True)

	poses = getConstr(Nodes_manh, Nodes)
	# print(poses[0])
	# drawConstr(poses)

	icpId = getIcpId(poses, LBL)
	# print(icpId)
	# writeIcp(icpId)

	return (poses, icpId)


def startPose(X, Y, THETA, LBL):
	# (X, Y, THETA, LBL) = read(fileName)
	
	# X = X[3100: 6000]; Y = Y[3100: 6000]; LBL = LBL[3100: 6000]; THETA = THETA[3100: 6000]	
	# draw(X, Y, LBL)

	Node_meta = meta(X, Y, LBL)
	Node_meta = outRemove(Node_meta)
	# drawMeta(Node_meta)

	Nodes = []

	thetas = []
	for line in Node_meta:
		lbl = line[4]
		x = [line[0], line[2]]
		y = [line[1], line[3]]

		theta = calcTheta(x[0], x[1], y[0], y[1])
		thetas.append(theta)
	# drawTheta(Node_meta, thetas)
	
	Nodes_manh = manh(Node_meta, thetas)
	
	Nodes, NodesN = extManh(Nodes_manh)
	drawManh(Nodes)

	# writeMlp(Nodes, False)

	poses = getConstr(Nodes_manh, Nodes)
	# drawConstr(poses)

	icpId = getIcpId(poses, LBL)
	# writeIcp(icpId)

	return (poses, NodesN)


if __name__ == '__main__':
	fileName = str(argv[1])
	
	poses = start(fileName)
