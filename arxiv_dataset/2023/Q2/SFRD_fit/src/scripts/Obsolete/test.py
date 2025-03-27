import subprocess
import time

jid = 16565878
command = "sacct  -j %s --format State "%(jid)
p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
nline = 0
while True:
	line = p.stdout.readline()
	nline +=1
	#print(line)
	if not line:
		break
	if nline == 3:
		break
#text = p.stdout.read()
result = str(line)
print('result = ', result)
if b"COMPLETE" in line:
	print('YAY your job finished!')
elif b"FAILED" in line:
	print('Job failed :( %s')%jid
elif np.logical_or(b"RUNNING" in line, b"PENDING" in line):
	print('darn, still running, check back in 10 sec')
	time.sleep(10) # Sleep until the coscmic integration slurms should be done

