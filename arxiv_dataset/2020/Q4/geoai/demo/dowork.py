from flask import Flask, request, redirect, url_for, make_response, abort
from werkzeug import secure_filename
from pymongo import Connection
from bson.objectid import ObjectId
from gridfs import GridFS
from gridfs.errors import NoFile
import numpy as np
import cv2
import sys
from sklearn.externals import joblib

from helpers import clarity_rating

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
DB = Connection().geoai
FS = GridFS(DB)

app = Flask(__name__)

def allowed_file(filename):
	return '.' in filename and \
			filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
	if request.method == 'POST':
		file = request.files['file']
		if file and allowed_file(file.filename):
			filename = secure_filename(file.filename)
			oid = FS.put(file, content_type=file.content_type, filename=filename)
			return redirect(url_for('serve_gridfs_file', oid=str(oid)))
	return '''
	<!DOCTYPE html>
	<html>
	<head>
	<title>Upload new file</title>
	</head>
	<body>
	<h1>Upload new file</h1>
	<form action="" method="post" enctype="multipart/form-data">
	<p><input type="file" name="file"></p>
	<p><input type="submit" value="Upload"></p>
	</form>
	<a href="%s">All files</a>
	</body>
	</html>
	''' % url_for('list_gridfs_files')

@app.route('/files')
def list_gridfs_files():
	classifier = joblib.load("demo")
	files = [FS.get_last_version(file) for file in FS.list()]
	i=0
	data=[]
	idset=[]
	result=[]
	while i<len(files):
		tempName=files[i].filename
		tmpid=files[i]._id
		data.append(tempName)
		idset.append(tmpid)
		file=FS.get(idset[i])
		f=open(data[i],"w")
		oof=file.read()
		f.write(oof)
		f.close()
		i=i+1

	i=0
	while i<len(data):
		image=cv2.imread(data[i])
		height=np.size(image,0)
		width=np.size(image,1)
		if height<200:
			cv2.resize(image,(200,200))
		if width<200:
			cv2.resize(image,(200,200))
			
		count=0.0
		pre=0;
		total=0.0
		y=200
		x=200
		while (y < height):
			while (x < width):
				cropped = image[y - 200:y, x - 200:x]
				if cropped is not None:
					cv2.imwrite("temp.jpg", cropped)
					hi = cv2.imread("temp.jpg")
					if hi is not None:
						prediction = classifier.predict(clarity_rating(hi))
						count = count + 1

						if prediction == '0':
							temp=0.0
						if prediction == '1':
							temp=1.0
						if prediction == '2':
							temp=2.0
						if prediction == '3':
							temp=3.0
						if prediction == '4':
							temp=4.0
						if prediction == '5':
							temp=5.0
						if prediction == '6':
							temp=6.0
						if prediction == '7':
							temp=7.0
						if prediction == '8':
							temp=8.0
						if prediction == '9':
							temp=9.0
						total = total + temp
						temp=0.0
				x=x+200
			y=y+200
			x=0
		total=total/count
		result.append(data[i])
		result.append(str(total))
	return result[0]+"___"+result[1]+"___"+result[2]



if __name__=="__main__":
	app.run();
