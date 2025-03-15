import time
import random
import string
import os
import copy
import json
import csv
import random

from concurrent.futures import ThreadPoolExecutor

from flask import Flask, request, render_template, redirect, url_for, make_response

app = Flask(__name__)
executor = ThreadPoolExecutor(2)

taskId = 0
taskNames = ["highlighting", "counting"]
print("task is ", taskNames[taskId])
count=0


def get_shuffle_order(length):
    l_range = list(range(0, length))
    random.seed(0)
    random.shuffle(l_range)
    return l_range


def append_to_file(filepath, line):
    with open(filepath, 'a') as outfile:
        outfile.write(line+'\n')


def generateRandomCode():
    src_digits = string.digits  # string_数字
    src_uppercase = string.ascii_uppercase  # string_大写字母
    src_lowercase = string.ascii_lowercase  # string_小写字母

    # 随机生成数字、大写字母、小写字母的组成个数（可根据实际需要进行更改）
    digits_num = random.randint(1, 6)
    uppercase_num = random.randint(1, 8-digits_num-1)
    lowercase_num = 8 - (digits_num + uppercase_num)
    # 生成字符串
    password = random.sample(src_digits, digits_num) + random.sample(
        src_uppercase, uppercase_num) + random.sample(src_lowercase, lowercase_num)
    # 打乱字符串
    random.shuffle(password)
    # 列表转字符串
    new_password = ''.join(password)

    return new_password


if(not os.path.exists('results/user_info.csv')):
    executor.submit(append_to_file, 'results/user_info.csv',
                    ','.join(("workerId", "age", "gender", "degree", "screenSize", "familiar", "comment", "code", "startTime", "endTime","effect")))


################################
# highlighting Task
################################
@app.route('/result/1', methods=['POST'])
def write_result_to_disk1():

    with open('results/{}_{}.json'.format(request.cookies.get('username'), 'tasks'),'r') as f:
        new_dict=json.load(f)
        task_name=new_dict['tasksname'][new_dict['currentNum']]
        print(task_name)
        filename = new_dict['tasksname'][new_dict['currentNum']] +'-'+ request.cookies.get('username') + '.csv'
        new_dict['currentNum']+=1
        f.close()

    fp = open('results/{}_{}.json'.format(request.cookies.get('username'), 'tasks'), 'w')
    json.dump(new_dict, fp, indent=4)
    fp.close()

    flag=False
    if len(new_dict['tasksname'])==new_dict['currentNum']:
        flag=True
        pass

    # filename = 'highlighting_task-'+'.csv'
    # get the result
    result = json.loads(request.form.get('result'))

    with open('results/'+filename, 'a') as outfile:
        outfile.write(','.join(("userName","label", "test_id", "Id", "predict_value")) + '\n')
        for trial in result:
            outfile.write(','.join((request.cookies.get('username'), str(trial["label"]), str(trial["test_id"]), str(trial["id"]), str(trial["predict_value"]))) + '\n')
        # outfile.write(','.join(("userName", "fileId", "conditionId",
        #               "clusterNum", "clusterType", "userResult", "totalTime", "lightnessValue"))+'\n')
        # for trial in result:
        #     outfile.write(','.join((request.cookies.get('username'), str(trial["fileId"]), str(trial["conditionId"]), str(
        #         trial["clusterNum"]), str(trial["clusterType"]), str(trial["result"]), str(trial["totalTime"]), str(trial["lightnessValue"])))+'\n')

    if flag:
        return url_for('experimentForm',label=task_name)

    return url_for('userguide')


@app.route('/experiment/1/')
def experiment1():
    if not os.path.exists('results/{}_{}.json'.format(request.cookies.get('username'), 'tasks')):
        with open("static/data/tasks.json",'r') as f:
            d=json.load(f)
            f.close()
        while True:
            rand=random.randint(0,len(d['tasks'])-1)
            if d['num'][rand]>0:
                break
            if sum(d['num'])==0:
                print('error')
                return render_template('error.html')
        tasks=d['tasks'][rand]
        d['num'][rand]-=1
        with open("static/data/tasks.json",'w') as f_:
            json.dump(d, f_, indent=4)
        new_dict={'username':request.cookies.get('username'),'tasksname':tasks,'currentNum':0}
        fp = open('results/{}_{}.json'.format(request.cookies.get('username'), 'tasks'), 'w')
        json.dump(new_dict, fp, indent=4)
        fp.close()
    else:
        with open('results/{}_{}.json'.format(request.cookies.get('username'), 'tasks'),'r') as f:
            new_dict=json.load(f)
            #做一步current与tasksname length的判断
            if len(new_dict['tasksname'])==new_dict['currentNum']:
                return render_template('error.html')
                pass
    # if 'posAngle_pie_tp' in new_dict['tasksname'][new_dict['currentNum']]:
    #     return render_template('highlighting-single-pilot.html',label=new_dict['tasksname'][new_dict['currentNum']])
    # elif 'posAngle_pie' in new_dict['tasksname'][new_dict['currentNum']]:
    #     return render_template('posAngle-single-pilot.html',label=new_dict['tasksname'][new_dict['currentNum']])
    # elif 'posAngle_bar' in new_dict['tasksname'][new_dict['currentNum']]:
    #     return render_template('posAngle-single-pilot.html',label=new_dict['tasksname'][new_dict['currentNum']])

    return render_template('single-pilot.html',label=new_dict['tasksname'][new_dict['currentNum']])


################################
# counting Task
################################
@app.route('/result/2', methods=['POST'])
def write_result_to_disk2():
    filename = 'counting_task-' + request.cookies.get('username') + '.csv'

    # get the result
    result = json.loads(request.form.get('result'))

    with open('results/'+filename, 'a') as outfile:
        outfile.write(','.join(("userName", "fileId", "conditionId",
                      "clusterNum", "clusterType", "userResult", "totalTime"))+'\n')
        for trial in result:
            outfile.write(','.join((request.cookies.get('username'), str(trial["fileId"]), str(trial["conditionId"]), str(
                trial["clusterNum"]), str(trial["clusterType"]), str(trial["result"]), str(trial["totalTime"])))+'\n')

    return url_for('experimentForm')


@app.route('/experiment/2/')
def experiment2():
    return render_template('counting-single-pilot.html')


################################
# process functions
################################
@app.route('/user_info', methods=['POST'])
def user_info():
    code = generateRandomCode()
    print(code)
    print(request.form)
    form=request.form.to_dict()
    noise=['Stroke width','Title position','Bar width','Background color','Stroke color']
    print(form['age'])
    effect=[]
    for k,v in form.items():
        if k in noise:
            effect.append(v)
    print(effect)

    executor.submit(append_to_file, 'results/user_info.csv',
                    ','.join((str(request.cookies.get('username')), str(request.form['age']), str(request.form['sex']),
                              str(request.form['degree']), str(request.form['screen_size']), str(
                                  request.form['vis_experience']),
                              "\""+str(request.form['comment_additional'])+"\"", str(code), str(request.cookies.get('startTime')), time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),str(effect))))
    return render_template('thankyou.html', code=code)

@app.route('/form/<label>')
def experimentForm(label):
    return render_template('form.html',label=label)


@app.route('/userguide')
def userguide():
    return render_template('guide.html', trialsNum=39, taskName=taskNames[taskId])


@app.route('/consent_info', methods=['POST'])
def consent_info():
    resp = make_response(redirect(url_for('userguide')))
    resp.set_cookie('username', request.form['workerId'])
    resp.set_cookie('startTime', time.strftime(
        "%Y-%m-%d %H:%M:%S", time.localtime()))
    return resp


@app.route('/userstudy')
def userstudy():
    rows = []
    with open("results/user_info.csv", 'r') as file:
        csvreader = csv.reader(file)
        header = next(csvreader)
        for row in csvreader:
            rows.append(row)
    return render_template('consent.html', taskName=taskNames[taskId], userInfo=rows)


@app.route('/')
def index():
    return redirect(url_for('userstudy'))

@app.route('/error')
def error():
    return render_template('noqualified.html')

@app.route('/noqualified', methods=['GET'])
def noqualified():
    return url_for('error')


if __name__ == '__main__':
    app.jinja_env.auto_reload = True
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.run(debug=True, host='0.0.0.0')
