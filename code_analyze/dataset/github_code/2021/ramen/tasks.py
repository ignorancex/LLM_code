#!/usr/bin/env python3
import glob
from fabric import Connection
from invoke import task

HOST        = 'ec2-13-212-116-131.ap-southeast-1.compute.amazonaws.com'
STORAGE     = '172.31.43.166'
USER        = 'ubuntu'
ROOT        = '/mnt/efs/ramen'
TRAINROOT   = f'/home/{USER}/ramen'
PROJECT_NAME= 'ramen'
TBPORT      =  6006
REMOTE      = '{user}@{host}:{root}'.format(user=USER, host=HOST, root=ROOT)
VENV        = 'pytorch_p36'
MODEL       = 'models'
OUTPUT      = 'output_tests'
LOGS        = 'logs'
DATA        = 'data'
DATASET_LOCATION = 'honours-datasets/dataset'
GIT_REPO = 'https://github.com/bhanukaManesha/ramen.git'

ALL = [
    'requirements.in',
    'scripts',
    'preprocess',
    'postprocess',
    'criterion',
    'components',
    'charts',
    'debug',
    'models',
    '*.py'
]

TRAIN_DATASET = 'CLEVR_CoGenTA'
TEST_DATASET = 'CLEVR_CoGenTA'
MODEL_NAME = 'ramen'

@task
def connect(ctx):
    ctx.conn = Connection(host=HOST, user=USER)

@task
def close(ctx):
    ctx.conn.close()

@task(pre=[connect], post=[close])
def ls(ctx):
    with ctx.conn.cd(ROOT):
        ctx.conn.run('find | sed \'s|[^/]*/|- |g\'')

@task(pre=[connect], post=[close])
def reset(ctx):
    ctx.conn.run('rm -rf {}'.format(ROOT), pty=True)

@task(pre=[connect], post=[close])
def killpython(ctx):
    ctx.conn.run('pkill -9 python', pty=True)


# Setup the environment
@task(pre=[connect], post=[close])
def setup(ctx):
    ctx.conn.run('sudo apt-get update')
    ctx.conn.run('sudo apt install -y dtach')
    ctx.conn.run('sudo apt-get install nfs-common -y')
    with ctx.conn.cd('/mnt/'):
        ctx.conn.run('sudo mkdir efs')
        ctx.conn.run(f'sudo mount -t nfs4 {STORAGE}:/ /mnt/efs')

    with ctx.conn.cd(f'/home/{USER}'):
        ctx.conn.run('sudo mkdir ramen')

    with ctx.conn.cd('/home/ubuntu/ramen/'):
        ctx.conn.run(f'sudo chmod -R 777 .')
        ctx.conn.run(f'sudo chmod -R +x .')

    with ctx.conn.cd('/mnt/'):
        ctx.conn.run('sudo rsync -ar --exclude dataset /home/ubuntu/ramen/')
        ctx.conn.run('sudo mkdir /home/ubuntu/ramen/dataset/')

@task(pre=[connect], post=[close])
def fix(ctx):
    # locked error
    ctx.conn.run('sudo rm /var/lib/dpkg/lock')
    ctx.conn.run('sudo dpkg --configure -a')

@task(pre=[connect], post=[close])
def gpustats(ctx):
    with ctx.conn.prefix('source activate pytorch_p36'):
        ctx.conn.run('watch -n0.1 nvidia-smi', pty=True)

@task(pre=[connect], post=[close])
def cpustats(ctx):
    with ctx.conn.prefix('source activate pytorch_p36'):
        ctx.conn.run('htop', pty=True)


@task
def push(ctx):
    ctx.run('rsync -rv --progress {files} {remote}'.format(files=' '.join(ALL), remote=REMOTE))

@task
def pull(ctx):
    for file in ALL:
        ctx.run(f'rsync -rv --progress {REMOTE}/{file} .')

@task
def pulllogs(ctx):
    ctx.run(f'rsync -zamrv --progress --include="*/" --include="*.log" --exclude="*" {REMOTE}/dataset/ logs/')



@task(pre=[connect], post=[close])
def clean(ctx):
    REMOVE_DATASET_NAME = 'CLEVR'
    with ctx.conn.cd('/home/ubuntu/ramen/dataset/'):
        ctx.conn.run(f'sudo rm -rfv {REMOVE_DATASET_NAME}', pty=True)

@task(pre=[connect], post=[close])
def cleanresults(ctx):
    REMOVE_DATASET_NAME = 'CLEVR'
    with ctx.conn.cd(f'/home/ubuntu/ramen/dataset/{REMOVE_DATASET_NAME}/'):
        ctx.conn.run(f'sudo rm -rfv {REMOVE_DATASET_NAME}_results', pty=True)

@task(pre=[connect], post=[close])
def moveresultsbacktoefs(ctx):
    with ctx.conn.cd('/mnt/efs/ramen/'):
        ctx.conn.run(f'sudo rsync -r --progress /home/ubuntu/ramen/dataset/{TRAIN_DATASET}/{TRAIN_DATASET}_results {ROOT}/dataset/{TRAIN_DATASET}/')
        ctx.conn.run(
            f'sudo rsync -r --progress /home/ubuntu/ramen/dataset/{TRAIN_DATASET}/features {ROOT}/dataset/{TRAIN_DATASET}/')

@task(pre=[connect], post=[close])
def train(ctx, model=''):
    ctx.run('rsync -rv --progress {files} {remote}'.format(files=' '.join(ALL), remote=REMOTE))

    # with ctx.conn.cd('/mnt/efs/ramen/'):
    #     ctx.conn.run('sudo rsync -ar --progress --exclude dataset . /home/ubuntu/ramen/')
    #
    #     # comment if dataset is already present in the home folder
    #     ctx.conn.run(f'sudo rsync -r --copy-links -h --progress dataset/{TRAIN_DATASET} /home/ubuntu/ramen/dataset/')

        # comment if dataset is already present in the home folder
        # ctx.conn.run(f'sudo rsync -r --copy-links -h --progress dataset/{TRAIN_DATASET} /home/ubuntu/ramen/dataset/')

    # with ctx.conn.cd(TRAINROOT):
    #     with ctx.conn.prefix('source activate pytorch_p36'):
    #         ctx.conn.run(f'dtach -A /tmp/{PROJECT_NAME} ./scripts/{TRAIN_DATASET}/{MODEL_NAME}_{TEST_DATASET}.sh', pty=True)

    with ctx.conn.cd('/mnt/efs/ramen/'):
        with ctx.conn.prefix('source activate pytorch_p36'):
            ctx.conn.run(f'dtach -A /tmp/{PROJECT_NAME} ./scripts/{TRAIN_DATASET}/{MODEL_NAME}_{TRAIN_DATASET}.sh', pty=True)

    # with ctx.conn.cd('/mnt/efs/ramen/'):
    #     with ctx.conn.prefix('source activate pytorch_p36'):
    #         ctx.conn.run(f'dtach -A /tmp/{PROJECT_NAME} ./scripts/{TRAIN_DATASET}/{MODEL_NAME}_{TRAIN_DATASET}.sh', pty=True)

@task
def pullalldata(ctx):
    ctx.run(f'rsync -rv --progress {REMOTE} /Users/bhanukagamage/HonoursProject')

    # with ctx.conn.cd('/mnt/efs/ramen/'):
    #     with ctx.conn.prefix('source activate pytorch_p36'):
    #         ctx.conn.run(f'dtach -A /tmp/{PROJECT_NAME} ./scripts/{TRAIN_DATASET}/{MODEL_NAME}_{TRAIN_DATASET}.sh', pty=True)


@task(pre=[connect], post=[close])
def test(ctx):
    ctx.run('rsync -rv --progress {files} {remote}'.format(files=' '.join(ALL), remote=REMOTE))
    with ctx.conn.cd('/mnt/efs/ramen/'):
        ctx.conn.run(f'sudo chmod -R 777 /mnt/efs/ramen/dataset/{TRAIN_DATASET}/{TRAIN_DATASET}_results/')
        ctx.conn.run(f'sudo chmod +x ./scripts/{TRAIN_DATASET}/test_{MODEL_NAME}_{TEST_DATASET}.sh')
        with ctx.conn.prefix('source activate pytorch_p36'):
            ctx.conn.run(f'dtach -A /tmp/{PROJECT_NAME} ./scripts/{TRAIN_DATASET}/test_{MODEL_NAME}_{TEST_DATASET}.sh',
                         pty=True)

@task(pre=[connect], post=[close])
def resume(ctx):
    ctx.conn.run('dtach -a /tmp/{}'.format(PROJECT_NAME), pty=True)