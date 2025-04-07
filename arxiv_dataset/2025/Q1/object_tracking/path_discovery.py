import lakefs_config as cfg
import boto3
import awswrangler as wr
from util import get_aux_json, get_aux_config, write_aux_json

def sensor_path_discovery(computer, scope, boto3_session):
    paths = []
    print('sessions:', scope["sessions"])
    for session in scope["sessions"]:
        # get all paths for this sensor that are trials of sessions
        print(f"s3://{cfg.repo}/{cfg.download_branch}/{computer}/session.{session.zfill(3)}*/*/*")
        paths.extend(wr.s3.list_objects(f"s3://{cfg.repo}/{cfg.download_branch}/{computer}/session.{session.zfill(3)}*/*/*", boto3_session=boto3_session))
    paths = ['/'.join(p.split('/')[4:]) for p in paths] # dont include repo and branch
    print('found paths:', paths)
    write_aux_json(name=f'{computer}_paths', data=paths)

def recovered_mocap_path_discovery(scope, boto3_session):
    paths = []
    for session in scope["sessions"]:
        # get all paths for this sensor that are trials of sessions
        paths.extend(wr.s3.list_objects(f"s3://{cfg.repo}/{cfg.download_branch}/mocap_recover/s{session.zfill(3)}*", boto3_session=boto3_session))
    paths = ['/'.join(p.split('/')[4:]) for p in paths] # dont include repo and branch
    write_aux_json(name=f'mocap_matched_paths', data=paths)

def annotations_path_discovery(scope, boto3_session):
    paths = []
    for session in scope["sessions"]:
        # get all paths for this sensor that are trials of sessions
        paths.extend(wr.s3.list_objects(f"s3://{cfg.repo}/{cfg.download_branch}/annotations/s{session.zfill(3)}*/*.annotation.*.eaf", boto3_session=boto3_session))
        paths.extend(wr.s3.list_objects(f"s3://{cfg.repo}/{cfg.download_branch}/annotations/s{session.zfill(3)}*/*.transcript.*.eaf", boto3_session=boto3_session))
    paths = ['/'.join(p.split('/')[4:]) for p in paths] # dont include repo and branch
    write_aux_json(name='annotations_paths', data=paths)

if __name__ == "__main__":
    scope = get_aux_config()

    my_session = boto3.Session()
    wr.config.s3_endpoint_url = cfg.endpoint

    sensor_path_discovery(computer='lsl', scope=scope, boto3_session=my_session)
    sensor_path_discovery(computer='video', scope=scope, boto3_session=my_session)
    recovered_mocap_path_discovery( scope=scope, boto3_session=my_session)
    annotations_path_discovery( scope=scope, boto3_session=my_session)

    print('path discovery successful')