_base_ = ["../convnext_large.py", "../imagenet-c-bs64-continual.py"]
runner_type = "SourceCls"
update_auxiliary = False

model = dict(
    auxiliary_model=None
)
