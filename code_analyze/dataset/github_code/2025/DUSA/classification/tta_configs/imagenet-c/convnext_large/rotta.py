_base_ = ["../convnext_large.py", "../imagenet-c-bs64.py"]
runner_type = "RottaCls"
update_auxiliary = False

model = dict(
    auxiliary_model=None
)
