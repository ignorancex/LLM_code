_base_ = ['./base.py']

# Custom model
model = dict(
    test_cfg=dict(
        step=1,),
    decode_head=dict(
        accuracy=[
            dict(type='CollisionAccuracy', acc_name='acc_collision_dynamic'),
            dict(type='L2Accuracy', reduction='mean', acc_name='acc_l2_dynamic'),],),)
