import tensorflow as tf
from keras import backend

class RandomNotMaxDirLoss(tf.keras.losses.Loss):
    '''
    get the gradient toward some number in range of random wrong label.
    y_pred should be logit output rather than softmax output.
    '''
    def __init__(self, reduction=tf.keras.losses.Reduction.SUM):
        #because of computing input gradients, just sum. the input gradients of each image is independent.
        super().__init__(reduction=reduction)
    def call(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_max_index = tf.argmax(y_pred, axis=-1)
        y_max_mask = tf.one_hot(y_max_index, y_pred.shape[-1])
        y_not_max_mask = 1-tf.cast(y_max_mask, y_pred.dtype)

        #loss
        pick_mask = y_not_max_mask*(0.1+tf.random.uniform(y_pred.shape))#shape=y_pred.shape. from 0.1 to 1.1. add 0.1 to avoid picking up y_true.
        y_mask = tf.cast(tf.one_hot(tf.argmax(pick_mask, axis=-1), y_pred.shape[-1]), y_pred.dtype)

        loss = -tf.reduce_sum(y_pred*y_mask, axis=-1)#push target up
        return loss
class RandomFalseDirLoss(tf.keras.losses.Loss):
    '''
    get the gradient toward some number in range of random wrong label.
    y_pred should be logit output rather than softmax output.
    '''
    def __init__(self, reduction=tf.keras.losses.Reduction.SUM):
        #because of computing input gradients, just sum. the input gradients of each image is independent.
        super().__init__(reduction=reduction)
    def call(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, tf.int32)
        y_true_mask = tf.one_hot(y_true, y_pred.shape[-1])

        #loss of false prediction
        y_false_mask = tf.cast(1-y_true_mask, y_pred.dtype)# shape=y_pred.shape. should not choose gradient toward true label
        pick_mask = y_false_mask*(0.1+tf.random.uniform(y_pred.shape))#shape=y_pred.shape+(pick_count,). from 0.1 to 1.1. add 0.1 to avoid picking up y_true.
        y_mask = tf.cast(tf.one_hot(tf.argmax(pick_mask, axis=-1), y_pred.shape[-1]), y_pred.dtype)

        loss = -tf.reduce_sum(y_pred*y_mask, axis=-1)#push target up
        return loss
class RandomSeveralFalseDirLoss(tf.keras.losses.Loss):
    '''
    get the gradient toward some number in range of random wrong label.
    y_pred should be logit output rather than softmax output.
    '''
    def __init__(self, pick_count, reduction=tf.keras.losses.Reduction.SUM):
        #because of computing input gradients, just sum. the input gradients of each image is independent.
        super().__init__(reduction=reduction)
        self.pick_count = pick_count# range = [minval, maxval)
    def call(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, tf.int32)
        y_true_mask = tf.one_hot(y_true, y_pred.shape[-1])

        #loss of false prediction
        y_false_mask = tf.cast(1-y_true_mask, y_pred.dtype)[..., tf.newaxis]# shape=y_pred.shape+(1,). should not choose gradient toward true label
        pick_mask = y_false_mask*(0.1+tf.random.uniform(y_pred.shape+(self.pick_count,)))#shape=y_pred.shape+(pick_count,). from 0.1 to 1.1. add 0.1 to avoid picking up y_true.
        temp_y_mask = tf.cast(tf.one_hot(tf.argmax(pick_mask, axis=1), y_pred.shape[-1], axis=1), tf.bool)
        y_mask = tf.reduce_any(temp_y_mask, axis=-1)
        y_mask = tf.cast(y_mask, y_pred.dtype)

        y_mask_base = tf.reduce_sum(y_mask, axis=-1)
        y_mask_base = tf.where(y_mask_base>0, y_mask_base, tf.ones_like(y_mask_base))
        loss = -tf.reduce_sum(y_pred*y_mask, axis=-1)/y_mask_base#push target up
        return loss

class MinWrongDirLoss(tf.keras.losses.Loss):
    '''
    get the gradient toward min wrong label.
    y_pred should be raw output rather than softmax output.
    '''
    def __init__(self, reduction=tf.keras.losses.Reduction.SUM):
        #because of computing input gradients, just sum. the input gradients of each image is independent.
        super().__init__(reduction=reduction)
    def call(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, tf.int32)
        y_true_mask = tf.one_hot(y_true, y_pred.shape[-1])

        y_false_mask = tf.cast(1-y_true_mask, y_pred.dtype)#should not choose gradient toward true label
        max_y = tf.reduce_max(y_pred, axis=-1, keepdims=True)#avoid choosing true prediction
        y_mask = tf.cast(tf.one_hot(tf.argmin((y_pred-max_y)*y_false_mask, axis=-1), y_pred.shape[-1]), y_pred.dtype)
        loss = -tf.reduce_sum(y_pred*y_mask, axis=-1)#push target up
        return loss
class MaxWrongDirLoss(tf.keras.losses.Loss):
    '''
    get the gradient toward max wrong label.
    y_pred should be raw output rather than softmax output.
    '''
    def __init__(self, reduction=tf.keras.losses.Reduction.SUM):
        #because of computing input gradients, just sum. the input gradients of each image is independent.
        super().__init__(reduction=reduction)
    def call(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, tf.int32)
        y_true_mask = tf.one_hot(y_true, y_pred.shape[-1])

        y_false_mask = tf.cast(1-y_true_mask, y_pred.dtype)#should not choose gradient toward true label
        min_y = tf.reduce_min(y_pred, axis=-1, keepdims=True)#avoid choosing true prediction
        y_mask = tf.cast(tf.one_hot(tf.argmax((y_pred-min_y)*y_false_mask, axis=-1), y_pred.shape[-1]), y_pred.dtype)
        loss = -tf.reduce_sum(y_pred*y_mask, axis=-1)#push target up
        return loss
class MeanWrongDirLoss(tf.keras.losses.Loss):
    '''
    get the gradient toward max wrong label.
    y_pred should be raw output rather than softmax output.
    '''
    def __init__(self, reduction=tf.keras.losses.Reduction.SUM):
        #because of computing input gradients, just sum. the input gradients of each image is independent.
        super().__init__(reduction=reduction)
    def call(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, tf.int32)
        y_true_mask = tf.one_hot(y_true, y_pred.shape[-1])

        y_false_mask = tf.cast(1-y_true_mask, y_pred.dtype)#should not choose gradient toward true label
        y_mask = y_false_mask/tf.reduce_sum(y_false_mask, axis=-1, keepdims=True)
        loss = -tf.reduce_sum(y_pred*y_mask, axis=-1)#push target up
        return loss
class MinDirLoss(tf.keras.losses.Loss):
    '''
    get the gradient toward lowest predicted label.
    y_pred should be raw output rather than softmax output.
    '''
    def __init__(self, reduction=tf.keras.losses.Reduction.SUM):
        #because of computing input gradients, just sum. the input gradients of each image is independent.
        super().__init__(reduction=reduction)
    def call(self, y_true, y_pred):
        # don't use y_true, just use y_pred
        y_pred = tf.convert_to_tensor(y_pred)
        y_max_index = tf.argmin(y_pred, axis=-1)
        y_mask = tf.one_hot(y_max_index, y_pred.shape[-1])

        y_mask = tf.cast(y_mask, y_pred.dtype)
        loss = -tf.reduce_sum(y_pred*y_mask, axis=-1)#push target up
        return loss
class MaxDirLoss(tf.keras.losses.Loss):
    '''
    get the gradient toward max label.
    y_pred should be raw output rather than softmax output.
    '''
    def __init__(self, reduction=tf.keras.losses.Reduction.SUM):
        #because of computing input gradients, just sum. the input gradients of each image is independent.
        super().__init__(reduction=reduction)
    def call(self, y_true, y_pred):
        # don't use y_true, just use y_pred
        y_pred = tf.convert_to_tensor(y_pred)
        y_max_index = tf.argmax(y_pred, axis=-1)
        y_mask = tf.one_hot(y_max_index, y_pred.shape[-1])

        y_mask = tf.cast(y_mask, y_pred.dtype)
        loss = -tf.reduce_sum(y_pred*y_mask, axis=-1)#push target up
        return loss
class TargetDirLoss(tf.keras.losses.Loss):
    '''
    get the gradient toward true label.
    y_pred should be raw output rather than softmax output.
    '''
    def __init__(self, reduction=tf.keras.losses.Reduction.SUM):
        #because of computing input gradients, just sum. the input gradients of each image is independent.
        super().__init__(reduction=reduction)
    def call(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, tf.int32)
        #loss of true prediction
        y_true_mask = tf.cast(tf.one_hot(y_true, y_pred.shape[-1]), y_pred.dtype)
        loss = -tf.reduce_sum(y_true_mask*y_pred, axis=-1)#push target up
        return loss

class ChosenDirLoss(tf.keras.losses.Loss):
    '''
    get the gradient toward min wrong label.
    y_pred should be raw output rather than softmax output.
    '''
    def __init__(self, target_index, reduction=tf.keras.losses.Reduction.SUM):
        #because of computing input gradients, just sum. the input gradients of each image is independent.
        super().__init__(reduction=reduction)
        self.target_index = target_index
    def call(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        target_index = tf.cast(self.target_index, tf.int32)
        #loss of true prediction
        target_mask = tf.cast(tf.one_hot(target_index, y_pred.shape[-1]), y_pred.dtype)
        loss = -tf.reduce_sum(target_mask*y_pred, axis=-1)#push target up
        return loss

class ReverseLoss(tf.keras.losses.Loss):
    '''
    reverse all kind of loss
    '''
    def __init__(self, loss_object):
        super().__init__()
        self.loss_object = loss_object
    def __call__(self, y_true, y_pred, sample_weight=None):#apply sample_weight, directly call
        loss = self.loss_object(y_true, y_pred, sample_weight=sample_weight)
        return -loss

class MultiHotKL(tf.keras.losses.Loss):#
    def __init__(self, axis=-1, sparse_label=True, reduction=tf.keras.losses.Reduction.AUTO):
        #y_pred should be raw prediction without softmax
        #y_true should be sparse. that is, the index of the true. otherwise, sparse_label should be False.
        super().__init__(reduction=reduction)
        self.axis = axis
        self.sparse_label = sparse_label

    def __call__(self, y_true, y_pred, sample_weight=None):
        #overwrite for special sample_weight 
        #reduction will be disabled
        return self.call(y_true, y_pred, sample_weight=sample_weight)

    def call(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.convert_to_tensor(y_pred)
        if not isinstance(sample_weight, type(None)):
            sample_mask = sample_weight[:, tf.newaxis]
            sample_mask = tf.cast(sample_mask, y_pred.dtype)
            sample_mask = sample_mask*tf.ones_like(y_pred)
        else:
            sample_mask = tf.ones_like(y_pred)
            sample_mask = tf.cast(sample_mask, y_pred.dtype)
        y_true = tf.cast(y_true, tf.int32)
        if self.sparse_label:
            y_true_mask = tf.cast(tf.one_hot(y_true, y_pred.shape[-1]), y_pred.dtype)
        else:
            y_true_mask = tf.cast(y_true[:, tf.newaxis], y_pred.dtype)
        y_true_mask = y_true_mask*sample_mask#mask out sample by sample_weight
        true_num = tf.reduce_sum(y_true_mask, axis=self.axis, keepdims=True)
        true_num_div = tf.where(true_num>0, true_num, tf.ones_like(true_num))
        y_false_mask = 1-y_true_mask
        y_false_mask = y_false_mask*sample_mask#mask out sample by sample_weight
        false_num = tf.reduce_sum(y_false_mask, axis=self.axis, keepdims=True)
        KL_target = y_true_mask/true_num_div

        #get bias for true. make sure the same pred from true get the same softmax output
        false_num_ratio = false_num/true_num_div
        true_bias = y_true_mask*tf.math.log(false_num_ratio/9)#compared to cifar10, 9 false num
        bias_y_pred = y_pred+true_bias

        #loss
        exp_pred = tf.exp(bias_y_pred-tf.reduce_max(bias_y_pred, axis=self.axis, keepdims=True))#minus max to avoid overflow
        exp_pred = exp_pred*sample_mask#mask out sample by sample_weight
        softmax_pred = exp_pred/(tf.reduce_sum(exp_pred, axis=self.axis, keepdims=True))

        KL_target = backend.clip(KL_target, backend.epsilon(), 1)
        softmax_pred = backend.clip(softmax_pred, backend.epsilon(), 1)
        loss = tf.reduce_sum(KL_target * tf.math.log(KL_target / softmax_pred), axis=self.axis)
        loss = tf.reduce_mean(loss)
        return loss

class FalseUniformKL(tf.keras.losses.Loss):
    def __init__(self, axis=-1, sparse_label=True, reduction=tf.keras.losses.Reduction.AUTO):
        #y_pred should be raw prediction without softmax
        #y_true should be sparse. that is, the index of the true. otherwise, sparse_label should be False.
        super().__init__(reduction=reduction)
        self.axis = axis
        self.sparse_label = sparse_label
        # self.block_incorrect = block_incorrect
    def __call__(self, y_true, y_pred, sample_weight=None):
        #overwrite for special sample_weight 
        #reduction will be disabled
        return self.call(y_true, y_pred, sample_weight=sample_weight)

    def _get_loss(self, KL_target, y_pred, y_false_mask):
        # temp_y_pred = y_pred*y_false_mask#avoid too small true output, should be useless.
        temp_y_pred = y_pred-tf.reduce_min(y_pred, axis=self.axis, keepdims=True)
        temp_y_pred = temp_y_pred*y_false_mask#set origin output to min to avoid too big true output making false output zero (overflow).
        exp_pred = tf.exp(temp_y_pred-tf.reduce_max(temp_y_pred, axis=self.axis, keepdims=True))#minus max to avoid overflow
        exp_pred = exp_pred*y_false_mask#mask out true prediction
        softmax_pred = exp_pred/tf.reduce_sum(exp_pred, axis=self.axis, keepdims=True)

        KL_target = backend.clip(KL_target, backend.epsilon(), 1)
        softmax_pred = backend.clip(softmax_pred, backend.epsilon(), 1)
        loss = tf.reduce_sum(KL_target * tf.math.log(KL_target / softmax_pred), axis=self.axis)
        return loss

    def call(self, y_true, y_pred, sample_weight):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, tf.int32)
        if self.sparse_label:
            y_true_mask = tf.cast(tf.one_hot(y_true, y_pred.shape[-1]), y_pred.dtype)
        else:
            y_true_mask = tf.cast(y_true[:, tf.newaxis], y_pred.dtype)
        y_false_mask = 1-y_true_mask
        # if self.block_incorrect:
        #     y_pred_index = tf.argmax(y_pred, output_type=tf.int32, axis=1)
        #     correct_mask = tf.cast(tf.equal(y_pred_index, y_true), tf.float32)[:, tf.newaxis]
        #     y_false_mask = y_false_mask*correct_mask
        KL_target = y_false_mask/tf.reduce_sum(y_false_mask, axis=self.axis, keepdims=True)
        
        #loss, two direction to avoid gradient clip
        pos_loss = self._get_loss(KL_target, y_pred, y_false_mask)
        neg_loss = self._get_loss(KL_target, -y_pred, y_false_mask)
        loss = (pos_loss+neg_loss)/2
        return loss

class FalseKL():#not a formal tensorflow loss
    def __init__(self, axis=-1, reduction=tf.keras.losses.Reduction.AUTO):
        #y_pred should be raw prediction without softmax
        self.KL = tf.keras.losses.KLDivergence(reduction=reduction)
        self.axis = axis
    def __false_softmax(self, y_label, raw_y):
        y_true_mask = tf.cast(tf.one_hot(y_label, raw_y.shape[-1]), raw_y.dtype)
        y_false_mask = 1-y_true_mask
        temp_raw_y = raw_y-tf.reduce_min(raw_y, axis=self.axis, keepdims=True)
        temp_raw_y = temp_raw_y*y_false_mask#set origin output to min to avoid too big true output making false output zero (overflow).
        exp_pred = tf.exp(temp_raw_y-tf.reduce_max(temp_raw_y, axis=self.axis, keepdims=True))#minus max to avoid overflow
        exp_pred = exp_pred*y_false_mask#mask out true prediction
        softmax_pred = exp_pred/tf.reduce_sum(exp_pred, axis=self.axis, keepdims=True)
        return softmax_pred
    def __call__(self, y_target, y_pred, y_label, sample_weight=None):
        y_pred = tf.convert_to_tensor(y_pred)
        y_target = tf.convert_to_tensor(y_target)
        y_label = tf.cast(y_label, tf.int32)
        KL_target = tf.stop_gradient(self.__false_softmax(y_label, y_target))#should not target to src
        KL_src = self.__false_softmax(y_label, y_pred)
        
        #loss
        loss = self.KL(KL_target, KL_src, sample_weight=sample_weight)
        return loss

class MeanWrapper(tf.keras.metrics.Metric):
    '''
    should be placed in "acc" of dict
    '''
    def __init__(self, value_func, name='mean_target', **kwargs):
        super().__init__(name=name, **kwargs)
        self.value_func = value_func
        self.mean_func = tf.keras.metrics.Mean(name="self_mean")
    def update_state(self, y_true, y_pred, sample_weight=None):
        self.mean_func(self.value_func(y_true, y_pred), sample_weight=sample_weight)
    def result(self):
        return self.mean_func.result()
class ShowTarget():
    # value_func should be used with MeanWrapper
    def __call__(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, tf.int32)
        #loss of true prediction
        y_true_mask = tf.cast(tf.one_hot(y_true, y_pred.shape[-1]), y_pred.dtype)
        target = tf.reduce_sum(y_true_mask*y_pred, axis=-1)
        return target
class ShowMax():
    # value_func should be used with MeanWrapper
    def __call__(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        max_pred = tf.reduce_max(y_pred, axis=-1)
        return max_pred
class ShowMaxWrong():
    # value_func should be used with MeanWrapper
    def __call__(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, tf.int32)
        y_true_mask = tf.one_hot(y_true, y_pred.shape[-1])
        y_false_mask = tf.cast(1-y_true_mask, y_pred.dtype)
        min_y = tf.reduce_min(y_pred, axis=-1, keepdims=True)#avoid choosing true prediction if all smaller than 0
        y_mask = tf.cast(tf.one_hot(tf.argmax((y_pred-min_y)*y_false_mask, axis=-1), y_pred.shape[-1]), y_pred.dtype)
        max_wrong = tf.reduce_sum(y_pred*y_mask, axis=-1)
        return max_wrong
class ShowMeanWrong():
    # value_func should be used with MeanWrapper
    def __call__(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, tf.int32)
        y_true_mask = tf.one_hot(y_true, y_pred.shape[-1])
        y_false_mask = tf.cast(1-y_true_mask, y_pred.dtype)
        mean_wrong = tf.reduce_sum(y_pred*y_false_mask, axis=-1)/tf.reduce_sum(y_false_mask, axis=-1)
        return mean_wrong

