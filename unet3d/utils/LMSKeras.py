from tensorflow.python.keras.callbacks import Callback

from tensorflow.contrib.lms import LMS
import tensorflow as tf
from tensorflow.python.framework import ops
tf.logging.set_verbosity(tf.logging.INFO)


class LMSKerasCallback(Callback):
    def set_model(self, model):
        self.model = model
        print('In LMSKerasCallback.  Model: %s' % self.model)
        optimizer_name = self.model.optimizer.__class__.__name__
        lmsMod = LMS({'training/'+optimizer_name+'/gradients'},
                     graph=ops.get_default_graph(),
                     starting_op_names={'input_1', 'conv3d/kernel/read'},
                     n_tensors=50,
                     lb=5,
                     branch_threshold=100,
                     swap_branches=True,
                     debug=True,
                     debug_level=1
        )
        lmsMod.run()
