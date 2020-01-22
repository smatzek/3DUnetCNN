# Copyright 2019, 2020. IBM All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import csv
import ctypes
import time
import tensorflow as tf

from tensorflow.keras.callbacks import Callback
from tensorflow.estimator import SessionRunHook

_cudart = ctypes.CDLL('libcudart.so')
nvtx=  ctypes.CDLL("libnvToolsExt.so")
nvtx.nvtxMarkA.restype = None


class CudaProfileCallback(Callback):
    def __init__(self, profile_epoch, profile_batch_start, profile_batch_end):
        self._epoch = profile_epoch - 1
        self._start = profile_batch_start
        self._end = profile_batch_end
        self.epoch_keeper = 0
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_keeper = epoch
    def on_batch_begin(self, batch, logs=None):
        if batch == self._start and self.epoch_keeper == self._epoch:
            print('Starting cuda profiler')
            _cudart.cudaProfilerStart()
        if batch == self._end and self.epoch_keeper == self._epoch:
            print('Stopping cuda profiler')
            _cudart.cudaProfilerStop()
        nvtx.nvtxRangePushA(ctypes.c_char_p("Iteration".encode("ascii")))
    def on_batch_end(self, batch, logs=None):
        ret = nvtx.nvtxRangePop()

class LMSStats():
    def __init__(self, logfile, gpu_id=0):
        self._logfile = logfile
        self._gpu_id = gpu_id
        self._batch_start = 0
        self._num_reclaims = 0
        self._num_reclaimAll = 0
        self._defrags = 0
        self._bytes_reclaimed = 0
        self._num_allocs = 0
        self._bytes_defragged = 0


    def write_step_log_header(self):
        with open(self._logfile, 'w', newline='') as csvfile:
            statswriter = csv.writer(csvfile)
            statswriter.writerow(['step type', 'epoch', 'step',
                                  'duration', 'allocs', 'reclaimOnes',
                                  'reclaimAlls', 'defrags',
                                  'GiB reclaimed', 'GiB defragged'])
    def step_begin(self):
        self._batch_start = time.time()
        self._num_reclaims = tf.experimental.get_num_single_reclaims(self._gpu_id)
        self._num_reclaimAll = tf.experimental.get_num_full_reclaims(self._gpu_id)
        self._defrags = tf.experimental.get_num_defragmentations(self._gpu_id)
        self._bytes_reclaimed = tf.experimental.get_bytes_reclaimed(self._gpu_id)
        self._num_allocs = tf.experimental.get_num_allocs(self._gpu_id)
        self._bytes_defragged = tf.experimental.get_bytes_defragged(self._gpu_id)

    def write_step_end(self, step_type, epoch, step_number):
        row = [step_type, epoch, step_number]
        row.append(time.time()-self._batch_start) # duration
        row.append(tf.experimental.get_num_allocs(self._gpu_id)-self._num_allocs) # allocs
        row.append(tf.experimental.get_num_single_reclaims(self._gpu_id)-self._num_reclaims) # reclaims
        row.append(tf.experimental.get_num_full_reclaims(self._gpu_id)-self._num_reclaimAll) # reclaimAlls
        row.append(tf.experimental.get_num_defragmentations(self._gpu_id) - self._defrags) # defrags
        row.append(((tf.experimental.get_bytes_reclaimed(self._gpu_id)-self._bytes_reclaimed) / 1073741824.0)) # GiB reclaimed
        row.append(((tf.experimental.get_bytes_defragged(self._gpu_id)-self._bytes_defragged) / 1073741824.0)) # GiB defragged

        with open(self._logfile, 'a+', newline='') as csvfile:
            statswriter = csv.writer(csvfile)
            statswriter.writerow(row)


class LMSStatsLogger(Callback):
    def __init__(self, logfile, gpu_id=0):
        self._epoch=0
        self._lms_stats = LMSStats(logfile, gpu_id=gpu_id)

    def set_params(self, params):
        self._lms_stats.write_step_log_header()

    def on_epoch_begin(self, epoch, logs=None):
        self._epoch = epoch

    def on_epoch_end(self, epoch, logs=None):
        self._epoch = epoch

    def on_train_batch_begin(self, batch, logs=None):
        self._lms_stats.step_begin()

    def on_test_batch_begin(self, batch, logs=None):
        self._lms_stats.step_begin()

    def on_train_batch_end(self, batch, logs=None):
        self._lms_stats.write_step_end('t', self._epoch, batch)

    def on_test_batch_end(self, batch, logs=None):
        self._lms_stats.write_step_end('v', self._epoch, batch)


class LMSStatsLoggerRunHook(SessionRunHook):
    def __init__(self, logfile, gpu_id=0):
        self._lms_stats = LMSStats(logfile, gpu_id=gpu_id)
        self._step = 0

    # Estimator SessionRunHook methods
    def begin(self):
        self._lms_stats.write_step_log_header()

    def before_run(self, run_context):
        self._lms_stats.step_begin()
        self._step += 1

    def after_run(self, run_context, run_values):
        self._lms_stats.write_step_end('t', 0, self._step)
