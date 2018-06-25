# (C) Copyright IBM Corp. 2018. All Rights Reserved.
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

"""
This program resizes prediction files from one directory tree and writes
them to another directory tree.

For example `python 192 prediction128 prediction192` will resize all the
prediction.nii.gz files it finds in the prediction128 directory tree to a
cube of size 192 and writes the resized prediction in the same subpath under
the prediction192 directory.
"""

import sys
import os
import glob
import nibabel as nib
from unet3d.utils.utils import resize

def main():
    if len(sys.argv) < 2:
        print('Usage: <target_size> <source_dir> <target_dir>')
        print('Example: 192 prediction prediction192')

        return
    target_dim = int(sys.argv[1])
    target_shape = (target_dim, target_dim, target_dim)
    source_dir = sys.argv[2]
    target_dir = sys.argv[3]

    for case_folder in glob.glob('%s/*' % source_dir):
        if not os.path.isdir(case_folder):
            continue

        truth_file = os.path.join(case_folder, "prediction.nii.gz")
        target_file = os.path.join(case_folder.replace(source_dir, target_dir),
                                   "prediction.nii.gz")
        print('Processing %s' % truth_file)
        image = nib.load(truth_file)
        new_img = resize(image, target_shape)
        new_img.to_filename(target_file)

if __name__ == "__main__":
    main()
