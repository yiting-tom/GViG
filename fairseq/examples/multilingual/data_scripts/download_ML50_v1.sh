#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

if [ -z $WORKDIR_ROOT ] ;
then
        echo "please specify your working directory root in environment variable WORKDIR_ROOT. Exitting..."
        exit
fi

# first run download_wmt20.sh; it will install a few useful tools for other scripts

bash ./download_wmt20.sh

python ./download_wmt19_and_before.py
bash ./download_wat19_my.sh
python ./download_ted_and_extract.py
bash ./download_lotus.sh
bash ./download_iitb.sh
bash ./download_af_xh.sh


# IWSLT downloading URLs have changed in between; TODO: fix them:
bash ./download_iwslt_and_extract.sh


bash ./download_flores_data.sh
