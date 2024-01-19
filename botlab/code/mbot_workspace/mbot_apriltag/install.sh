#!/bin/bash
set -e

sudo pip install dt-apriltags
#As of November 1, 2023, the dt-apriltags repository is newer than the release that we can install from Pip. This means some crucial updates are not included in the pip installed version that we have to manually edit.
#One of these differences is in the parameter `max_hamming`, which has a default of 2, which tries to allocate a significant chunk of memory that usually fails especially on the Jetson. To set this parameter, we have to copy this newer version of apriltags.py over.
sudo cp ./_new_dt_apriltags.py /usr/local/lib/python3.8/dist-packages/dt_apriltags/apriltags.py