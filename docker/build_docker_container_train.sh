nvidia-docker run -ti \
    -v $(pwd):/mmcaptioning/ \
    --ipc=host \
    --net=host \
    --name=mmcap_train \
    mmcaptioning \
    /bin/bash
