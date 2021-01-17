nvidia-docker run -ti \
    -v $(pwd):/mmcaptioning/ \
    --ipc=host \
    --net=host \
    --name=mmcap_test \
    mmcaptioning \
    /bin/bash
