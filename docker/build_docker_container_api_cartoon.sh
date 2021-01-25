nvidia-docker run -ti \
    -v $(pwd):/mmcaptioning/ \
    --ipc=host \
    --net=host \
    --name=mmcap_api_cartoon_ver \
    mmcaptioning \
    /bin/bash
