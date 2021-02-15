도커 이미지 빌드
$ chmod 755 build_docker_img.sh
$ ./build_docker_img.sh

도커 컨테이너 빌드
$ ./docker/build_*.sh

도커 컨테이너에서 
$ python setup.py develop


Config 명명 규칙
[Encoder]_[Decoder]_[Tokenizer]_[TotalEpochs]_[MixedPrecisionTraining]_[Dataset]_[Distributed]
