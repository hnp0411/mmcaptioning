import os.path as osp


BASE_DIR = '/mmcaptioning/'
API_TMPDIR = osp.join(BASE_DIR,
                      'data',
                      'cartoon_api_tmpfile')
API_CHECKPOINT = osp.join(BASE_DIR,
                          'checkpoints',
                          'res2net101_transformer_koelectra_30e_fp16_cocoHayao_nondist',
                          'epoch_20.pth')
API_MODELCONFIG = osp.join(BASE_DIR,
                           'configs',
                           'resnet_transformer',
                           'res2net101_transformer_koelectra_30e_fp16_cocoHayao_nondist.py')
API_DEVICE = 'cuda:1'
API_HOST = '0.0.0.0'
API_PORT = '50010'
