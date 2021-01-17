"""
Luke
"""

import os.path as osp
import time
import pprint
import requests
from flask import Flask, request
from flask_cors import CORS, cross_origin
from werkzeug.utils import secure_filename

from mmcap.apis.inference import init_caption, generate_caption
from mmcap.tokenizers import build_tokenizer


BASE_DIR = '/mmcaptioning/'
API_TMPDIR = osp.join(BASE_DIR, 
                      'data', 
                      'api_tmpfile')
API_CHECKPOINT = osp.join(BASE_DIR, 
                          'checkpoints', 
                          'res2net101_transformer_koelectra_30e_fp16_coco_nondist',
                          'epoch_10.pth')
API_MODELCONFIG = osp.join(BASE_DIR,
                           'configs',
                           'resnet_transformer',
                           'res2net101_transformer_koelectra_30e_fp16_coco_nondist.py')
API_DEVICE = 'cuda:1'
API_HOST = '0.0.0.0'
API_PORT = '50000'


# Initialize Flask App
app = Flask(__name__)
cors = CORS(app, resources={r"": {"origins": "*"}})

# Initialize Caption Model
model = init_caption(API_MODELCONFIG, 
                     device=API_DEVICE, 
                     checkpoint=API_CHECKPOINT)

# Initialize Tokenizer
tokenizer_cfg = model.cfg.tokenizer
tokenizer = build_tokenizer(tokenizer_cfg)


def allowed_file(filename:str):
    """Check whether given file has an allowed extension or not.

    """
    return '.' in filename and filename.rsplit('.', 1)[1] in ['jpg']


@app.route('/captioning', methods=['POST'])
@cross_origin()
def captioning():
    """Image Captioning API

    """
    st = time.time()
    try:
        image_file = request.files['image']
        if image_file and allowed_file(image_file.filename):
            filename = secure_filename(image_file.filename)
            image_file.save(osp.join(API_TMPDIR, filename))
            image_file_path = osp.join(API_TMPDIR, filename)
    except:
        return {'status' : 'FAIL', 'result' : None}

    caption = generate_caption(model, 
                               tokenizer,
                               tokenizer_cfg,
                               image_file_path)

    result = {
            'status' : 'SUCCESS',
            'result' : caption,
            'elapsed_time' : str(time.time() - st),
            }
    print('\nResult : \n')
    pprint.pprint(result)
    print()

    return result


def main():
    app.run(host=API_HOST, port=API_PORT, debug = False)


if __name__ == '__main__':
    main()
