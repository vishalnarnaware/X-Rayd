import os
import cv2
import numpy as np
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from models import TB, Cancer, Covid, Multiple

UPLOAD_FOLDER = os.path.join('static', 'inference')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

tb = TB()
cancer = Cancer()
covid = Covid()
# multiple = Multiple()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')


@app.route('/uploader', methods=['GET', 'POST'])
def uploader():
    if request.method == 'POST':
        f = request.files['file']
        img_path = os.path.join(os.getcwd(), app.config['UPLOAD_FOLDER'], f.filename)
        f.save(img_path)
        return result(img_path)


@app.route('/result')
def result(img_path):
    img = img_to_array(load_img(img_path, target_size=(600, 600)))
    plt.imshow(np.uint8(img))

    path_to_orig = os.path.join('static', 'inference', 'orig_resized.png')
    plt.savefig(path_to_orig, transparent=True)

    pred_tb = tb.predict(img_path)
    tb.explain()

    pred_cancer = cancer.predict(img_path)
    pred_cancr = dict(zip(['Adenocarcinoma', 'Large Cell Carcinoma', 'normal', 'Squamous Cell Carcinoma'], pred_cancer))
    del pred_cancr['normal']
    cancer.explain()

    pred_covid = covid.predict(img_path)
    pred_cov = dict(zip(['Covid', 'Lung Opacity', 'normal', 'Viral Pneumonia'], pred_covid))
    del pred_cov['normal']
    covid.explain()

    """
    pred_multiple = multiple.predict(img_path)
    pred_mult = dict(zip(['Cardiomegaly', 'Hernia', 'Infiltration', 'Nodule', 'Emphysema', 'Effusion',
                  'Atelectasis', 'Pleural Thickening', 'Pneumothorax', 'Mass', 'Fibrosis', 
                  'Consolidation', 'Edema', 'Pneumonia'], pred_multiple))
    #del pred_mult['Pneumonia']
    multiple.explain()
    """
    return render_template('result.html',
                           pred_tb=pred_tb,
                           path_tb=os.path.join('static', 'explain', 'explain_tb.png'),
                           path_to_orig=path_to_orig,
                           pred_cancer=pred_cancr,
                           path_can=os.path.join('static', 'explain', 'explain_can.png'),
                           pred_cov=pred_cov,
                           path_cov=os.path.join('static', 'explain', 'explain_cov.png'),
                           # pred_mult=pred_mult,
                           # path_mult=os.path.join('static', 'explain', 'explain_mult.png'),
                           )


if __name__ == '__main__':
    # from werkzeug.serving import run_simple
    # run_simple('localhost', 5000, app)
    app.run(debug=False)
