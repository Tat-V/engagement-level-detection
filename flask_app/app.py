import os
import cv2
from flask import Flask, flash, request, redirect, url_for, \
    send_from_directory, render_template
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = '../resources/uploaded_video/'
ALLOWED_EXTENSIONS = {'mp4', 'wbm', 'avi', 'gif'}
EXTRACTED_IMAGES_FOLDER = '../resources/images_from_video/'
IMG_HEIGHT, IMG_WIDTH = 256, 256

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('Cannot read the file')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('File is not chosen')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            os.mkdir(UPLOAD_FOLDER)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], 'video.mp4'))
            return redirect(url_for('output_result'))

    return render_template('form.html')


@app.route('/result')
def output_result():
    from video_parsing.utils import images_from_video
    from prediction.predict import make_emotions_dataset
    from prediction.ml_utils import make_data_mean_agg, define_engagement

    images_from_video(UPLOAD_FOLDER + 'video.mp4', IMG_HEIGHT, IMG_WIDTH, EXTRACTED_IMAGES_FOLDER)
    make_emotions_dataset()
    define_engagement(make_data_mean_agg())

    with open('model_results/result.txt', 'r') as f:
        flash(f'\nModel result - engagement level: {f.readline()}')
    # flash('OK')
    # Clean csv with emotions result
    with open('../resources/prediction_answer.csv', 'w+') as f:
        f.write('')
    return render_template('result.html')


app.run(host='localhost', port=5000)
