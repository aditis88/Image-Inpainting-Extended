from flask import Flask, render_template, request, redirect, url_for, send_file
import os
import cv2
import numpy as np
from inpainting import create_mask, inpaint_image

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            return redirect(request.url)

        file = request.files['image']
        if file.filename == '':
            return redirect(request.url)

        if file:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            # Read image and perform inpainting
            threshold = int(request.form.get('threshold', 20))
            kernel_dim = int(request.form.get('kernel_dim', 5))

            image = cv2.imread(filepath)
            mask = create_mask(filepath, threshold, (kernel_dim, kernel_dim))
            inpainted = inpaint_image(image.copy(), mask, radius=7)

            result_path = os.path.join(RESULT_FOLDER, 'result_' + file.filename)
            cv2.imwrite(result_path, inpainted)

            return render_template('index.html', original=filepath, result=result_path)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)



