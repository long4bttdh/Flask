from flask import Flask, render_template, request, send_from_directory
import os
import numpy as np
from keras.models import load_model
import pickle
import tensorflow as tf
import uuid

app = Flask(__name__)

# Đường dẫn tạm để lưu trữ ảnh đã upload
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load mô hình từ file .h5
model_path = 'E:/PhamLong/code/AI/Keybroard/phan_loai_am_thanh.h5'
model = load_model(model_path)

# Đọc nhãn từ file pickle
with open('E:/PhamLong/code/AI/Keybroard/model_indices.pickle', 'rb') as handle:
    labels = pickle.load(handle)

# Hàm dự đoán từ ảnh đã upload
def predict_images_in_folder(image_folder):
    prediction_result = ''
    image_paths = []

    for root, dirs, files in os.walk(image_folder):
        for file_name in files:
            if file_name.endswith(".png") or file_name.endswith(".jpg"):
                image_path = os.path.join(root, file_name)
                # predicted_label = predict_image(image_path)
                # prediction_result += f'File {file_name}: {predicted_label}<br>'
                image_paths.append(image_path.replace('\\', '/'))  # Thêm đường dẫn ảnh vào danh sách
                return image_paths, image_paths
# Hàm dự đoán từ ảnh đã upload
def predict_image(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(64, 64))  # Đảm bảo kích thước ảnh phù hợp với mô hình
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.  # Chuẩn hóa dữ liệu
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)

    # Lấy nhãn từ labels dựa vào predicted_class
    predicted_label = labels[predicted_class]

    return predicted_label
# Route chính để xử lý upload ảnh và dự đoán

@app.route("/", methods=['GET', 'POST'])
def index():
    prediction_result = ''
    image_paths = []

    if request.method == "GET":
        return render_template("index.html")
    else:
        # Xử lý yêu cầu POST từ form
        if 'image_folder' not in request.files:
            return render_template('index.html', prediction_result='No folder selected')

        files = request.files.getlist('image_folder')
        folder_name = str(uuid.uuid4())  # Tạo tên thư mục duy nhất
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], folder_name)
        os.makedirs(save_path)  # Tạo thư mục mới để lưu các tệp

        for file in files:
            filename = file.filename
            file_path = os.path.join(save_path, filename.replace('\\', '/'))  # Đảm bảo sử dụng dấu gạch chéo
            # Tạo thư mục cha nếu chưa tồn tại
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            file.save(file_path)

        # Thực hiện dự đoán từng ảnh trong thư mục
        prediction_result, image_paths = predict_images_in_folder(save_path)
        return render_template('index.html', prediction_result=prediction_result, image_paths=image_paths)

# Route để hiển thị ảnh đã upload
@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
