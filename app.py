import os
import base64
import json
import torch
from werkzeug.utils import secure_filename
from flask import Flask, Response, stream_with_context, request, render_template, jsonify
from models.model import load_models, predict as model_predict
from models.retrain import retrain

device = 'cuda' if torch.cuda.is_available() else 'cpu'
app = Flask(__name__)

# 设置文件上传路径
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 加载模型
load_models()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    return model_predict(request.json)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': '未找到文件'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '未选择文件'})

    if file and file.filename.endswith('.npy'):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # 获取用户选择的输入变量
        selected_inputs = request.form.get('inputs', '[]')
        selected_inputs = json.loads(selected_inputs)

        # 将输入变量信息保存到文件名中
        input_str = '_'.join(selected_inputs)
        new_file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{input_str}_{filename}")
        print(f"new_file_path:{new_file_path}")

        # 如果文件已存在，直接覆盖
        if os.path.exists(new_file_path):
            os.remove(new_file_path)  # 删除已存在的文件

        os.rename(file_path, new_file_path)  # 重命名文件

        return jsonify({'status': '文件上传成功', 'path': new_file_path})
    else:
        return jsonify({'error': '无效的文件格式。请上传一个 .npy 文件'})

@app.route('/retrain', methods=['GET'])
def retrain_route():
    data_path = request.args.get('data_path')
    epoch = int(request.args.get('epoch', 10))
    gamma = float(request.args.get('gamma', 0.4))
    batch_size = int(request.args.get('batch_size', 64))

    def generate_logs():
        for log in retrain(data_path, epoch, gamma, batch_size):
            yield f"data: {log}\n\n"

    return Response(stream_with_context(generate_logs()), mimetype='text/event-stream')

@app.route('/download_model', methods=['POST'])
def download_model():
    model_base64 = request.json.get('model_base64')
    if not model_base64:
        return jsonify({'error': '未提供模型数据'})

    # 将 Base64 编码解码为字节流
    model_bytes = base64.b64decode(model_base64)

    # 返回文件下载响应
    return Response(
        model_bytes,
        mimetype='application/octet-stream',
        headers={'Content-Disposition': 'attachment; filename=model.pth'}
    )

if __name__ == '__main__':
    app.run(debug=True)