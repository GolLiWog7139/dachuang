let uploadedFilePath = null;

// 页面加载时默认显示预测功能
document.addEventListener('DOMContentLoaded', () => {
    document.getElementById('predictTab').addEventListener('click', () => {
        document.getElementById('predictSection').style.display = 'block';
        document.getElementById('retrainSection').style.display = 'none';
        document.getElementById('predictTab').classList.add('active');
        document.getElementById('retrainTab').classList.remove('active');
    });

    document.getElementById('retrainTab').addEventListener('click', () => {
        document.getElementById('predictSection').style.display = 'none';
        document.getElementById('retrainSection').style.display = 'block';
        document.getElementById('retrainTab').classList.add('active');
        document.getElementById('predictTab').classList.remove('active');
    });
});

// 预测功能
function predict() {
    const Ma = document.getElementById('Ma').value.trim();
    const x = document.getElementById('x').value.trim();
    const y = document.getElementById('y').value.trim();
    const rho = document.getElementById('rho').value || null;
    const u = document.getElementById('u').value || null;
    const v = document.getElementById('v').value || null;
    const p = document.getElementById('p').value || null;

    // 检查 Ma, x, y 是否为空
    if (!Ma || !x || !y) {
        let missingFields = [];
        if (!Ma) missingFields.push('马赫数 (Ma)');
        if (!x) missingFields.push('X 坐标 (x)');
        if (!y) missingFields.push('Y 坐标 (y)');

        document.getElementById('result').innerHTML = `<p class="error">ERROR：${missingFields.join(', ')} 未输入</p>`;
        return;
    }

    // 构造输入数据
    const inputs = { Ma, x, y };
    if (rho) inputs.rho = rho;
    if (u) inputs.u = u;
    if (v) inputs.v = v;
    if (p) inputs.p = p;

    document.getElementById('result').innerText = '正在预测...';

    fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(inputs)
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            document.getElementById('result').innerHTML = `<p class="error">ERROR：${data.error}</p>`;
            return;
        }

        // 显示预测结果
        let resultHTML = `
            <h3>预测结果：</h3>
            <p>密度：${data.density.toFixed(4)}</p>
            <p>X 方向速度：${data.u.toFixed(4)}</p>
            <p>Y 方向速度：${data.v.toFixed(4)}</p>
            <p>压力：${data.pressure.toFixed(4)}</p>
        `;
    // 处理图片
         if (data.image_paths && data.image_paths.length > 0) {
            console.log("图片路径：", data.image_paths);  // 打印图片路径
            resultHTML += `<h3>模型结果图：</h3><div id="image-container">`;
            data.image_paths.forEach(imagePath => {
                resultHTML += `<img src="${imagePath}" alt="预测图片" style="width: 870px; height: auto;">`;
            });
            resultHTML += `</div>`;
        } else {
            resultHTML += `<p>没有找到相关图片。</p>`;  // 没有图片时提示
        }

        document.getElementById('result').innerHTML = resultHTML;
    })
    .catch(error => {
        document.getElementById('result').innerText = `ERROR：${error}`;
    });
}

// 上传数据
function uploadData() {
    const formData = new FormData();
    const fileInput = document.getElementById('dataFile');
    const inputs = document.querySelectorAll('.input-checkbox:checked');
    const selectedInputs = Array.from(inputs).map(input => input.value);

    formData.append('file', fileInput.files[0]);
    formData.append('inputs', JSON.stringify(selectedInputs));

    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            alert(`ERROR：${data.error}`);
        } else {
            uploadedFilePath = data.path;  // 保存上传文件路径
            document.getElementById('uploadResult').innerText = `文件上传成功。路径：${data.path}`;
            document.getElementById('retrainButton').disabled = false;  // 启用重新训练按钮
        }
    })
    .catch(error => {
        alert(`ERROR：${error}`);
    });
}

// 触发模型下载
function downloadModel(modelBase64) {
    fetch('/download_model', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_base64: modelBase64 })
    })
    .then(response => response.blob())
    .then(blob => {
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'model.pth';
        a.click();
        window.URL.revokeObjectURL(url);
    })
    .catch(error => {
        alert(`下载失败：${error}`);
    });
}

function retraining() {
    // 获取用户选择的输入变量
    const inputs = [];
    document.querySelectorAll('.input-checkbox').forEach((checkbox) => {
        if (checkbox.checked) {
            inputs.push(checkbox.value);
        }
    });

    // 获取训练参数
    const epoch = document.getElementById('epoch').value;
    const gamma = document.getElementById('gamma').value;
    const batchSize = document.getElementById('batch_size').value;

    // 创建 FormData 对象
    const formData = new FormData();
    formData.append('file', document.getElementById('dataFile').files[0]);
    formData.append('inputs', inputs.join(','));
    formData.append('epoch', epoch);
    formData.append('gamma', gamma);
    formData.append('batch_size', batchSize);

    // 发送请求到后端
    fetch('/retraining', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            document.getElementById('testResult').innerHTML = `<p>模型训练完成！</p>`;
            document.getElementById('downloadModelButton').disabled = false;
        } else {
            document.getElementById('testResult').innerHTML = `<p>模型训练失败：${data.message}</p>`;
        }
    })
    .catch(error => {
        console.error('Error:', error);
        document.getElementById('testResult').innerHTML = `<p>模型训练失败：${error}</p>`;
    });
}
// 重新训练模型
// 修改 retrain 函数
function retrain() {
    if (!uploadedFilePath) {
        alert('请先上传一个 .npy 文件。');
        return;
    }

    // 获取训练参数
    const epoch = document.getElementById('epoch').value;
    const gamma = document.getElementById('gamma').value;
    const batchSize = document.getElementById('batch_size').value;

    // 显示训练状态提示
    document.getElementById('trainingStatusContainer').style.display = 'block';
    document.getElementById('trainingStatus').innerText = '正在重新训练模型...';
    document.getElementById('downloadModelButton').disabled = true;

    // 清空训练结果区域
    const testResult = document.getElementById('testResult');
    testResult.innerHTML = '<h3>训练模型过程：</h3><pre id="logStream"></pre>';
    const logStream = document.getElementById('logStream');

    // 使用 EventSource 实时接收日志
    const url = `/retrain?epoch=${epoch}&gamma=${gamma}&batch_size=${batchSize}&data_path=${encodeURIComponent(uploadedFilePath)}`;
    const evtSource = new EventSource(url);

    evtSource.onmessage = function (event) {
        if (event.data.startsWith('DONE|||')) {
            const result = JSON.parse(event.data.replace('DONE|||', ''));

            const errors = result.errors;
            const modelBase64 = result.model_base64;

            // 显示测试结果
            testResult.innerHTML += `
                <h3>模型重新训练成功。测试结果：</h3>
                <p>特征 1：L2 相对误差 = ${errors[0][0].toFixed(6)}, MSE = ${errors[0][1].toFixed(6)}</p>
                <p>特征 2：L2 相对误差 = ${errors[1][0].toFixed(6)}, MSE = ${errors[1][1].toFixed(6)}</p>
                <p>特征 3：L2 相对误差 = ${errors[2][0].toFixed(6)}, MSE = ${errors[2][1].toFixed(6)}</p>
                <p>特征 4：L2 相对误差 = ${errors[3][0].toFixed(6)}, MSE = ${errors[3][1].toFixed(6)}</p>
            `;

            // 加载预测结果图
            testResult.innerHTML += `<h3>模型结果图：</h3><div id="image-container">`;
            const imagePaths = ["feature_1.png", "feature_2.png", "feature_3.png", "feature_4.png"];
            imagePaths.forEach(imagePath => {
                testResult.innerHTML += `<img src="./static/result/${imagePath}" alt="预测图片" style="width: 830px; height: auto; margin: 10px 0;">`;
            });
            testResult.innerHTML += `</div>`;

            // 启用下载按钮
            const downloadButton = document.getElementById('downloadModelButton');
            downloadButton.disabled = false;
            downloadButton.onclick = () => downloadModel(modelBase64);

            evtSource.close();
            document.getElementById('trainingStatusContainer').style.display = 'none';
        } else {
            // 实时追加日志
            logStream.textContent += event.data + "\n";
        }
    };

    evtSource.onerror = function (err) {
        logStream.textContent += "\n❌ 出错：" + err;
        evtSource.close();
        document.getElementById('trainingStatusContainer').style.display = 'none';
    };
}