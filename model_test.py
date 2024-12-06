import torch
import onnxruntime as ort
import numpy as np

# 加載儲存的資料
data = torch.load("Sling_Shot.pth")
X_tensor = data["X_tensor"]
y_tensor = data["y_tensor"]
scaler = data["scaler"]

# 加載 ONNX 模型
onnx_model_path = "gru_model.onnx"
session = ort.InferenceSession(onnx_model_path)

# 選取測試資料進行推論
X_test = X_tensor.numpy()  # 轉為 NumPy 格式
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# 進行模型推論
predictions = session.run([output_name], {input_name: X_test})
predictions = np.array(predictions[0])

# 將模型輸出反轉標準化
predictions_inverse = scaler.inverse_transform(predictions)
print("模型預測結果(遊樂設施等待時間):", predictions_inverse)
