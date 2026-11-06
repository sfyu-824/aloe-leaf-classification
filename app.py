"""Flask 简易 Web 演示原型。"""
import torch
from flask import Flask, request
from PIL import Image

from config import DEVICE, NUM_CLASSES, WEIGHT_DIR, CLASS_NAMES
from utils.preprocess import eval_transform
from models.mobilenetv3_se import get_mobilenetv3_se

app = Flask(__name__)

model = get_mobilenetv3_se(NUM_CLASSES)
model.load_state_dict(
    torch.load(f"{WEIGHT_DIR}/mobilenetv3_se.pth", map_location=DEVICE)
)
model.to(DEVICE).eval()


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return '''
        <h2>芦荟叶片智能诊断演示</h2>
        <form method="post" enctype="multipart/form-data">
            <input type="file" name="image" accept="image/*" required>
            <button type="submit">提交识别</button>
        </form>
        '''

    image = Image.open(request.files["image"].stream).convert("RGB")
    tensor = eval_transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        prob = torch.softmax(model(tensor), dim=1)
    conf, idx = torch.max(prob, dim=1)
    name = CLASS_NAMES[int(idx)]
    return f"<h3>识别结果：{name}</h3><p>置信度：{float(conf):.4f}</p>"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
