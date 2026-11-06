"""Flask 简易 Web 演示原型：上传图片并返回识别结果。"""
from flask import Flask, request
from PIL import Image
import torch

from configs import settings
from aloe_classifier.data import eval_transform
from aloe_classifier.models import build_improved_mobilenetv3

app = Flask(__name__)

model = build_improved_mobilenetv3(settings.NUM_CLASSES)
model.load_state_dict(
    torch.load(f"{settings.WEIGHT_DIR}/mobilenetv3_se.pth",
               map_location=settings.DEVICE)
)
model.to(settings.DEVICE).eval()


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
    tensor = eval_transform(image).unsqueeze(0).to(settings.DEVICE)
    with torch.no_grad():
        prob = torch.softmax(model(tensor), dim=1)
    conf, idx = torch.max(prob, dim=1)
    name = settings.CLASS_NAMES[int(idx)]
    return f"<h3>识别结果：{name}</h3><p>置信度：{float(conf):.4f}</p>"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
