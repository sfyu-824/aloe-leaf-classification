"""Flask 简易 Web 演示原型。"""
import torch
from flask import Flask, request
from PIL import Image

from config import DEVICE, NUM_CLASSES, WEIGHT_DIR, CLASS_NAMES
from utils.preprocess import eval_transform
from models.mobilenetv3_eca import get_mobilenetv3_eca

app = Flask(__name__)

SUGGESTION = {
    "锈病": "疑似锈病，及时摘除病叶并清理病残体，加强通风降湿，必要时喷施对应杀菌剂。",
    "炭疽病": "疑似炭疽病，剪除病斑叶片，避免叶面长时间积水，喷施炭疽病防治药剂。",
    "叶斑病": "疑似叶斑病，控制浇水与叶面湿度，及时清除病叶并喷施针对性杀菌剂。",
    "晒伤": "疑似强光晒伤，适当遮阴或移至散射光环境，减少暴晒并补充适量水分。",
    "健康叶片": "叶片状态健康，维持现有水肥与光照管理即可。",
}

model = get_mobilenetv3_eca(NUM_CLASSES)
model.load_state_dict(
    torch.load(f"{WEIGHT_DIR}/mobilenetv3_eca.pth", map_location=DEVICE)
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
    return (
        f"<h3>识别结果：{name}</h3>"
        f"<p>置信度：{float(conf):.4f}</p>"
        f"<p>防治建议：{SUGGESTION[name]}</p>"
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
