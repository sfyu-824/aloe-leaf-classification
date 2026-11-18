"""单张图片推理脚本。"""
import argparse

import torch
from PIL import Image

from config import DEVICE, NUM_CLASSES, WEIGHT_DIR, CLASS_NAMES
from utils.preprocess import eval_transform
from models.mobilenetv3_eca import get_mobilenetv3_eca

SUGGESTION = {
    "锈病": "疑似锈病，及时摘除病叶并清理病残体，加强通风降湿，必要时喷施对应杀菌剂。",
    "炭疽病": "疑似炭疽病，剪除病斑叶片，避免叶面长时间积水，喷施炭疽病防治药剂。",
    "叶斑病": "疑似叶斑病，控制浇水与叶面湿度，及时清除病叶并喷施针对性杀菌剂。",
    "晒伤": "疑似强光晒伤，适当遮阴或移至散射光环境，减少暴晒并补充适量水分。",
    "健康叶片": "叶片状态健康，维持现有水肥与光照管理即可。",
}


def load_model(weight_path=None):
    weight_path = weight_path or f"{WEIGHT_DIR}/mobilenetv3_eca.pth"
    model = get_mobilenetv3_eca(NUM_CLASSES)
    model.load_state_dict(
        torch.load(weight_path, map_location=DEVICE)
    )
    model.to(DEVICE).eval()
    return model


def predict(image_path, model):
    image = Image.open(image_path).convert("RGB")
    tensor = eval_transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        prob = torch.softmax(model(tensor), dim=1)
    confidence, idx = torch.max(prob, dim=1)
    class_name = CLASS_NAMES[int(idx)]
    return class_name, float(confidence), SUGGESTION[class_name]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="芦荟叶片单图推理")
    parser.add_argument("--image", required=True, help="图片路径")
    parser.add_argument("--weight", default=None, help="权重路径")
    args = parser.parse_args()

    name, conf, tip = predict(args.image, load_model(args.weight))
    print(f"识别类别：{name}")
    print(f"置信度：{conf:.4f}")
    print(f"建议：{tip}")
