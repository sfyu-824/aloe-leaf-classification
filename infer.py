"""单图推理：加载改进模型权重，输出类别、置信度与建议。"""
import argparse

import torch
from PIL import Image

from configs import settings
from aloe_classifier.data import eval_transform
from aloe_classifier.models import build_improved_mobilenetv3

SUGGESTION = {
    "病害叶片": "及时摘除病叶，加强通风降湿，必要时喷施对应杀菌剂。",
    "干枯叶片": "多为生理性干枯，检查水肥与光照，移除干枯叶片。",
    "成熟健康叶片": "叶片状态健康，维持现有水肥与光照管理。",
    "幼年健康叶片": "幼苗健康，避免强光暴晒，做好日常养护。",
}


def load_model(weight_path=None):
    weight_path = weight_path or \
        f"{settings.WEIGHT_DIR}/mobilenetv3_se.pth"
    model = build_improved_mobilenetv3(settings.NUM_CLASSES)
    model.load_state_dict(
        torch.load(weight_path, map_location=settings.DEVICE)
    )
    model.to(settings.DEVICE).eval()
    return model


def predict(image_path, model):
    image = Image.open(image_path).convert("RGB")
    tensor = eval_transform(image).unsqueeze(0).to(settings.DEVICE)
    with torch.no_grad():
        prob = torch.softmax(model(tensor), dim=1)
    confidence, idx = torch.max(prob, dim=1)
    class_name = settings.CLASS_NAMES[int(idx)]
    return class_name, float(confidence), SUGGESTION[class_name]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="芦荟叶片单图推理")
    parser.add_argument("--image", required=True, help="图片路径")
    parser.add_argument("--weight", default=None, help="模型权重路径")
    args = parser.parse_args()

    name, conf, tip = predict(args.image, load_model(args.weight))
    print(f"识别类别：{name}")
    print(f"置信度：{conf:.4f}")
    print(f"建议：{tip}")
