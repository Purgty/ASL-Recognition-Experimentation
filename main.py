from models.cnn import build_custom_cnn
from models.mobilenetv2 import build_mobilenet
from models.resnet50 import build_resnet50
from models.alexnet import build_alexnet
from models.efficientnet import build_efficientnet
from utils.dataset import get_datasets
from train import train_model

MODEL_MAP = {
    "cnn": build_custom_cnn,
    "mobilenetv2": build_mobilenet,
    "resnet50": build_resnet50,
    "alexnet": build_alexnet,
    "efficientnet": build_efficientnet
}

if __name__ == "__main__":
    train_ds, val_ds = get_datasets("asl_dataset", img_size=(224,224), batch_size=32)
    model_name = "mobilenetv2"  # change here
    model = MODEL_MAP[model_name](input_shape=(224,224,3), num_classes=29)
    trained_model, history = train_model(model, train_ds, val_ds, model_name=model_name, epochs=20, lr=1e-4)
