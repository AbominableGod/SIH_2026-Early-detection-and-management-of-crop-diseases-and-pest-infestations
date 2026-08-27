from pathlib import Path
import shutil

import opendatasets as od
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DOWNLOAD_DIR = DATA_DIR / "tomato-leaf-disease"
PREPARED_DIR = DATA_DIR / "tomato_classes"
MODEL_PATH = BASE_DIR / "tomato_model.h5"
IMAGE_SIZE = (256, 256)
BATCH_SIZE = 32
EPOCHS = 3

CLASS_NAMES = [
    "Bacterial spot", "Early blight", "Healthy", "Late blight", "Leaf Mold",
    "Septoria leaf spot", "Spider mites", "Target Spot",
    "Yellow Leaf Curl Virus", "Mosaic virus",
]


def normalized(value):
    return "".join(character.lower() for character in value if character.isalnum())


def find_source_directories(dataset_dir):
    expected = {normalized(name): name for name in CLASS_NAMES}
    found = {}
    for path in dataset_dir.rglob("*"):
        if path.is_dir():
            label = normalized(path.name.replace("tomato", ""))
            for expected_key, class_name in expected.items():
                if label == expected_key or expected_key in label:
                    found[class_name] = path
                    break
    missing = [name for name in CLASS_NAMES if name not in found]
    if missing:
        raise RuntimeError(f"Could not find dataset folders for: {', '.join(missing)}")
    return found


def prepare_dataset(dataset_dir):
    source_directories = find_source_directories(dataset_dir)
    if PREPARED_DIR.exists():
        shutil.rmtree(PREPARED_DIR)
    PREPARED_DIR.mkdir(parents=True)
    for class_name in CLASS_NAMES:
        destination = PREPARED_DIR / class_name
        try:
            destination.symlink_to(source_directories[class_name], target_is_directory=True)
        except OSError:
            shutil.copytree(source_directories[class_name], destination)


def build_model():
    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3), include_top=False, weights="imagenet"
    )
    backbone.trainable = False
    inputs = layers.Input(shape=(*IMAGE_SIZE, 3))
    features = backbone(inputs, training=False)
    features = layers.GlobalAveragePooling2D()(features)
    features = layers.Dropout(0.2)(features)
    outputs = layers.Dense(len(CLASS_NAMES), activation="softmax")(features)
    model = models.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def main():
    DATA_DIR.mkdir(exist_ok=True)
    od.download(
        "https://www.kaggle.com/datasets/naveedgull/tomato-leaf-disease",
        data_dir=str(DATA_DIR),
    )
    prepare_dataset(DOWNLOAD_DIR)

    generator = ImageDataGenerator(rescale=1.0 / 255.0, validation_split=0.2)
    train_data = generator.flow_from_directory(
        PREPARED_DIR, target_size=IMAGE_SIZE, classes=CLASS_NAMES,
        class_mode="categorical", batch_size=BATCH_SIZE, subset="training",
        shuffle=True,
    )
    validation_data = generator.flow_from_directory(
        PREPARED_DIR, target_size=IMAGE_SIZE, classes=CLASS_NAMES,
        class_mode="categorical", batch_size=BATCH_SIZE, subset="validation",
        shuffle=False,
    )

    model = build_model()
    model.fit(train_data, validation_data=validation_data, epochs=EPOCHS)
    model.save(MODEL_PATH)
    print("CLASS_NAMES =", CLASS_NAMES)
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
