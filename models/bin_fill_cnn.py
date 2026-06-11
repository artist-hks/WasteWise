import numpy as np
import os, json
from PIL import Image, ImageDraw
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import joblib

os.makedirs("assets/sample_bins", exist_ok=True)
os.makedirs("models/saved", exist_ok=True)

def generate_bin_image(fill_percent, img_size=64):
    """Create a synthetic bin image with colored fill."""
    img = Image.new("RGB", (img_size, img_size), color=(240, 240, 240))
    draw = ImageDraw.Draw(img)
    bx, by, bw, bh = 14, 10, 36, 44
    draw.rectangle([bx, by, bx+bw, by+bh], outline=(100,100,100), width=2, fill=(200,200,200))
    fill_h = int((fill_percent / 100) * bh)
    if fill_h > 0:
        fy = by + bh - fill_h
        if fill_percent < 30:    fc = (16, 185, 129)
        elif fill_percent < 60:  fc = (245, 158, 11)
        elif fill_percent < 85:  fc = (249, 115, 22)
        else:                    fc = (239, 68, 68)
        noise = np.random.randint(-15, 15, (fill_h, bw, 3))
        fill_arr = np.array([list(fc)] * (fill_h * bw), dtype=np.int16).reshape(fill_h, bw, 3)
        fill_arr = np.clip(fill_arr + noise, 0, 255).astype(np.uint8)
        fill_img = Image.fromarray(fill_arr)
        img.paste(fill_img, (bx+1, fy))
    return img

def get_class(fp):
    if fp < 25:   return 0
    elif fp < 50: return 1
    elif fp < 75: return 2
    else:         return 3

# Generate dataset
N = 800
X, y_cls, y_reg = [], [], []
for i in range(N):
    fp = np.random.uniform(0, 100)
    img = generate_bin_image(fp)
    if i < 20:
        img.save(f"assets/sample_bins/bin_{i:03d}_{int(fp)}pct.png")
    arr = np.array(img.resize((64,64))) / 255.0
    X.append(arr)
    y_cls.append(get_class(fp))
    y_reg.append(fp / 100.0)

X = np.array(X)
y_cls = tf.keras.utils.to_categorical(y_cls, 4)
y_reg = np.array(y_reg)

X_train, X_val, yc_train, yc_val, yr_train, yr_val = train_test_split(
    X, y_cls, y_reg, test_size=0.2, random_state=42)

# Build model
inp = layers.Input(shape=(64, 64, 3))
x = layers.Conv2D(32, 3, activation='relu', padding='same')(inp)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.3)(x)
out_cls = layers.Dense(4, activation='softmax', name='class_out')(x)
out_reg = layers.Dense(1, activation='sigmoid', name='fill_out')(x)

model = models.Model(inp, [out_cls, out_reg])
model.compile(
    optimizer='adam',
    loss={'class_out': 'categorical_crossentropy', 'fill_out': 'mse'},
    metrics={'class_out': 'accuracy', 'fill_out': 'mae'}
)

history = model.fit(
    X_train, {'class_out': yc_train, 'fill_out': yr_train},
    validation_data=(X_val, {'class_out': yc_val, 'fill_out': yr_val}),
    epochs=10, batch_size=32, verbose=1
)

model.save("models/saved/bin_fill_model.h5")
history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
with open("models/saved/cnn_history.json", "w") as f:
    json.dump(history_dict, f)

print("CNN model saved to models/saved/bin_fill_model.h5")

def predict_fill_level(image_path):
    """Predict fill level from an image path."""
    model = tf.keras.models.load_model("models/saved/bin_fill_model.h5")
    img = Image.open(image_path).resize((64, 64))
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, 0)
    cls_pred, reg_pred = model.predict(arr, verbose=0)
    classes = ["Empty", "Low", "Medium", "Full"]
    cls_idx = np.argmax(cls_pred[0])
    return {
        "class": classes[cls_idx],
        "percent": round(float(reg_pred[0][0]) * 100, 1),
        "confidence": round(float(cls_pred[0][cls_idx]), 3)
    }

if __name__ == "__main__":
    history_dict  # already saved above
