import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from PIL import Image
import io

app = FastAPI()
model = load_model("cnn333.h5")


# 예측
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 이미지를 메모리(램)에서 직접 처리
    contents = await file.read()
    img = Image.open(io.BytesIO(contents))  # PIL 라이브러리 사용
    img = img.resize((150, 150))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction)
    classes = ['1', '2', '3', '4']
    predicted_class = classes[class_idx]

    return JSONResponse(content={"predicted_class": predicted_class})


# 서버 실행
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
