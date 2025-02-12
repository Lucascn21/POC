
import json
import base64
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import JSONResponse
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

app = FastAPI()

def encode_image(image_file):
    return base64.b64encode(image_file.read()).decode("utf-8")

@app.get("/models")
async def get_models():
    return client.models.list()

@app.post("/analyze/")
async def analyze_image(file: UploadFile = File(...)):
    try:
        base64_image = encode_image(file.file)
        response = client.chat.completions.create(
            model="chatgpt-4o-latest",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Eres un experto en identificación de razas de animales domésticos. "
                        "Tu tarea es identificar la especie y la raza predominante de un animal en una imagen proporcionada. "
                        "Solo aceptas imagenes de gatos y perros. "
                        "Proporciona la información en un objeto JSON con los siguientes campos: especie, razaPredominante, nivelDeConfianzaDeRazaPredominante, razaMixta, color, edad, tamaño, personalidad, recomendaciones (de higene, cuidados especiales y actividades)."
                        "Responde únicamente con el JSON, sin ningún delimitador de formato ni etiquetas adicionales (por ejemplo, no uses triple backticks)."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Aquí tienes una imagen para analizar:"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                    ],
                },
            ],
            max_tokens=600,
        )
        print(response.choices[0].message.content)
        result = json.loads(response.choices[0].message.content)

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error: {str(e)}"
        )

    return JSONResponse(result)


