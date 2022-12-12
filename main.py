from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from utils import process_sequence, classifier_model, detector_model
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequestModel(BaseModel):
    seq: str


@app.post("/predict")
async def predict_riboswitch(predict_request: PredictRequestModel):
    seq = process_sequence(predict_request.seq)
    p = detector_model.predict(seq)
    d_pred = np.argmax(p)
    if not d_pred:
        return {
            'is_riboswitch': False,
            'annotations': None,
            'confidence': f"{p[0][d_pred]*100:.2f}",
        }
    
    seq = process_sequence(predict_request.seq, for_classifier=True)
    p = classifier_model.predict(seq)
    c_pred = np.argmax(p)

    return {
        'is_riboswitch': True,
        'annotations': f"Ribo-{c_pred}",
        'confidence': f"{p[0][c_pred]*100:.2f}",
    }

