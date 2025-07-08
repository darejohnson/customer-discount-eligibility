import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel
from datetime import datetime
import numpy as np
import logging

# Initialize FastAPI and logging
app = FastAPI()
logger = logging.getLogger("uvicorn.error")

# ---
# Model & Tokenizer Loading (Do this once at startup)
# ---
class DiscountClassifier(torch.nn.Module):
    # Define your model architecture (same as training)
    def __init__(self, input_dim):
        super().__init__()
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, 1)
        )
    
    def forward(self, x):
        return self.classifier(x)

# Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DiscountClassifier(input_dim=773)  # 768 (BERT) + 5 structured features
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.to(device)
model.eval()

# Load BERT tokenizer and model for text embeddings
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
bert_model = AutoModel.from_pretrained("bert-base-uncased").to(device)

# ---
# Pydantic Models for Input Validation
# ---
class PredictionRequest(BaseModel):
    text: str                   # Review text (e.g., "The product broke after 2 days")
    score: int                  # Rating (1-5)
    helpfulness_numerator: int  # Number of users who found the review helpful
    helpfulness_denominator: int # Total users who voted on helpfulness
    user_review_count: int      # How many reviews the user has written
    product_avg_score: float    # Average score of the product
    time: datetime              # Timestamp of the review

# ---
# Helper Functions
# ---
def get_bert_embeddings(text: str) -> np.ndarray:
    """Convert text to BERT embeddings."""
    inputs = tokenizer(
        text, return_tensors="pt", padding=True, truncation=True, max_length=128
    ).to(device)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]  # [CLS] embedding

def preprocess_features(request: PredictionRequest) -> torch.Tensor:
    """Convert request data to model input tensor."""
    # Calculate structured features
    helpfulness_ratio = (request.helpfulness_numerator + 1e-6) / (request.helpfulness_denominator + 1e-6)
    days_since_review = (datetime.now() - request.time).days
    
    # Combine BERT embeddings + structured features
    text_embedding = get_bert_embeddings(request.text)
    structured_features = np.array([
        request.score,
        helpfulness_ratio,
        days_since_review,
        request.user_review_count,
        request.product_avg_score
    ], dtype=np.float32)
    
    combined = np.concatenate([text_embedding, structured_features])
    return torch.tensor(combined, dtype=torch.float32).to(device)

# ---
# API Endpoint
# ---
@app.post("/predict")
async def predict(request: PredictionRequest):
    try:
        # Preprocess input
        input_tensor = preprocess_features(request)
        
        # Predict
        with torch.no_grad():
            prediction = torch.sigmoid(model(input_tensor.unsqueeze(0)))
        
        return {
            "probability": prediction.item(),
            "discount_eligible": bool(prediction > 0.5)  # Threshold at 0.5
        }
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))