from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List

app = FastAPI()

# 1. DEFINE WHAT THE API ACCEPTS (The Input Structure)
class TeamSubmission(BaseModel):
    team_name: str
    project_title: str
    scores: List[float] = Field(..., description="A list of judge scores from 0.0 to 10.0")

# 2. DEFINE WHAT THE API SPITS OUT (The Output Structure)
class ProcessedResult(BaseModel):
    team_name: str
    final_score: float
    status: str
    
# Root will be used to obtain analytics about the API and endpoints.
@app.get("/api/")
def read_root():
    # We return a json showing all endpoints.

    return {
        "message": "O(1)",
        "status": "active",
        "description": "This is a simple FastAPI application that demonstrates basic routing and response handling.",            
        }



# 3. CREATE THE ENDPOINTTHAT ACCEPTS AND SPITS OUT DATA - Exemplar
@app.post("/api/evaluate", response_model=ProcessedResult)
def evaluate_team(payload: TeamSubmission):
    # FastAPI automatically parsed the JSON input into 'payload'
    
    if not payload.scores:
        raise HTTPException(status_code=400, detail="Entry invalid.")
        
    # Process the data (Business Logic)
    average_score = sum(payload.scores) / len(payload.scores)
    status_label = "Passed to Finals" if average_score >= 7.5 else "Eliminated"
    
    # Spit out the data structured exactly like our ProcessedResult model
    return {
        "team_name": payload.team_name,
        "final_score": round(average_score, 2),
        "status": status_label
    }
