from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import requests
import os
from dotenv import load_dotenv
import logging
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import io
import base64
from typing import List
from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI()

# Load environment variables from storekey.env file
load_dotenv(dotenv_path='storekey.env')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise RuntimeError("OpenAI API key is missing. Please set the OPENAI_API_KEY environment variable.")

# Update to the new OpenAI model endpoint
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"

logging.basicConfig(level=logging.DEBUG)

templates = Jinja2Templates(directory="templates")

class ProjectData(BaseModel):
    description: str

class RiskAnalysisData(BaseModel):
    descriptions: List[str]
    human_risks: List[List[str]]

def call_openai_api(prompt: str):
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    data = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1000,
        "temperature": 0.5,
    }
    response = requests.post(OPENAI_API_URL, headers=headers, json=data)
    logging.debug(f"Response Status Code: {response.status_code}")
    logging.debug(f"Response Text: {response.text}")
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail="Error calling OpenAI API")
    return response.json()['choices'][0]['message']['content'].strip()


@app.get("/performance_metrics/")
async def get_performance_metrics():
    # Define your performance metrics here
    metrics = {
        "precision": "0.85",
        "recall": "0.80",
        "f1_score": "0.82",
        "accuracy": "0.88"
    }
    return JSONResponse(content=metrics)

@app.post("/identify_risks/")
def identify_risks(project: ProjectData):
    prompt = f"Identify potential risks in the following project description:\n\n{project.description}\n\nRisks:"
    risks = call_openai_api(prompt)
    prompt_assessment = f"Assess the likelihood and impact of the following risks:\n\n{risks}\n\nAssessment:"
    assessment = call_openai_api(prompt_assessment)
    return {"risks": risks.split('\n'), "assessment": assessment}

@app.post("/mitigate_risks/")
def mitigate_risks(project: ProjectData):
    prompt = f"Provide the ways through which the Risks can be avoided according to the agile project management:\n\n{project.description}\n\nRisk Mitigation:"
    mitigation = call_openai_api(prompt)
    return {"mitigation": mitigation}

@app.post("/efectiveness/")
def define_efectiveness(project: ProjectData):
    prompt = f"Determine the effectiveness of OpenAPI's LLM in identifying the Risks according to agile management for the following project description:\n\n{project.description}\n\nEffectiveness of LLM:"
    efectiveness = call_openai_api(prompt)
    return {"efectveness": efectiveness}

###############################################################################
@app.get("/maintenance", response_class=HTMLResponse)
async def maintenance(request: Request):
    return templates.TemplateResponse("maintenance.html", {"request": request})

# @app.get("/design", response_class=HTMLResponse)
# async def design(request: Request):
#     return templates.TemplateResponse("design.html", {"request": request})

@app.get("/design", response_class=HTMLResponse)
def get_design(request: Request):
    prompt = "Generate a detailed plan for the design stage of an agile project. "
    design_content = call_openai_api(prompt)
    return templates.TemplateResponse("design.html", {"request": request, "stage": "Design", "content": design_content})

@app.get("/prototyping", response_class=HTMLResponse)
def get_prototyping(request: Request):
    prompt = "Generate a detailed plan for the prototyping stage of an agile project."
    prototyping_content = call_openai_api(prompt)
    return templates.TemplateResponse("prototyping.html", {"request": request, "stage": "Prototyping", "content": prototyping_content})

@app.get("/customer_evaluation", response_class=HTMLResponse)
def get_customer_evaluation(request: Request):
    prompt = "Generate a detailed plan for the customer evaluation stage of an agile project."
    customer_evaluation_content = call_openai_api(prompt)
    return templates.TemplateResponse("customer_evaluation.html", {"request": request, "stage": "Customer Evaluation", "content": customer_evaluation_content})

@app.get("/review_and_update", response_class=HTMLResponse)
def get_review_and_update(request: Request):
    prompt = "Generate a detailed plan for the review and update stage of an agile project."
    review_and_update_content = call_openai_api(prompt)
    return templates.TemplateResponse("review_and_update.html", {"request": request, "stage": "Review and Update", "content": review_and_update_content})

@app.get("/development", response_class=HTMLResponse)
def get_development(request: Request):
    prompt = "Generate a detailed plan for the development stage of an agile project."
    development_content = call_openai_api(prompt)
    return templates.TemplateResponse("development.html", {"request": request, "stage": "Development", "content": development_content})

@app.get("/testing", response_class=HTMLResponse)
def get_testing(request: Request):
    prompt = "Generate a detailed plan for the testing stage of an agile project."
    testing_content = call_openai_api(prompt)
    return templates.TemplateResponse("testing.html", {"request": request, "stage": "Testing", "content": testing_content})

####################################################################################

@app.post("/clean_description/")
def clean_description(project: ProjectData):
    prompt = f"Clean the following project description by removing unnecessary information and ensuring clarity:\n\n{project.description}\n\nCleaned Description:"
    cleaned_description = call_openai_api(prompt)
    return {"cleaned_description": cleaned_description}

@app.post("/analyze_risks/")
async def analyze_risks(request: Request):
    try:
        # Extract the data from the request
        data = await request.json()
        descriptions = data.get('project_descriptions', [])
        human_risks = data.get('human_risks', [])

        # Ensure both lists have the same length
        if len(descriptions) != len(human_risks):
            raise ValueError("Mismatch between number of descriptions and number of risk lists")

        # Process each description and corresponding human risks
        results = []
        for desc, risks in zip(descriptions, human_risks):
            # Perform risk analysis
            predicted_risks = await call_openai_api(f"Identify potential risks in the following project description:\n\n{desc}\n\nRisks:")
            results.append({
                "description": desc,
                "human_risks": risks,
                "predicted_risks": predicted_risks
            })

        return {"results": results}

    except ValueError as ve:
        logging.error(f"ValueError occurred: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logging.error(f"Error occurred: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while processing the request")

@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/metrics", response_class=HTMLResponse)
def get_metrics(request: Request):
    # Example data
    true_risks = [
        ["Risk A", "Risk B"],  # True risks for Project 1
        ["Risk C", "Risk D"]   # True risks for Project 2
    ]

    predicted_risks = [
        ["Risk A", "Risk B"],  # Predicted risks for Project 1
        ["Risk C", "Risk E"]   # Predicted risks for Project 2
    ]

    # Flatten lists for calculation
    true_labels = [risk for sublist in true_risks for risk in sublist]
    predicted_labels = [risk for sublist in predicted_risks for risk in sublist]

    # Example binary encoding for metrics calculation
    all_risks = sorted(set(true_labels + predicted_labels))
    true_binary = [1 if risk in true_labels else 0 for risk in all_risks]
    predicted_binary = [1 if risk in predicted_labels else 0 for risk in all_risks]

    # Calculate metrics
    precision = precision_score(true_binary, predicted_binary, zero_division=0)
    recall = recall_score(true_binary, predicted_binary, zero_division=0)
    f1 = f1_score(true_binary, predicted_binary, zero_division=0)

    metrics = f"""
        <h2>Performance Metrics for LLM</h2>
        <p>Precision: {precision:.2f}</p>
        <p>Recall: {recall:.2f}</p>
        <p>F1 Score: {f1:.2f}</p>
    """
    return HTMLResponse(content=metrics)

# Define the remaining endpoints as you have them

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=True)



