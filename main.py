from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import openai
import os

# Initialize FastAPI
app = FastAPI()

# Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")


# ----------- Request Model -----------

class CommentRequest(BaseModel):
    comment: str


# ----------- Response Schema for Structured Output -----------

sentiment_schema = {
    "name": "sentiment_analysis",
    "schema": {
        "type": "object",
        "properties": {
            "sentiment": {
                "type": "string",
                "enum": ["positive", "negative", "neutral"]
            },
            "rating": {
                "type": "integer",
                "minimum": 1,
                "maximum": 5
            }
        },
        "required": ["sentiment", "rating"],
        "additionalProperties": False
    }
}


# ----------- Endpoint -----------

@app.post("/comment", response_class=JSONResponse)
async def analyze_comment(request: CommentRequest):
    
    if not request.comment or request.comment.strip() == "":
        raise HTTPException(status_code=400, detail="Comment cannot be empty")

    try:
        response = openai.responses.create(
            model="gpt-4.1-mini",
            input=[
                {
                    "role": "system",
                    "content": "You are a sentiment analysis engine."
                },
                {
                    "role": "user",
                    "content": request.comment
                }
            ],
            response_format={
                "type": "json_schema",
                "json_schema": sentiment_schema
            }
        )

        result = response.output[0].content[0].text

        return JSONResponse(
            content=result,
            media_type="application/json"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI API Error: {str(e)}")