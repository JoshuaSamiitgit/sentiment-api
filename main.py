from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from openai import OpenAI
import os

app = FastAPI()
client = OpenAI()

class CommentRequest(BaseModel):
    comment: str

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

@app.post("/comment")
async def analyze_comment(request: CommentRequest):

    if not request.comment.strip():
        raise HTTPException(status_code=400, detail="Comment cannot be empty")

    try:
        response = client.responses.create(
            model="gpt-4.1-mini",
            input=[
                {"role": "system", "content": "You are a sentiment analysis engine."},
                {"role": "user", "content": request.comment}
            ],
            response_format={
                "type": "json_schema",
                "json_schema": sentiment_schema
            }
        )

        return JSONResponse(
            content=response.output_parsed,
            media_type="application/json"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))