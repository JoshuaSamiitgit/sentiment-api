from openai import OpenAI
client = OpenAI()

@app.post("/comment")
async def analyze_comment(request: CommentRequest):

    if not request.comment or request.comment.strip() == "":
        raise HTTPException(status_code=400, detail="Comment cannot be empty")

    try:
        response = client.responses.create(
            model="gpt-4.1-mini",
            input=[
                {
                    "role": "system",
                    "content": "You are a sentiment analysis engine. Return sentiment and rating."
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

        result = response.output_parsed

        return JSONResponse(
            content=result,
            media_type="application/json"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI API Error: {str(e)}")