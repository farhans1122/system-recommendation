import os
from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
from recommendationSystem import (
    get_recommendations_by_keyword,
    get_recommendations_by_purchased_products,
)
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

load_dotenv()

FRONTEND_URL = os.getenv('FRONTEND_URL', '*')  # fallback to allow all during dev

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL] if FRONTEND_URL != '*' else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input model untuk cart-based recommendation
class ProductIds(BaseModel):
    product_ids: list[str]

# === ROUTE: GET Keyword Recommendation ===
@app.get("/recommend")
def recommend_by_keyword(keyword: str = Query(..., description="Kata kunci pencarian produk")):
    try:
        results = get_recommendations_by_keyword(keyword)
        return JSONResponse(content=jsonable_encoder({
            "keyword": keyword,
            "matched_count": len(results["matched_products"]),
            "results": results
        }))
    except Exception as e:
        print("ERROR:", e)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/recommend/purchased-items")
def recommendation_by_all_completed_purchases():
    try:
        results = get_recommendations_by_purchased_products()
        return JSONResponse(content=jsonable_encoder({
            "results": results
        }))
    except Exception as e:
        print("ERROR:", e)
        raise HTTPException(status_code=500, detail=str(e))