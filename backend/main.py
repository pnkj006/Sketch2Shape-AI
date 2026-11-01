
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routes import router


app = FastAPI(
    title="Sketch2Shape AI",
    description=(
        "AI-powered 2D sketch and image "
        "to 3D model generation system."
    ),
    version="1.0.0"
)


# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Register API routes
app.include_router(
    router,
    prefix="/api",
    tags=["3D Generation"]
)


@app.get("/")
async def root():
    return {
        "message": "Welcome to Sketch2Shape AI",
        "status": "running",
        "documentation": "/docs"
    }


@app.get("/api/info")
async def application_info():
    return {
        "name": "Sketch2Shape AI",
        "version": "1.0.0",
        "features": [
            "Sketch to 3D",
            "Photo to 3D",
            "Multi-view generation",
            "GLB export",
            "OBJ export"
        ]
    }
