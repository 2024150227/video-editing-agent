from fastapi import FastAPI
from app.core.config import get_settings
from app.api.projects import router as projects_router
from app.api.story import router as story_router
from app.api.materials import router as materials_router
from app.api.edit import router as edit_router

settings = get_settings()


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
    )

    app.include_router(projects_router)
    app.include_router(story_router)
    app.include_router(materials_router)
    app.include_router(edit_router)

    @app.get("/health")
    async def health():
        return {"status": "ok", "version": settings.APP_VERSION}

    return app


app = create_app()
