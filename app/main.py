from fastapi import FastAPI
from app.core.config import get_settings
from app.api.projects import router as projects_router
from app.api.story import router as story_router

settings = get_settings()


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
    )

    app.include_router(projects_router)
    app.include_router(story_router)

    @app.get("/health")
    async def health():
        return {"status": "ok", "version": settings.APP_VERSION}

    return app


app = create_app()
