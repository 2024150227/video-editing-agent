from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db
from app.core.llm_client import LLMClient
from app.agents.director import DirectorAgent
from app.agents.material import MaterialAgent
from app.agents.editor import EditorAgent


async def get_llm_client():
    return LLMClient()


def get_director_agent(llm: LLMClient = Depends(get_llm_client)):
    return DirectorAgent(llm_client=llm)


def get_material_agent(llm: LLMClient = Depends(get_llm_client)):
    return MaterialAgent(llm_client=llm)


def get_editor_agent():
    return EditorAgent()
