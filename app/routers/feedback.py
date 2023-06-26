from collections import defaultdict

from fastapi import APIRouter
from pydantic import BaseModel

from app.util import logger


class Feedback(BaseModel):
    client_name: str
    client_version: str
    success: bool

    def __str__(self):
        return "__".join([self.client_name, self.client_version, str(self.success)])


def get_feedback_router() -> APIRouter:
    counter: dict[str, int] = defaultdict(int)
    router = APIRouter(
        prefix="/feedback", tags=["feedback"]
    )

    @router.get("/")
    def get_feedback() -> dict[str, int]:
        return counter

    @router.post("/")
    def create_feedback(feedback: Feedback):
        counter[str(feedback)] += 1
        logger.info(f"Send feedback [success={feedback.success}]")
        logger.info(f"Feedback result: [{dict(counter)}]")
        return

    return router
