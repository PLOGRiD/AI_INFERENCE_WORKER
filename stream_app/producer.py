import os

from redis import asyncio as aioredis

REDIS_URL = os.getenv("REDIS_URL")
RESULT_STREAM = "trash-analysis-results"


class ResultProducer:
    """분류 결과를 trash-analysis-results 스트림에 발행하는 프로듀서."""

    def __init__(self):
        self.redis = aioredis.from_url(REDIS_URL, decode_responses=True)

    async def close(self) -> None:
        await self.redis.aclose()

    def _base_payload(self, fields: dict) -> dict:
        return {
            "ploggingId": fields.get("ploggingId", ""),
            "imageUrl": fields.get("imageUrl", ""),
            "latitude": fields.get("latitude", ""),
            "longitude": fields.get("longitude", ""),
        }

    async def publish_success(self, fields: dict, result: dict) -> None:
        payload = self._base_payload(fields)
        payload["status"] = "success"
        payload["finalLabel"] = result["final_label"]
        payload["materialVerified"] = str(result["material_verified"]).lower()
        payload["labelOverridden"] = str(result["label_overridden"]).lower()
        await self.redis.xadd(RESULT_STREAM, payload)

    async def publish_error(self, fields: dict, message: str) -> None:
        payload = self._base_payload(fields)
        payload["status"] = "error"
        payload["errorMessage"] = message
        await self.redis.xadd(RESULT_STREAM, payload)
