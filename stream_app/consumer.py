import asyncio
import io
import logging
import os

import httpx
from PIL import Image
from redis import asyncio as aioredis

from pipeline import PipelineError, classify_material
from stream_app.producer import ResultProducer

logger = logging.getLogger("plogrid-ai.stream")

REDIS_URL = os.getenv("REDIS_URL")
REQUEST_STREAM = "trash-analysis-requests"
CONSUMER_GROUP = "plogrid-ai-workers"
CONSUMER_NAME = os.getenv("REDIS_CONSUMER_NAME")

BLOCK_MS = 2000
BATCH_SIZE = 1
RECONNECT_DELAY_S = 5

CLAIM_IDLE_MS = 40_000
CLAIM_INTERVAL_S = 30
MAX_DELIVERIES = 3
PROCESS_TIMEOUT_S = 20

class StreamConsumer:
    """trash-analysis-requests 스트림을 소비해 YOLO 분류를 수행하는 컨슈머."""

    def __init__(self, yolo_predictor):
        self.yolo_predictor = yolo_predictor
        self.redis = aioredis.from_url(REDIS_URL, decode_responses=True)
        self.http = httpx.AsyncClient(timeout=15.0)
        self.producer = ResultProducer()
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        self._task = asyncio.create_task(self._run(), name="redis-stream-consumer")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        await self.http.aclose()
        await self.redis.aclose()
        await self.producer.close()

    async def _run(self) -> None:
        await self._ensure_group()
        await self._claim_abandoned(0)
        logger.info(
            "Redis Streams consumer started (stream=%s group=%s consumer=%s)",
            REQUEST_STREAM, CONSUMER_GROUP, CONSUMER_NAME,
        )
        loop = asyncio.get_running_loop()
        next_claim_at = loop.time() + CLAIM_INTERVAL_S
        while True:
            try:
                response = await self.redis.xreadgroup(
                    groupname=CONSUMER_GROUP,
                    consumername=CONSUMER_NAME,
                    streams={REQUEST_STREAM: ">"},
                    count=BATCH_SIZE,
                    block=BLOCK_MS,
                )
            except asyncio.CancelledError:
                raise
            except Exception as e:
                if "NOGROUP" in str(e):
                    logger.warning("스트림/컨슈머 그룹이 사라짐, 다시 생성 시도")
                    await self._ensure_group()
                else:
                    logger.exception("Redis Streams 읽기 실패, %d초 후 재시도", RECONNECT_DELAY_S)
                    await asyncio.sleep(RECONNECT_DELAY_S)
                continue

            if response:
                for _stream_name, messages in response:
                    for record_id, fields in messages:
                        # _handle()의 xack 자체가 실패하는 등 예상 밖 예외가 나도
                        # 이 컨슈머 루프(_run) 전체가 죽지 않도록 메시지 단위로 격리한다.
                        try:
                            await self._handle(record_id, fields)
                        except asyncio.CancelledError:
                            raise
                        except Exception:
                            logger.exception(
                                "메시지 처리 중 예상 밖 예외, 다음 메시지로 진행 (recordId=%s)", record_id
                            )

            now = loop.time()
            if now >= next_claim_at:
                await self._claim_abandoned(CLAIM_IDLE_MS)
                next_claim_at = now + CLAIM_INTERVAL_S

    async def _ensure_group(self) -> None:
        while True:
            try:
                await self.redis.xgroup_create(REQUEST_STREAM, CONSUMER_GROUP, id="0", mkstream=True)
                return
            except asyncio.CancelledError:
                raise
            except Exception as e:
                if "BUSYGROUP" in str(e):
                    return
                logger.exception("Redis 연결/컨슈머 그룹 생성 실패, %d초 후 재시도", RECONNECT_DELAY_S)
                await asyncio.sleep(RECONNECT_DELAY_S)

    async def _claim_abandoned(self, min_idle_ms: int) -> None:
        """min_idle_ms 이상 ACK도 재청구도 안 된 메시지를 (누구 소유든) 가져와 처리한다."""
        while True:
            _next_id, claimed, _deleted = await self.redis.xautoclaim(
                REQUEST_STREAM, CONSUMER_GROUP, CONSUMER_NAME,
                min_idle_time=min_idle_ms, start_id="0-0", count=BATCH_SIZE,
            )
            if not claimed:
                return
            for record_id, fields in claimed:
                # 아래 _handle_claimed()에서 xack 자체가 실패하는 등 예상 밖 예외가 나도
                # 스윕 루프가 죽지 않도록 메시지 단위로 격리한다 (이 스윕은 _run 안에서 돌기 때문에,
                # 여기서 죽으면 _run 전체가 멈추고 이후 회수 시도 자체가 사라진다).
                try:
                    await self._handle_claimed(record_id, fields)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception(
                        "회수한 메시지 처리 중 예상 밖 예외, 다음 메시지로 진행 (recordId=%s)", record_id
                    )

    async def _handle_claimed(self, record_id: str, fields: dict) -> None:
        """스윕(_claim_abandoned)이 회수한 메시지 하나를 처리한다.

        배달 횟수(times_delivered)를 먼저 확인해서, MAX_DELIVERIES를 넘겼으면
        poison message로 보고 더 재시도하지 않고 에러로 확정 처리한다.
        아니면 평소 새 메시지와 동일하게 _handle()로 넘긴다.
        """
        pending = await self.redis.xpending_range(
            REQUEST_STREAM, CONSUMER_GROUP, min=record_id, max=record_id, count=1
        )
        times_delivered = pending[0]["times_delivered"] if pending else 1
        if times_delivered > MAX_DELIVERIES:
            logger.error(
                "%d회 재시도 후에도 실패, 포기하고 에러 처리 (recordId=%s ploggingId=%s)",
                times_delivered, record_id, fields.get("ploggingId"),
            )
            await self.producer.publish_error(fields, f"{MAX_DELIVERIES}회 재시도 후에도 처리 실패")
            await self.redis.xack(REQUEST_STREAM, CONSUMER_GROUP, record_id)
            return
        await self._handle(record_id, fields)

    async def _handle(self, record_id: str, fields: dict) -> None:
        try:
            result = await self._process(fields)
            await self.producer.publish_success(fields, result)
        except Exception as e:
            logger.exception(
                "메시지 처리 실패 (recordId=%s ploggingId=%s)", record_id, fields.get("ploggingId")
            )
            await self.producer.publish_error(fields, str(e))
        finally:
            await self.redis.xack(REQUEST_STREAM, CONSUMER_GROUP, record_id)

    async def _process(self, fields: dict) -> dict:
        resp = await self.http.get(fields["imageUrl"])
        resp.raise_for_status()
        image = Image.open(io.BytesIO(resp.content)).convert("RGB")

        try:
            return await asyncio.wait_for(
                asyncio.to_thread(classify_material, image, self.yolo_predictor),
                timeout=PROCESS_TIMEOUT_S,
            )
        except PipelineError as e:
            raise RuntimeError(str(e))
        except asyncio.TimeoutError:
            raise RuntimeError(f"AI 처리 {PROCESS_TIMEOUT_S}초 타임아웃")
