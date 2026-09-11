import asyncio
import logging
import signal

from stream_app.consumer import StreamConsumer
from yolo_app.predictor import YOLOPredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("plogrid-ai")


async def run() -> None:
    logger.info("YOLO model 로딩...")
    yolo_predictor = YOLOPredictor()

    logger.info("모델 로드 완료")

    consumer = StreamConsumer(yolo_predictor)
    consumer.start()

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop_event.set)
        except NotImplementedError:
            pass

    try:
        await stop_event.wait()
    finally:
        await consumer.stop()


if __name__ == "__main__":
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass
