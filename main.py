import logging
import os
from typing import Any

import aiohttp
import uvicorn
from fastapi import APIRouter, FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pipecat.services.openai.tts import OpenAITTSService
from pipecat.services.openai.stt import OpenAISTTService
from pipecat.frames.frames import (
    EndFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.serializers.protobuf import ProtobufFrameSerializer
from pipecat.services.openai import OpenAILLMService

from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.transcriptions.language import Language
from pipecat.transports.network.fastapi_websocket import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.transports.base_output import BaseOutputTransport, TransportParams
from heygen_video_service import HeyGenVideoService

logger = logging.getLogger(__name__)

deepgram_api_key = os.getenv("DEEPGRAM_API_KEY")
if not deepgram_api_key:
    raise ValueError("DEEPGRAM_API_KEY must be set")

openai_api_key=os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY")

elevenlabs_api_key=os.getenv("ELEVENLABS_API_KEY")
if not elevenlabs_api_key:
    raise ValueError("ELEVENLABS_API_KEY must be set")

async def run_bot(
        websocket_client: WebSocket,
        session_id: str,
        session_token: str,
        realtime_endpoint: str,
) -> None:
    async with aiohttp.ClientSession() as session:
        params = VADParams(
            min_volume=0.6,
            start_secs=0.2,
            stop_secs=1.2,
            confidence=0.7,
        )
        transport = FastAPIWebsocketTransport(
            websocket=websocket_client,
            params=FastAPIWebsocketParams(
                audio_out_enabled=True,
                add_wav_header=True,
                vad_enabled=True,
                vad_analyzer=SileroVADAnalyzer(params=params),
                vad_audio_passthrough=True,
                serializer=ProtobufFrameSerializer(),
            ),
        )

        stt = OpenAISTTService(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o-transcribe",
            prompt="日本語の言葉を期待する",
            language=Language.JA,
        )


        llm = OpenAILLMService(api_key=openai_api_key, model="gpt-4o-mini")
        messages = [
            {
                "role": "system",
                "content": """
#命令書：
あなたの属性は以下です。
- 性別: 男性
- 年齢: 30歳
- 職業: 漫才師
- 趣味: 映画鑑賞
- 口調: ルー大柴のように日本語と英語を混ぜて話す
- 性格: 明るくてお調子者
- 特技: 料理

以下の制約条件もとに、最高の結果を出力してください。

#制約条件：
 - ユーザの問いに対して、回答する。


""",
            },
        ]

        context = OpenAILLMContext(messages=messages)
        context_aggregator = llm.create_context_aggregator(context)

        tts = OpenAITTSService(api_key=os.getenv("OPENAI_API_KEY"), voice="ballad")

        heygen_video_service = HeyGenVideoService(session_id=session_id, session_token=session_token, session=session, realtime_endpoint=realtime_endpoint)
        output_transport = BaseOutputTransport(TransportParams(audio_out_enabled=True))
        pipeline = Pipeline(
            [
                transport.input(),  # Websocket input from client
                stt,  # Speech-To-Text
                context_aggregator.user(),
                llm,
                tts,
                heygen_video_service,
                output_transport,
                context_aggregator.assistant(),
            ]
        )
        task = PipelineTask(
            pipeline
        )

        @transport.event_handler("on_client_connected")
        async def on_client_connected(transport: Any, client: Any) -> None:
            logger.info("Client connected.")

        @transport.event_handler("on_client_disconnected")
        async def on_client_disconnected(transport: Any, client: Any) -> None:
            logger.info("Client disconnected.")
            await task.queue_frames([EndFrame()])
        runner = PipelineRunner(handle_sigint=False)

        await runner.run(task)

api_router = APIRouter()

@api_router.websocket("/user-audio-input")
async def websocket_endpoint(
        websocket: WebSocket,
        session_id: str,
        session_token: str,
        realtime_endpoint: str,
) -> None:
    logger.info(f"WebSocket connection established with session_id: {session_id}")
    logger.info(f"realtime_endpoint: {realtime_endpoint}")
    await websocket.accept()
    await run_bot(
        websocket,
        session_id,
        session_token,
        realtime_endpoint,
    )

app = FastAPI()
app.include_router(router=api_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


if __name__ == "__main__":
    port = int(os.getenv("PORT", 3001))
    workers = int(os.getenv("WORKERS", 1))
    uvicorn.run("main:app", host="0.0.0.0", port=port, workers=workers)
