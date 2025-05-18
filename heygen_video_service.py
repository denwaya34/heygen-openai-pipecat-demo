import json
import aiohttp
import base64
import uuid
import ssl
import certifi  # certifi をインポート

from pipecat.frames.frames import (
    Frame,
    TTSAudioRawFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    StartFrame,
    EndFrame,
    CancelFrame,
    ErrorFrame
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.ai_services import AIService
from pipecat.audio.utils import create_default_resampler
import websockets

from loguru import logger



class HeyGenVideoService(AIService):
    """Class to send agent audio to HeyGen using the streaming audio input api"""

    def __init__(
            self,
            *,
            session_id: str,
            session_token: str,
            realtime_endpoint: str,
            session: aiohttp.ClientSession,
            api_base_url: str = "https://api.heygen.com",
            **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._session_id = session_id
        self._session_token = session_token
        self._session = session
        self._api_base_url = api_base_url
        self._websocket = None
        self._buffered_audio_duration_ms = 0
        self._event_id = None
        self._realtime_endpoint = realtime_endpoint

    # AI Service class methods
    async def start(self, frame: StartFrame):
        logger.info("HeyGenVideoService starting")
        await super().start(frame)
        await self._ws_connect()

    async def stop(self, frame: EndFrame):
        logger.info("HeyGenVideoService stopping")
        await super().stop(frame)
        await self._ws_disconnect()

    async def cancel(self, frame: CancelFrame):
        logger.info("HeyGenVideoService canceling")
        await super().cancel(frame)
        await self._ws_disconnect()
        await self.stop_ttfb_metrics()
        await self.stop_processing_metrics()

    # websocket connection methods
    async def _ws_connect(self):
        """Connect to HeyGen websocket endpoint"""
        try:
            logger.info("HeyGenVideoService ws connecting")
            if self._websocket:
                # assume connected
                return

            # Create a default SSL context
            # ssl_context = ssl.create_default_context()
            # If you have a specific CA bundle to use, you can load it here:
            # ssl_context.load_verify_locations(cafile="/path/to/your/ca_bundle.pem")
            # certifi を使用してCA証明書バンドルを指定
            ssl_context = ssl.create_default_context(cafile=certifi.where())

            self._websocket = await websockets.connect(
                uri=self._realtime_endpoint,
                ssl=ssl_context  # Use the created SSL context
            )
            self._receive_task = self.get_event_loop().create_task(self._ws_receive_task_handler())
        except Exception as e:
            logger.error(f"{self} initialization error: {e}")
            self._websocket = None

    async def _ws_disconnect(self) -> None:
        """Disconnect from HeyGen websocket endpoint"""
        try:
            if self._websocket:
                await self._websocket.close()
        except Exception as e:
            logger.error(f"{self} disconnect error: {e}")
        finally:
            self._websocket = None

    async def _ws_receive_task_handler(self) -> None:
        """Handle incoming messages from HeyGen websocket"""
        try:
            while True:
                message = await self._websocket.recv()
                try:
                    parsed_message = json.loads(message)
                    await self._handle_ws_server_event(parsed_message)
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse websocket message as JSON: {e}")
                    continue
                if message:
                    logger.info(f"HeyGenVideoService ws received message: {message}")

        except Exception as e:
            logger.error(f"Error receiving message from websocket: {e}")

    async def _handle_ws_server_event(self, event: dict) -> None:
        """Handle an event from HeyGen websocket"""
        event_type = event.get("type")
        if event_type == "agent.status":
            logger.info(f"HeyGenVideoService ws received agent status: {event}")
        elif event_type == "agent.state":
            logger.info(f"HeyGenVideoService ws received agent.state: {event}")
        else:
            logger.error(f"HeyGenVideoService ws received unknown event: {event_type}")

    async def _ws_send(self, message: dict) -> None:
        """Send a message to HeyGen websocket"""
        try:
            logger.info(f"HeyGenVideoService ws sending message: {message.get('type')}")
            if self._websocket:
                await self._websocket.send(json.dumps(message))
            else:
                logger.error(f"{self} websocket not connected")
        except Exception as e:
            logger.error(f"Error sending message to websocket: {e}")
            await self.push_error(ErrorFrame(error=f"Error sending client event: {e}", fatal=True))

    # heygen api methods
    async def _stop_session(self) -> None:
        """Stop the current session"""
        try:
            await self._ws_disconnect()
        except Exception as e:
            logger.error(f"{self} stop ws error: {e}")
        url = f"{self._api_base_url}/v1/streaming.stop"
        headers = {"Content-Type": "application/json", "accept": "application/json", "Authorization": f"Bearer {self._session_token}"}
        body = {"session_id": self._session_id}
        async with self._session.post(url, headers=headers, json=body) as r:
            r.raise_for_status()

    async def _interrupt(self) -> None:
        """Interrupt the current session"""
        await self._ws_send({
            "type": "agent.interrupt",
            "event_id": str(uuid.uuid4()),
        })

    async def _start_agent_listening(self) -> None:
        await self._ws_send({
            "type": "agent.start_listening",
            "event_id": str(uuid.uuid4()),
        })

    async def _stop_agent_listening(self) -> None:
        """Stop listening animation"""
        await self._ws_send({
            "type": "agent.stop_listening",
            "event_id": str(uuid.uuid4()),
        })

    # audio buffer methods
    async def _send_audio(self, audio_bytes_in: bytes, sample_rate_in: int, event_id: str, finish: bool = False) -> None:
        TARGET_SAMPLE_RATE = 24000

        processed_audio_bytes: bytes
        if sample_rate_in == TARGET_SAMPLE_RATE:
            processed_audio_bytes = audio_bytes_in
        else:
            resampler = create_default_resampler()
            # Assuming default num_channels=1 and sample_width=2 are appropriate
            # based on common usage and how _calculate_audio_duration_ms implies 16-bit samples.
            processed_audio_bytes = await resampler.resample(
                audio=audio_bytes_in,
                in_rate=sample_rate_in,
                out_rate=TARGET_SAMPLE_RATE
            )

        self._buffered_audio_duration_ms += self._calculate_audio_duration_ms(processed_audio_bytes, TARGET_SAMPLE_RATE)
        await self._agent_audio_buffer_append(processed_audio_bytes, event_id)

        if finish and self._buffered_audio_duration_ms < 80:
            await self._agent_audio_buffer_clear()
            self._buffered_audio_duration_ms = 0

        if finish or self._buffered_audio_duration_ms > 1000:
            logger.info(f"Audio buffer duration from buffer: {self._buffered_audio_duration_ms:.2f}ms")
            await self._agent_audio_buffer_commit(event_id)
            self._buffered_audio_duration_ms = 0

    def _calculate_audio_duration_ms(self, audio: bytes, sample_rate: int) -> float:
        # Each sample is 2 bytes (16-bit audio)
        num_samples = len(audio) / 2
        return (num_samples / sample_rate) * 1000

    async def _agent_audio_buffer_append(
            self, audio: bytes, event_id: str
    ) -> None:
        audio_base64 = base64.b64encode(audio).decode("utf-8")
        await self._ws_send({
            "type": "agent.audio_buffer_append",
            "audio": audio_base64,
            "event_id": str(uuid.uuid4()),
        })

    async def _agent_audio_buffer_clear(self) -> None:
        await self._ws_send({
            "type": "agent.audio_buffer_clear",
            "event_id": str(uuid.uuid4()),
        })

    async def _agent_audio_buffer_commit(self, event_id: str) -> None:
        audio_base64 = base64.b64encode(b"\x00").decode("utf-8")
        await self._ws_send({
            "type": "agent.audio_buffer_commit",
            "audio": audio_base64,
            "event_id": str(uuid.uuid4()),
        })

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        try:
            if isinstance(frame, UserStartedSpeakingFrame):
                await self._interrupt()
                await self._start_agent_listening()
            elif isinstance(frame, UserStoppedSpeakingFrame):
                await self._stop_agent_listening()
            if isinstance(frame, TTSStartedFrame):
                logger.info("HeyGenVideoService TTS started")
                await self.start_processing_metrics()
                await self.start_ttfb_metrics()
                self._event_id = str(uuid.uuid4())
                await self._agent_audio_buffer_clear()
            elif isinstance(frame, TTSAudioRawFrame):
                logger.info("HeyGenVideoService TTS audio raw frame")
                await self._send_audio(frame.audio, frame.sample_rate, self._event_id, finish=False)
                await self.stop_ttfb_metrics()
            elif isinstance(frame, TTSStoppedFrame):
                logger.info("HeyGenVideoService TTS stopped")
                await self._send_audio(b"\x00\x00", 24000, self._event_id, finish=True)
                await self.stop_processing_metrics()
                self._event_id = None
            elif isinstance(frame, (EndFrame, CancelFrame)):
                logger.info("HeyGenVideoService session ended")
                await self._stop_session()
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
        finally:
            await self.push_frame(frame, direction)
