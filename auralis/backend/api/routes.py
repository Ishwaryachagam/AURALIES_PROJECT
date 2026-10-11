"""
HTTP API: health check + WebRTC SDP offer/answer exchange.

WebRTC signaling here uses the standard aiortc pattern: the browser POSTs
its SDP offer to /api/offer, the server creates an RTCPeerConnection,
attaches the Auralis pipeline to the incoming audio track, adds an outgoing
AI voice track, and returns its SDP answer. Actual audio then flows
peer-to-peer over WebRTC, not through this HTTP endpoint.
"""

import logging
from fastapi import APIRouter, Request
from aiortc import RTCPeerConnection, RTCSessionDescription

from backend.services.streaming_service import AIVoiceTrack, ConversationSession
from backend.models.model_manager import model_manager

logger = logging.getLogger("auralis.api")
router = APIRouter()

active_connections: set[RTCPeerConnection] = set()
active_sessions: dict[RTCPeerConnection, ConversationSession] = {}


@router.get("/health")
async def health():
    return {
        "status": "ok",
        "models_loaded": model_manager.loaded,
    }


@router.post("/api/offer")
async def offer(request: Request):
    params = await request.json()
    offer_desc = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    active_connections.add(pc)
    ai_track = AIVoiceTrack()

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info("Connection state: %s", pc.connectionState)
        if pc.connectionState in ("failed", "closed"):
            session = active_sessions.pop(pc, None)
            if session:
                session.stop()
            active_connections.discard(pc)
            await pc.close()

    @pc.on("datachannel")
    def on_datachannel(channel):
        logger.info("Data channel established: %s", channel.label)
        session = active_sessions.get(pc)
        if session:
            session.data_channel = channel

    @pc.on("track")
    def on_track(track):
        logger.info("Track received: %s", track.kind)
        if track.kind == "audio":
            session = ConversationSession(track, ai_track, data_channel=None)
            active_sessions[pc] = session
            session.start()

    pc.addTrack(ai_track)

    await pc.setRemoteDescription(offer_desc)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}


async def close_all_connections():
    for pc in list(active_connections):
        session = active_sessions.pop(pc, None)
        if session:
            session.stop()
        await pc.close()
    active_connections.clear()
