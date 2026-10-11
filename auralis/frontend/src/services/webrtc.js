/**
 * WebRTC client: captures the mic, negotiates a peer connection with the
 * FastAPI/aiortc backend via the /api/offer endpoint, opens a data channel
 * for control messages (transcript, emotion, latency, status), and plays
 * back the AI's audio track through a hidden <audio> element.
 */

import { BACKEND_URL } from "./api";

export class AuralisClient {
  constructor({ onMessage, onRemoteStream, onConnectionStateChange }) {
    this.pc = null;
    this.dataChannel = null;
    this.localStream = null;
    this.onMessage = onMessage;
    this.onRemoteStream = onRemoteStream;
    this.onConnectionStateChange = onConnectionStateChange;
  }

  async connect() {
    this.localStream = await navigator.mediaDevices.getUserMedia({ audio: true });

    this.pc = new RTCPeerConnection({
      iceServers: [{ urls: "stun:stun.l.google.com:19302" }],
    });

    this.pc.onconnectionstatechange = () => {
      this.onConnectionStateChange?.(this.pc.connectionState);
    };

    this.pc.ontrack = (event) => {
      this.onRemoteStream?.(event.streams[0]);
    };

    this.localStream.getTracks().forEach((track) => {
      this.pc.addTrack(track, this.localStream);
    });

    this.dataChannel = this.pc.createDataChannel("control");
    this.dataChannel.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        this.onMessage?.(data);
      } catch (err) {
        console.error("Failed to parse control message:", err);
      }
    };

    const offer = await this.pc.createOffer();
    await this.pc.setLocalDescription(offer);

    const response = await fetch(`${BACKEND_URL}/api/offer`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sdp: offer.sdp, type: offer.type }),
    });

    if (!response.ok) {
      throw new Error(`Backend rejected offer: ${response.status}`);
    }

    const answer = await response.json();
    await this.pc.setRemoteDescription(answer);
  }

  sendInterrupt() {
    if (this.dataChannel && this.dataChannel.readyState === "open") {
      this.dataChannel.send(JSON.stringify({ type: "manual_interrupt" }));
    }
  }

  disconnect() {
    this.localStream?.getTracks().forEach((track) => track.stop());
    this.dataChannel?.close();
    this.pc?.close();
    this.pc = null;
    this.dataChannel = null;
    this.localStream = null;
  }
}
