"""
Voice agent side — routes its TTS audio to the MuseTalk avatar worker instead of
publishing audio directly. The worker republishes audio+lip-synced video to the room.

Key lines:
  • session.output.audio = DataStreamAudioOutput(room, destination_identity="avatar_worker")
  • start the AgentSession with RoomOutputOptions(audio_enabled=False)  (worker owns the audio track)
  • dispatch the "musetalk-avatar" worker into the same room so it's present to receive the audio.
"""

import os

from livekit import agents
from livekit.agents import Agent, AgentSession, JobContext, RoomOutputOptions, WorkerOptions, cli
from livekit.agents.voice.avatar import DataStreamAudioOutput
from livekit.api import RoomAgentDispatch, RoomConfiguration
from livekit.plugins import openai, silero  # swap for your STT/LLM/TTS (incl. your Darija TTS)

AVATAR_IDENTITY = "avatar_worker"


async def entrypoint(ctx: JobContext):
    await ctx.connect()

    session = AgentSession(
        stt=openai.STT(),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=openai.TTS(),          # <-- replace with your Darija TTS (or any streaming TTS)
        vad=silero.VAD.load(),
    )

    # Route the agent's spoken audio to the MuseTalk worker over DataStream.
    session.output.audio = DataStreamAudioOutput(
        ctx.room, destination_identity=AVATAR_IDENTITY, wait_playback_start=True,
    )

    await session.start(
        agent=Agent(instructions="You are a helpful assistant."),
        room=ctx.room,
        # The worker publishes the audio track; the agent must NOT also publish audio.
        room_output_options=RoomOutputOptions(audio_enabled=False),
    )


if __name__ == "__main__":
    # Dispatch BOTH this agent and the avatar worker into the room. One way: explicit dispatch
    # via RoomConfiguration so the "musetalk-avatar" worker joins the same room automatically.
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name="voice-agent",
            # When a job is created for this agent, also dispatch the avatar worker:
            # room_config = RoomConfiguration(agents=[RoomAgentDispatch(agent_name="musetalk-avatar")])
            # (set this on the dispatch/room-creation side per your deployment).
        )
    )
