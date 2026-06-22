"""Interactive terminal chat with the LiteLLM (Mercury) agent — non-streaming."""

from __future__ import annotations

import asyncio
import os

from dotenv import load_dotenv
from icecream import ic

from google.adk.agents.llm_agent import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
load_dotenv()

SESSION_ID = "12904022-1234-5678-9012-345678901234"
APP_NAME = "litellm_non_streaming_demo"
USER_ID = "user_12345"


# --- Model config (same as the streaming sample, but stream=False) ---
model = LiteLlm(
    model="openai/mercury-2",
    api_key=os.getenv("MERCURY_API_KEY"),
    api_base="https://api.inceptionlabs.ai/v1",
    max_tokens=8000,
    stream=False,
    extra_body={"reasoning_effort": "high"},
)

root_agent = LlmAgent(
    name="root_agent",
    model=model,
    instruction=("You are a helpful assistant with access to tools. "),
    tools=[],
)


async def create_session() -> InMemorySessionService:
    session_service = InMemorySessionService()
    await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID,
    )
    return session_service


async def call_agent(
    runner: Runner, user_input: str, session_id: str, user_id: str
) -> dict:
    content = types.Content(role="user", parts=[types.Part(text=user_input)])

    final_response_text = None
    final_event = None
    all_events = []

    async for event in runner.run_async(
        user_id=user_id, session_id=session_id, new_message=content
    ):
        all_events.append(event)

        if event.is_final_response():
            final_event = event
            if event.content and event.content.parts:
                final_response_text = event.content.parts[0].text
            elif event.actions and event.actions.escalate:
                final_response_text = (
                    f"Agent escalated: {event.error_message or 'No specific message.'}"
                )
            break

    return {
        "author": final_event.author if final_event else "unknown",
        "content": final_event.content if final_event else None,
        "type": type(final_event).__name__ if final_event else "unknown",
        "final_response_text": final_response_text,
        "final_event": final_event,
        "all_events": all_events,
    }


async def main():
    try:
        session_service = await create_session()
        runner = Runner(
            session_service=session_service,
            app_name=APP_NAME,
            agent=root_agent,
        )
        print("Session created successfully!")
        print("Type 'exit' or 'quit' to end the session.\n")
    except Exception as e:
        print(f"Error creating session and runner: {e}")
        return

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ("exit", "quit"):
            print("Exiting the session.")
            break

        if not user_input.strip():
            continue

        try:
            response = await call_agent(
                runner=runner,
                user_input=user_input,
                session_id=SESSION_ID,
                user_id=USER_ID,
            )
            print(f"\nAssistant: {response['final_response_text']}")
            ic(response)
        except Exception as e:
            print(f"Error calling agent: {e}")
            ic(e)
            break


if __name__ == "__main__":
    print("Starting LiteLLM Non-Streaming Assistant...")
    asyncio.run(main())
