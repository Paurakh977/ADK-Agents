
from __future__ import annotations

import asyncio
import os

from dotenv import load_dotenv
from google.adk.agents.run_config import RunConfig
from google.adk.agents.run_config import StreamingMode
from google.genai import types

from google.adk.agents.llm_agent import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import InMemoryRunner
import warnings
import logging

load_dotenv()

class _LLMBlockOnly(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return msg.startswith("LLM Request") or msg.startswith("LLM Response")

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(name)s -\n%(message)s",
)
_adk_log = logging.getLogger("google_adk")
_adk_log.setLevel(logging.DEBUG)
_adk_log.addFilter(_LLMBlockOnly())
# os.environ["OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"] = "true"

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

model = LiteLlm(
    model="openai/mercury-2",
    api_key=os.getenv("MERCURY_API_KEY"),
    api_base="https://api.inceptionlabs.ai/v1",
    max_tokens=8000,
    stream=True,
    extra_body={"reasoning_effort": "high"},
)

root_agent = LlmAgent(
    name="root_agent",
    model=model,
    instruction=(
        "You are a helpful assistant with access to tools. "
    ),
    tools=[],
)


async def _run_prompt(
    *,
    runner: InMemoryRunner,
    user_id: str,
    session_id: str,
    prompt: str,
) -> None:
  """Runs one prompt and prints partial chunks in real time."""
  content = types.Content(
      role='user',
      parts=[types.Part.from_text(text=prompt)],
  )

  print('Agent: ', end='', flush=True)
  saw_text = False
  saw_partial_text = False

  async for event in runner.run_async(
      user_id=user_id,
      session_id=session_id,
      new_message=content,
      run_config=RunConfig(streaming_mode=StreamingMode.SSE),
  ):
    if not event.content:
      continue
    text = ''.join(part.text for part in event.content.parts if part.text)
    if not text:
      continue

    if event.partial:
      print(text, end='', flush=True)
      saw_text = True
      saw_partial_text = True
      continue

    # With SSE mode, ADK emits a final aggregated event after partial chunks.
    if not saw_partial_text:
      print(text, end='', flush=True)
      saw_text = True

  if saw_text:
    print()
  else:
    print('(no text response)')
  print('------------------------------------')


async def main() -> None:
  app_name = 'litellm_streaming_demo'
  user_id = 'user_1'
  runner = InMemoryRunner(agent=root_agent, app_name=app_name)
  session = await runner.session_service.create_session(
      app_name=app_name,
      user_id=user_id,
  )

  print('Interactive chat started. Type "exit" or "quit" to stop.')
  print('------------------------------------')

  loop = asyncio.get_event_loop()

  while True:
    # input() is blocking, so run it in a thread executor to avoid
    # blocking the asyncio event loop.
    prompt = await loop.run_in_executor(None, input, 'You: ')
    prompt = prompt.strip()

    if not prompt:
      continue
    if prompt.lower() in ('exit', 'quit'):
      print('Goodbye!')
      break

    await _run_prompt(
        runner=runner,
        user_id=user_id,
        session_id=session.id,
        prompt=prompt,
    )


if __name__ == '__main__':
  asyncio.run(main())