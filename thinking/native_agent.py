from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field

from dotenv import load_dotenv
from google import genai
from google.adk.agents.run_config import RunConfig
from google.adk.agents.run_config import StreamingMode
from google.genai import types

from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.llm_agent import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import InMemoryRunner
import warnings
import logging

import colorama
from colorama import Fore, Style

load_dotenv()
colorama.init()

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------
THOUGHT_COLOR = Fore.LIGHTBLACK_EX
ANSWER_COLOR = Fore.GREEN
USAGE_COLOR = Fore.CYAN
CONTEXT_COLOR = Fore.YELLOW
SESSION_COLOR = Fore.MAGENTA
RESET = Style.RESET_ALL

MODEL_NAME = 'gemini-flash-latest'
FALLBACK_INPUT_LIMIT = 1_048_576
FALLBACK_OUTPUT_LIMIT = 65_535

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
adk_log_handler = logging.FileHandler("adk_debug.log", encoding="utf-8")
adk_log_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(levelname)s - %(name)s -\n%(message)s")
)
_adk_log = logging.getLogger("google_adk")
_adk_log.setLevel(logging.DEBUG)
_adk_log.addHandler(adk_log_handler)
_adk_log.propagate = False

class _LLMBlockOnly(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return msg.startswith("LLM Request") or msg.startswith("LLM Response")

_adk_log.addFilter(_LLMBlockOnly())
logging.basicConfig(level=logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")


# ---------------------------------------------------------------------------
# Session-wide (cumulative) usage tracker
# ---------------------------------------------------------------------------
@dataclass
class SessionUsage:
  """Accumulates billed tokens across every turn since the app started.

  This is DIFFERENT from '[context] used' below: context usage reflects how
  much of the model's window the *current* conversation state occupies
  (which already includes all prior history resent as the prompt). Session
  usage is the sum of what you were actually billed for, turn by turn —
  useful for cost tracking, not for window-occupancy tracking.
  """
  turn_count: int = 0
  total_prompt_tokens: int = 0
  total_thoughts_tokens: int = 0
  total_candidates_tokens: int = 0
  total_tokens: int = 0

  def add(self, usage) -> None:
    self.turn_count += 1
    self.total_prompt_tokens += usage.prompt_token_count or 0
    self.total_thoughts_tokens += getattr(usage, 'thoughts_token_count', 0) or 0
    self.total_candidates_tokens += getattr(usage, 'candidates_token_count', 0) or 0
    self.total_tokens += usage.total_token_count or 0


root_agent = Agent(
    model=MODEL_NAME,
    name='root_agent',
    description="Ai agent that can answer questions and perform tasks.",
    instruction="You are a helpful assistant.",
    generate_content_config=types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(
            include_thoughts=True,
            thinking_budget=-1,
        ),
    ),
)


async def _get_context_window_limit(model_name: str) -> int:
  try:
    client = genai.Client()
    model_info = await asyncio.to_thread(client.models.get, model=model_name)
    input_limit = getattr(model_info, 'input_token_limit', None) or FALLBACK_INPUT_LIMIT
    output_limit = getattr(model_info, 'output_token_limit', None) or FALLBACK_OUTPUT_LIMIT
    print(f'{Fore.MAGENTA}[debug] live model info: '
          f'name={getattr(model_info, "name", model_name)} '
          f'input_limit={input_limit:,} output_limit={output_limit:,}{RESET}')
    return input_limit + output_limit
  except Exception as exc:
    print(f'{Fore.RED}[warn] Could not fetch live context limit ({exc}); '
          f'using fallback.{RESET}')
    return FALLBACK_INPUT_LIMIT + FALLBACK_OUTPUT_LIMIT


async def _run_prompt(
    *,
    runner: InMemoryRunner,
    user_id: str,
    session_id: str,
    prompt: str,
    context_limit: int,
    session_usage: SessionUsage,
) -> None:
  content = types.Content(role='user', parts=[types.Part.from_text(text=prompt)])

  saw_partial_thought = False
  saw_partial_answer = False
  printed_thought_header = False
  printed_answer_header = False
  last_usage = None  # THIS invocation's usage only

  async for event in runner.run_async(
      user_id=user_id,
      session_id=session_id,
      new_message=content,
      run_config=RunConfig(streaming_mode=StreamingMode.SSE),
  ):
    if getattr(event, 'usage_metadata', None):
      last_usage = event.usage_metadata  # overwritten each event -> ends up as final/largest for this turn

    if not event.content or not event.content.parts:
      continue

    for part in event.content.parts:
      if not part.text:
        continue
      is_thought = bool(getattr(part, 'thought', False))

      if event.partial:
        if is_thought:
          if not printed_thought_header:
            print(f'\n{THOUGHT_COLOR}🤔 Thinking: ', end='', flush=True)
            printed_thought_header = True
          print(f'{THOUGHT_COLOR}{part.text}', end='', flush=True)
          saw_partial_thought = True
        else:
          if not printed_answer_header:
            print(f'{RESET}\n\n{ANSWER_COLOR}Agent: ', end='', flush=True)
            printed_answer_header = True
          print(f'{ANSWER_COLOR}{part.text}', end='', flush=True)
          saw_partial_answer = True
        continue

      if is_thought and saw_partial_thought:
        continue
      if not is_thought and saw_partial_answer:
        continue

      if is_thought:
        if not printed_thought_header:
          print(f'\n{THOUGHT_COLOR}🤔 Thinking: ', end='', flush=True)
          printed_thought_header = True
        print(f'{THOUGHT_COLOR}{part.text}', end='', flush=True)
      else:
        if not printed_answer_header:
          print(f'{RESET}\n\n{ANSWER_COLOR}Agent: ', end='', flush=True)
          printed_answer_header = True
        print(f'{ANSWER_COLOR}{part.text}', end='', flush=True)

  print(RESET)

  if not printed_thought_header and not printed_answer_header:
    print('(no response)')

  if last_usage:
    session_usage.add(last_usage)  # fold this turn into the running session total

    total = last_usage.total_token_count
    pct = (total / context_limit * 100) if context_limit else 0.0

    # --- THIS INVOCATION ONLY ---
    print(
        f'{USAGE_COLOR}[this turn] prompt_tokens={last_usage.prompt_token_count} '
        f'thoughts_tokens={getattr(last_usage, "thoughts_token_count", 0)} '
        f'candidates_tokens={getattr(last_usage, "candidates_token_count", 0)} '
        f'total_tokens={total}{RESET}'
    )
    print(
        f'{CONTEXT_COLOR}[context now] {total:,} / {context_limit:,} tokens '
        f'({pct:.2f}% of window occupied by current conversation state){RESET}'
    )

    # --- CUMULATIVE ACROSS THE WHOLE SESSION (SUM of every turn's billing) ---
    print(
        f'{SESSION_COLOR}[session total, {session_usage.turn_count} turns] '
        f'prompt={session_usage.total_prompt_tokens:,} '
        f'thoughts={session_usage.total_thoughts_tokens:,} '
        f'candidates={session_usage.total_candidates_tokens:,} '
        f'billed_total={session_usage.total_tokens:,}{RESET}'
    )

  print('------------------------------------')


async def main() -> None:
  app_name = 'litellm_streaming_demo'
  user_id = 'user_1'
  runner = InMemoryRunner(agent=root_agent, app_name=app_name)
  session = await runner.session_service.create_session(app_name=app_name, user_id=user_id)

  context_limit = await _get_context_window_limit(MODEL_NAME)
  print(f'{CONTEXT_COLOR}Context window for {MODEL_NAME}: {context_limit:,} tokens{RESET}')

  session_usage = SessionUsage()

  print('Interactive chat started. Type "exit" or "quit" to stop.')
  print('------------------------------------')

  loop = asyncio.get_event_loop()

  while True:
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
        context_limit=context_limit,
        session_usage=session_usage,
    )


if __name__ == '__main__':
  asyncio.run(main())