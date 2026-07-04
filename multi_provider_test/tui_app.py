"""
OpenCode-style TUI on top of Google ADK + LiteLLM.

Flow:
  ctrl+k -> search/pick a provider from models.dev's full list -> enter
            creds (just an API key for ~90% of providers, more fields for
            the handful that need them) -> saved locally.
  ctrl+j -> search/pick a model from any provider you've already configured.
  type + enter -> chat, streamed, with [THINK]/[ANSWER] sections.

Run:  python tui_app.py
"""

from __future__ import annotations

import asyncio
from typing import Any

from textual import work
from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Footer, Header, Input, Label, ListItem, ListView, RichLog

import credential_store
import llm_runner
import provider_registry
from provider_registry import normalize_provider_id


class ProviderPickerScreen(ModalScreen[str | None]):
    """Search box + filtered list of all providers from the registry."""

    DEFAULT_CSS = """
    ProviderPickerScreen { align: center middle; }
    #dialog { width: 70; height: 24; border: round $accent; background: $surface; padding: 1; }
    """

    def __init__(self, providers: list[tuple[str, str]]) -> None:
        super().__init__()
        self._all = providers  # [(id, name)]

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Label("Connect a provider  [esc to cancel]")
            yield Input(placeholder="Search providers...", id="search")
            yield ListView(id="results")

    def on_mount(self) -> None:
        self.call_later(self._populate, self._all)
        self.query_one("#search", Input).focus()

    def _populate(self, items: list[tuple[str, str]]) -> None:
        lv = self.query_one("#results", ListView)
        lv.clear()
        configured = set(credential_store.list_configured_providers())
        for pid, name in items[:200]:
            tag = " (connected)" if pid in configured else ""
            item = ListItem(Label(f"{name}{tag}"))
            item.provider_id = pid  # type: ignore[attr-defined]
            lv.append(item)

    def on_input_changed(self, event: Input.Changed) -> None:
        q = event.value.lower().strip()
        if not q:
            self._populate(self._all)
            return
        filtered = [
            (pid, name)
            for pid, name in self._all
            if q in name.lower() or q in pid.lower()
        ]
        self._populate(filtered)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        lv = self.query_one("#results", ListView)
        if lv.index is not None and lv.children:
            item = lv.children[lv.index]
            self.dismiss(item.provider_id)  # type: ignore[attr-defined]

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        self.dismiss(event.item.provider_id)  # type: ignore[attr-defined]


class CredentialScreen(ModalScreen[dict[str, Any] | None]):
    """Dynamically renders one Input per field in the provider's credential schema."""

    DEFAULT_CSS = """
    CredentialScreen { align: center middle; }
    #dialog { width: 70; border: round $accent; background: $surface; padding: 1; }
    Input { margin-bottom: 1; }
    """

    def __init__(self, provider_id: str, provider_name: str) -> None:
        super().__init__()
        self.provider_id = provider_id
        self.provider_name = provider_name
        self.schema = provider_registry.schema_for(provider_id)

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Label(f"Connect {self.provider_name}  [esc to cancel]")
            for field in self.schema:
                yield Label(field["label"])
                yield Input(
                    password=field.get("secret", False), id=f"field_{field['key']}"
                )
            yield Label("[enter on last field submits]", classes="hint")

    def on_mount(self) -> None:
        inputs = self.query(Input)
        if inputs:
            inputs.first().focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        inputs = list(self.query(Input))
        idx = inputs.index(event.input)
        if idx < len(inputs) - 1:
            inputs[idx + 1].focus()
            return
        creds = {}
        for field, inp in zip(self.schema, inputs):
            val = inp.value.strip()
            if val:
                creds[field["key"]] = val
        missing = [
            f["label"]
            for f in self.schema
            if not f.get("optional") and f["key"] not in creds
        ]
        if missing:
            self.app.bell()
            return
        self.dismiss(creds)


class ModelPickerScreen(ModalScreen[tuple[str, str] | None]):
    """Search box + filtered list of models across configured providers only."""

    DEFAULT_CSS = """
    ModelPickerScreen { align: center middle; }
    #dialog { width: 80; height: 26; border: round $accent; background: $surface; padding: 1; }
    """

    def __init__(self, registry: dict[str, Any]) -> None:
        super().__init__()
        self.registry = registry
        configured = credential_store.list_configured_providers()
        self._all: list[tuple[str, str, str]] = []  # (provider_id, model_id, display)
        for pid in configured:
            pname = registry.get(pid, {}).get("name", pid)
            for mid in provider_registry.list_models(registry, pid):
                self._all.append((pid, mid, f"{mid}   [{pname}]"))

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            if not self._all:
                yield Label("No providers connected yet. Press esc, then ctrl+k first.")
            else:
                yield Label("Select model  [esc to cancel]")
                yield Input(placeholder="Search models...", id="search")
                yield ListView(id="results")

    def on_mount(self) -> None:
        if self._all:
            self.call_later(self._populate, self._all)
            self.query_one("#search", Input).focus()

    def _populate(self, items: list[tuple[str, str, str]]) -> None:
        lv = self.query_one("#results", ListView)
        lv.clear()
        for pid, mid, display in items[:300]:
            item = ListItem(Label(display))
            item.provider_id = pid  # type: ignore[attr-defined]
            item.model_id = mid  # type: ignore[attr-defined]
            lv.append(item)

    def on_input_changed(self, event: Input.Changed) -> None:
        q = event.value.lower().strip()
        filtered = self._all if not q else [t for t in self._all if q in t[2].lower()]
        self._populate(filtered)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        lv = self.query_one("#results", ListView)
        if lv.index is not None and lv.children:
            item = lv.children[lv.index]
            self.dismiss((item.provider_id, item.model_id))  # type: ignore[attr-defined]

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        self.dismiss((event.item.provider_id, event.item.model_id))  # type: ignore[attr-defined]


class ChatApp(App):
    ENABLE_COMMAND_PALETTE = False
    BINDINGS = [
        ("ctrl+k", "connect_provider", "Connect provider"),
        ("ctrl+j", "select_model", "Select model"),
        ("ctrl+c", "quit", "Quit"),
    ]
    CSS = """
    #chatlog { border: round $accent; height: 1fr; }
    #prompt { dock: bottom; }
    """

    def __init__(self) -> None:
        super().__init__()
        self.registry: dict[str, Any] = {}
        self.active_provider: str | None = None
        self.active_model: str | None = None
        self.active_credential_key: str | None = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield RichLog(id="chatlog", wrap=True, markup=True)
        yield Input(
            placeholder="Connect a provider with ctrl+k, pick a model with ctrl+j, then type here...",
            id="prompt",
        )
        yield Footer()

    def on_mount(self) -> None:
        self.title = "ai-tui"
        self.sub_title = "no model selected"
        self._load_registry()

    @work(exclusive=True, thread=True)
    def _load_registry(self) -> None:
        registry = provider_registry.fetch_registry()
        self.call_from_thread(self._on_registry_loaded, registry)

    def _on_registry_loaded(self, registry: dict[str, Any]) -> None:
        self.registry = registry
        log = self.query_one("#chatlog", RichLog)
        log.write(f"[dim]Loaded {len(registry)} providers from models.dev.[/dim]")

    def action_connect_provider(self) -> None:
        if not self.registry:
            self.bell()
            return
        providers = provider_registry.list_providers(self.registry)

        def after_pick(provider_id: str | None) -> None:
            if not provider_id:
                return
            name = self.registry.get(provider_id, {}).get("name", provider_id)
            internal_id = normalize_provider_id(provider_id)

            def after_creds(creds: dict[str, Any] | None) -> None:
                if not creds:
                    return
                credential_store.save_credentials(internal_id, creds)
                self.query_one("#chatlog", RichLog).write(
                    f"[green]Connected {name}.[/green]"
                )

            self.push_screen(CredentialScreen(provider_id, name), after_creds)

        self.push_screen(ProviderPickerScreen(providers), after_pick)

    def action_select_model(self) -> None:
        if not self.registry:
            self.bell()
            return

        def after_pick(result: tuple[str, str] | None) -> None:
            if not result:
                return
            provider_id, model_id = result
            self.active_provider = provider_id
            self.active_model = model_id
            self.active_credential_key = normalize_provider_id(provider_id)
            self.sub_title = f"{provider_id}/{model_id}"
            self.query_one("#chatlog", RichLog).write(
                f"[cyan]Active model: {provider_id}/{model_id}[/cyan]"
            )

        self.push_screen(ModelPickerScreen(self.registry), after_pick)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "prompt":
            return
        prompt = event.value.strip()
        event.input.value = ""
        if not prompt:
            return
        if not self.active_provider or not self.active_model:
            self.query_one("#chatlog", RichLog).write(
                "[red]No model selected. Press ctrl+k then ctrl+j.[/red]"
            )
            return
        self.run_chat(prompt)

    @work(exclusive=False)
    async def run_chat(self, prompt: str) -> None:
        log = self.query_one("#chatlog", RichLog)
        log.write(f"\n[bold]> {prompt}[/bold]")
        creds = credential_store.get_credentials(self.active_credential_key) or {}
        think_buf, answer_buf = "", ""
        try:
            async for kind, text in llm_runner.stream_chat(
                provider_id=self.active_provider,
                model_id=self.active_model,
                creds=creds,
                prompt=prompt,
            ):
                if kind == "think":
                    think_buf += text
                else:
                    answer_buf += text
        except Exception as e:
            log.write(f"[red]Error: {e}[/red]")
            return
        if think_buf:
            log.write(f"[dim]{think_buf}[/dim]")
        log.write(answer_buf or "[dim](no text response)[/dim]")


if __name__ == "__main__":
    ChatApp().run()
