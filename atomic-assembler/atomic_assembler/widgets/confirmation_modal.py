import logging
from textual.screen import ModalScreen
from textual.widgets import Static
from textual.containers import Vertical
from textual.app import ComposeResult
from atomic_assembler.constants import PRIMARY_COLOR
from typing import Callable


class ConfirmationModal(ModalScreen):
    """A modal widget for confirming file selection."""

    def __init__(self, message: str, callback: Callable[[bool], None], mode: str = "yes_no"):
        super().__init__()
        self.message = message
        self.callback = callback
        self.mode = mode
        logging.info(f"ConfirmationModal initialized with message: {message} and mode: {mode}")

    BINDINGS = [
        ("y", "confirm", "Yes"),
        ("n", "dismiss", "No"),
    ]

    def compose(self) -> ComposeResult:
        logging.debug("Composing ConfirmationModal")
        if self.mode == "yes_no":
            yield Vertical(
                Static(self.message, id="modal-content", markup=False),
                Static("[Y]es / [N]o", id="options"),
                id="dialog",
            )
        elif self.mode == "continue":
            yield Vertical(
                Static(self.message, id="modal-content", markup=False),
                Static("Press any key to continue", id="options"),
                id="dialog",
            )

    def action_confirm(self) -> None:
        logging.info("Confirmation action triggered")
        self.app.pop_screen()
        self.callback(True)

    def action_dismiss(self) -> None:
        logging.info("Dismissal action triggered")
        self.app.pop_screen()
        self.callback(False)

    def on_mount(self):
        logging.debug("ConfirmationModal mounted")

    def on_key(self, event) -> None:
        if self.mode == "continue":
            logging.info(f"Key '{event.key}' pressed in continue mode")
            self.app.pop_screen()
            self.callback(True)
        # Removed the call to super().on_key(event)

    CSS = f"""
    ModalScreen {{
        align: center middle;
    }}

    #dialog {{
        width: 40%;
        height: auto;
        border: solid {PRIMARY_COLOR};
        background: $surface;
    }}

    Vertical {{
        align: center middle;
        background: $surface;
        padding: 1 2;
    }}

    #modal-content {{
        content-align: center middle;
        width: 100%;
        margin-bottom: 1;
        text-align: center;
        color: {PRIMARY_COLOR};
        text-style: bold;
    }}

    #options {{
        text-align: center;
        color: $text;
    }}

    Static {{
        width: 100%;
    }}
    """
