from dataclasses import dataclass
from logging import Logger
from typing import ClassVar, List

import requests  # type: ignore

from .notification import PushNotificationConfig
from .utils import hilight


@dataclass
class NtfyNotificationConfig(PushNotificationConfig):
    notify_method = "ntfy"
    required_fields: ClassVar[List[str]] = ["ntfy_server", "ntfy_topic"]

    message_format: str | None = None
    ntfy_server: str | None = None
    ntfy_topic: str | None = None

    def handle_ntfy_server(self: "NtfyNotificationConfig") -> None:
        if self.ntfy_server is None:
            # Documented default ("ntfy_server - Optional - default to
            # https://ntfy.sh") was never actually applied here -- it was
            # left as None forever, which then made required_fields treat
            # it as permanently "missing" and silently drop every ntfy
            # notification for anyone who (correctly, per the docs) omitted
            # this field. Confirmed live: multiple 5-star matches never
            # notified because of exactly this.
            self.ntfy_server = "https://ntfy.sh"
            return
        if not isinstance(self.ntfy_server, str) or not self.ntfy_server:
            raise ValueError("An non-empty ntfy_server is needed.")

        if not self.ntfy_server.startswith("https://") and not self.ntfy_server.startswith(
            "http://"
        ):
            raise ValueError("ntfy_server must start with https:// or http://")

    def handle_ntfy_topic(self: "NtfyNotificationConfig") -> None:
        if self.ntfy_topic is None:
            return

        if not isinstance(self.ntfy_topic, str) or not self.ntfy_topic:
            raise ValueError("user requires an non-empty ntfy_topic.")

        self.ntfy_topic = self.ntfy_topic.strip()

    def send_message(
        self: "NtfyNotificationConfig",
        title: str,
        message: str,
        logger: Logger | None = None,
    ) -> bool:
        msg = f"{message}\n\nSent by https://github.com/BoPeng/ai-marketplace-monitor"
        assert self.ntfy_server is not None
        assert self.ntfy_topic is not None
        try:
            response = requests.post(
                f"{self.ntfy_server.rstrip('/')}/{self.ntfy_topic}",
                msg,
                headers={
                    "Title": title,
                    "Markdown": "yes" if self.message_format == "markdown" else "no",
                },
                # Separate connect and read timeouts, so "could not reach the
                # server" is distinguishable from "sent it, answer was slow".
                timeout=(5, 10),
            )
            # The status was previously ignored entirely: a rejected message
            # was reported as sent.
            response.raise_for_status()
        except requests.exceptions.ReadTimeout:
            # The request WAS sent and ntfy very likely delivered it; only the
            # reply was slow. The caller retries on any exception, and ntfy has
            # no idempotency key, so re-sending here delivers the notification
            # a second time. A possible duplicate is worse than a possible
            # missed log line, so this counts as sent.
            if logger:
                logger.info(
                    f"""{hilight("[Notify]", "info")} {self.name}: ntfy did not answer in time; """
                    f"""treating as sent rather than risking a duplicate."""
                )
            return True

        if logger:
            logger.info(
                f"""{hilight("[Notify]", "succ")} Sent {self.name} a message {hilight(msg)}"""
            )
        return True
