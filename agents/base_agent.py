"""
agents/base_agent.py
--------------------
Abstract base class that every agent inherits from.

Each agent:
  - connects to the EventBus
  - subscribes to its input topic(s)
  - processes messages in a loop
  - publishes results to its output topic

Subclasses must implement:
  - input_topics  : list[str]  — topics to subscribe to
  - output_topic  : str        — topic to publish results on
  - process(channel, payload)  — core logic, returns dict to publish
"""

from abc import ABC, abstractmethod
from loguru import logger
from event_bus.bus import EventBus


class BaseAgent(ABC):

    def __init__(self, name: str, bus: EventBus):
        self.name = name
        self.bus  = bus
        logger.info(f"[{self.name}] Initialised.")

    # ------------------------------------------------------------------
    # Subclasses must define these
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def input_topics(self) -> list[str]:
        """Topics this agent subscribes to."""
        ...

    @property
    @abstractmethod
    def output_topic(self) -> str:
        """Topic this agent publishes results to."""
        ...

    @abstractmethod
    def process(self, channel: str, payload: dict) -> dict | None:
        """
        Core logic. Receives a message and returns a dict to publish,
        or None to skip publishing (e.g. waiting for more data).
        """
        ...

    # ------------------------------------------------------------------
    # Run loop
    # ------------------------------------------------------------------

    def run(self):
        """Subscribe and start processing messages indefinitely."""
        self.bus.subscribe(*self.input_topics)
        logger.info(f"[{self.name}] Listening on {self.input_topics} ...")

        for channel, payload in self.bus.listen():
            try:
                result = self.process(channel, payload)
                if result is not None:
                    self.bus.publish(self.output_topic, result)
                    self.bus.set_latest(self.output_topic, result)
            except Exception as e:
                logger.exception(f"[{self.name}] Error processing message: {e}")
