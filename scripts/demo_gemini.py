"""Demo script to interact with Zoya's Brain (LLM integration).

You can run this to talk to the AI and test tool calls.
Note: You must have GEMINI_API_KEY set in your .env file!
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Any

# Add the 'src' directory to Python's path so it can find 'zoya'
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / "src"))

from zoya.automation.tools.base import BaseTool
from zoya.core.config import load_settings
from zoya.llm.facade import ZoyaBrain

logging.basicConfig(level=logging.INFO)


# A dummy tool for testing the LLM's function calling
class GetWeatherTool(BaseTool):
    name = "get_weather"
    description = "Get the current weather for a specific location."
    readonly = True
    
    from pydantic import BaseModel, Field
    class ParamsModel(BaseModel):
        location: str = Field(..., description="The city name to get the weather for.")
        
    def _run(self, params: Any) -> dict[str, Any]:
        # Dummy weather data
        return {
            "location": params.location,
            "temperature_celsius": 24,
            "condition": "Sunny",
            "humidity_percent": 45
        }


async def main() -> None:
    settings = load_settings()
    
    if not settings.app.gemini_api_key:
        print("ERROR: GEMINI_API_KEY is not set in your .env file.")
        print("Please copy .env.example to .env and add your Gemini API key.")
        return

    # Instantiate our dummy tool
    tools = [GetWeatherTool()]

    # Initialize the brain with the settings and the tool
    brain = ZoyaBrain(settings=settings, tools=tools)

    print("--- Zoya Brain Test ---")
    print("Type 'exit' or 'quit' to stop.")
    
    while True:
        try:
            prompt = input("\nYou: ")
            if prompt.lower() in ("exit", "quit"):
                break
            if not prompt.strip():
                continue

            print("Zoya is thinking...")
            response = await brain.chat(prompt)
            print(f"\nZoya: {response}")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
            break

if __name__ == "__main__":
    asyncio.run(main())
