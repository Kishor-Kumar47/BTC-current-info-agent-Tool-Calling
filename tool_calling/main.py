from agents import Agent, Runner, OpenAIChatCompletionsModel, AsyncOpenAI, set_tracing_disabled, function_tool
import os
from dotenv import load_dotenv
import requests


load_dotenv()
set_tracing_disabled(disabled=True)

gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY environment variable is not set.")

provider = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    openai_client=provider,
    model="gemini-1.5-flash"
)

@function_tool
def get_bitcoin_price(currency: str = "USD") -> str:
    """
    Fetch the current price of Bitcoin.

    Args:
        currency (str): The fiat currency to convert the price to, e.g., USD.
    
    Returns:
        str: The current price of Bitcoin in the specified currency.
    """
    try:
        if currency.upper() != "USD":
            return "Only USD is currently supported."

        response = requests.get("https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT")
        data = response.json()
        price = data.get("price")
        return f"The current price of Bitcoin in USD is ${price}."
    except requests.RequestException as e:
        return f"Error fetching Bitcoin price: {str(e)}"


agent = Agent(
    name="Bitcoin Price Tracker",
    instructions="You are a helpful assistant that tracks the price of Bitcoin.",
    model=model,
    tools=[get_bitcoin_price],
)

results = Runner.run_sync(
    agent,
    input="What is the current price of Bitcoin in usd?",
    
)

print("Agent response:", results.final_output)