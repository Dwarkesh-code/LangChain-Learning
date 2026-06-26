from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Literal, Optional
from pydantic import BaseModel, Field

load_dotenv()
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")

class Summary(BaseModel):

    key_theme : list[str] = Field(description="Write down all the key themes that discused in the review")
    summary : str = Field(description="A brief summary of this review")
    sentiment : Literal["Positive", "Negative", "Neutral"] = Field(description="Write down the setiments Either Positive, Negative or Neutral")
    pros : Optional[list[str]] = Field(description="write down all pros of this review")
    cons : Optional[list[str]] = Field(description="write down all cons of this review")    
    name : Optional[str] = Field(description="Return the full name of the person who wrote this review. Only return a proper human name. If no human name is mentioned, return 'NoName'")

structhured_model = model.with_structured_output(Summary)

result = structhured_model.invoke("""
I've been using this PC for about 8 months now and my feelings are genuinely complicated. The processor handles my video editing workloads surprisingly well — I was not expecting this at this price point, and for that I'm grateful. But then the GPU drivers crashed twice in the first month, wiping out 3 hours of unsaved work each time. After a firmware update, the crashes stopped completely, so that problem technically no longer exists. The display has beautiful color accuracy for creative work, which I love, but the glossy finish makes it nearly unusable near a window, which is exactly where my desk is. RAM is 16GB which should be plenty, yet somehow Chrome + Premiere Pro together bring it to its knees — I don't know if that's the PC's fault or my own workflow. Build quality feels premium in the hand but two of the rubber feet fell off within weeks. Customer support replaced them for free and were very polite about it. The keyboard is the best I've ever typed on, no complaints at all. Thermals run hot under load but never hit throttling territory. I wanted to return it at month two. Now at month eight I'd probably buy it again, but I'd hesitate before recommending it to someone else.

""")

print(result.key_theme)
