from uuid import uuid4
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

# Define agent tools
def add_to_order(item: str, quantity: int) -> str:
    """Add an item to the customer's sandwich order."""
    return f"Added {quantity} x {item} to the order."

def confirm_order(order_summary: str) -> str:
    """Confirm the final order with customer."""
    return f"Order confirmed: {order_summary}. Sending to kitchen."

# Create agent with tools and memory
agent = create_agent(
        model="deepseek-chat",
        tools=[add_to_order, confirm_order],
        system_prompt="""You are a helpful sandwich shop assistant.
        Your goal is to take the user's order. Be concise and friendly.
        Do NOT use emojis, special characters, or markdown.
        Your responses will be read by a text-to-speech engine.""",
        checkpointer=InMemorySaver()
)

async def agent_stream(
        event_stream: AsyncIterator[VoiceAgentEvent]
        ) -> AsyncIterator[VoiceAgentEvent]:
    """
    Transform stream: Voice Events -> Voice Events (with Agent Responses)

    Pauses through all upstream events and adds agent_chunk events
    when processing STT transcript.
    """
    # Generate unique thread ID for conversation memory
    thread_id = str(uuid4())

    async for event in event_stream:
        # Pass through all upstream events
        yield event

        # Process final transcript through the agent
        if event.type == "stt_output":
            stream = agent.astream(
                {"messages": [HumanMessage(content=event.transcript)]},
                {"configurable": {"thread_id": thread_id}},
                stream_mode="messages",
            )

            # Yield agent response chunks as they arrive
            async for message, _ in stream:
                if message.text:
                    yield AgentChunkEvent.create(message.text)
