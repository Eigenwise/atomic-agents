"""FastAPI application for conversational AI with session management."""

import asyncio
import json
import uuid
from contextlib import aclosing, asynccontextmanager
from datetime import datetime
from typing import Dict

from atomic_agents import AtomicAgent
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from fastapi_memory.lib.agents.chat_agent import create_async_chat_agent
from fastapi_memory.lib.schemas import (
    ChatRequest,
    ChatResponse,
    ConversationHistory,
    ConversationMessage,
    SessionCreateResponse,
    SessionDeleteResponse,
    SessionInfo,
    UserSessionsResponse,
)

# One asynchronous agent per session, shared by standard and streaming responses.
async_sessions: Dict[str, Dict[str, AtomicAgent[ChatRequest, ChatResponse]]] = {}

# Session metadata: user_id -> session_id -> creation_timestamp
session_metadata: Dict[str, Dict[str, str]] = {}

# Conversation history: user_id -> session_id -> list of messages
conversation_history: Dict[str, Dict[str, list]] = {}

# Serialize turns and deletion within a session, without blocking other sessions.
session_locks: Dict[str, Dict[str, asyncio.Lock]] = {}


def _generate_session_id() -> str:
    """Generate a unique session identifier.

    Returns:
        UUID-based session identifier
    """
    return str(uuid.uuid4())


def _ensure_user_exists(user_id: str) -> None:
    """Ensure user exists in all storage dictionaries.

    Args:
        user_id: User identifier
    """
    if user_id not in async_sessions:
        async_sessions[user_id] = {}
    if user_id not in session_metadata:
        session_metadata[user_id] = {}
    if user_id not in conversation_history:
        conversation_history[user_id] = {}
    if user_id not in session_locks:
        session_locks[user_id] = {}


def _ensure_session_history_exists(user_id: str, session_id: str) -> None:
    """Ensure conversation history exists for a session.

    Args:
        user_id: User identifier
        session_id: Session identifier
    """
    _ensure_user_exists(user_id)
    if session_id not in conversation_history[user_id]:
        conversation_history[user_id][session_id] = []


def _add_message_to_history(
    user_id: str,
    session_id: str,
    role: str,
    content: str,
    suggested_questions: list[str] = None,
) -> None:
    """Add a message to the conversation history.

    Args:
        user_id: User identifier
        session_id: Session identifier
        role: Message role (user or assistant)
        content: Message content
        suggested_questions: Optional list of suggested questions
    """
    _ensure_session_history_exists(user_id, session_id)
    message = {
        "role": role,
        "content": content,
        "timestamp": datetime.now().isoformat(),
        "suggested_questions": suggested_questions,
    }
    conversation_history[user_id][session_id].append(message)


def _require_session(user_id: str, session_id: str) -> None:
    """Validate the session registry without creating state on read/error paths."""
    if session_id not in session_metadata.get(user_id, {}):
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found for user '{user_id}'")


def get_or_create_async_agent(user_id: str, session_id: str) -> AtomicAgent[ChatRequest, ChatResponse]:
    """Lazily create the shared model conversation for an existing session."""
    _require_session(user_id, session_id)
    if session_id not in async_sessions[user_id]:
        async_sessions[user_id][session_id] = create_async_chat_agent()
    return async_sessions[user_id][session_id]


@asynccontextmanager
async def _session_turn(user_id: str, session_id: str):
    """Serialize a model turn and discard its model history on failure or cancellation."""
    _require_session(user_id, session_id)
    async with session_locks[user_id][session_id]:
        # A queued operation may have waited behind session deletion.
        _require_session(user_id, session_id)
        agent = get_or_create_async_agent(user_id, session_id)
        previous_history = agent.history.copy()
        try:
            yield agent
        except BaseException:
            agent.history = previous_history
            raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager to clean up resources on shutdown.

    Args:
        app: FastAPI application instance

    Yields:
        None
    """
    yield
    async_sessions.clear()
    session_metadata.clear()
    conversation_history.clear()
    session_locks.clear()


app = FastAPI(
    title="Atomic Agents FastAPI Example",
    description="Simple example showing FastAPI integration with Atomic Agents",
    version="1.0.0",
    lifespan=lifespan,
)


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest) -> ChatResponse:
    """Process a chat message using non-streaming response.

    Args:
        request: Chat request containing message, user_id, and optional session ID

    Returns:
        ChatResponse with agent's reply and suggested questions

    Raises:
        HTTPException: If message processing fails
    """
    try:
        if not request.session_id:
            raise HTTPException(
                status_code=400, detail="session_id is required. Create a session first using POST /users/{user_id}/sessions"
            )

        async with _session_turn(request.user_id, request.session_id) as agent:
            result = await agent.run_async(ChatRequest(message=request.message, user_id=request.user_id))
            _add_message_to_history(request.user_id, request.session_id, "user", request.message)
            _add_message_to_history(
                request.user_id, request.session_id, "assistant", result.response, result.suggested_questions
            )
            return ChatResponse(
                response=result.response, session_id=request.session_id, suggested_questions=result.suggested_questions
            )

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to process message: {str(e)}")


@app.post("/chat/stream", tags=["Chat"])
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    """Process a chat message using streaming response.

    Args:
        request: Chat request containing message, user_id, and optional session ID

    Returns:
        StreamingResponse with Server-Sent Events format

    Raises:
        HTTPException: If streaming setup fails
    """
    try:
        if not request.session_id:
            raise HTTPException(
                status_code=400, detail="session_id is required. Create a session first using POST /users/{user_id}/sessions"
            )

        _require_session(request.user_id, request.session_id)

        async def generate():
            """Hold the session lock throughout streaming, including cancellation cleanup."""
            try:
                async with _session_turn(request.user_id, request.session_id) as agent:
                    last_chunk = None
                    async with aclosing(
                        agent.run_async_stream(ChatRequest(message=request.message, user_id=request.user_id))
                    ) as chunks:
                        async for chunk in chunks:
                            last_chunk = chunk.model_dump()
                            data = {
                                "response": last_chunk.get("response") or "",
                                "session_id": request.session_id,
                                "suggested_questions": last_chunk.get("suggested_questions"),
                            }
                            yield f"data: {json.dumps(data)}\n\n"
                    if last_chunk is None:
                        raise RuntimeError("Model stream returned no response.")
                    _add_message_to_history(request.user_id, request.session_id, "user", request.message)
                    _add_message_to_history(
                        request.user_id,
                        request.session_id,
                        "assistant",
                        last_chunk.get("response") or "",
                        last_chunk.get("suggested_questions"),
                    )
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e), 'session_id': request.session_id})}\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to setup stream: {str(e)}")


@app.post("/users/{user_id}/sessions", response_model=SessionCreateResponse, tags=["Sessions"])
async def create_session(user_id: str) -> SessionCreateResponse:
    """Create a new chat session for a user.

    Args:
        user_id: User identifier

    Returns:
        SessionCreateResponse with generated session ID

    Raises:
        HTTPException: If session creation fails
    """
    try:
        _ensure_user_exists(user_id)
        session_id = _generate_session_id()
        session_metadata[user_id][session_id] = datetime.now().isoformat()
        conversation_history[user_id][session_id] = []
        session_locks[user_id][session_id] = asyncio.Lock()

        return SessionCreateResponse(session_id=session_id, message=f"Session '{session_id}' created successfully")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create session: {str(e)}")


@app.get("/users/{user_id}/sessions", response_model=UserSessionsResponse, tags=["Sessions"])
async def get_user_sessions(user_id: str) -> UserSessionsResponse:
    """Get all sessions for a specific user.

    Args:
        user_id: User identifier

    Returns:
        UserSessionsResponse with list of user's sessions
    """
    # Metadata registers sessions before a model agent is needed.
    metadata = session_metadata.get(user_id, {})
    session_list = [SessionInfo(session_id=sid, created_at=metadata[sid]) for sid in sorted(metadata)]

    return UserSessionsResponse(user_id=user_id, sessions=session_list)


@app.get("/users/{user_id}/sessions/{session_id}/history", response_model=ConversationHistory, tags=["Sessions"])
async def get_conversation_history(user_id: str, session_id: str) -> ConversationHistory:
    """Get conversation history for a specific session.

    Args:
        user_id: User identifier
        session_id: Session identifier

    Returns:
        ConversationHistory with all messages in the session

    Raises:
        HTTPException: If session is not found
    """
    _require_session(user_id, session_id)

    messages = conversation_history.get(user_id, {}).get(session_id, [])

    return ConversationHistory(session_id=session_id, messages=[ConversationMessage(**msg) for msg in messages])


@app.delete("/users/{user_id}/sessions/{session_id}", response_model=SessionDeleteResponse, tags=["Sessions"])
async def delete_session(user_id: str, session_id: str) -> SessionDeleteResponse:
    """Delete a specific session for a user.

    Args:
        user_id: User identifier
        session_id: Session identifier to delete

    Returns:
        SessionDeleteResponse with success message

    Raises:
        HTTPException: If session is not found
    """
    _require_session(user_id, session_id)
    async with session_locks[user_id][session_id]:
        _require_session(user_id, session_id)
        async_sessions[user_id].pop(session_id, None)
        conversation_history[user_id].pop(session_id, None)
        del session_metadata[user_id][session_id]
        del session_locks[user_id][session_id]

    return SessionDeleteResponse(message=f"Session '{session_id}' deleted successfully")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
