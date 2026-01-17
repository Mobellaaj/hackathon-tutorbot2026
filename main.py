import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import httpx
from dotenv import load_dotenv
import logging
from pydantic import BaseModel
import json

# Load environment variables from .env
load_dotenv()

app = FastAPI()

# Get API keys from environment
KNOWUNITY_API_KEY = os.getenv("KNOWUNITY_API_KEY")
if not KNOWUNITY_API_KEY:
    raise RuntimeError("KNOWUNITY_API_KEY not set in environment variables")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_ASSESSOR_MODEL = os.getenv("OPENAI_ASSESSOR_MODEL", "gpt-4o-mini")

KNOWUNITY_BASE_URL = "https://knowunity-agent-olympics-2026-api.vercel.app"
OPENAI_BASE_URL = "https://api.openai.com/v1"

logger = logging.getLogger(__name__)

# In-memory conversation state
CONV_STATE = {}


# Default assessment state
DEFAULT_ASSESSMENT = {
    "predicted_level": 3,
    "confidence": 0.2,
    "evidence": [],
    "misconceptions": [],
    "student_style": {"brevity": 3, "emoji_level": 1, "tone": "neutral"},
    "updated_turn": 0,
    "next_diagnostic_goal": "Initial assessment"
}


# Request/Response models
class StartConversationRequest(BaseModel):
    student_id: str
    topic_id: str
    subject_name: str = ""
    topic_name: str = ""


class SendMessageRequest(BaseModel):
    conversation_id: str
    tutor_message: str


class TutorNextRequest(BaseModel):
    conversation_id: str
    subject_name: str
    topic_name: str
    max_turns: int = 10


@app.get("/", response_class=HTMLResponse)
async def get_home():
    """Serve the main UI page"""
    with open("templates/index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.get("/chat", response_class=HTMLResponse)
async def get_chat():
    """Serve the chat page"""
    with open("templates/chat.html", "r", encoding="utf-8") as f:
        return f.read()


@app.get("/results", response_class=HTMLResponse)
async def get_results():
    """Serve the results page"""
    with open("templates/results.html", "r", encoding="utf-8") as f:
        return f.read()


@app.get("/api/students")
async def get_students(set_type: str = "mini_dev"):
    """
    Proxy endpoint to fetch students from Knowunity API
    Injects X-Api-Key header
    """
    if set_type not in ["mini_dev", "dev", "eval"]:
        raise HTTPException(status_code=400, detail="Invalid set_type")
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{KNOWUNITY_BASE_URL}/students",
                params={"set_type": set_type},
                headers={"X-Api-Key": KNOWUNITY_API_KEY},
                timeout=10.0
            )
            response.raise_for_status()
            return response.json()
    except httpx.HTTPError as e:
        logger.error(f"Error fetching students: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch students: {str(e)}")


@app.get("/api/topics/{student_id}")
async def get_topics(student_id: str):
    """
    Proxy endpoint to fetch topics for a student from Knowunity API
    Injects X-Api-Key header
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{KNOWUNITY_BASE_URL}/students/{student_id}/topics",
                headers={"X-Api-Key": KNOWUNITY_API_KEY},
                timeout=10.0
            )
            response.raise_for_status()
            return response.json()
    except httpx.HTTPError as e:
        logger.error(f"Error fetching topics for student {student_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch topics: {str(e)}")


@app.post("/api/conversation/start")
async def start_conversation(request: StartConversationRequest):
    """
    Proxy endpoint to start a conversation with the Knowunity API
    Calls POST /interact/start with student_id and topic_id
    Initializes conversation state for history tracking and assessment
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{KNOWUNITY_BASE_URL}/interact/start",
                json={"student_id": request.student_id, "topic_id": request.topic_id},
                headers={"x-api-key": KNOWUNITY_API_KEY},
                timeout=10.0
            )
            response.raise_for_status()
            result = response.json()
            
            # Initialize conversation state
            conv_id = result.get("conversation_id")
            if conv_id:
                CONV_STATE[conv_id] = {
                    "history": [],
                    "subject_name": request.subject_name,
                    "topic_name": request.topic_name,
                    "max_turns": result.get("max_turns", 10),
                    "student_id": request.student_id,
                    "topic_id": request.topic_id,
                    "assessment": DEFAULT_ASSESSMENT.copy()
                }
            
            return result
    except httpx.HTTPError as e:
        logger.error(f"Error starting conversation: {e}")
        try:
            error_detail = response.text
        except:
            error_detail = str(e)
        raise HTTPException(status_code=500, detail=f"Failed to start conversation: {error_detail}")


async def update_assessment(conversation_id: str):
    """
    Call OpenAI assessor to continuously update student understanding level
    Based on last ~10 turns of conversation
    """
    if not OPENAI_API_KEY or conversation_id not in CONV_STATE:
        return
    
    state = CONV_STATE[conversation_id]
    subject = state.get("subject_name", "Subject")
    topic = state.get("topic_name", "Topic")
    history = state.get("history", [])
    
    # Limit context to last 10 turns (5 exchanges)
    recent_history = history[-10:] if len(history) > 10 else history
    
    # Build conversation context for assessor
    context_messages = [
        {
            "role": "user",
            "content": m["content"]
        } if m["role"] == "student" else {
            "role": "assistant",
            "content": m["content"]
        }
        for m in recent_history
    ]
    
    messages = [
        {
            "role": "system",
            "content": (
                f"You are an expert learning assessor. Your task is to infer the student's understanding level for: {subject} - {topic}\n"
                "Based ONLY on their responses, analyze:\n"
                "1. Correctness and depth of answers\n"
                "2. Quality of reasoning and explanation\n"
                "3. Any misconceptions or gaps\n"
                "4. Communication style (brevity, tone, emoji use)\n"
                "Output ONLY valid JSON in this exact schema:\n"
                "{\n"
                '  "predicted_level": 1-5 (1=novice, 5=mastery),\n'
                '  "confidence": 0-1,\n'
                '  "evidence": ["quote or paraphrase from student"],\n'
                '  "misconceptions": ["if any"],\n'
                '  "student_style": {"brevity": 1-5, "emoji_level": 0-3, "tone": "..."},\n'
                '  "next_diagnostic_goal": "what to assess next"\n'
                "}"
            )
        }
    ]
    
    # Add recent conversation
    messages.extend(context_messages)
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{OPENAI_BASE_URL}/chat/completions",
                json={
                    "model": OPENAI_ASSESSOR_MODEL,
                    "messages": messages,
                    "temperature": 0.5,
                    "max_tokens": 400
                },
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json"
                },
                timeout=15.0
            )
            response.raise_for_status()
            data = response.json()
            
            # Extract assistant response
            assistant_message = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            if not assistant_message:
                logger.warning(f"Empty assessor response for {conversation_id}")
                return
            
            # Parse JSON
            try:
                json_start = assistant_message.find("{")
                json_end = assistant_message.rfind("}") + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = assistant_message[json_start:json_end]
                else:
                    json_str = assistant_message
                
                assessment = json.loads(json_str)
                
                # Validate and set defaults
                assessment.setdefault("predicted_level", 3)
                assessment.setdefault("confidence", 0.5)
                assessment.setdefault("evidence", [])
                assessment.setdefault("misconceptions", [])
                assessment.setdefault("student_style", {"brevity": 3, "emoji_level": 1, "tone": "neutral"})
                assessment.setdefault("next_diagnostic_goal", "Continue assessment")
                assessment["updated_turn"] = len(history)
                
                # Store assessment
                CONV_STATE[conversation_id]["assessment"] = assessment
                logger.info(f"Assessment updated for {conversation_id[:8]}: level {assessment['predicted_level']}")
                
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse assessor JSON for {conversation_id}: {str(e)[:100]}")
                # Keep previous assessment
                
    except httpx.HTTPError as e:
        logger.warning(f"Assessor call failed for {conversation_id}: {str(e)[:100]}")
        # Keep previous assessment


async def finalize_assessment(conversation_id: str):
    """
    Compute final assessment for conversation completion.
    Called when turn_number >= max_turns or is_complete=true.
    Stores final_assessment in CONV_STATE.
    """
    if not OPENAI_API_KEY or conversation_id not in CONV_STATE:
        return
    
    state = CONV_STATE[conversation_id]
    subject = state.get("subject_name", "Subject")
    topic = state.get("topic_name", "Topic")
    history = state.get("history", [])
    current_assessment = state.get("assessment", DEFAULT_ASSESSMENT)
    
    # Build final assessment prompt
    context_messages = [
        {
            "role": "user",
            "content": m["content"]
        } if m["role"] == "student" else {
            "role": "assistant",
            "content": m["content"]
        }
        for m in history
    ]
    
    messages = [
        {
            "role": "system",
            "content": (
                f"You are an expert learning assessor. The student conversation on '{subject} - {topic}' has ended.\n"
                "Based on the FULL conversation history, provide a final comprehensive assessment.\n"
                "Output ONLY valid JSON:\n"
                "{\n"
                '  "final_level": 1-5,\n'
                '  "confidence": 0-1,\n'
                '  "key_evidence": ["most important quotes/paraphrases"],\n'
                '  "misconceptions": ["if any"],\n'
                '  "short_summary": "1-3 sentence summary of understanding"\n'
                "}"
            )
        }
    ]
    
    messages.extend(context_messages)
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{OPENAI_BASE_URL}/chat/completions",
                json={
                    "model": OPENAI_ASSESSOR_MODEL,
                    "messages": messages,
                    "temperature": 0.3,
                    "max_tokens": 400
                },
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json"
                },
                timeout=15.0
            )
            response.raise_for_status()
            data = response.json()
            
            assistant_message = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            if not assistant_message:
                logger.warning(f"Empty final assessment for {conversation_id}")
                # Fallback to current assessment
                CONV_STATE[conversation_id]["final_assessment"] = {
                    "final_level": current_assessment.get("predicted_level", 3),
                    "confidence": current_assessment.get("confidence", 0.5),
                    "key_evidence": current_assessment.get("evidence", []),
                    "misconceptions": current_assessment.get("misconceptions", []),
                    "short_summary": "Conversation completed."
                }
                return
            
            # Parse JSON
            try:
                json_start = assistant_message.find("{")
                json_end = assistant_message.rfind("}") + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = assistant_message[json_start:json_end]
                else:
                    json_str = assistant_message
                
                final_assessment = json.loads(json_str)
                
                # Validate fields
                final_assessment.setdefault("final_level", current_assessment.get("predicted_level", 3))
                final_assessment.setdefault("confidence", current_assessment.get("confidence", 0.5))
                final_assessment.setdefault("key_evidence", current_assessment.get("evidence", []))
                final_assessment.setdefault("misconceptions", current_assessment.get("misconceptions", []))
                final_assessment.setdefault("short_summary", "Conversation completed.")
                
                CONV_STATE[conversation_id]["final_assessment"] = final_assessment
                logger.info(f"Final assessment saved for {conversation_id[:8]}: level {final_assessment['final_level']}")
                
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse final assessment JSON: {str(e)[:100]}")
                # Fallback to current assessment
                CONV_STATE[conversation_id]["final_assessment"] = {
                    "final_level": current_assessment.get("predicted_level", 3),
                    "confidence": current_assessment.get("confidence", 0.5),
                    "key_evidence": current_assessment.get("evidence", []),
                    "misconceptions": current_assessment.get("misconceptions", []),
                    "short_summary": "Conversation completed."
                }
    
    except httpx.HTTPError as e:
        logger.warning(f"Final assessment call failed: {str(e)[:100]}")
        # Fallback to current assessment
        CONV_STATE[conversation_id]["final_assessment"] = {
            "final_level": current_assessment.get("predicted_level", 3),
            "confidence": current_assessment.get("confidence", 0.5),
            "key_evidence": current_assessment.get("evidence", []),
            "misconceptions": current_assessment.get("misconceptions", []),
            "short_summary": "Conversation completed."
        }


@app.post("/api/conversation/message")
async def send_message(request: SendMessageRequest):
    """
    Proxy endpoint to send a message in an ongoing conversation
    Calls POST /interact with conversation_id and tutor_message
    Stores message history for auto-tutor context
    Continuously updates assessment based on student response
    Finalizes assessment when session completes
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{KNOWUNITY_BASE_URL}/interact",
                json={"conversation_id": request.conversation_id, "tutor_message": request.tutor_message},
                headers={"x-api-key": KNOWUNITY_API_KEY},
                timeout=10.0
            )
            response.raise_for_status()
            result = response.json()
            
            # Store messages in conversation history
            conv_id = request.conversation_id
            if conv_id in CONV_STATE:
                # Add tutor message
                CONV_STATE[conv_id]["history"].append({
                    "role": "tutor",
                    "content": request.tutor_message
                })
                # Add student response
                if "student_response" in result:
                    CONV_STATE[conv_id]["history"].append({
                        "role": "student",
                        "content": result["student_response"]
                    })
                
                # Update assessment based on new student response
                await update_assessment(conv_id)
                
                # Include assessment in response
                result["assessment"] = CONV_STATE[conv_id].get("assessment", DEFAULT_ASSESSMENT)
                
                # Check if session is complete and finalize if needed
                turn_num = len(CONV_STATE[conv_id]["history"]) // 2
                max_turns = CONV_STATE[conv_id].get("max_turns", 10)
                is_complete = result.get("is_complete", False)
                
                if is_complete or turn_num >= max_turns:
                    # Compute and store final assessment
                    await finalize_assessment(conv_id)
                    result["final_assessment"] = CONV_STATE[conv_id].get("final_assessment")
            
            return result
    except httpx.HTTPError as e:
        logger.error(f"Error sending message: {e}")
        try:
            error_detail = response.text
        except:
            error_detail = str(e)
        raise HTTPException(status_code=500, detail=f"Failed to send message: {error_detail}")


@app.post("/api/tutor/next")
async def generate_next_tutor_message(request: TutorNextRequest):
    """
    Generate the next tutor message using OpenAI based on conversation context
    Returns: { "tutor_message": "...", "next_goal": "..." }
    """
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured")
    
    conv_id = request.conversation_id
    
    # Initialize state if missing
    if conv_id not in CONV_STATE:
        CONV_STATE[conv_id] = {
            "history": [],
            "subject_name": request.subject_name,
            "topic_name": request.topic_name,
            "max_turns": request.max_turns,
            "assessment": DEFAULT_ASSESSMENT.copy()
        }
    
    state = CONV_STATE[conv_id]
    assessment = state.get("assessment", DEFAULT_ASSESSMENT)
    
    # Build context for tutor
    turn_num = len(state["history"]) // 2 + 1
    misconceptions_text = ", ".join(assessment.get("misconceptions", [])) or "none identified"
    student_style = assessment.get("student_style", {})
    next_goal = assessment.get("next_diagnostic_goal", "Continue assessment")
    
    # Build OpenAI messages
    messages = [
        {
            "role": "system",
            "content": (
                "You are an experienced tutor helping a student learn. Speak ENGLISH only.\n"
                "Stay strictly on the selected topic. Keep responses to 3-6 sentences.\n"
                "Ask exactly ONE question or give ONE exercise per turn to guide the student.\n"
                "Adapt your tone based on the student's communication style, but be natural (not cringe).\n"
                "Do NOT reveal the student's predicted understanding level or scoring.\n"
                "Output ONLY valid JSON:\n"
                "{\"tutor_message\": \"...\", \"next_goal\": \"...\"}"
            )
        },
        {
            "role": "user",
            "content": (
                f"Subject: {state['subject_name']}\n"
                f"Topic: {state['topic_name']}\n"
                f"Turn: {turn_num}/{state['max_turns']}\n"
                f"Predicted student level: {assessment.get('predicted_level', 3)}/5\n"
                f"Known misconceptions: {misconceptions_text}\n"
                f"Assessment goal: {next_goal}\n"
                f"Student style: brief={student_style.get('brevity', 3)}/5, tone={student_style.get('tone', 'neutral')}\n\n"
                "Guide this student's learning. Focus on their misconceptions and diagnosed gaps."
            )
        }
    ]
    
    # Add recent history (last ~12 messages)
    recent_history = state["history"][-12:] if len(state["history"]) > 12 else state["history"]
    for msg in recent_history:
        # Map tutor->assistant, student->user for OpenAI
        openai_role = "assistant" if msg["role"] == "tutor" else "user"
        messages.append({
            "role": openai_role,
            "content": msg["content"]
        })
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{OPENAI_BASE_URL}/chat/completions",
                json={
                    "model": OPENAI_MODEL,
                    "messages": messages,
                    "temperature": 0.7,
                    "max_tokens": 500
                },
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json"
                },
                timeout=15.0
            )
            response.raise_for_status()
            data = response.json()
            
            # Extract assistant response
            assistant_message = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            if not assistant_message:
                raise ValueError("No message content from OpenAI")
            
            # Parse JSON from assistant response
            try:
                # Try to extract JSON from response (in case there's extra text)
                json_start = assistant_message.find("{")
                json_end = assistant_message.rfind("}") + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = assistant_message[json_start:json_end]
                else:
                    json_str = assistant_message
                
                result = json.loads(json_str)
                
                # Validate required fields
                if "tutor_message" not in result:
                    result["tutor_message"] = assistant_message[:300]
                result.setdefault("next_goal", "Continue learning")
                
                return result
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse tutor JSON for {conv_id[:8]}: {str(e)[:80]}")
                # Fallback: use raw response as tutor message
                return {
                    "tutor_message": assistant_message[:300],
                    "next_goal": "Continue with topic"
                }
    except httpx.HTTPError as e:
        logger.error(f"Error calling OpenAI tutor: {str(e)[:100]}")
        raise HTTPException(status_code=500, detail=f"Failed to generate tutor message")


@app.get("/api/assessment/{conversation_id}")
async def get_assessment(conversation_id: str):
    """
    Get the current assessment for a conversation
    """
    if conversation_id not in CONV_STATE:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    return CONV_STATE[conversation_id].get("assessment", DEFAULT_ASSESSMENT)


@app.get("/api/results/{conversation_id}")
async def get_results(conversation_id: str):
    """
    Get full conversation results including history and final assessment
    """
    if conversation_id not in CONV_STATE:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    state = CONV_STATE[conversation_id]
    
    return {
        "conversation_id": conversation_id,
        "student_id": state.get("student_id", ""),
        "topic_id": state.get("topic_id", ""),
        "subject_name": state.get("subject_name", ""),
        "topic_name": state.get("topic_name", ""),
        "max_turns": state.get("max_turns", 10),
        "history": state.get("history", []),
        "assessment": state.get("assessment", DEFAULT_ASSESSMENT),
        "final_assessment": state.get("final_assessment")
    }
