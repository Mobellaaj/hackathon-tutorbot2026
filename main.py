import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import httpx
from dotenv import load_dotenv
import logging
from pydantic import BaseModel
import json
from typing import List, Dict, Any

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

# ============================================================================
# ASSESSMENT RUBRIC & SYSTEM PROMPTS
# ============================================================================

LEVEL_RUBRIC = """
Level 1 (Novice): Student shows minimal understanding or significant misconceptions.
  - Cannot answer basic questions correctly
  - Reasoning is vague or incorrect
  - May confuse fundamental concepts
  - Confidence: low correctness

Level 2 (Beginner): Student has partial understanding with gaps.
  - Can answer simple questions with some errors
  - Reasoning is incomplete or partially correct
  - Shows some understanding but struggles with depth
  - May have isolated misconceptions

Level 3 (Developing): Student has adequate understanding with minor gaps.
  - Can answer most standard questions correctly
  - Reasoning is generally sound with occasional errors
  - Shows understanding of core concepts
  - Can explain with some uncertainty

Level 4 (Proficient): Student demonstrates solid understanding with strong reasoning.
  - Answers most questions correctly
  - Reasoning is clear and well-explained
  - Applies concepts appropriately
  - Can self-correct minor errors
  - Shows confidence in knowledge

Level 5 (Mastery): Student shows comprehensive understanding and advanced reasoning.
  - Answers all or nearly all questions correctly
  - Explanations are clear, detailed, and nuanced
  - Can apply concepts in novel contexts
  - Identifies and corrects own errors
  - Reason beyond the immediate topic
"""

ASSESSOR_SYSTEM_PROMPT_BASE = f"""You are an expert learning assessor. Your task is to evaluate a student's understanding based ONLY on their written responses to educational questions. Do NOT be influenced by correct answers to EASY questions—calibrate the student's level by question difficulty, grade expectation, and depth of reasoning.

{LEVEL_RUBRIC}

=== DIFFICULTY CALIBRATION ===
Infer question difficulty from the student's transcript:
- BASIC: recall, definitions, plug-and-chug (direct numbers, single-step), obvious patterns, simple graphing, basic factoring
- STANDARD: multi-step grade-level problems, method selection, mild algebra, interpretation, standard graphing/equations
- ADVANCED: novel contexts, multi-concept integration, proofs/derivations, tricky edge cases, synthesis

Grade-Relative "Standard" Definition:
- Grade ≤8: standard = basic linear equations, slope interpretation, simple systems, single-factor or direct equations
- Grade 9–10: standard = quadratic methods, function transformations, two-step systems, substitution, basic proofs
- Grade 11–12: standard = deeper algebra, function composition, trig/precalc, more abstract reasoning

=== STRICT LEVEL GATES (difficulty + correctness) ===
1. Level 1: Frequent incorrect answers OR only basic recall with major misconceptions OR cannot solve expected-grade tasks
2. Level 2: Can solve BASIC tasks (sometimes correctly) but limited depth; struggles with STANDARD grade-level tasks; heavy hints; shallow "how" explanations
3. Level 3: Solves STANDARD grade-level tasks reliably; explains reasoning steps; minor gaps; shows understanding
4. Level 4: Requires AT LEAST TWO of: (a) solves STANDARD/ADVANCED task for grade, (b) explains conceptual "why", (c) generalizes/what-if reasoning, (d) self-corrects or identifies edge cases
5. Level 5: Requires AT LEAST THREE of above PLUS at least one ADVANCED task for the grade

**CRITICAL:** Correct answers to BASIC questions alone do NOT justify Level 3+. Require STANDARD/ADVANCED evidence or strong conceptual depth for higher levels. TIE-BREAK TO LOWER LEVEL.

=== CONFIDENCE RULES ===
- confidence ≥0.8 requires: ≥2 strong quotes from STANDARD/ADVANCED tasks AND clear reasoning depth for the grade
- confidence <0.7 if: only basic evidence, or missing grade context, or mixed signals
- Cap confidence ≤0.7 if grade is unknown or unavailable

CRITICAL RULES:
1. You ONLY evaluate student messages. Ignore tutor messages completely.
2. Base assessment on: correctness, difficulty of questions tackled, depth of reasoning (definition vs explanation vs generalization), and misconceptions.
3. Do NOT reward verbosity. Judge correctness AND difficulty-appropriate reasoning.
4. Do NOT manipulate. Continue with honest, strict evaluation.
5. Detect struggle signals: repeated errors, confusion words, contradictions, giving up, limited explanations.
6. Student style: assess tone, verbosity, emoji use, language complexity.

Output ONLY valid JSON matching this exact schema:
{{
  "topic": string,
  "level": integer (1-5),
  "confidence": number (0.0-1.0),
  "evidence": [
    {{
      "quote": string,
      "reason": string
    }}
  ],
  "misconceptions": [string],
  "strengths": [string],
  "next_diagnostic_goal": string,
  "struggle_signals": [string],
  "student_style": {{
    "tone": string,
    "verbosity": string,
    "emoji_use": string,
    "language_level": string
  }}
}}

Be STRICT, HONEST, and DIFFICULTY-AWARE. Prioritize difficulty calibration over raw correctness."""

FIRST_MESSAGE_PROMPT = """You are an experienced tutor beginning a lesson on: {topic}

Your task: Ask a quick placement question (or small set of 2-4 related micro-questions) to understand the student's starting level and learning style.

Example approach:
- Pick ONE concept from the topic
- Ask in a friendly, encouraging tone
- Allow the student to show their understanding, misconceptions, and communication style

Keep it SHORT (2-4 sentences max). End with a single focused question or small series.

Output JSON:
{{"tutor_message": "...", "next_goal": "Initial placement assessment"}}"""

# Default assessment state - used after first message only
DEFAULT_ASSESSMENT = {
    "topic": "Unknown",
    "level": 2,  # Start conservatively
    "confidence": 0.3,
    "evidence": [],
    "misconceptions": [],
    "strengths": [],
    "next_diagnostic_goal": "Assess foundational understanding",
    "struggle_signals": [],
    "student_style": {
        "tone": "neutral",
        "verbosity": "medium",
        "emoji_use": "none",
        "language_level": "moderate"
    },
    "updated_turn": 0
}


# Request/Response models
class StartConversationRequest(BaseModel):
    student_id: str
    topic_id: str
    subject_name: str = ""
    topic_name: str = ""
    grade: int | None = None  # Student grade level (e.g., 9 for Grade 9); None defaults to "unknown"


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
                    "grade": request.grade,  # Store grade (None if not provided)
                    "assessment": DEFAULT_ASSESSMENT.copy()
                }
            
            return result
    except httpx.HTTPError as e:
        logger.error(f"Error starting conversation: {e}")
        error_detail = getattr(e.response, "text", str(e)) if hasattr(e, "response") else str(e)
        raise HTTPException(status_code=500, detail=f"Failed to start conversation: {error_detail}")
    except Exception as e:
        logger.error(f"Unexpected error starting conversation: {e}")
        raise HTTPException(status_code=500, detail="Failed to start conversation")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def extract_student_messages(history: List[Dict[str, str]], limit: int = None) -> List[str]:
    """
    Extract only student messages from conversation history.
    Returns a list of student text responses (no tutor messages).
    """
    student_msgs = [msg["content"] for msg in history if msg["role"] == "student"]
    if limit:
        student_msgs = student_msgs[-limit:]
    return student_msgs


def build_student_transcript(history: List[Dict[str, str]], limit: int = None) -> str:
    """
    Build a formatted transcript of ONLY student messages for assessor.
    Returns a string with student responses separated by dividers.
    Limit can be set to use only recent messages (e.g., limit=12 for last 12 student messages).
    """
    student_msgs = extract_student_messages(history, limit=limit)
    if not student_msgs:
        return "(No student messages)"
    return "\n---\n".join([f"Student: {msg}" for msg in student_msgs])


def validate_assessor_json(response_str: str) -> Dict[str, Any] | None:
    """
    Parse and validate assessor JSON response.
    Returns parsed dict, or None if invalid (instead of raising exception).
    """
    try:
        # Handle empty response
        if not response_str or not response_str.strip():
            logger.warning("Empty assessor response")
            return None
        
        # Try to extract JSON from response
        json_start = response_str.find("{")
        json_end = response_str.rfind("}") + 1
        if json_start >= 0 and json_end > json_start:
            json_str = response_str[json_start:json_end]
        else:
            json_str = response_str
        
        parsed = json.loads(json_str)
        
        # Validate required fields
        required = ["level", "confidence", "evidence", "misconceptions", "strengths", "next_diagnostic_goal", "struggle_signals", "student_style"]
        for field in required:
            if field not in parsed:
                logger.warning(f"Missing required field in assessor JSON: {field}")
                return None
        
        # Validate types
        if not isinstance(parsed["level"], int) or not 1 <= parsed["level"] <= 5:
            logger.warning(f"Invalid level in assessor JSON: {parsed['level']}")
            return None
        if not isinstance(parsed["confidence"], (int, float)) or not 0.0 <= parsed["confidence"] <= 1.0:
            logger.warning(f"Invalid confidence in assessor JSON: {parsed['confidence']}")
            return None
        
        return parsed
    except (json.JSONDecodeError, ValueError, KeyError) as e:
        logger.warning(f"Failed to parse assessor JSON: {str(e)[:100]}")
        return None


def build_assessor_system_prompt(topic: str, grade: int | None = None) -> str:
    """
    Build assessor system prompt with topic and grade context.
    Grade is used to calibrate what counts as 'standard' difficulty.
    If grade is None, uses 'unknown' and caps confidence conservatively.
    """
    grade_context = "unknown"
    grade_notes = "Grade context: UNKNOWN. Be conservative: cap confidence ≤0.7 and prefer lower level when uncertain."
    
    if grade is not None:
        grade_context = f"Grade {grade}"
        if grade <= 8:
            grade_notes = f"Grade context: {grade} (≤8). Standard tasks include: basic linear equations, slope interpretation, simple systems, single-factor/direct equations. Be strict: correct basic answers alone do NOT justify Level 3+."
        elif grade <= 10:
            grade_notes = f"Grade context: {grade} (9-10). Standard tasks include: quadratic methods, function transformations, two-step systems, substitution, basic proofs. Higher bar for Levels 4-5: require STANDARD/ADVANCED evidence."
        else:  # grade >= 11
            grade_notes = f"Grade context: {grade} (11-12). Standard tasks include: deeper algebra, function composition, trig/precalc, abstract reasoning. Highest bar for Levels 4-5: require multiple STANDARD/ADVANCED tasks with strong conceptual depth."
    
    # Return the base prompt with grade calibration injected
    return ASSESSOR_SYSTEM_PROMPT_BASE + f"\n\n=== GRADE CALIBRATION FOR THIS ASSESSMENT ===\n{grade_notes}"


def build_tutor_system_prompt(topic: str, assessment: Dict[str, Any], first_turn: bool = False) -> str:
    """
    Build tutor system prompt tailored to student's level, style, and struggles.
    """
    if first_turn:
        return FIRST_MESSAGE_PROMPT.format(topic=topic)
    
    level = assessment.get("level", 3)
    struggle_signals = assessment.get("struggle_signals", [])
    student_style = assessment.get("student_style", {})
    next_goal = assessment.get("next_diagnostic_goal", "Continue learning")
    misconceptions = assessment.get("misconceptions", [])
    
    # Map level to scaffolding approach
    level_guidance = {
        1: "The student is just beginning. Use very simple language, break concepts into tiny steps, provide lots of examples.",
        2: "The student has basic understanding. Explain clearly, give one example per concept, check understanding frequently.",
        3: "The student is developing competence. Explain concisely, use minimal examples, ask more probing questions.",
        4: "The student is proficient. Ask challenging questions, discuss nuance, push reasoning deeper.",
        5: "The student shows mastery. Extend to complex applications, ask them to teach or derive concepts, explore edge cases."
    }
    
    tone_guidance = ""
    if "anxious" in student_style.get("tone", ""):
        tone_guidance = "The student appears anxious. Be encouraging, validate emotions briefly, then proceed with clear steps."
    elif "frustrated" in student_style.get("tone", ""):
        tone_guidance = "The student may be frustrated. Acknowledge effort, simplify the problem, break into smaller pieces."
    elif "confident" in student_style.get("tone", ""):
        tone_guidance = "The student is confident. You can explore deeper concepts and ask challenging questions."
    
    struggle_text = ""
    if struggle_signals:
        struggle_text = f"NOTE: Student signals: {', '.join(struggle_signals[:2])}. Simplify, confirm understanding, use smaller steps."
    
    misconception_text = ""
    if misconceptions:
        misconception_text = f"Key misconception to address: {misconceptions[0]}. Plan diagnostic questions around this."
    
    system_prompt = f"""You are an experienced, caring tutor helping a student learn: {topic}

STUDENT PROFILE & ADAPTATION:
{level_guidance[level]}
{tone_guidance}

LEARNING GOAL THIS TURN:
{next_goal}

APPROACH:
1. Ask ONE focused diagnostic question OR
2. Give a SHORT explanation + 1 concrete example + 1 check-question OR
3. Respond to their answer: validate, clarify misconceptions, move forward

CRITICAL:
- NEVER mention the student's "level" or "score" numerically.
- Match their tone and verbosity slightly (casual/formal/brief/detailed), but stay clear and kind.
- If they struggle, simplify: use shorter sentences, fewer concepts, validate feelings, then proceed.
- Focus on reasoning + correctness, not perfection.
- One diagnostic goal per turn.

{struggle_text}
{misconception_text}

Output JSON:
{{"tutor_message": "Your response here", "next_goal": "What to focus on next"}}
"""
    return system_prompt


async def initial_assessment(conversation_id: str, first_student_response: str, topic: str, grade: int | None = None) -> Dict[str, Any]:
    """
    Run initial assessment after first student response.
    Sets starting level, confidence, and learning goals.
    Uses temperature=0 for stability and includes grade context for calibration.
    """
    if not OPENAI_API_KEY:
        logger.warning(f"Cannot run initial assessment: OPENAI_API_KEY not set")
        return DEFAULT_ASSESSMENT.copy()
    
    messages = [
        {
            "role": "system",
            "content": build_assessor_system_prompt(topic, grade=grade)
        },
        {
            "role": "user",
            "content": f"Topic: {topic}\n\nStudent's first response:\n{first_student_response}\n\nProvide initial assessment based on this response. Remember to calibrate by question difficulty and grade expectation, not just correctness."
        }
    ]
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{OPENAI_BASE_URL}/chat/completions",
                json={
                    "model": OPENAI_ASSESSOR_MODEL,
                    "messages": messages,
                    "temperature": 0,  # Strict, no randomness
                    "top_p": 1,
                    "max_tokens": 600
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
                logger.warning(f"Empty initial assessment response")
                return DEFAULT_ASSESSMENT.copy()
            
            assessment = validate_assessor_json(assistant_message)
            if not assessment:
                logger.warning(f"Initial assessment validation failed, using default")
                return DEFAULT_ASSESSMENT.copy()
            
            assessment["topic"] = topic
            assessment["updated_turn"] = 1
            
            # If grade is unknown, cap confidence conservatively
            if grade is None:
                assessment["confidence"] = min(assessment.get("confidence", 0.5), 0.7)
            
            num_student_msgs = 1
            logger.info(f"Initial assessment for {conversation_id[:8]}: grade={grade}, messages={num_student_msgs}, level={assessment['level']}/5, confidence={assessment['confidence']:.2f}")
            return assessment
            
    except Exception as e:
        logger.warning(f"Initial assessment failed: {str(e)[:100]}")
        return DEFAULT_ASSESSMENT.copy()


async def update_assessment(conversation_id: str):
    """
    Call OpenAI assessor to update student understanding level.
    Uses ONLY student messages (last 12 for longer evidence window).
    Temperature=0 for strict, stable ratings.
    Applies ±1 level clamp unless confidence ≥0.8 AND ≥2 strong quotes.
    """
    if not OPENAI_API_KEY or conversation_id not in CONV_STATE:
        return
    
    state = CONV_STATE[conversation_id]
    topic = state.get("topic_name", "Unknown")
    grade = state.get("grade")
    history = state.get("history", [])
    previous_assessment = state.get("assessment", DEFAULT_ASSESSMENT.copy())
    
    # Extract ONLY recent student messages (last 12 for longer evidence window)
    student_messages = extract_student_messages(history, limit=12)
    
    if not student_messages:
        logger.warning(f"No student messages to assess in {conversation_id[:8]}")
        return
    
    # Build transcript of only student responses
    student_transcript = build_student_transcript(history, limit=12)
    
    messages = [
        {
            "role": "system",
            "content": build_assessor_system_prompt(topic, grade=grade)
        },
        {
            "role": "user",
            "content": (
                f"Topic: {topic}\n\n"
                f"Previous assessment (turn {previous_assessment.get('updated_turn', 0)}):\n"
                f"- Level: {previous_assessment.get('level', 3)}/5\n"
                f"- Confidence: {previous_assessment.get('confidence', 0.5):.2f}\n\n"
                f"Student responses (recent, evaluate ONLY these, ignore tutor):\n"
                f"{student_transcript}\n\n"
                f"Provide updated assessment. Calibrate by difficulty, not just correctness. Limit level change to ±1 unless confidence ≥0.8 AND ≥2 strong quotes from STANDARD/ADVANCED tasks."
            )
        }
    ]
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{OPENAI_BASE_URL}/chat/completions",
                json={
                    "model": OPENAI_ASSESSOR_MODEL,
                    "messages": messages,
                    "temperature": 0,  # STRICT, no randomness
                    "top_p": 1,
                    "max_tokens": 600
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
                logger.warning(f"Empty assessor response for {conversation_id[:8]}")
                return
            
            # Validate and parse JSON
            assessment = validate_assessor_json(assistant_message)
            if not assessment:
                logger.warning(f"Assessment validation failed for {conversation_id[:8]}, keeping current assessment")
                return
            
            # Enforce level change constraint: ±1 unless confidence ≥0.8 AND ≥2 strong quotes
            prev_level = previous_assessment.get("level", 3)
            new_level = assessment.get("level", prev_level)
            new_confidence = assessment.get("confidence", 0.5)
            num_quotes = len(assessment.get("evidence", []))
            
            # Allow ±1 change unless confidence ≥ 0.8 and evidence is strong (≥2 quotes)
            if abs(new_level - prev_level) > 1:
                if new_confidence < 0.8 or num_quotes < 2:
                    # Clamp to ±1
                    assessment["level"] = max(prev_level - 1, min(prev_level + 1, new_level))
                    assessment["confidence"] = min(new_confidence, 0.7)  # Also cap confidence
                    logger.info(f"Clamped level change: {prev_level} → {new_level} → {assessment['level']} (confidence={new_confidence:.2f}, quotes={num_quotes})")
            
            # If grade unknown, cap confidence further
            if grade is None:
                assessment["confidence"] = min(assessment.get("confidence", 0.5), 0.7)
            
            assessment["topic"] = topic
            assessment["updated_turn"] = len(history)
            
            # Store assessment
            CONV_STATE[conversation_id]["assessment"] = assessment
            logger.info(f"Assessment updated for {conversation_id[:8]}: grade={grade}, messages={len(student_messages)}, level={assessment['level']}/5, confidence={assessment['confidence']:.2f}")
            
    except Exception as e:
        logger.warning(f"Assessor call failed: {str(e)[:100]}")
        return



async def finalize_assessment(conversation_id: str):
    """
    Compute final assessment for conversation completion.
    Called when turn_number >= max_turns or is_complete=true.
    Uses FULL student-only transcript (no tutor messages).
    Temperature=0 for strict, stable final ratings.
    Includes previous assessment and applies clamp logic.
    """
    if not OPENAI_API_KEY or conversation_id not in CONV_STATE:
        return
    
    state = CONV_STATE[conversation_id]
    subject = state.get("subject_name", "Subject")
    topic = state.get("topic_name", "Topic")
    grade = state.get("grade")
    history = state.get("history", [])
    current_assessment = state.get("assessment", DEFAULT_ASSESSMENT)
    
    # Build FULL student-only transcript (not full conversation)
    student_transcript = build_student_transcript(history)
    
    # Build final assessment prompt using updated assessor prompt
    system_prompt = build_assessor_system_prompt(topic, grade=grade)
    
    messages = [{"role": "system", "content": system_prompt}]
    
    # Add instruction with previous assessment context
    messages.append({
        "role": "user",
        "content": (
            f"Subject: {subject}\n"
            f"Topic: {topic}\n"
            f"The conversation has completed after {len(history)//2} turns.\n\n"
            f"Previous assessment:\n"
            f"- Level: {current_assessment.get('level', 3)}/5\n"
            f"- Confidence: {current_assessment.get('confidence', 0.5):.2f}\n\n"
            f"Student responses (full transcript, evaluate ONLY these, ignore tutor):\n"
            f"{student_transcript}\n\n"
            f"Provide FINAL comprehensive assessment based on FULL student transcript.\n"
            f"Calibrate by question difficulty and grade expectation.\n"
            f"Update all fields: level (1-5), confidence (0-1), evidence with quotes, misconceptions, strengths, struggle_signals, and next_diagnostic_goal.\n"
            f"Apply clamp logic: limit level change to ±1 unless confidence ≥0.8 AND ≥2 strong quotes from STANDARD/ADVANCED tasks."
        )
    })
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{OPENAI_BASE_URL}/chat/completions",
                json={
                    "model": OPENAI_ASSESSOR_MODEL,
                    "messages": messages,
                    "temperature": 0,  # STRICT for final rating
                    "top_p": 1,
                    "max_tokens": 600
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
                CONV_STATE[conversation_id]["final_assessment"] = current_assessment.copy()
                return
            
            # Parse and validate JSON
            final_assessment = validate_assessor_json(assistant_message)
            if not final_assessment:
                logger.warning(f"Final assessment JSON invalid for {conversation_id}, using current assessment")
                CONV_STATE[conversation_id]["final_assessment"] = current_assessment.copy()
                return
            
            # Apply clamp logic to final assessment
            prev_level = current_assessment.get("level", 3)
            new_level = final_assessment.get("level", prev_level)
            new_confidence = final_assessment.get("confidence", 0.5)
            num_quotes = len(final_assessment.get("evidence", []))
            
            # Clamp to ±1 unless confidence ≥0.8 AND ≥2 strong quotes
            if abs(new_level - prev_level) > 1:
                if new_confidence < 0.8 or num_quotes < 2:
                    final_assessment["level"] = max(prev_level - 1, min(prev_level + 1, new_level))
                    final_assessment["confidence"] = min(new_confidence, 0.7)
                    logger.info(f"Final assessment clamped: {prev_level} → {new_level} → {final_assessment['level']}")
            
            # If grade unknown, cap confidence further
            if grade is None:
                final_assessment["confidence"] = min(final_assessment.get("confidence", 0.5), 0.7)
            
            CONV_STATE[conversation_id]["final_assessment"] = final_assessment
            num_student_msgs = len(extract_student_messages(history))
            logger.info(f"Final assessment for {conversation_id[:8]}: grade={grade}, messages={num_student_msgs}, level={final_assessment['level']}/5, confidence={final_assessment['confidence']:.2f}")
    
    except httpx.HTTPError as e:
        logger.warning(f"Final assessment call failed: {str(e)[:100]}")
        # Fallback to current assessment
        CONV_STATE[conversation_id]["final_assessment"] = current_assessment.copy()



@app.post("/api/conversation/message")
async def send_message(request: SendMessageRequest):
    """
    Proxy endpoint to send a message in an ongoing conversation
    Calls POST /interact with conversation_id and tutor_message
    Stores message history for auto-tutor context
    Triggers initial_assessment() after first student response
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
                state = CONV_STATE[conv_id]
                
                # Add tutor message
                state["history"].append({
                    "role": "tutor",
                    "content": request.tutor_message
                })
                
                # Add student response
                student_response = result.get("student_response", "")
                if student_response:
                    state["history"].append({
                        "role": "student",
                        "content": student_response
                    })
                    
                    # Check if this is first student response (after tutor's first message)
                    if len(state["history"]) == 2:  # 1 tutor + 1 student
                        # Run initial assessment on first student response
                        topic = state.get("topic_name", "Unknown Topic")
                        grade = state.get("grade")
                        state["assessment"] = await initial_assessment(conv_id, student_response, topic, grade=grade)
                    else:
                        # Subsequent responses: update assessment with student-only messages
                        await update_assessment(conv_id)
                
                # Include assessment, tutor message, and history in response
                result["assessment"] = state.get("assessment", DEFAULT_ASSESSMENT)
                result["tutor_message"] = request.tutor_message
                result["history"] = state["history"]
                
                # Check if session is complete and finalize if needed
                turn_num = len(state["history"]) // 2
                max_turns = state.get("max_turns", 10)
                is_complete = result.get("is_complete", False)
                
                if is_complete or turn_num >= max_turns:
                    # Compute and store final assessment
                    await finalize_assessment(conv_id)
                    result["final_assessment"] = state.get("final_assessment")
            
            return result
    except httpx.HTTPError as e:
        logger.error(f"Error sending message: {e}")
        error_detail = getattr(e.response, "text", str(e)) if hasattr(e, "response") else str(e)
        raise HTTPException(status_code=500, detail=f"Failed to send message: {error_detail}")
    except Exception as e:
        logger.error(f"Unexpected error sending message: {e}")
        raise HTTPException(status_code=500, detail="Failed to send message")


@app.post("/api/tutor/next")
async def generate_next_tutor_message(request: TutorNextRequest):
    """
    Generate the next tutor message using OpenAI.
    Uses adapted system prompt based on student level, style, and struggles.
    For first turn (no history), uses placement questions.
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
            "topic_name": request.topic_name,
            "assessment": DEFAULT_ASSESSMENT.copy()
        }
    
    state = CONV_STATE[conv_id]
    assessment = state.get("assessment", DEFAULT_ASSESSMENT.copy())
    topic = request.topic_name
    
    # Determine if first turn
    is_first_turn = len(state["history"]) == 0
    
    # Build system prompt
    system_prompt = build_tutor_system_prompt(topic, assessment, first_turn=is_first_turn)
    
    # Build messages
    messages = [{"role": "system", "content": system_prompt}]
    
    if is_first_turn:
        # First turn: just ask placement questions
        messages.append({
            "role": "user",
            "content": f"Topic: {topic}. Begin with 2-3 quick calibration questions to understand my level."
        })
    else:
        # Subsequent turns: provide context
        messages.append({
            "role": "user",
            "content": (
                f"Subject: {state['subject_name']}\n"
                f"Topic: {topic}\n"
                f"Turn: {len(state['history']) // 2 + 1}/{state['max_turns']}\n"
                f"Current level: {assessment.get('level', 3)}/5\n"
                f"Goal: {assessment.get('next_diagnostic_goal', 'Continue learning')}\n\n"
                "Continue the lesson. Ask 1 focused question, give brief explanation + example if needed, then check understanding."
            )
        })
        
        # Add recent conversation (last ~12 messages)
        recent_history = state["history"][-12:] if len(state["history"]) > 12 else state["history"]
        for msg in recent_history:
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
                    "temperature": 0.6,  # Slightly less random for consistency
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
            
            assistant_message = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            if not assistant_message:
                raise ValueError("No message content from OpenAI")
            
            # Parse JSON
            try:
                json_start = assistant_message.find("{")
                json_end = assistant_message.rfind("}") + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = assistant_message[json_start:json_end]
                else:
                    json_str = assistant_message
                
                result = json.loads(json_str)
                
                if "tutor_message" not in result:
                    result["tutor_message"] = assistant_message
                result.setdefault("next_goal", "Continue learning")
                
                return result
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse tutor JSON: {str(e)[:80]}")
                return {
                    "tutor_message": assistant_message,
                    "next_goal": "Continue learning"
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
