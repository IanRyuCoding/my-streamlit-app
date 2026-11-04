



# 김대리 개인비서 에이블의 Streamlit 화면
import asyncio
import base64
import io
import json
import os
import smtplib
import wave
from datetime import date, datetime, timedelta, timezone
from email.mime.text import MIMEText

import streamlit as st
from google import genai
from google.auth.transport.requests import Request
from google.genai import types
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from openai import OpenAI
from streamlit_calendar import calendar

def load_local_env():
    if not os.path.exists("api_key.txt"):
        return
    with open("api_key.txt", encoding="utf-8") as f:
        lines = f.read().splitlines()
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        index += 1
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if value in ('"""', "'''"):
            quote = value
            chunks = []
            while index < len(lines) and lines[index].strip() != quote:
                chunks.append(lines[index])
                index += 1
            if index < len(lines) and lines[index].strip() == quote:
                index += 1
            value = "\n".join(chunks).strip()
        else:
            value = value.strip('"').strip("'")
        os.environ.setdefault(key, value)


load_local_env()
client = OpenAI()
MODEL = "gpt-6-luna"
SCOPES = ["https://www.googleapis.com/auth/calendar.events"]
KST = timezone(timedelta(hours=9))
MAX_TURNS = 10

#응답 부분 프롬프트

INSTRUCTIONS = """
당신은 '김 대리'의 AI 개인비서 '에이블'입니다.

회의에 참석하지 못한 김 대리가 회의 내용을 이해하고,
자신의 담당 업무와 마감일을 정확하게 파악하도록 돕습니다.

[기본 대화 원칙]
1. 사용자의 질문을 이해하고 자연스러운 한국어로 답합니다.
2. 입력으로 제공된 최근 대화 기록을 참고하여
   이전 질문과 연결되는 후속 질문을 이해합니다.
3. 일상적인 대화는 도구를 호출하지 않고 직접 답합니다.
4. 회의록과 저장된 회의 분석 결과에 없는 내용을
   추측하거나 만들어내지 않습니다.
5. 사용자가 요청하지 않은 도구는 호출하지 않습니다.
6. 도구 실행 결과를 받기 전에는
   작업이 완료됐다고 말하지 않습니다.

[회의 요약 및 후속 질문]
1. 새로운 회의 요약이나 미팅 정리를 요청받으면
   summarize_meeting 도구를 호출합니다.
2. 담당 업무를 처음 확인하는 요청이며
   이전 회의 분석 결과가 제공되지 않았다면
   summarize_meeting 도구를 호출합니다.
3. 이전 대화 또는 저장된 회의 분석 결과로
   답변할 수 있는 후속 질문에는
   불필요하게 summarize_meeting을 다시 호출하지 않습니다.
4. 첫 회의 요약 요청에는 도구가 반환한 summary를
   참고하여 핵심 내용을 간결하게 안내합니다.
5. 김 대리의 업무 전체를 요청하면
   배정된 업무와 마감일을 빠짐없이 설명합니다.
6. 특정 업무를 질문하면
   해당 업무의 내용과 마감일을 중심으로 설명합니다.
7. 상세한 설명이나 원본 근거를 요청하면
   제공된 회의 분석 결과에 근거하여 설명합니다.
8. 확정된 결정 사항과 검토 중인 의견을 구분합니다.
9. 담당자가 변경되거나 업무가 취소됐다면
   최종 확정된 내용을 기준으로 설명합니다.
10. 마감일이 없거나 취소됐다면
    마감일이 정해지지 않았다고 안내합니다.

[이메일 기능]
1. 사용자가 이메일 발송을 명시적으로 요청한 경우에만
   send_email 도구를 호출합니다.
2. 수신자나 발송 내용이 불분명하면 확인을 요청합니다.
3. 사용자가 발송 내용을 확인할 수 있도록 안내합니다.
4. 도구 실행 결과를 확인하기 전에는
   이메일 발송이 완료됐다고 말하지 않습니다.
5. 발송 실패 시 성공했다고 안내하지 않습니다.

[캘린더 기능]
1. 사용자가 일정 등록을 명시적으로 요청한 경우에만
   register_calendar_events 도구를 호출합니다.
2. 사용자가 회의 기준일을 명시한 경우
   meeting_date에 YYYY-MM-DD 형식으로 전달합니다.
3. 사용자가 회의 기준일을 말하지 않았다면
   meeting_date에는 빈 문자열을 전달합니다.
4. 기준일이 없다는 이유만으로
   등록 도구 호출 전에 날짜를 다시 묻지 않습니다.
5. 회의록에 명확한 날짜가 없는 업무에 대해
   임의로 마감일을 확정했다고 말하지 않습니다.
6. 도구가 반환한 등록 결과를 확인하고
   등록된 업무와 등록되지 않은 업무를 구분합니다.
7. 등록에 실패하거나 제외된 업무를
   등록 완료했다고 말하지 않습니다.
8. 일정이 등록되지 않은 이유가 날짜 불명확성이라면
   사용자에게 해당 사실을 안내합니다.

[최종 응답 원칙]
1. 사용자의 질문 의도에 맞춰 답변합니다.
2. 첫 요약은 간결하게 안내합니다.
3. 후속 질문에는 필요한 만큼 자세하게 답변합니다.
4. 김 대리의 담당 업무와 마감일은 정확하게 전달합니다.
5. 업무의 정확성을 답변 길이보다 우선합니다.
6. 도구 실행 결과와 제공된 회의 정보를 근거로 답변합니다.
7. 같은 내용을 불필요하게 반복하지 않습니다.
"""

TOOLS = [
    {
        "type": "function",
        "name": "summarize_meeting",
        "description": "업로드된 회의록을 요약하고 김대리 업무 체크리스트를 만듭니다. 회의 요약, 미팅 정리, 배정 업무 확인을 요청할 때만 호출합니다.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "send_email",
        "description": "저장된 회의 요약을 이메일로 보냅니다. 사용자가 이메일 발송을 명시적으로 요청할 때만 호출합니다.",
        "parameters": {
            "type": "object",
            "properties": {
                "to": {"type": "string", "description": "수신 이메일 주소"},
                "subject": {"type": "string", "description": "이메일 제목"},
            },
            "required": ["to", "subject"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "register_calendar_events",
        "description": "날짜가 확정된 김대리 업무를 구글 캘린더와 앱 캘린더에 함께 등록합니다. 사용자가 캘린더 등록을 명시적으로 요청할 때만 호출합니다.",
        "parameters": {
            "type": "object",
            "properties": {
                "meeting_date": {
                    "type": "string",
                    "description": "사용자가 말한 회의 기준일. YYYY-MM-DD. 말하지 않았으면 빈 문자열.",
                },
            },
            "required": ["meeting_date"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]

st.set_page_config(page_title="에이블", page_icon="🤖", layout="wide")
st.title("김대리의 개인비서 에이블")
st.write("음성이나 텍스트로 요청하면 최근 대화 10개까지 이어서 답합니다.")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "tasks" not in st.session_state:
    st.session_state.tasks = []
if "summary" not in st.session_state:
    st.session_state.summary = ""
if "events" not in st.session_state:
    st.session_state.events = []
if "calendar_tasks" not in st.session_state:
    st.session_state.calendar_tasks = []
if "meeting_text" not in st.session_state:
    st.session_state.meeting_text = ""


def meeting_text():
    if st.session_state.meeting_text.strip():
        return st.session_state.meeting_text
    if os.path.exists("meeting.txt"):
        with open("meeting.txt", encoding="utf-8") as f:
            return f.read()
    return ""


def load_google_credentials():
    info = None
    raw = os.getenv("GOOGLE_TOKEN", "").strip()
    if raw:
        try:
            info = json.loads(raw)
        except json.JSONDecodeError:
            info = None
    if not info and os.path.exists("token.json"):
        with open("token.json", encoding="utf-8") as f:
            info = json.load(f)
    if not info:
        raise RuntimeError("token.json 또는 GOOGLE_TOKEN 비밀값이 없습니다.")
    creds = Credentials.from_authorized_user_info(info, SCOPES)
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
    return creds


def get_calendar_service():
    return build("calendar", "v3", credentials=load_google_credentials())


def send_email(to, subject, body):
    sender = os.getenv("GMAIL_ADDRESS")
    password = os.getenv("GMAIL_APP_PASSWORD")
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = to
    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender, password)
            server.sendmail(sender, to, msg.as_string())
        return {"status": "success"}
    except Exception as e:
        return {"status": "failed", "error": str(e)}


def sync_checklist(events):
    previous = {task["title"]: task for task in st.session_state.tasks}
    tasks = []
    seen = set()
    for event in events:
        title = (event.get("summary") or "").strip()
        if not title or title in seen:
            continue
        seen.add(title)
        current = previous.get(title, {"title": title, "done": False, "event_id": ""})
        current["due_text"] = (event.get("due_text") or "").strip()
        tasks.append(current)
    st.session_state.tasks = tasks


def summarize_meeting():
    text = meeting_text()
    if not text.strip():
        return {"status": "failed", "error": "회의록 txt를 먼저 업로드하세요."}

    prompt = f"""
다음 회의록을 분석하여 JSON만 작성해 주세요.

[회의록]
{meeting_text}

[분석 원칙]
1. 회의록에 명시된 사실만 사용하고 추측하지 않습니다.
2. 회의 핵심 내용을 3~5줄로 요약합니다.
3. 확정된 결정 사항과 논의 중인 의견을 구분합니다.
4. 김 대리에게 최종 배정된 모든 업무를 빠짐없이 추출합니다.
5. 다른 사람에게 배정된 업무는 포함하지 않습니다.
6. 담당자가 변경됐다면 최종 담당자를 기준으로 합니다.
7. 최종 취소된 업무는 제외합니다.
8. 업무는 유지되고 마감일만 취소됐다면
   해당 업무를 유지하고 마감일은 미정으로 처리합니다.
9. 반복 언급된 동일 업무는 하나로 정리합니다.
10. 원본 근거는 회의록 문장을 그대로 인용합니다.
11. 업무의 우선순위가 명시되지 않았다면
    임의로 우선순위를 정하지 않습니다.

[summary 작성 기준]
summary에는 사람이 읽기 좋은 한국어 보고서를 작성합니다.

첫 줄:
회의 주제를 작성하고
"미팅 내용을 요약해 전달드립니다."라고 안내합니다.

[미팅 요약]
- 핵심 내용을 3~5줄로 작성합니다.
- 확정된 사항과 검토 중인 사항을 구분합니다.

[김대리님께 배정된 업무]
김 대리의 모든 업무를 아래 형식으로 작성합니다.

1. 업무 제목
   - 담당자: 김 대리
   - 해야 할 일: 회의록에 나온 업무 내용
   - 마감일: 회의록에 나온 최종 기한
   - 구체적인 요청사항: 회의록에 나온 요청사항
   - 원본 근거: 실제 회의록 문장

업무가 여러 개면 같은 형식을 반복합니다.

김 대리에게 확정된 업무가 없다면
"배정된 업무 없음"이라고 작성합니다.

[마감일 처리]
- 날짜가 명시된 경우 회의록 표현을 유지합니다.
- "다음 주까지", "금요일까지" 등의 표현은
  특정 날짜로 임의 계산하지 않습니다.
- "빨리", "조속히" 등은 원문 표현을 유지하면서
  구체적인 기한은 미정이라고 안내합니다.
- 마감일이 없다면 "마감일 미정"으로 작성합니다.
- 최종적으로 취소된 마감일은 사용하지 않습니다.

[events 작성 기준]
events에는 김 대리에게 최종 배정된 업무만 넣습니다.

기존 JSON 필드명과 자료형을 변경하지 않습니다.

- summary: 업무 제목
- description: 해야 할 일과 구체적인 요청사항
- owner: "김대리"
- due_text: 회의록에 나온 최종 기한 표현
- has_clock_time: 명확한 시각이 있을 때만 true
- start_date: 명확한 YYYY-MM-DD 날짜가 있을 때만 입력
- end_date: 명확한 YYYY-MM-DD 날짜가 있을 때만 입력

[events 날짜 기준]
1. 마감일 표현은 due_text에 원문 그대로 기록합니다.
2. 기한이 없거나 취소됐다면 due_text는 빈 문자열입니다.
3. YYYY-MM-DD 날짜가 명시되지 않았다면
   start_date와 end_date는 빈 문자열입니다.
4. "다음 주", "금요일", "빨리" 등의 표현을
   임의로 특정 날짜로 변환하지 않습니다.
5. 시각이 명시되지 않았다면 has_clock_time은 false입니다.
6. 동일 업무는 events에 중복 추가하지 않습니다.
7. 김 대리에게 확정된 업무가 없다면 events는 빈 배열입니다.

[최종 확인]
1. 회의록 전체를 확인하여 김 대리 업무 누락 여부를 검토합니다.
2. summary와 events에 포함된 업무가 일치하는지 확인합니다.
3. 취소되거나 변경된 업무와 기한이 정확하게 반영됐는지 확인합니다.
4. 담당자, 업무, 마감일, 원본 근거를 만들어내지 않습니다.
5. JSON 외의 설명이나 코드 블록을 출력하지 않습니다.

[회의록]
{text}
"""
    response = client.responses.create(
        model=MODEL,
        instructions="당신은 김대리의 개인 비서 에이블입니다. 회의록에 있는 사실만 사용합니다.",
        input=prompt,
        text={
            "format": {
                "type": "json_schema",
                "name": "meeting_pack",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "summary": {"type": "string"},
                        "events": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "summary": {"type": "string"},
                                    "description": {"type": "string"},
                                    "owner": {"type": "string"},
                                    "due_text": {"type": "string"},
                                    "has_clock_time": {"type": "boolean"},
                                    "start_date": {"type": "string"},
                                    "end_date": {"type": "string"},
                                },
                                "required": [
                                    "summary", "description", "owner", "due_text",
                                    "has_clock_time", "start_date", "end_date",
                                ],
                                "additionalProperties": False,
                            },
                        },
                    },
                    "required": ["summary", "events"],
                    "additionalProperties": False,
                },
            }
        },
    )
    data = json.loads(response.output_text)
    events = [
        event for event in data["events"]
        if event.get("owner", "").replace(" ", "").removesuffix("님") == "김대리"
    ]
    st.session_state.summary = data["summary"]
    st.session_state.events = events
    sync_checklist(events)
    return {"summary": data["summary"]}


def send_meeting_email(to, subject):
    if not to:
        return {"status": "failed", "error": "수신 이메일 주소가 없습니다."}
    if not st.session_state.summary:
        summarized = summarize_meeting()
        if summarized.get("status") == "failed":
            return summarized
    return send_email(to, subject or "회의 요약", st.session_state.summary)


def next_week_friday(meeting_date):
    base = date.fromisoformat(meeting_date)
    days_until_next_monday = (7 - base.weekday()) % 7 or 7
    friday = base + timedelta(days=days_until_next_monday + 4)
    return friday.isoformat(), (friday + timedelta(days=1)).isoformat()


def event_dates(event, meeting_date):
    due_text = (event.get("due_text") or "").strip()
    if "다음 주" in due_text and not event.get("has_clock_time"):
        return next_week_friday(meeting_date)

    start_date = (event.get("start_date") or "").strip()
    end_date = (event.get("end_date") or "").strip()
    try:
        start = date.fromisoformat(start_date)
    except ValueError:
        return None
    try:
        date.fromisoformat(end_date)
    except ValueError:
        end_date = (start + timedelta(days=1)).isoformat()
    return start_date, end_date


def put_app_calendar(title, start_date, event_id=""):
    for event in st.session_state.calendar_tasks:
        if event["title"] == title:
            event["start"] = start_date
            event["allDay"] = True
            if event_id:
                event["event_id"] = event_id
            return
    st.session_state.calendar_tasks.append({
        "title": title,
        "start": start_date,
        "allDay": True,
        "event_id": event_id,
    })


def remember_event_id(title, event_id):
    for task in st.session_state.tasks:
        if task["title"] == title:
            task["event_id"] = event_id


def create_calendar_event(summary, start_date, end_date, description):
    try:
        service = get_calendar_service()
        created = service.events().insert(
            calendarId="primary",
            body={
                "summary": summary,
                "description": description,
                "start": {"date": start_date},
                "end": {"date": end_date},
            },
        ).execute()
        return {
            "status": "success",
            "summary": summary,
            "start_date": start_date,
            "event_id": created["id"],
        }
    except Exception as e:
        return {"status": "failed", "summary": summary, "error": str(e)}


def delete_calendar_event(event_id):
    if not event_id:
        return {"status": "success"}
    try:
        service = get_calendar_service()
        service.events().delete(calendarId="primary", eventId=event_id).execute()
        return {"status": "success"}
    except Exception as e:
        return {"status": "failed", "error": str(e)}


def register_calendar_events(meeting_date=""):
    meeting_date = (meeting_date or "").strip()
    if not meeting_date:
        meeting_date = datetime.now(KST).date().isoformat()
    else:
        try:
            date.fromisoformat(meeting_date)
        except ValueError:
            return {"status": "failed", "error": "기준일은 YYYY-MM-DD 형식이어야 합니다."}
    if not st.session_state.events:
        summarized = summarize_meeting()
        if summarized.get("status") == "failed":
            return summarized

    registered = []
    skipped = []
    for event in st.session_state.events:
        title = event.get("summary", "")
        dates = event_dates(event, meeting_date)
        if not dates:
            skipped.append({"summary": title, "reason": "등록할 날짜가 없습니다."})
            continue
        description = event.get("description", "")
        due_text = (event.get("due_text") or "").strip()
        if due_text:
            description = f"{description}\n기한 원문: {due_text}\n기준일: {meeting_date}"
        put_app_calendar(title, dates[0])
        result = create_calendar_event(title, dates[0], dates[1], description)
        if result["status"] == "success":
            event["event_id"] = result["event_id"]
            put_app_calendar(title, dates[0], result["event_id"])
            remember_event_id(title, result["event_id"])
            registered.append(result)
        else:
            skipped.append({"summary": title, "reason": result["error"]})
    return {"status": "success", "registered": registered, "skipped": skipped, "meeting_date": meeting_date}


def run_tool(name, arguments):
    if isinstance(arguments, str):
        arguments = json.loads(arguments or "{}")
    arguments = arguments or {}
    if name == "summarize_meeting":
        return summarize_meeting()
    if name == "send_email":
        return send_meeting_email(arguments.get("to", ""), arguments.get("subject", "회의 요약"))
    if name == "register_calendar_events":
        return register_calendar_events(arguments.get("meeting_date", ""))
    return {"error": f"unknown tool: {name}"}


def pcm_to_wav(pcm, rate=24000):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(pcm)
    return buf.getvalue()


async def live_speak(text):
    if not os.getenv("GEMINI_API_KEY"):
        raise RuntimeError("GEMINI_API_KEY가 없습니다.")
    gemini = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    spoken = bytearray()
    async with gemini.aio.live.connect(
        model="gemini-3.8-live",
        config={
            "response_modalities": ["AUDIO"],
            "system_instruction": "사용자가 준 문장을 바꾸거나 덧붙이지 말고 그대로 읽습니다.",
            "speech_config": {
                "voice_config": {
                    "prebuilt_voice_config": {"voice_name": "Leda"}
                }
            },
        },
    ) as session:
        await session.send_client_content(
            turns=types.Content(role="user", parts=[types.Part(text=text)]),
            turn_complete=True,
        )
        async for msg in session.receive():
            if msg.data:
                spoken.extend(msg.data)
            server_content = msg.server_content
            if server_content and server_content.turn_complete:
                break
    if not spoken:
        raise RuntimeError("음성 응답이 비어 있습니다.")
    return pcm_to_wav(bytes(spoken))


def ask_assistant(user_text):
    items = [
        {"role": message["role"], "content": message["content"]}
        for message in st.session_state.messages[-(MAX_TURNS * 2):]
    ]
    items.append({"role": "user", "content": user_text})
    response = client.responses.create(
        model=MODEL,
        instructions=INSTRUCTIONS,
        input=items,
        tools=TOOLS,
    )
    for _ in range(3):
        calls = [item for item in response.output if getattr(item, "type", None) == "function_call"]
        if not calls:
            break
        outputs = []
        for call in calls:
            result = run_tool(call.name, call.arguments)
            outputs.append({
                "type": "function_call_output",
                "call_id": call.call_id,
                "output": json.dumps(result, ensure_ascii=False),
            })
        items = [*items, *response.output, *outputs]
        response = client.responses.create(
            model=MODEL,
            instructions=INSTRUCTIONS,
            input=items,
            tools=TOOLS,
        )
    answer = response.output_text or "요청을 처리하지 못했습니다."
    st.session_state.messages.append({"role": "user", "content": user_text})
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.messages = st.session_state.messages[-(MAX_TURNS * 2):]
    return answer


if st.session_state.get("voice_error"):
    st.error(st.session_state.pop("voice_error"))
autoplay_voice = st.session_state.pop("autoplay_voice", False)
last_voice = max(
    (index for index, message in enumerate(st.session_state.messages) if message.get("audio")),
    default=-1,
)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# 실행 클릭으로 오디오를 먼저 깨운 뒤, 답변 음성이 도착하면 버튼 없이 재생한다.
voice_player = st.components.v2.component(
    "able_voice",
    js="""
    export default function(component) {
      if (!window.__ableUnlock) {
        window.__ableUnlock = true;
        window.__ableCtx = new AudioContext();
        document.addEventListener("pointerdown", () => {
          if (window.__ableCtx && window.__ableCtx.state !== "running") {
            window.__ableCtx.resume();
          }
        }, true);
      }
      const data = component.data || {};
      if (!data.wav || window.__ablePlayed === data.token) return;
      window.__ablePlayed = data.token;
      const ctx = window.__ableCtx;
      const binary = atob(data.wav);
      const bytes = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
      ctx.decodeAudioData(bytes.buffer.slice(0)).then((buffer) => {
        const source = ctx.createBufferSource();
        source.buffer = buffer;
        source.connect(ctx.destination);
        source.start(0);
      }).catch(() => {
        ctx.resume().then(() => {
          return ctx.decodeAudioData(bytes.buffer.slice(0));
        }).then((buffer) => {
          const source = ctx.createBufferSource();
          source.buffer = buffer;
          source.connect(ctx.destination);
          source.start(0);
        });
      });
    }
    """,
)
play_wav = ""
play_token = 0
if autoplay_voice and last_voice >= 0:
    spoken = st.session_state.messages[last_voice]
    play_wav = base64.b64encode(spoken["audio"]).decode("ascii")
    play_token = spoken.get("voice_token", 0)
voice_player(data={"wav": play_wav, "token": play_token}, key="able_voice_player", height=0)

with st.form("task_form"):
    request_text = st.text_area("텍스트 요청")
    audio_value = st.audio_input("음성 요청")
    uploaded_file = st.file_uploader("회의록 텍스트", type=["txt"])
    submit = st.form_submit_button("실행")

if st.session_state.meeting_text:
    st.caption("회의록이 저장되어 있습니다. 다음 요청에도 그대로 사용합니다.")
else:
    st.caption("요약을 하려면 회의록 txt를 함께 올려 주세요.")

if submit:
    if uploaded_file is not None:
        st.session_state.meeting_text = uploaded_file.getvalue().decode("utf-8")
    parts = []
    if request_text and request_text.strip():
        parts.append(request_text.strip())
    if audio_value is not None:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=("audio.wav", audio_value, "audio/wav"),
        )
        if transcript.text.strip():
            parts.append(transcript.text.strip())
    if not parts:
        st.warning("음성이나 텍스트로 요청해 주세요.")
    else:
        with st.spinner("답변을 준비하고 음성으로 읽고 있습니다..."):
            try:
                answer = ask_assistant("\n".join(parts))
                try:
                    st.session_state.messages[-1]["audio"] = asyncio.run(live_speak(answer))
                    st.session_state.voice_token = st.session_state.get("voice_token", 0) + 1
                    st.session_state.messages[-1]["voice_token"] = st.session_state.voice_token
                    st.session_state.autoplay_voice = True
                except Exception as e:
                    st.session_state.voice_error = str(e)
                st.rerun()
            except Exception as e:
                st.error(str(e))

if st.session_state.summary:
    st.divider()
    st.subheader("회의 요약")
    st.write(st.session_state.summary)

st.divider()
st.subheader("김대리 업무 체크리스트")
st.write("요약을 요청하면 김대리 업무가 여기에 생깁니다. 완료를 체크하면 연결된 캘린더 일정을 지웁니다.")

completed = 0
for task in st.session_state.tasks:
    label = task["title"]
    if task.get("due_text"):
        label = f"{label} ({task['due_text']})"
    checked = st.checkbox(label, key=f"done_{task['title']}")
    if checked:
        completed += 1
    if checked == task["done"]:
        continue
    task["done"] = checked
    if checked and task.get("event_id"):
        delete_calendar_event(task["event_id"])
        event_id = task["event_id"]
        task["event_id"] = ""
        st.session_state.calendar_tasks = [
            event for event in st.session_state.calendar_tasks
            if event.get("event_id") != event_id and event.get("title") != task["title"]
        ]
    st.rerun()

total = len(st.session_state.tasks)
st.progress(completed / total if total else 0)
st.write(f"완료한 업무: {completed}/{total}개")

st.divider()
st.subheader("업무 캘린더")
with st.form("calendar_form"):
    titles = [task["title"] for task in st.session_state.tasks]
    task_name = st.selectbox("앱 캘린더에 표시할 업무", options=titles) if titles else None
    deadline = st.date_input("마감일", value=date.today())
    submitted = st.form_submit_button("앱 캘린더에 등록/날짜 수정")
    if submitted:
        if task_name:
            put_app_calendar(task_name, deadline.isoformat())
            st.success("앱 캘린더에 반영했습니다.")
        else:
            st.warning("먼저 요약을 요청해 업무를 만들어 주세요.")

calendar(
    events=st.session_state.calendar_tasks,
    options={
        "initialView": "dayGridMonth",
        "locale": "ko",
        "height": 550,
    },
    key="work_calendar",
)
