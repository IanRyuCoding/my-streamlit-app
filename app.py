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



INSTRUCTIONS = """

당신은 김대리의 개인 비서 에이블입니다.

입력에 들어 있는 최근 대화를 이어서 이해합니다.

일상적인 대화는 도구 없이 직접 답합니다.

회의 요약, 미팅 정리, 배정 업무 확인 요청일 때만 summarize_meeting을 호출합니다.

이메일로 보내 달라는 말이 있을 때만 send_email을 호출합니다.

캘린더에 등록해 달라는 말이 있으면 register_calendar_events를 호출합니다.

사용자가 기준일을 말하면 meeting_date에 YYYY-MM-DD로 넣고, 말하지 않으면 빈 문자열을 넘깁니다.

기준일이 없어도 등록 도구를 호출하고, 날짜를 다시 물어보지 않습니다.

사용자가 요청하지 않은 도구는 호출하지 않습니다.

도구 결과가 오기 전에는 요약, 발송 완료, 등록 완료를 말하지 않습니다.

요약 도구가 돌려준 summary만 짧게 말합니다.

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





# [신규 추가 1] 이메일 편집창 상태는 Streamlit 세션에 보관합니다.
if "email_editor" not in st.session_state:
    st.session_state.email_editor = ""
if "email_draft_task" not in st.session_state:
    st.session_state.email_draft_task = ""


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

다음 회의록을 읽고 JSON만 작성해 주세요.



summary에는 아래 형식의 한국어 요약만 넣습니다.

- 회의록에 명시된 사실만 사용하고, 추측하거나 세부사항을 만들지 마세요.
- 제안이나 의견을 확정된 결정으로 표현하지 마세요.
- 담당자나 기한이 없으면 

"명시되지 않음"이라고 적으세요.

- 
"빨리", "조속히"처럼 날짜가 아닌 표현은 그대로 적고, 구체적인 기한은 미정이라고 덧붙이세요.

- 김대리의 업무와 다른 사람의 업무를 섞지 마세요.



summary 형식:

첫 줄은 회의 주제를 쓰고 "미팅 내용을 요약해 전달드립니다."라고 이어서 작성합니다.

[미팅 요약]

- 주제별로 한 줄씩 작성합니다.

[김대리님께 배정된 업무]

1. 업무 제목

  - 해야 할 일을 회의록 표현 그대로 적습니다.

  - 관련 의견을 정리합니다.

  - 합의된 기한 표현을 적고, 없으면 "별도의 완료 기한은 회의에서 정해지지 않았습니다."라고 적습니다.

김대리 업무가 없으면 [김대리님께 배정된 업무] 아래에 "배정된 업무 없음"이라고 적습니다.



events에는 김대리에게 배정된 업무만 넣습니다. 다른 사람 업무는 넣지 않습니다.

- summary는 업무 제목입니다.
- description은 해야 할 일입니다.
- owner는 김대리로만 적습니다.
- due_text는 회의록의 기한 표현 그대로 적고, 없으면 빈 문자열입니다.
- 회의록에 YYYY-MM-DD 날짜가 있을 때만 start_date와 end_date를 채웁니다.
- 날짜가 없으면 start_date와 end_date는 빈 문자열입니다.
- 
"다음 주", "빨리"를 날짜로 바꾸지 마세요.

- 시각이 없으면 has_clock_time은 false입니다.



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





# ================================================================
# [신규 추가 2] 지출/영수증 기록 기능 (원본의 record_expense를 Streamlit에 맞게 연결)
# ================================================================
EXPENSE_FILE = "expenses.json"  # Colab의 path 대신 Streamlit 실행 위치 사용


def read_expenses():
    """기록된 지출을 읽어 화면에 표시합니다."""
    if not os.path.exists(EXPENSE_FILE):
        return []
    with open(EXPENSE_FILE, "r", encoding="utf-8") as f:
        records = json.load(f)
    if not isinstance(records, list):
        raise ValueError("expenses.json 데이터가 목록 형식이 아닙니다.")
    return records


def record_expense(amount: int, category: str, description: str):
    """텍스트/음성 도구 호출 또는 Streamlit 입력창에서 지출을 기록합니다."""
    try:
        amount = int(amount)
        category = str(category).strip() or "기타"
        description = str(description).strip()
        if amount <= 0:
            return {"status": "failed", "error": "지출 금액은 1원 이상이어야 합니다."}
        if not description:
            return {"status": "failed", "error": "지출 내용을 입력해 주세요."}

        now = datetime.now(KST)
        new_record = {
            "date": now.strftime("%Y-%m-%d %H:%M:%S"),
            "amount": amount,
            "category": category,
            "description": description,
        }
        expenses = read_expenses()
        expenses.append(new_record)
        with open(EXPENSE_FILE, "w", encoding="utf-8") as f:
            json.dump(expenses, f, ensure_ascii=False, indent=2)
        return {"status": "success", "recorded_data": new_record}
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as e:
        return {"status": "failed", "error": str(e)}


# ================================================================
# [신규 추가 3] 이메일 초안 작성/어조 변환 기능
# 전역변수 대신 st.session_state에 보관하여 수정한 초안이 유지됩니다.
# ================================================================
def draft_business_email(task_description: str):
    """선택된 김대리 업무를 바탕으로 비즈니스 이메일 초안을 만듭니다."""
    task_description = str(task_description).strip()
    if not task_description:
        return {"status": "failed", "error": "이메일을 작성할 업무를 선택해 주세요."}

    selected_task = next(
        (task for task in st.session_state.tasks if task["title"] == task_description),
        None,
    )
    # 체크리스트의 기한 및 업무명을 우선 사용하되, 음성 요청 문장도 받을 수 있게 합니다.
    task_info = task_description
    if selected_task:
        task_info = f"업무: {selected_task['title']}\n회의록의 기한 표현: {selected_task.get('due_text') or '명시되지 않음'}"

    prompt = f"""
[회의 요약]
{st.session_state.summary or '회의 요약 없음'}

[사용자 요청 업무]
{task_info}

위 정보에 근거해 업무 수행에 필요한 한국어 비즈니스 이메일 초안을 작성하세요.
- 회의록에 없는 사실, 확정되지 않은 결정 및 기한은 꾸며내지 마세요.
- 수신인, 제목, 본문을 포함해 주세요.
- 이름, 회사명, 날짜 등 미제공 정보는 [수신인], [이름]처럼 표시하세요.
- 발송하지 말고 초안만 작성해 주세요.
"""
    try:
        response = client.responses.create(
            model=MODEL,
            instructions=(
                "당신은 전문 비즈니스 이메일 작성가입니다. "
                "입력된 업무와 회의 요약에 근거한 명확한 이메일 초안만 작성합니다."
            ),
            input=prompt,
        )
        draft = (response.output_text or "").strip()
        if not draft:
            return {"status": "failed", "error": "이메일 초안을 생성하지 못했습니다."}
        st.session_state.email_editor = draft
        st.session_state.email_draft_task = (
            selected_task["title"] if selected_task else task_description
        )
        return {"status": "success", "draft": draft}
    except Exception as e:
        return {"status": "failed", "error": str(e)}


def convert_business_tone(text_to_convert: str = ""):
    """사용자가 편집한 내용을 기준으로 정중한 비즈니스 이메일 어조로 변환합니다."""
    target_text = (text_to_convert or st.session_state.get("email_editor", "")).strip()
    if not target_text:
        return {"status": "failed", "error": "먼저 이메일 초안을 작성하거나 입력해 주세요."}

    try:
        response = client.responses.create(
            model=MODEL,
            instructions=(
                "당신은 비즈니스 커뮤니케이션 전문가입니다. "
                "내용의 의미, 사실관계, 날짜, 수신인, 요청 사항 및 대괄호 자리표시자는 보존하세요. "
                "말투만 격식 있고 정중한 비즈니스 이메일 스타일로 다듬으세요."
            ),
            input=f"다음 이메일을 비즈니스 매너에 맞게 다듬어 주세요.\n\n{target_text}",
        )
        converted = (response.output_text or "").strip()
        if not converted:
            return {"status": "failed", "error": "어조를 변환하지 못했습니다."}
        st.session_state.email_editor = converted
        return {"status": "success", "converted_text": converted}
    except Exception as e:
        return {"status": "failed", "error": str(e)}


# [신규 추가 4] 기존 OpenAI 도구 목록을 건드리지 않고 새 도구 2개만 등록합니다.
TOOLS.extend([
    {
        "type": "function",
        "name": "record_expense",
        "description": "사용자가 지출 내역 기록을 요청할 때만 호출합니다.",
        "parameters": {
            "type": "object",
            "properties": {
                "amount": {"type": "integer", "description": "지출 금액(원)"},
                "category": {"type": "string", "description": "식비, 교통비 등 지출 분류"},
                "description": {"type": "string", "description": "지출 내용"},
            },
            "required": ["amount", "category", "description"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "draft_business_email",
        "description": "사용자가 특정 업무에 대한 이메일 초안 작성을 요청할 때 호출합니다. 발송하지 않습니다.",
        "parameters": {
            "type": "object",
            "properties": {
                "task_description": {"type": "string", "description": "작성할 업무명 또는 목적"},
            },
            "required": ["task_description"],
            "additionalProperties": False,
        },
        "strict": True,
    },
])

# [신규 추가 5] 기존 기본 지시문은 수정하지 않고 필요한 작동 규칙을 덧붙입니다.
INSTRUCTIONS += """
사용자가 금액, 분류, 지출 내용 기록을 요청하면 record_expense를 호출합니다.
사용자가 이메일 초안 작성을 요청하면 draft_business_email을 호출합니다.
초안은 이메일 편집창에 표시되고 발송되지 않습니다.
이메일 어조 변환은 사용자가 화면의 '비즈니스 매너에 맞게 어조 변환' 버튼을 누르면 실행됩니다.
새로 추가된 도구의 결과를 확인하기 전에는 저장 또는 초안 작성 완료를 알리지 않습니다.
"""

# [신규 추가 6] 원본 run_tool은 그대로 두고 새 기능을 처리하는 래퍼로 확장합니다.
_original_run_tool = run_tool


def run_tool(name, arguments):
    if name in ("record_expense", "draft_business_email"):
        try:
            args = json.loads(arguments or "{}") if isinstance(arguments, str) else (arguments or {})
        except json.JSONDecodeError:
            return {"status": "failed", "error": "도구 인자를 분석할 수 없습니다."}
        if name == "record_expense":
            return record_expense(
                args.get("amount", 0),
                args.get("category", "기타"),
                args.get("description", ""),
            )
        return draft_business_email(args.get("task_description", ""))
    return _original_run_tool(name, arguments)


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



# ================================================================
# [신규 추가 7] 체크리스트 바로 아래: 업무 선택 → 이메일 생성 → 직접 수정 → 어조 변환
# 화면 위젯을 그리기 전에 session_state를 갱신하여 수정 결과가 즉시 반영됩니다.
# ================================================================
st.divider()
st.subheader("비즈니스 이메일 초안 작성 및 어조 변환")

email_task_titles = [task["title"] for task in st.session_state.tasks]
if email_task_titles:
    selected_email_task = st.selectbox(
        "이메일 초안을 작성할 김대리 업무 선택",
        options=email_task_titles,
        key="selected_email_task",
    )
    if st.button("선택한 업무로 이메일 초안 작성", key="create_business_email"):
        with st.spinner("선택한 업무의 이메일 초안을 작성하고 있습니다..."):
            result = draft_business_email(selected_email_task)
        if result["status"] == "success":
            st.success("이메일 초안을 작성했습니다. 아래에서 직접 수정할 수 있습니다.")
        else:
            st.error(result["error"])
else:
    st.info("먼저 회의록을 요약해 김대리 업무 체크리스트를 생성해 주세요.")

if st.button(
    "비즈니스 매너에 맞게 어조 변환",
    key="convert_business_email_tone",
    disabled=not bool(st.session_state.get("email_editor", "").strip()),
):
    with st.spinner("수정한 이메일의 어조를 다듬고 있습니다..."):
        result = convert_business_tone(st.session_state.email_editor)
    if result["status"] == "success":
        st.success("어조 변환을 완료했습니다. 아래 편집창에 결과가 반영되었습니다.")
    else:
        st.error(result["error"])

st.text_area(
    "이메일 초안 (내용을 직접 수정할 수 있습니다)",
    key="email_editor",
    height=290,
    placeholder="업무를 선택해 초안을 생성하거나 여기에 직접 입력해 주세요.",
)
if st.session_state.email_draft_task:
    st.caption(f"현재 초안 작성 기준 업무: {st.session_state.email_draft_task}")
st.caption("이 초안은 자동 발송되지 않습니다.")


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


# ================================================================
# [신규 추가 8] 업무 캘린더 바로 아래: 지출 기록 및 최근 내역 확인
# ================================================================
st.divider()
st.subheader("지출 기록")
with st.form("expense_record_form", clear_on_submit=True):
    expense_amount = st.number_input("지출 금액 (원)", min_value=1, value=1000, step=100)
    expense_category = st.selectbox(
        "지출 분류", options=["식비", "교통비", "카페", "쇼핑", "업무", "기타"]
    )
    expense_description = st.text_input("지출 내용", placeholder="예: 팀 점심 식사")
    expense_submitted = st.form_submit_button("지출 기록 저장")

if expense_submitted:
    saved = record_expense(expense_amount, expense_category, expense_description)
    if saved["status"] == "success":
        st.success(
            f"{saved['recorded_data']['category']} 지출 "
            f"{saved['recorded_data']['amount']:,}원을 저장했습니다."
        )
    else:
        st.error(saved["error"])

try:
    expense_history = read_expenses()
    if expense_history:
        st.write(f"누적 지출: {sum(item['amount'] for item in expense_history):,}원")
        st.write("최근 지출 내역")
        st.dataframe(
            [
                {
                    "날짜": item["date"],
                    "금액 (원)": item["amount"],
                    "분류": item["category"],
                    "내용": item["description"],
                }
                for item in reversed(expense_history[-10:])
            ],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.caption("아직 기록된 지출 내역이 없습니다.")
except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as e:
    st.error(f"지출 내역을 읽는 중 문제가 발생했습니다: {e}")
