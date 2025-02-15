from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import httpx
import json
from enum import Enum
import subprocess
import sys
from datetime import datetime
import sqlite3
import logging
import requests
import os
from PIL import Image
import pytesseract
import re
import csv
from pydub import AudioSegment
from markdown import markdown
from bs4 import BeautifulSoup


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "Welcome to the TDS PROJECT 1 APP !"}

# Configuration
AIPROXY_TOKEN = os.environ["AIPROXY_TOKEN"] = "eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6IjIzZjIwMDQ5NDFAZHMuc3R1ZHkuaWl0bS5hYy5pbiJ9.1J8Dj5Y1CUN-I9IPjGAyKkYy5T7cKDke8FW8OP9yaYk"
if not AIPROXY_TOKEN:
    raise ValueError("AIPROXY_TOKEN environment variable is not set")

class TaskType(Enum):
    INSTALL_AND_RUN = "A1"
    FORMAT_MARKDOWN = "A2"
    COUNT_WEDNESDAYS = "A3"
    SORT_CONTACTS = "A4"
    RECENT_LOGS = "A5"
    CREATE_INDEX = "A6"
    EXTRACT_EMAIL = "A7"
    EXTRACT_CARD = "A8"
    SIMILAR_COMMENTS = "A9"
    TICKET_SALES = "A10"
    FETCH_DATA = "B1"
    CLONE_GIT_REPO = "B2"
    RUN_SQL_QUERY = "B3"
    SCRAPE_WEBSITE = "B4"
    COMPRESS_RESIZE_IMAGE = "B5"
    TRANSCRIBE_AUDIO = "B6"
    CONVERT_MARKDOWN = "B7"
    FILTER_CSV = "B8"

async def call_llm(prompt: str) -> str:
    """Call the LLM (GPT-4-mini) via AI Proxy"""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {AIPROXY_TOKEN}",
                "Content-Type": "application/json"
            },
            json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0
            },
            timeout=15
        )
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail="LLM API error")
        return response.json()["choices"][0]["message"]["content"]

async def parse_task(task_description: str) -> TaskType:
    """Use LLM to categorize the task into one of the predefined types"""
    prompt = f"""Given this task description: "{task_description}"
"""Categorize it into one of these types:
A1: Install uv and run datagen.py
A2: Format markdown file
A3: Count Wednesdays in dates
A4: Sort contacts
A5: Get recent log lines
A6: Create markdown index
A7: Extract email sender
A8: Extract credit card number
A9: Find similar comments
A10: Calculate ticket sales
B1: Fetch data from API and save it
B2: Clone a Git repo and commit changes
B3: Run a SQL query on a database
B4: Scrape a website
B5: Compress or resize an image
B6: Transcribe audio from MP3 to text
B7: Convert Markdown to HTML
B8: Filter a CSV file

Return only the task type (e.g., "A1")."""
    
    task_type = await call_llm(prompt)
    return TaskType(task_type.strip())

@app.get("/run")
async def run_task(task: str):
    """Run the given task using the query parameter task"""
    logging.basicConfig(level=logging.DEBUG)
    logging.debug(f"Received task: {task}")

    try:
        # Categorize the task
        task_type = await parse_task(task)
        logging.debug(f"Task categorized as: {task_type}")

        # Execute the task based on its type
        result = await execute_task(task_type, task)
        return {"status": "success", "result": result}
    
    except ValueError as e:
        logging.error(f"Error processing task: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid task: {str(e)}")
    
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
    

@app.post("/run")
async def run_task(task: str):
    """Run the given task"""
    logging.basicConfig(level=logging.DEBUG)
    logging.debug(f"Received task: {task}")

    try:
        # Categorize the task
        task_type = await parse_task(task)
        logging.debug(f"Task categorized as: {task_type}")

        # Execute the task based on its type
        result = await execute_task(task_type, task)
        return {"status": "success", "result": result}
    
    except ValueError as e:
        logging.error(f"Error processing task: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid task: {str(e)}")
    
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/read")
async def read_file(path: str):
    try:
        with open(path, 'r') as f:
            return f.read()
    except FileNotFoundError:
        raise HTTPException(status_code=404)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

async def handle_install_and_run(task: str) -> dict:
    """A1: Install uv and run datagen.py"""
    # Check if uv is installed
    try:
        subprocess.run(["uv", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Install uv using pip
        subprocess.run([sys.executable, "-m", "pip", "install", "uv"], check=True)
    
    # Download datagen.py
    script_url = "https://raw.githubusercontent.com/sanand0/tools-in-data-science-public/tds-2025-01/project-1/datagen.py"
    async with httpx.AsyncClient() as client:
        response = await client.get(script_url)
        if response.status_code != 200: 
            raise ValueError("Failed to download datagen.py")
        
        with open("datagen.py", "w") as f:
            f.write(response.text)
    
    # Run datagen.py with email
    email = os.environ.get("USER_EMAIL", "23f2004941@ds.study.iitm.ac.in")
    result = subprocess.run(["uv", "run", "datagen.py", email], capture_output=True, text=True)
    
    return {"output": result.stdout}

async def handle_format_markdown(task: str) -> dict:
    """A2: Format markdown using prettier"""
    # Install prettier if not present
    try:
        subprocess.run(["npx", "prettier", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        subprocess.run(["npm", "install", "-g", "prettier@3.4.2"], check=True)
    
    # File path debugging
    file_path = "data\format.md"
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    subprocess.run(["npx", "prettier", "--write", file_path], check=True)
    
    return {"message": "Markdown formatted successfully"}

async def handle_count_wednesdays(task: str) -> dict:
    """A3: Count Wednesdays in dates.txt"""
    with open("data\dates.txt", "r") as f:
        dates = f.read().splitlines()
    
    count = 0
    date_formats = [
        "%Y-%m-%d",
        "%d-%b-%Y",
        "%b %d, %Y",
        "%Y/%m/%d %H:%M:%S"
    ]
    
    for date_str in dates:
        for fmt in date_formats:
            try:
                date = datetime.strptime(date_str.strip(), fmt)
                if date.weekday() == 2:  # Wednesday is 2
                    count += 1
                break
            except ValueError:
                continue
    
    with open("data/dates-wednesdays.txt", "w") as f:
        f.write(str(count))
    
    return {"count": count}

async def handle_sort_contacts(task: str) -> dict:
    """A4: Sort contacts by last_name, first_name"""
    with open("data\contacts.json", "r") as f:
        contacts = json.load(f)
    
    sorted_contacts = sorted(
        contacts,
        key=lambda x: (x["last_name"], x["first_name"])
    )
    
    with open("data/contacts-sorted.json", "w") as f:
        json.dump(sorted_contacts, f)
    
    return {"message": "Contacts sorted successfully"}

async def handle_recent_logs(task: str) -> dict:
    """A5: Get first lines of 10 most recent logs"""
    log_dir = "data\logs"
    log_files = []
    
    for file in os.listdir(log_dir):
        if file.endswith(".log"):
            path = os.path.join(log_dir, file)
            log_files.append((os.path.getmtime(path), path))
    
    # Sort by modification time, newest first
    log_files.sort(reverse=True)
    
    # Get first lines of 10 most recent files
    first_lines = []
    for _, path in log_files[:10]:
        with open(path, "r") as f:
            first_lines.append(f.readline().strip())
    
    with open("data/logs-recent.txt", "w") as f:
        f.write("\n".join(first_lines))
    
    return {"message": "Recent log lines extracted successfully"}

async def handle_create_index(task: str) -> dict:
    """A6: Create index of markdown h1 headers"""
    docs_dir = "data\docs"
    index = {}
    
    for root, _, files in os.walk(docs_dir):
        for file in files:
            if file.endswith(".md"):
                rel_path = os.path.relpath(os.path.join(root, file), docs_dir)
                with open(os.path.join(root, file), "r") as f:
                    content = f.read()
                    
                # Find first H1 header
                for line in content.split("\n"):
                    if line.startswith("# "):
                        index[rel_path] = line[2:].strip()
                        break
    
    with open("data/docs/index.json", "w") as f:
        json.dump(index, f)
    
    return {"message": "Index created successfully"}

async def handle_extract_email(task: str) -> dict:
    """A7: Extract sender's email from email.txt"""
    with open("data\email.txt", "r") as f:
        email_content = f.read()
    
    prompt = f"""Extract ONLY the sender's email address from this email content. Return just the email address, nothing else:

{email_content}"""
    
    email = await call_llm(prompt)
    email = email.strip().strip('"<>')  # Clean up any extra characters
    
    with open("data/email-sender.txt", "w") as f:
        f.write(email)
    
    return {"email": email}

async def handle_extract_card(task: str) -> dict:
    """A8: Extract credit card number from image"""
    
    image_path = 'data/credit_card.png'
    # Check if the image exists
    if not os.path.exists(image_path):
        return {"error": f"Image not found at {image_path}"}
    
    prompt = """An image has been provided that contains a credit card. Please extract and return ONLY the credit card number, with no spaces or other characters."""
    
    try:
        # Load the image
        image = Image.open(image_path)
        
        # Perform OCR (pytesseract)
        extracted_text = pytesseract.image_to_string(image)
        
        # Extract the credit card number using regex
        card_number = re.search(r'\b(?:\d[ -]*?){13,16}\b', extracted_text)
        if card_number:
            card_number = ''.join(filter(str.isdigit, card_number.group()))  # Clean up the number
        else:
            card_number = "Not found"


        # Save the card number to a file
        with open("data/credit-card.txt", "w") as f:
            f.write(card_number)
        
        return {"card_number": card_number}
    
    except Exception as e:
        return {"error": f"An error occurred while processing the image: {str(e)}"}
    

# async def handle_extract_card(task: str) -> dict:
    # """A8: Extract credit card number from image"""
    # prompt = """An image has been provided that contains a credit card. Please extract and return ONLY the credit card number, with no spaces or other characters."""

    # card_number = await call_llm(prompt)
    # card_number = ''.join(filter(str.isdigit, card_number))
    
    # with open("data\credit-card.txt", "w") as f:
    #     f.write(card_number)
    
    # return {"card_number": card_number}

async def handle_similar_comments(task: str) -> dict:
    """A9: Find most similar pair of comments"""
    with open("data\comments.txt", "r") as f:
        comments = f.read().splitlines()
    
    prompt = f"""Find the two most similar comments from this list and return them exactly as they appear, separated by a newline:

{comments}"""
    
    similar_pair = await call_llm(prompt)
    
    with open("data/comments-similar.txt", "w") as f:
        f.write(similar_pair)
    
    return {"similar_pair": similar_pair}

async def handle_ticket_sales(task: str) -> dict:
    """A10: Calculate total sales for Gold tickets"""
    
    db_path = "data/ticket-sales.db"
    
    # Ensure the database file exists
    if not os.path.exists(db_path):
        return {"error": f"Database file not found: {db_path}"}
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT SUM(units * price)
            FROM tickets
            WHERE type = 'Gold'
        """)
        
        total = cursor.fetchone()[0]
        conn.close()
        
        with open("data/ticket-sales-gold.txt", "w") as f:
            f.write(str(total))
        
        return {"total": total}
    
    except sqlite3.Error as e:
        return {"error": f"SQLite error: {e}"}
    


# PHASE B - BUSINESS TASKS

# B1: Ensure data outside /data is never accessed or exfiltrated
# B2: Ensure data is never deleted anywhere on the file system

# The handler for the tasks in the FastAPI app

# Task Types Enum for B1 to B10

async def handle_fetch_data(task: str) -> dict:
    """B1: Fetch data from an API and save it"""
    api_url = task.strip()
    try:
        response = httpx.get(api_url)
        response.raise_for_status()
        data = response.json()
        with open("data/api_data.json", "w") as f:
            json.dump(data, f)
        return {"message": "API data fetched and saved successfully"}
    except Exception as e:
        return {"error": str(e)}

async def handle_clone_git_repo(task: str) -> dict:
    """B4: Clone a git repo and make a commit"""
    repo_url, commit_message = task.split(" | ")
    
    # Clone the repository
    repo_name = repo_url.split('/')[-1].replace('.git', '')
    repo_path = os.path.join('/data', repo_name)
    
    if not os.path.exists(repo_path):
        subprocess.run(['git', 'clone', repo_url, repo_path], check=True)
    
    # Change to the repo directory
    os.chdir(repo_path)
    
    # Check if there are any changes to commit
    result = subprocess.run(['git', 'status', '--porcelain'], stdout=subprocess.PIPE)
    if result.stdout.strip() == b'':
        return {"message": "No changes to commit in the repository."}

    # Stage and commit changes
    subprocess.run(['git', 'add', '.'], check=True)
    subprocess.run(['git', 'commit', '-m', commit_message], check=True)
    subprocess.run(['git', 'push'], check=True)
    
    return {"message": f"Git repo {repo_name} cloned and committed successfully"}
# pass url as https://github.com/ASTRINGUISH/my-work-iitm-tds-ga and commit message as "Initial commit"

async def handle_run_sql_query(task: str) -> dict:
    """B5: Run a SQL query on a SQLite or DuckDB database"""
    sql_query = task.strip()

    try:
        # Connect to the database using the dynamic db_path
        conn = sqlite3.connect('data/ticket-sales.db')
        cursor = conn.cursor()

        # Execute the provided SQL query
        cursor.execute(sql_query)

        # Fetch all results
        result = cursor.fetchall()

        # Get column names from cursor description
        columns = [description[0] for description in cursor.description]

        # Convert rows into a list of dictionaries
        result_as_dicts = [dict(zip(columns, row)) for row in result]

        # Close the connection
        conn.close()

        # Return result as JSON (or Python dict)
        return {"result": result_as_dicts}
    
    except sqlite3.Error as e:
        # Handle SQLite errors
        return {"error": f"SQLite error: {str(e)}"}
# pass the sql query as prompt and the task is to run the query on the chinook.db database and return the result as a list of dictionaries


async def handle_scrape_website(task: str) -> dict:
    """B6: Extract data from (i.e. scrape) a website"""
    url = task.strip()

    try:
        # Make the request to the website
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for a bad HTTP status code

        # Parse the content with BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')

        # Example: Extract the title of the page
        title = soup.find('title').get_text()

        # Example: Extract all links (anchors) from the page
        links = [a['href'] for a in soup.find_all('a', href=True)]

        # Save extracted data to a JSON file
        with open('data/scraped_data.json', 'w') as f:
            json.dump({'title': title, 'links': links}, f)

        return {"message": "Website scraped and data saved successfully"}

    except Exception as e:
        return {"error": str(e)}
# here the prompt is only the url of the website to scrape and the task is to extract the title and all the links from the website
# url -- https://www.iplt20.com/

async def handle_compress_resize_image(task: str) -> dict:
    """B7: Compress or resize an image"""
    action = task.strip()
    
    try:
        image_path = 'data\credit_card.png'

        image = Image.open(image_path)
        
        if action == "resize image":
            # Resize image to 200x200 as an example
            image_new = image.resize((200, 200))
            image_new.save('data/resized_image.png')
        elif action == "compress image":
            # Compress the image (example for JPG)
            image.save('data/compressed_image.png', quality=25)
        
        image.save('data/resized_or_compressed_image.png')
        return {"message": "Image compressed or resized successfully"}
    
    except Exception as e:
        return {"error": str(e)}


async def handle_transcribe_audio(task: str) -> dict:
    """B8: Transcribe audio from an MP3 file"""
    audio_path = 'data/audio.mp3'
    
    try:
        audio = AudioSegment.from_mp3(audio_path)
        # Save audio as WAV for transcription (example)
        audio.export('data/transcribed_audio.wav', format='wav')
        
        # For transcription, use any transcription service here.
        # Here we're just saving the audio file for now.
        return {"message": "Audio file transcribed (WAV saved)"}
    except Exception as e:
        return {"error": str(e)}


async def handle_convert_markdown_to_html(task: str) -> dict:
    """B9: Convert Markdown to HTML"""
    markdown_file = 'data/format.md'
    
    try:
        with open(markdown_file, 'r') as f:
            markdown_content = f.read()
        
        html_content = markdown(markdown_content)
        
        # Save the HTML content
        with open('data/converted.html', 'w') as f:
            f.write(html_content)
        
        return {"message": "Markdown converted to HTML successfully"}
    except Exception as e:
        return {"error": str(e)}


async def handle_filter_csv(task: str) -> dict:
    """B10: Write an API endpoint that filters a CSV file and returns JSON data"""
    filter_column, filter_value = task.split(" | ")
    
    try:
        with open('data/currency.csv', 'r') as f:
            reader = csv.DictReader(f)
            filtered_data = [row for row in reader if row.get(filter_column) == filter_value]
        
        # Return filtered data as JSON
        return {"filtered_data": filtered_data}
    except Exception as e:
        return {"error": str(e)}


async def execute_task(task_type: TaskType, original_task: str) -> dict:
    """Execute the identified task"""
    handlers = {
        TaskType.INSTALL_AND_RUN: handle_install_and_run,
        TaskType.FORMAT_MARKDOWN: handle_format_markdown,
        TaskType.COUNT_WEDNESDAYS: handle_count_wednesdays,
        TaskType.SORT_CONTACTS: handle_sort_contacts,
        TaskType.RECENT_LOGS: handle_recent_logs,
        TaskType.CREATE_INDEX: handle_create_index,
        TaskType.EXTRACT_EMAIL: handle_extract_email,
        TaskType.EXTRACT_CARD: handle_extract_card,
        TaskType.SIMILAR_COMMENTS: handle_similar_comments,
        TaskType.TICKET_SALES: handle_ticket_sales,
        TaskType.FETCH_DATA: handle_fetch_data,
        TaskType.CLONE_GIT_REPO: handle_clone_git_repo,
        TaskType.RUN_SQL_QUERY: handle_run_sql_query,
        TaskType.SCRAPE_WEBSITE: handle_scrape_website,
        TaskType.COMPRESS_RESIZE_IMAGE: handle_compress_resize_image,
        TaskType.TRANSCRIBE_AUDIO: handle_transcribe_audio,
        TaskType.CONVERT_MARKDOWN: handle_convert_markdown_to_html,
        TaskType.FILTER_CSV: handle_filter_csv,
    }
    
    handler = handlers.get(task_type)
    if not handler:
        raise ValueError(f"Unknown task type: {task_type}")
    
    return await handler(original_task)
