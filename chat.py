import base64
import collections
import subprocess
from collections import OrderedDict
import itertools
from tkinter.font import Font
import requests
import openai
from openai import OpenAI, AsyncOpenAI
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox
from threading import Thread
import configparser
import re
import os
import platform
import asyncio
import tiktoken
import json
from datetime import datetime
import random
import string
import anthropic
import logging

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

model_info_params = (
    'max_context_tokens', 'max_rsp_tokens', 'selectable',
    'input_price', 'output_price', 'points_to', 'vision',
    'image', 'image_prices', 'claude', 'image_styles'
)
ModelInfo = collections.namedtuple(
    'ModelInfo', model_info_params, defaults=(None,) * len(model_info_params)
)

STYLE_NONE = 'none'

# Pricing information per 1M tokens, or per image
all_models = {
    'gpt-4o': ModelInfo(
        points_to='gpt-4o-2024-05-13',
        selectable=True
    ),
    'gpt-4o-2024-05-13': ModelInfo(
        max_context_tokens=128000,
        selectable=False,
        input_price=5,
        output_price=15,
        vision=True
    ),
    'gpt-4-turbo': ModelInfo(
        points_to='gpt-4-turbo-2024-04-09',
        selectable=True
    ),
    'gpt-4-turbo-2024-04-09': ModelInfo(
        max_context_tokens=128000,
        selectable=False,
        input_price=10,
        output_price=30
    ),
    'gpt-4-turbo-preview': ModelInfo(
        points_to='gpt-4-0125-preview',
        selectable=True
    ),
    'gpt-4-0125-preview': ModelInfo(
        max_context_tokens=12800,
        max_rsp_tokens=4096,
        selectable=False,
        input_price=10,
        output_price=30
    ),
    'gpt-4-1106-preview': ModelInfo(
        max_context_tokens=12800,
        max_rsp_tokens=4096,
        selectable=True,
        input_price=10,
        output_price=30
    ),
    'gpt-4-vision-preview': ModelInfo(
        points_to='gpt-4-1106-vision-preview',
        selectable=True,
        vision=True
    ),
    'gpt-4-1106-vision-preview': ModelInfo(
        max_context_tokens=12800,
        max_rsp_tokens=4096,
        selectable=False,
        input_price=10,
        output_price=30,
        vision=True
    ),
    'gpt-4': ModelInfo(
        points_to='gpt-4-0613',
        selectable=True,
    ),
    'gpt-4-0613': ModelInfo(
        max_context_tokens=8192,
        selectable=False,
        input_price=30,
        output_price=60
    ),
    'gpt-4-32k': ModelInfo(
        points_to='gpt-4-32k-0613',
        selectable=True
    ),
    'gpt-4-32k-0613': ModelInfo(
        max_context_tokens=32768,
        selectable=False,
        input_price=60,
        output_price=120
    ),
    'gpt-3.5-turbo': ModelInfo(
        points_to='gpt-3.5-turbo-0125',
        selectable=True
    ),
    'gpt-3.5-turbo-0125': ModelInfo(
        max_context_tokens=16385,
        max_rsp_tokens=4096,
        selectable=False,
        input_price=0.5,
        output_price=1.5
    ),
    'gpt-3.5-turbo-1106': ModelInfo(
        max_context_tokens=16385,
        max_rsp_tokens=4096,
        selectable=True,
        input_price=1,
        output_price=2
    ),
    'gpt-3.5-turbo-instruct': ModelInfo(
        max_context_tokens=4096,
        selectable=True,
        input_price=1.5,
        output_price=2
    ),
    'dall-e-3': ModelInfo(
        selectable=True,
        image=True,
        image_prices=OrderedDict([
            ('standard,1024x1024', 0.04),
            ('standard,1024x1792', 0.08),
            ('standard,1792x1024', 0.08),
            ('hd,1024x1024', 0.08),
            ('hd,1024x1792', 0.12),
            ('hd,1792x1024', 0.12),
        ]),
        image_styles=[STYLE_NONE, 'vivid', 'natural']
    ),
    'dall-e-2': ModelInfo(
        selectable=True,
        image=True,
        image_prices=OrderedDict([
            ('1024x1024', 0.02),
            ('512x512', 0.018),
            ('256x256', 0.016)
        ])
    ),
    'claude-3-opus-20240229': ModelInfo(
        max_context_tokens=16384,
        selectable=True,
        input_price=15,
        output_price=75,
        claude=True
    ),
    'claude-3-sonnet-20240229': ModelInfo(
        max_context_tokens=16384,
        selectable=True,
        input_price=3,
        output_price=15,
        claude=True
    ),
    'claude-3-haiku-20240307': ModelInfo(
        max_context_tokens=16384,
        selectable=True,
        input_price=0.25,
        output_price=1.25,
        claude=True
    ),
    'claude-2.1': ModelInfo(
        max_context_tokens=16384,
        selectable=True,
        input_price=8,
        output_price=24,
        claude=True
    ),
    'claude-2.0': ModelInfo(
        max_context_tokens=16384,
        selectable=True,
        input_price=8,
        output_price=24,
        claude=True
    ),
    'claude-instant-1.2': ModelInfo(
        max_context_tokens=16384,
        selectable=True,
        input_price=8,
        output_price=24,
        claude=True
    ),
}

selectable_models = [k for k in all_models if all_models[k].selectable]
selectable_models.sort()


def get_model_info(model):
    model_info = all_models[model]
    return all_models[model_info.points_to] if model_info.points_to else model_info


def current_model_info():
    return get_model_info(model_var.get())


class ToolTip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip_window = None
        self.tooltip_id = None  # ID of the scheduled tooltip
        self.delay = 650  # Delay in milliseconds (0.65 seconds)
        self.widget.bind("<Enter>", self.schedule_tooltip)
        self.widget.bind("<Leave>", self.hide_tooltip)

    def schedule_tooltip(self, event=None):
        # Cancel any existing scheduled tooltip
        if self.tooltip_id:
            self.widget.after_cancel(self.tooltip_id)
        # Schedule the tooltip to be shown after the delay
        self.tooltip_id = self.widget.after(self.delay, self.show_tooltip)

    def show_tooltip(self):
        x, y, _, _ = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 25
        self.tooltip_window = tk.Toplevel(self.widget)
        self.tooltip_window.wm_overrideredirect(True)
        self.tooltip_window.wm_geometry(f"+{x}+{y}")
        label = tk.Label(
            self.tooltip_window, text=self.text, background="black", foreground="white", relief="solid", borderwidth=1
        )
        label.pack()

    def hide_tooltip(self, event=None):
        # Cancel any scheduled tooltip
        if self.tooltip_id:
            self.widget.after_cancel(self.tooltip_id)
            self.tooltip_id = None
        # Destroy the tooltip window if it exists
        if self.tooltip_window:
            self.tooltip_window.destroy()
            self.tooltip_window = None


# configure config file
config_filename = "config.ini"
if not os.path.exists(config_filename):
    with open(config_filename, "w") as f:
        f.write("[openai]\n")
        f.write("[anthropic]\n")
        f.write("[app]\n")
        f.write("[proxy]\n")
config = configparser.ConfigParser()
config.read(config_filename)
if not config.has_section("openai"):
    config.add_section("openai")
    with open(config_filename, "w") as f:
        config.write(f)
if not config.has_section("anthropic"):
    config.add_section("anthropic")
    with open(config_filename, "w") as f:
        config.write(f)
if not config.has_section("app"):
    config.add_section("app")
    with open(config_filename, "w") as f:
        config.write(f)
if not config.has_section('proxy'):
    config.add_section('proxy')
    with open(config_filename, "w") as f:
        config.write(f)


def update_proxy_environ():
    http_proxy = config.get('proxy', 'http', fallback='')
    if http_proxy:
        os.environ['http_proxy'] = http_proxy
    else:
        os.environ.pop('http_proxy', None)

    https_proxy = config.get('proxy', 'https', fallback='')
    if https_proxy:
        os.environ['https_proxy'] = https_proxy
    else:
        os.environ.pop('https_proxy', None)


system_message_default_text = "You are a helpful assistant."
os_name = platform.system()
if os_name == 'Linux' and "ANDROID_BOOTLOGO" in os.environ:
    os_name = 'Android'

update_proxy_environ()

client = OpenAI(
    api_key=config.get("openai", "api_key", fallback="insert-key"),
    organization=config.get("openai", "organization", fallback="")
)
aclient = AsyncOpenAI(
    api_key=config.get("openai", "api_key", fallback="insert-key"),
    organization=config.get("openai", "organization", fallback="")
)
anthropic_client = anthropic.Anthropic(api_key=config.get("anthropic", "api_key", fallback=""))

is_streaming_cancelled = False

if not os.path.exists("chat_logs"):
    os.makedirs("chat_logs")


def clear_chat_history():
    for row in reversed(range(len(chat_history))):
        delete_message(row + 1)

    chat_history.clear()
    

def save_chat_history():
    filename = chat_filename_var.get()
    chat_data = {
        "system_message": system_message_widget.get("1.0", tk.END).strip(),
        "chat_history": [
            {
                "role": message["role"].get(),
                "content": message["content_widget"].get("1.0", tk.END).strip()
            }
            for message in chat_history
        ]
    }

    if filename == "<new-log>":
        # Get a file name suggestion from the API
        suggested_filename = request_file_name()
        file_path = filedialog.asksaveasfilename(
            defaultextension=".json", initialdir="chat_logs", initialfile=suggested_filename, title="Save Chat Log",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")]  # Add a file type filter for JSON
        )
    else:
        file_path = os.path.join("chat_logs", filename)
        # Check for overwrite confirmation
        if not messagebox.askokcancel("Overwrite Confirmation", f"Do you want to overwrite '{filename}'?"):
            return

    if not file_path:
        return

    with open(file_path, "w", encoding='utf-8') as f:
        json.dump(chat_data, f, indent=4, ensure_ascii=False)

    update_chat_file_dropdown(file_path)


def count_tokens(messages, model):
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found for token counter. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    else:
        tokens_per_message = 3
        tokens_per_name = 1
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


def get_messages_from_chat_history():
    messages = [
        {"role": "system", "content": system_message_widget.get("1.0", tk.END).strip()}
    ]
    for message in chat_history:
        messages.append(
            {
                "role": message["role"].get(),
                "content": message["content_widget"].get("1.0", tk.END).strip()
            }
        )
    return messages


def request_file_name():
    # add to messages a system message informing the AI to create a title
    messages = get_messages_from_chat_history()
    messages.append(
        {
            "role": "system",
            "content": "The user is saving this chat log. In your next message, "
                       "please write only a suggested name for the file. "
                       "It should be in the format 'file-name-is-separated-by-hyphens', "
                       "it should be descriptive of the chat you had with the user, "
                       "and it should be very concise - no more than 4 words (and ideally just 2 or 3). "
                       "Do not acknowledge this system message with any additional words, "
                       "please simply write the suggested filename."
        }
    )
    # remove excess messages beyond context window limit for gpt-3.5-turbo
    num_tokens = count_tokens(messages, "gpt-3.5-turbo")
    num_messages = len(messages)
    if num_tokens > 4096:
        for i in range(num_messages):
            if i < 0:
                break
            num_tokens_in_this_message = count_tokens([messages[num_messages-i-2]], "gpt-3.5-turbo")
            messages[num_messages-i-2]["content"] = ""
            num_tokens = num_tokens - num_tokens_in_this_message
            if num_tokens <= 4096:
                break
    # get completion
    response = client.chat.completions.create(model="gpt-3.5-turbo", messages=messages)
    # return the filename
    suggested_filename = response.choices[0].message.content.strip()
    return suggested_filename


def show_error_popup(message):
    error_popup = tk.Toplevel(app)
    error_popup.title("Error")
    error_popup.geometry("350x100")

    error_label = ttk.Label(error_popup, text=message, wraplength=300)
    error_label.pack(padx=20, pady=20)

    error_popup.focus_force()
    center_popup_over_main_window(error_popup, app, 0, -150)


def show_error_and_open_settings(message):
    if popup is not None:
        popup.focus_force()
    else:
        show_popup()
    show_error_popup(message)


def parse_and_create_image_messages(content):
    image_url_pattern = r"https?://[^\s,\"\{\}]+"
    image_urls = re.findall(image_url_pattern, content, re.IGNORECASE)

    parts = re.split(image_url_pattern, content)
    messages = []

    for i, text in enumerate(parts):
        text = text.strip()
        if text:
            messages.append({"type": "text", "text": text})
        if i < len(image_urls):
            messages.append({"type": "image_url", "image_url": {"url": image_urls[i], "detail": image_detail_var.get()}})

    return {"role": "user", "content": messages}


def parse_and_create_image_messages_v2(content):
    image_pattern = re.compile(r'@image\[([^\[\]]+)\]')
    parts = image_pattern.split(content)
    if len(parts) == 1:
        return {"role": "user", "content": content}
    text_parts = parts[::2]
    image_parts = parts[1::2]
    messages = []
    for text, image in itertools.zip_longest(text_parts, image_parts):
        text = text.strip()
        if text:
            messages.append({"type": "text", "text": text})
        if image:
            if image.startswith('http://') or image.startswith('https://'):
                messages.append(
                    {"type": "image_url", "image_url": {"url": image, "detail": image_detail_var.get()}}
                )
            else:
                if not os.path.exists(image):
                    show_error_popup(f'Image [{image}] does not exist')
                    return
                with open(image, 'rb') as image_file:
                    image_data = image_file.read()
                base64_encoded = base64.b64encode(image_data).decode('utf8')
                messages.append(
                    {
                        "type": "image_url", "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_encoded}",
                            "detail": image_detail_var.get()
                        }
                    }
                )
    return {"role": "user", "content": messages}


def send_request():
    if current_model_info().image:
        send_image_request()
    else:
        send_chat_request()

def download_image(url, save_path, session=None):
    if url == 'mock-success':
        return True
    if url == 'mock-fail':
        return False
    # Send a GET request to the URL
    session = session or requests
    try:
        rsp = session.get(url)
    except Exception as e:
        LOGGER.warning('Downloading file error: %s', str(e))
        rsp = None

    # Check if the request was successful
    if rsp is None or rsp.status_code != 200:
        return False
    # Open the file and write the content of the response
    with open(save_path, 'wb') as file:
        file.write(rsp.content)
    return True


def send_image_request():
    global is_streaming_cancelled
    if not chat_history:
        return
    last_message = chat_history[-1]
    if last_message['role'].get() != 'user':
        show_error_popup('Last message is not from user')
    prompt = last_message["content_widget"].get("1.0", tk.END).strip()
    model = model_var.get()
    qualities = dall_quality_var.get().split(',')
    if len(qualities) == 2:
        quality, size = qualities
    else:
        quality = None
        size = qualities[0]
    n = dall_n_var.get()
    image_style = dall_style_var.get() if current_model_info().image_styles else None
    image_style = None if image_style == STYLE_NONE else image_style
    request_params = dict(
        prompt=prompt,
        model=model,
        size=size,
        n=n
    )
    if quality:
        request_params['quality'] = quality
    if image_style:
        request_params['style'] = image_style

    def request_thread():
        global is_streaming_cancelled
        try:
            image_rsp = client.images.generate(**request_params)
            class MockData:
                def __init__(self, **kwargs):
                    self.kwargs = kwargs
                def __getattr__(self, item):
                    return self.kwargs.get(item)

            # image_rsp = MockData(data=[
            #     MockData(url='mock-success', revised_prompt='This is an image')
            # ] * n)

            app.after(0, add_to_last_message, 'Images are generated. Start downloading...\n\n')
            image_file_prefix = f'images/{model}_{quality}_{size}_{image_style}_' \
                                f'{datetime.now().strftime("%y%m%d-%H%M%S")}'
            for i, image_data in enumerate(image_rsp.data):
                image_file = f'{image_file_prefix}_{i}.jpg'
                rt = download_image(image_data.url, image_file)
                if not rt:
                    download_result = f'Failed to download image {image_data.url}.'
                else:
                    LOGGER.info('Image (%s) downloaded as %s', image_data.url, image_file)
                    download_result = f'Downloaded as @image[{image_file}]'
                content = f'Image {i}:\n  {download_result}\n  {image_data.revised_prompt}\n\n'
                app.after(0, add_to_last_message, content, True)
                open_with_preview(image_file)
        except Exception as e:
            error_message = f"An unexpected error occurred: {e}"
            show_error_popup(error_message)
        if not is_streaming_cancelled and chat_history[-1]['role'] != 'user':
            app.after(0, finish_request)

    is_streaming_cancelled = False
    set_submit_button(False)
    Thread(target=request_thread).start()


def send_chat_request():
    global is_streaming_cancelled
    # get messages
    messages = get_messages_from_chat_history()
    # check if too many tokens
    model_max_context_window = current_model_info().max_context_tokens
    num_prompt_tokens = count_tokens(messages, model_var.get())
    num_completion_tokens = int(max_length_var.get())
    if num_prompt_tokens + num_completion_tokens > model_max_context_window:
        show_error_popup(
            f"combined prompt and completion tokens "
            f"({num_prompt_tokens} + {num_completion_tokens} = {num_prompt_tokens+num_completion_tokens})"
            f" exceeds this model's maximum context window of {model_max_context_window}."
        )
        return
    # convert messages to image api format, if necessary
    if current_model_info().vision:
        # Update the messages to include image data if any image URLs are found in the user's input
        new_messages = []
        for message in messages:
            if message["role"] == "user" and "content" in message:
                # Check for image URLs and create a single message with a 'content' array
                message_with_images = parse_and_create_image_messages_v2(message["content"])
                if not message_with_images:
                    return
                new_messages.append(message_with_images)
            else:
                # System or assistant messages are added unchanged
                new_messages.append(message)
        messages = new_messages

    # send request
    def request_thread():
        global is_streaming_cancelled
        model_name = model_var.get()
        if get_model_info(model_name).claude:
            # Streaming for Anthropic's Claude model
            # Create a copy of messages for Anthropic API
            # It has a bunch of extra requirements not present in OpenAI's API
            anthropic_messages = []
            system_content = ""
            for message in messages:
                if message["role"] == "system":
                    system_content += message["content"] + "\n"
                elif message["content"]:
                    anthropic_messages.append({"role": message["role"], "content": message["content"]})
            if len(anthropic_messages) == 0 or anthropic_messages[0]["role"] == "assistant":
                anthropic_messages.insert(0, {"role": "user", "content": "<no message>"})
            for i in range(len(anthropic_messages) - 1, 0, -1):
                if anthropic_messages[i]["role"] == anthropic_messages[i - 1]["role"]:
                    anthropic_messages.insert(i, {
                        "role": "user" if anthropic_messages[i]["role"] == "assistant" else
                        "assistant", "content": "<no message>"
                    })
            if anthropic_messages[-1]["role"] == "assistant":
                anthropic_messages.append({"role": "user", "content": "<no message>"})
            async def streaming_anthropic_chat_completion():
                global is_streaming_cancelled
                try:
                    with anthropic_client.messages.stream(
                        model=model_name,
                        max_tokens=min(max_length_var.get(), 4000),
                        messages=anthropic_messages,
                        system=system_content.strip(),
                        temperature=temperature_var.get()
                    ) as stream:
                        for text in stream.text_stream:
                            app.after(0, add_to_last_message, text)
                            if is_streaming_cancelled:
                                break
                except Exception as e:
                    error_message = f"An unexpected error occurred: {e}"
                    loop.call_soon_threadsafe(show_error_popup, error_message)
                if not is_streaming_cancelled:
                    app.after(0, finish_request)
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(streaming_anthropic_chat_completion())
        else:
            # Existing streaming code for OpenAI models
            async def streaming_chat_completion():
                global is_streaming_cancelled
                try:
                    async for chunk in await aclient.chat.completions.create(
                            model=model_name,
                            messages=messages,
                            temperature=temperature_var.get(),
                            max_tokens=max_length_var.get(),
                            top_p=1,
                            frequency_penalty=0,
                            presence_penalty=0,
                            stream=True
                    ):
                        content = chunk.choices[0].delta.content
                        if content is not None:
                            app.after(0, add_to_last_message, content)
                        if is_streaming_cancelled:
                            break
                except openai.AuthenticationError as e:
                    error_message = str(e)
                    if "Incorrect API key" in str(e):
                        error_message = "API key is incorrect, please configure it in the settings."
                    elif "No such organization" in str(e):
                        error_message = "Organization not found, please configure it in the settings."
                    loop.call_soon_threadsafe(show_error_and_open_settings, error_message)
                except Exception as e:
                    error_message = f"An unexpected error occurred: {e}"
                    loop.call_soon_threadsafe(show_error_popup, error_message)
                if not is_streaming_cancelled:
                    app.after(0, finish_request)
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(streaming_chat_completion())
    is_streaming_cancelled = False
    set_submit_button(False)
    Thread(target=request_thread).start()


def update_chat_file_dropdown(new_file_path):
    global chat_files  # Ensure chat_files is accessible globally
    
    # Refresh the list of chat files from the directory
    chat_files = sorted(
        [f for f in os.listdir("chat_logs") if os.path.isfile(os.path.join("chat_logs", f)) and f.endswith('.json')],
        key=lambda x: os.path.getmtime(os.path.join("chat_logs", x)),
        reverse=True
    )
    
    new_file_name = os.path.basename(new_file_path)
    
    # Check if the new file name is already in the list of chat files
    if new_file_name not in chat_files:
        chat_files.insert(0, new_file_name)  # Insert the file at the beginning if it's not there
    
    chat_filename_var.set(new_file_name)  # Select the newly created log

    # Clear and repopulate the dropdown menu with the refreshed list of files
    menu = chat_file_dropdown["menu"]
    menu.delete(0, "end")
    menu.add_command(label="<new-log>", command=lambda value="<new-log>": chat_filename_var.set(value))
    for file in chat_files:
        menu.add_command(label=file, command=lambda value=file: chat_filename_var.set(value))


def load_chat_history():
    filename = chat_filename_var.get()
    
    if not filename or filename == "<new-log>":
        clear_chat_history()
        system_message_widget.delete("1.0", tk.END)
        system_message_widget.insert(tk.END, system_message_default_text)
        add_message("user", "")
        return

    filepath = os.path.join("chat_logs", filename)
    if os.path.exists(filepath) and filepath.endswith('.json'):
        with open(filepath, "r", encoding='utf-8') as f:
            chat_data = json.load(f)

        clear_chat_history()

        system_message = chat_data["system_message"]
        system_message_widget.delete("1.0", tk.END)
        system_message_widget.insert(tk.END, system_message)

        for entry in chat_data["chat_history"]:
            add_message(entry["role"], entry["content"])

    app.after(100, update_height_of_all_messages)


def update_height_of_all_messages():
    for message in chat_history:
        update_content_height(None, message["content_widget"])


def add_to_last_message(content, check_file=False):
    last_message = chat_history[-1]
    if last_message["role"].get() == "assistant":
        last_message["content_widget"].insert(tk.END, content)
        update_content_height(None, last_message["content_widget"])
        if check_file:
            content_check_all_files(last_message["content_widget"])
    else:
        add_message("assistant", content)


def cancel_streaming():
    global is_streaming_cancelled
    is_streaming_cancelled = True
    set_submit_button(True)


def finish_request():
    add_empty_user_message()
    set_submit_button(True)


def add_empty_user_message():
    if chat_history and chat_history[-1]['role'] != 'user':
        add_message("user", "")


def update_entry_widths(event=None):
    window_width = app.winfo_width()
    screen_width = app.winfo_screenwidth()
    dpi = app.winfo_fpixels('1i')
    if os.name == 'posix':
        scaling_factor = 0.08 * (96/dpi)
    else:
        scaling_factor = 0.12 * (96/dpi)
    # Calculate the new width of the Text widgets based on the window width
    new_entry_width = int((window_width - scaling_factor*1000) * scaling_factor)

    for message in chat_history:
        message["content_widget"].configure(width=new_entry_width)


def strLen(s):
    count = 0
    for c in s:
        count += 1 if ord(c) < 256 else 2
    return count


def update_content_height(event, content_widget):
    if event and event.type == tk.EventType.Configure:
        if hasattr(content_widget, 'prev_width') and event.width == content_widget.prev_width:
            return
        content_widget.prev_width = event.width
    lines = content_widget.get("1.0", "end-1c").split("\n")
    # widget_width = int(content_widget["width"])
    widget_width = content_widget.winfo_width()
    measure_font = Font(font=content_widget["font"])
    single_char_width = measure_font.measure('a')
    wrapped_lines = 0
    for line in lines:
        # if line == "":
        #     wrapped_lines += 1
        # else:
            # line_length = strLen(line)
        line_length = measure_font.measure(line) + single_char_width
        wrapped_lines += -(-line_length // widget_width)  # Equivalent to math.ceil(line_length / widget_width)
    content_widget.configure(height=wrapped_lines)


def content_check_all_files(text_widget, text=None, only_last=False):
    image_pattern = re.compile(r'(@image\[[^\[\]]+\])')
    if text is None:
        text = text_widget.get('1.0', 'end')
    parts = image_pattern.split(text)
    prev_chars = 0
    parts_len = len(parts)
    for i in range(0, parts_len, 2):
        if i+1 == parts_len:
            break
        image_start = len(parts[i]) + prev_chars
        image_end = image_start + len(parts[i + 1])
        prev_chars = image_end
        if only_last and parts[-1] == '' and i < parts_len - 3:
            continue
        text_widget.tag_add('file', f'1.0 + {image_start} chars', f'1.0 + {image_end} chars')


def content_check_file(event, content_widget):
    e = event
    if event and event.type == tk.EventType.KeyRelease:
        if event.char == ']':
            key_pos = f'@{e.x},{e.y}+1c'
            text = content_widget.get('1.0', key_pos)
            if not text or len(text) < 5:
                return
            if text[-1] != ']' and text[-2] != ']':
                return
            if text[-2] == ']':
                text = text[:-1]
            content_check_all_files(content_widget, text=text, only_last=True)


def content_key_release(event, content_widget):
    update_content_height(event, content_widget)
    content_check_file(event, content_widget)


def content_loose_focus(event, content_widget):
    content_check_all_files(content_widget)


def open_with_preview(image_path):
    if not os.path.exists(image_path):
        LOGGER.info('Image %s does not exist', image_path)
        return
    # Use the 'open' command to open the image with the Preview app
    subprocess.run(["open", "-a", "Preview", image_path])


def content_image_click(event, content_widget):
    click_index = content_widget.index(f'@{event.x},{event.y}')
    all_ranges = content_widget.tag_ranges('file')
    all_ranges = list(zip(all_ranges[0::2], all_ranges[1::2]))
    for l, r in all_ranges:
        if content_widget.compare(l, '<=', click_index) and content_widget.compare(click_index, '<=', r):
            image_part = content_widget.get(l, r)
            image_pattern = re.compile(r'@image\[([^\[\]]+)\]')
            image_sch = image_pattern.search(image_part)
            if not image_sch:
                break
            image_path = image_sch.groups()[0]
            if os.path.exists(image_path):
                open_with_preview(image_path)
            else:
                show_error_popup(f'Image [{image_path}] does not exist')
            break


def add_message(role="user", content=""):
    global add_button_row
    message = {
        "role": tk.StringVar(value=role),
        "content": tk.StringVar(value=content)
    }
    chat_history.append(message)

    row = len(chat_history)
    message["role_button"] = ttk.Button(
        inner_frame, textvariable=message["role"], command=lambda: toggle_role(message),width=8
    )
    message["role_button"].grid(row=row, column=0, sticky="nw")
    message["content_widget"] = tk.Text(inner_frame, wrap=tk.WORD, height=1, width=50, undo=True)
    message["content_widget"].tag_configure('file', underline=True, foreground='blue')
    message["content_widget"].tag_bind(
        'file', '<Double-1>',
        lambda event, content_widget=message["content_widget"]: content_image_click(event, content_widget)
    )
    message["content_widget"].grid(row=row, column=1, sticky="we")
    message["content_widget"].insert(tk.END, content)
    key_release_fn = content_key_release if role == 'user' else update_content_height
    message["content_widget"].bind(
        "<KeyRelease>",
        lambda event, content_widget=message["content_widget"]: key_release_fn(event, content_widget)
    )
    if role == 'user':
        message["content_widget"].bind(
            "<FocusOut>",
            lambda event, content_widget=message["content_widget"]: content_loose_focus(event, content_widget)
        )
    message["content_widget"].bind(
        "<Configure>",
        lambda event, content_widget=message["content_widget"]: update_content_height(event, content_widget)
    )
    update_content_height(None, message["content_widget"])
    content_check_all_files(message["content_widget"])

    add_button_row += 1
    align_add_button()

    message["delete_button"] = ttk.Button(inner_frame, text="-", width=3, command=lambda: delete_message(row))
    message["delete_button"].grid(row=row, column=2, sticky="ne")

    chat_frame.yview_moveto(1.5)


def align_add_button():
    add_button.grid(row=add_button_row, column=0, sticky="e", pady=(5, 0))
    add_button_label.grid(row=add_button_row, column=1, sticky="sw")


def delete_message(row):
    for widget in inner_frame.grid_slaves():
        if int(widget.grid_info()["row"]) == row:
            widget.destroy()

    del chat_history[row - 1]

    for i, message in enumerate(chat_history[row - 1:], start=row):
        for widget in inner_frame.grid_slaves():
            if int(widget.grid_info()["row"]) == i + 1:
                widget.grid(row=i)

        message["delete_button"].config(command=lambda row=i: delete_message(row))

    global add_button_row
    add_button_row -= 1
    align_add_button()
    cancel_streaming()


def toggle_role(message):
    current_role = message["role"].get()
    if current_role == "user":
        message["role"].set("assistant")
    elif current_role == "assistant":
        message["role"].set("system")
    else:
        message["role"].set("user")


def configure_scrollregion(event):
    chat_frame.configure(scrollregion=chat_frame.bbox("all"))


def save_api_key():
    global client, aclient, anthropic_client
    if apikey_var.get() != "":
        client = OpenAI(api_key=apikey_var.get(), organization=orgid_var.get())
        aclient = AsyncOpenAI(api_key=apikey_var.get(), organization=orgid_var.get())
        config.set("openai", "api_key", client.api_key)
        config.set("openai", "organization", client.organization)
    if anthropic_apikey_var.get() != "":
        anthropic_client = anthropic.Anthropic(api_key=anthropic_apikey_var.get())
        config.set("anthropic", "api_key", anthropic_client.api_key)

    with open("config.ini", "w") as config_file:
        config.write(config_file)


def save_proxy_config():
    http_proxy = http_proxy_var.get()
    if http_proxy:
        config.set('proxy', "http", http_proxy)
    else:
        config.remove_option('proxy', 'http')

    https_proxy = https_proxy_var.get()
    if https_proxy:
        config.set('proxy', "https", https_proxy)
    else:
        config.remove_option('proxy', 'https')

    with open("config.ini", "w") as config_file:
        config.write(config_file)


def add_message_via_button():
    add_message("user" if len(chat_history) == 0 or chat_history[-1]["role"].get() == "assistant" else "assistant", "")


def prompt_paste_from_clipboard(event, entry):
    global previous_focused_widget
    # Check if the previously focused widget is the same as the clicked one
    if previous_focused_widget != entry:
        clipboard_content = app.clipboard_get()
        if messagebox.askyesno(
                "Paste from Clipboard", f"Do you want to paste the following content from the clipboard?\n\n"
                                        f"{clipboard_content}"
        ):
            entry.delete(0, tk.END)
            entry.insert(0, clipboard_content)
    previous_focused_widget = entry


def update_previous_focused_widget(event):
    global previous_focused_widget
    previous_focused_widget = event.widget


# Functions for synchronizing slider and entry
def on_temp_entry_change(*args):
    try:
        value = float(temp_entry_var.get())
        if 0 <= value <= 1:
            temperature_var.set(value)
        else:
            raise ValueError
    except ValueError:
        temp_entry_var.set(f"{temperature_var.get():.2f}")


def on_max_len_entry_change(*args):
    try:
        value = int(max_len_entry_var.get())
        if 1 <= value <= 8000:
            max_length_var.set(value)
        else:
            raise ValueError
    except ValueError:
        max_len_entry_var.set(max_length_var.get())


def add_app_section_to_config_if_not_present():
    if not config.has_section("app"):
        config.add_section("app")
        config.set("app", "dark_mode", "False")
        with open(config_filename, "w") as f:
            config.write(f)


def save_dark_mode_state():
    config.set("app", "dark_mode", str(dark_mode_var.get()))
    with open(config_filename, "w") as f:
        config.write(f)


def load_dark_mode_state():
    return config.getboolean("app", "dark_mode", fallback=False)


def toggle_dark_mode():
    global popup_frame
    if dark_mode_var.get():
        app.configure(bg="#2c2c2c")
        main_frame.configure(style="Dark.TFrame")
        configuration_frame.configure(style="Dark.TFrame")
        chat_frame.configure(bg="#2c2c2c") # Change chat_frame background color
        inner_frame.configure(style="Dark.TFrame")
        
        for widget in main_frame.winfo_children():
            if isinstance(widget, (ttk.Label, ttk.OptionMenu, ttk.Checkbutton)):
                widget.configure(style="Dark." + widget.winfo_class())
        for widget in configuration_frame.winfo_children():
            if isinstance(widget, (ttk.Label, ttk.OptionMenu, ttk.Checkbutton)):
                widget.configure(style="Dark." + widget.winfo_class())
        if popup_frame is not None:
            popup_frame.configure(style="Dark.TFrame")
            for widget in popup_frame.winfo_children():
                if isinstance(widget, (ttk.Label, ttk.OptionMenu, ttk.Checkbutton)):
                    widget.configure(style="Dark." + widget.winfo_class())
    else:
        app.configure(bg=default_bg_color)
        main_frame.configure(style="")
        configuration_frame.configure(style="")
        chat_frame.configure(bg=default_bg_color) # Reset chat_frame background color
        inner_frame.configure(style="")
        
        for widget in main_frame.winfo_children():
            if isinstance(widget, (ttk.Label, ttk.Button, ttk.OptionMenu, ttk.Checkbutton, ttk.Scrollbar)):
                widget.configure(style=widget.winfo_class())
        for widget in configuration_frame.winfo_children():
            if isinstance(widget, (ttk.Label, ttk.Button, ttk.OptionMenu, ttk.Checkbutton, ttk.Scrollbar)):
                widget.configure(style=widget.winfo_class())
        if popup_frame is not None:
            popup_frame.configure(style="")
            for widget in popup_frame.winfo_children():
                if isinstance(widget, (ttk.Label, ttk.Button, ttk.OptionMenu, ttk.Checkbutton, ttk.Scrollbar)):
                    widget.configure(style=widget.winfo_class())
    save_dark_mode_state()


def get_default_bg_color(root):
    # Create a temporary button widget to get the default background color
    temp_button = tk.Button(root)
    default_bg_color = temp_button.cget('bg')
    # Destroy the temporary button
    temp_button.destroy()
    return default_bg_color


def on_close():
    # Generate a timestamp string with the format "YYYYMMDD_HHMMSS"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Generate a random 6-character alphanumeric ID
    random_id = ''.join(random.choices(string.ascii_letters + string.digits, k=6))
    # Combine the timestamp and random ID to create the filename
    filename = f"{timestamp}_{random_id}.json"

    # Create the 'temp/backup/' directory if it doesn't exist
    backup_path = "temp/backup"
    if not os.path.exists(backup_path):
        os.makedirs(backup_path)

    # Construct the full file path
    file_path = os.path.join(backup_path, filename)

    # Get the chat history data
    chat_data = {
        "system_message": system_message_widget.get("1.0", tk.END).strip(),
        "chat_history": [
            {
                "role": message["role"].get(),
                "content": message["content_widget"].get("1.0", tk.END).strip()
            }
            for message in chat_history
        ]
    }
   
    # Save the chat history to the file
    with open(file_path, "w", encoding='utf-8') as f:
        json.dump(chat_data, f, indent=4)

    # Save the last used model
    config.set("app", "last_used_model", model_var.get())
    with open("config.ini", "w") as config_file:
        config.write(config_file)

    # Close the application
    app.destroy()


popup = None
popup_frame = None


def on_config_changed(*args):
    save_api_key()


def on_proxy_config_changed(*args):
    save_proxy_config()
    update_proxy_environ()
    save_api_key()


def on_popup_close():
    global popup
    popup.destroy()
    popup = None


def close_popup():
    global popup
    if popup is not None:
        popup.destroy()
        popup = None


def center_popup_over_main_window(popup_window, main_window, x_offset=0, y_offset=0):
    main_window.update_idletasks()

    main_window_width = main_window.winfo_width()
    main_window_height = main_window.winfo_height()
    main_window_x = main_window.winfo_rootx()
    main_window_y = main_window.winfo_rooty()

    popup_width = popup_window.winfo_reqwidth()
    popup_height = popup_window.winfo_reqheight()

    x_position = main_window_x + (main_window_width // 2) - (popup_width // 2) + x_offset
    y_position = main_window_y + (main_window_height // 2) - (popup_height // 2) + y_offset

    popup_window.geometry(f"+{x_position}+{y_position}")


def show_popup():
    global popup, popup_frame, apikey_var, orgid_var
    # If the popup already exists, close it and set popup to None
    if popup is not None:
        popup.destroy()
        popup = None
        return
        
    popup = tk.Toplevel(app)
    popup.title("Settings")
    popup_frame = ttk.Frame(popup, padding="3")
    popup_frame.grid(row=0, column=0, sticky="new")

    # Add API key / Org ID configurations
    ttk.Label(popup_frame, text="API Key:").grid(row=0, column=0, sticky="e")
    apikey_entry = ttk.Entry(popup_frame, textvariable=apikey_var, width=60)
    apikey_entry.grid(row=0, column=1, sticky="e")

    ttk.Label(popup_frame, text="Org ID:").grid(row=1, column=0, sticky="e")
    orgid_entry = ttk.Entry(popup_frame, textvariable=orgid_var, width=60)
    orgid_entry.grid(row=1, column=1, sticky="e")

    # Add Anthropic API key configuration
    ttk.Label(popup_frame, text="Anthropic API Key:").grid(row=2, column=0, sticky="e")
    anthropic_apikey_entry = ttk.Entry(popup_frame, textvariable=anthropic_apikey_var, width=60)
    anthropic_apikey_entry.grid(row=2, column=1, sticky="e")

    # Create a Checkbutton widget for dark mode toggle
    dark_mode_var.set(load_dark_mode_state())
    dark_mode_checkbutton = ttk.Checkbutton(
        popup_frame, text="Dark mode", variable=dark_mode_var, command=toggle_dark_mode
    )
    dark_mode_checkbutton.grid(row=3, column=0, columnspan=2, padx=10, pady=10, sticky="w")

    ttk.Label(popup_frame, text="HTTP Proxy:").grid(row=4, column=0, sticky="e")
    http_proxy_entry = ttk.Entry(popup_frame, textvariable=http_proxy_var, width=60)
    http_proxy_entry.grid(row=4, column=1, sticky="e")
    ttk.Label(popup_frame, text="HTTPS Proxy:").grid(row=5, column=0, sticky="e")
    https_proxy_entry = ttk.Entry(popup_frame, textvariable=https_proxy_var, width=60)
    https_proxy_entry.grid(row=5, column=1, sticky="e")

    # Add a button to close the popup
    close_button = ttk.Button(popup_frame, text="Close", command=close_popup)
    close_button.grid(row=100, column=0, columnspan=2, pady=10)
    toggle_dark_mode()
    # Bind the on_popup_close function to the WM_DELETE_WINDOW protocol
    popup.protocol("WM_DELETE_WINDOW", on_popup_close)
    # Bind events for api/org clipboard prompts, only in Android
    if os_name == 'Android':
        apikey_entry.bind("<Button-1>", lambda event, entry=apikey_entry: prompt_paste_from_clipboard(event, entry))
        orgid_entry.bind("<Button-1>", lambda event, entry=orgid_entry: prompt_paste_from_clipboard(event, entry))
        apikey_entry.bind("<FocusOut>", update_previous_focused_widget)
        orgid_entry.bind("<FocusOut>", update_previous_focused_widget)
        
    # Center the popup over the main window
    center_popup_over_main_window(popup, app)
    
    popup.focus_force()
    
def set_submit_button(active):
    if active:
        submit_button_text.set("Submit")
        submit_button.configure(command=send_request)
    else:
        submit_button_text.set("Cancel")
        submit_button.configure(command=cancel_streaming)


def update_parameters_visibility(*args):
    model_info = current_model_info()
    if model_info.image:
        temperature_label.grid_remove()
        temperature_scale.grid_remove()
        temp_entry.grid_remove()
        max_length_label.grid_remove()
        max_length_scale.grid_remove()
        max_len_entry.grid_remove()
        dall_n_label.grid(row=0, column=6, sticky="e")
        dall_n_option.grid(row=0, column=7, sticky="w")
        dall_quality_label.grid(row=0, column=6, sticky="se")
        option_menu = dall_quality_option['menu']
        option_menu.delete(0, 'end')
        qualities = [quality for quality in model_info.image_prices]
        dall_quality_var.set(qualities[0])
        for option in qualities:
            option_menu.add_command(label=option, command=lambda value=option: dall_quality_var.set(value))
        dall_quality_option.grid(row=0, column=7, columnspan=2, sticky="sw")
        if model_info.image_styles:
            dall_style_option.grid(row=0, column=8, sticky="e")
        else:
            dall_style_option.grid_remove()
    else:
        dall_n_label.grid_remove()
        dall_n_option.grid_remove()
        dall_quality_label.grid_remove()
        dall_quality_option.grid_remove()
        dall_style_option.grid_remove()
        temperature_label.grid(row=0, column=6, sticky="e")
        temperature_scale.grid(row=0, column=7, sticky="w")
        temp_entry.grid(row=0, column=8, sticky="w")
        max_length_label.grid(row=0, column=6, sticky="se")
        max_length_scale.grid(row=0, column=7, sticky="sw")
        max_len_entry.grid(row=0, column=8, sticky="sw")
    if model_info.vision:
        image_detail_dropdown.grid(row=0, column=8, sticky="ne")
    else:
        image_detail_dropdown.grid_remove()


def show_token_count():
    messages = get_messages_from_chat_history()
    num_input_tokens = count_tokens(messages, model_var.get())
    num_output_tokens = max_length_var.get()
    total_tokens = num_input_tokens + num_output_tokens
    model_info = current_model_info()
    
    # Estimation for high detail image cost based on a 1024x1024 image
    # todo: get the actual image sizes for a more accurate estimation
    high_detail_cost_per_image = (170 * 4 + 85) / 1000 * 0.01  # 4 tiles for 1024x1024 + base tokens
    
    # Count the number of images in the messages
    num_images = sum(1 for message in messages if "image_url" in message.get("content", ""))
    
    # Calculate vision cost if the model is vision preview
    vision_cost = 0
    if model_info.vision:
        if image_detail_var.get() == "low":
            # Fixed cost for low detail images
            vision_cost_per_image = 0.00085
            vision_cost = vision_cost_per_image * num_images
        else:
            # Estimated cost for high detail images
            vision_cost = high_detail_cost_per_image * num_images
        total_cost = vision_cost
        cost_message = f"Vision Cost: ${total_cost:.5f} for {num_images} images"
    else:
        # Calculate input and output costs for non-vision models
        input_cost = model_info.input_price * num_input_tokens / 1000000
        output_cost = model_info.output_price * num_output_tokens / 1000000
        total_cost = input_cost + output_cost
        cost_message = f"Input Cost: ${input_cost:.5f}\nOutput Cost: ${output_cost:.5f}\nTotal Cost: ${total_cost:.5f}"
    
    messagebox.showinfo(
        "Token Count and Cost", f"Number of tokens: {total_tokens} "
                                f"(Input: {num_input_tokens}, Output: {num_output_tokens})\n"
                                f"{cost_message}"
    )


# Initialize the main application window
app = tk.Tk()
app.geometry("800x600")
app.title("Chat Completions GUI")

add_app_section_to_config_if_not_present()

# Hide console window on Windows
if os.name == 'nt':
    import ctypes
    ctypes.windll.user32.ShowWindow(ctypes.windll.kernel32.GetConsoleWindow(), 0)

# Create the main_frame for holding the chat and other widgets
main_frame = ttk.Frame(app, padding="10")
main_frame.grid(row=1, column=0, sticky="nsew")

# System message and model selection
system_message = tk.StringVar(value=system_message_default_text)
ttk.Label(main_frame, text="System message:").grid(row=0, column=0, sticky="w")
system_message_widget = tk.Text(main_frame, wrap=tk.WORD, height=5, width=50, undo=True)
system_message_widget.grid(row=0, column=1, sticky="we", pady=3)
system_message_widget.insert(tk.END, system_message.get())

last_used_model = config.get("app", "last_used_model", fallback="gpt-3.5-turbo")
model_var = tk.StringVar(value=last_used_model)
ttk.Label(main_frame, text="Model:").grid(row=0, column=6, sticky="ne")

def change_model(e):
    model = model_var.get()
    model_info = get_model_info(model)
    if model_info.max_rsp_tokens:
        change_max_length_scale(model_info.max_rsp_tokens)
    else:
        change_max_length_scale(8000)


def change_max_length_scale(max_value):
    cur = max_length_scale.get()
    if cur > max_value:
        max_length_scale.set(max_value)
    max_length_scale.config(to=max_value)


model_option = ttk.OptionMenu(
    main_frame, model_var, last_used_model, *selectable_models, command=change_model
)
model_option.grid(row=0, column=7, sticky="nw")

# Add sliders for temperature, max length, and top p
temperature_var = tk.DoubleVar(value=0.7)
temperature_label = ttk.Label(main_frame, text="Temperature:")
temperature_scale = ttk.Scale(main_frame, variable=temperature_var, from_=0, to=1, orient="horizontal")

max_length_var = tk.IntVar(value=4000)
max_length_label = ttk.Label(main_frame, text="Max Length:")
max_length_scale = ttk.Scale(main_frame, variable=max_length_var, from_=1, to=8000, orient="horizontal")
change_max_length_scale(current_model_info().max_rsp_tokens or 8000)

# Add Entry widgets for temperature and max length
temp_entry_var = tk.StringVar()
temp_entry = ttk.Entry(main_frame, textvariable=temp_entry_var, width=5)
temp_entry_var.set(temperature_var.get())
temperature_var.trace("w", lambda *args: temp_entry_var.set(f"{temperature_var.get():.2f}"))
temp_entry_var.trace("w", on_temp_entry_change)

max_len_entry_var = tk.StringVar()
max_len_entry = ttk.Entry(main_frame, textvariable=max_len_entry_var, width=5)
max_len_entry_var.set(max_length_var.get())
max_length_var.trace("w", lambda *args: max_len_entry_var.set(max_length_var.get()))
max_len_entry_var.trace("w", on_max_len_entry_change)

dall_n_var = tk.IntVar(value=1)
dall_n_label = ttk.Label(main_frame, text="Images:")
dall_n_option = ttk.OptionMenu(main_frame, dall_n_var, str(1), *range(1, 11, 1))

dall_quality_var = tk.StringVar(value='')
dall_quality_label = ttk.Label(main_frame, text="Quality:")
dall_quality_option = ttk.OptionMenu(main_frame, dall_quality_var, '')

# Currently only dall-e-3 has styles
dall_style_var = tk.StringVar(value=all_models['dall-e-3'].image_styles[0])
dall_style_option = ttk.OptionMenu(
    main_frame, dall_style_var, dall_style_var.get(), *all_models['dall-e-3'].image_styles
)

# Chat frame and scrollbar
chat_history = []
chat_frame = tk.Canvas(main_frame, highlightthickness=0)
chat_frame.grid(row=1, column=0, columnspan=9, sticky="nsew")

inner_frame = ttk.Frame(chat_frame)
inner_frame.rowconfigure(0, weight=1)
chat_frame.create_window((0, 0), window=inner_frame, anchor="nw")

chat_scroll = ttk.Scrollbar(main_frame, orient="vertical", command=chat_frame.yview)
chat_scroll.grid(row=1, column=9, sticky="ns")
chat_frame.configure(yscrollcommand=chat_scroll.set)
chat_frame.bind('<MouseWheel>', lambda mw_event: chat_frame.yview_scroll(-1*mw_event.delta, 'units'))
inner_frame.bind('<MouseWheel>', lambda mw_event: chat_frame.yview_scroll(-1*mw_event.delta, 'units'))

# Add button for chat messages
add_button_row = 1
add_button = ttk.Button(inner_frame, text="+", width=2, command=add_message_via_button)
add_button_label = ttk.Label(inner_frame, text="Add")
ToolTip(add_button, "Add new message")

# Submit button
submit_button_text = tk.StringVar()  # Create a StringVar variable to control the text of the submit button
submit_button_text.set("Submit")  # Set the initial text of the submit button to "Submit"
# Use textvariable instead of text
submit_button = ttk.Button(main_frame, textvariable=submit_button_text, command=send_request)
submit_button.grid(row=7, column=7, sticky="e")

# Add a new button for counting tokens (new code)
token_count_button = ttk.Button(main_frame, text="Check Cost", command=show_token_count)
token_count_button.grid(row=7, column=0, sticky="w")  # Place it on the bottom left, same row as 'Submit'

add_message("user", "")

# Configuration frame for API key, Org ID, and buttons
configuration_frame = ttk.Frame(app, padding="3")
configuration_frame.grid(row=0, column=0, sticky="new")
config_row = 0

# Add a dropdown menu to select a chat log file to load
chat_filename_var = tk.StringVar()
chat_files = sorted(
    [f for f in os.listdir("chat_logs") if os.path.isfile(os.path.join("chat_logs", f)) and f.endswith('.json')],
    key=lambda x: os.path.getmtime(os.path.join("chat_logs", x)),
    reverse=True
)
ttk.Label(configuration_frame, text="Chat Log:").grid(row=config_row, column=0, sticky="w")
default_chat_file = "<new-log>"
chat_files.insert(0, default_chat_file)
chat_file_dropdown = ttk.OptionMenu(configuration_frame, chat_filename_var, default_chat_file, *chat_files)
chat_file_dropdown.grid(row=config_row, column=1, sticky="w")

# Add a button to load the selected chat log
load_button = ttk.Button(configuration_frame, text="Load Chat", command=load_chat_history)
load_button.grid(row=config_row, column=2, sticky="w")

# Add a button to save the chat history
save_button = ttk.Button(configuration_frame, text="Save Chat", command=save_chat_history)
save_button.grid(row=config_row, column=3, sticky="w")

http_proxy_var = tk.StringVar(value=config.get("proxy", "http", fallback=""))
https_proxy_var = tk.StringVar(value=config.get("proxy", "https", fallback=""))
apikey_var = tk.StringVar(value=client.api_key)
orgid_var = tk.StringVar(value=client.organization)
anthropic_apikey_var = tk.StringVar(value=anthropic_client.api_key)

http_proxy_var.trace("w", on_proxy_config_changed)
https_proxy_var.trace("w", on_proxy_config_changed)
apikey_var.trace("w", on_config_changed)
orgid_var.trace("w", on_config_changed)
anthropic_apikey_var.trace("w", on_config_changed)

# Add image detail dropdown
image_detail_var = tk.StringVar(value="low")
image_detail_dropdown = ttk.OptionMenu(main_frame, image_detail_var, "low", "low", "high", "auto")
update_parameters_visibility()

# Update image detail visibility based on selected model
model_var.trace("w", update_parameters_visibility)

# Create the hamburger menu button and bind it to the show_popup function
hamburger_button = ttk.Button(configuration_frame, text="≡", command=show_popup)
hamburger_button.grid(row=config_row, column=9, padx=10, pady=10, sticky="w")

default_bg_color = get_default_bg_color(app)
# Create styles for light and dark modes
style = ttk.Style(app)
style.configure("Dark.TFrame", background="#2c2c2c")
style.configure("Dark.TLabel", background="#2c2c2c", foreground="#ffffff")
# style.configure("Dark.TButton", background="#2c2c2c", foreground="2c2c2c")
style.configure("Dark.TOptionMenu", background="#2c2c2c", foreground="#ffffff")
style.configure("Dark.TCheckbutton", background="#2c2c2c", foreground="#ffffff")

dark_mode_var = tk.BooleanVar()
if load_dark_mode_state():
    dark_mode_var.set(True)
    toggle_dark_mode()
    
# Add a separator
ttk.Separator(configuration_frame, orient='horizontal').grid(
    row=config_row+1, column=0, columnspan=10, sticky="we", pady=3
)

# Set the weights for the configuration frame
configuration_frame.columnconfigure(3, weight=1)

# Configure weights for resizing behavior
app.columnconfigure(0, weight=1)
app.rowconfigure(0, weight=0)
app.rowconfigure(1, weight=1)
main_frame.columnconfigure(1, weight=1)
main_frame.rowconfigure(1, weight=1)

# Initialize the previous_focused_widget variable
previous_focused_widget = None

# Bind events
inner_frame.bind("<Configure>", configure_scrollregion)
app.bind("<Configure>", update_entry_widths)
app.bind_class('Entry', '<FocusOut>', update_previous_focused_widget)
app.bind("<Escape>", lambda event: show_popup())

# Add a protocol to handle the close event
app.protocol("WM_DELETE_WINDOW", on_close)
# Start the application main loop
app.mainloop()
