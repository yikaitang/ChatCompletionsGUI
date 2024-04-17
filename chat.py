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
import urllib.parse
import anthropic
import requests
import sys

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
        label = tk.Label(self.tooltip_window, text=self.text, background="black", foreground="white", relief="solid", borderwidth=1)
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

os_name = platform.system()
if os_name == 'Linux' and "ANDROID_BOOTLOGO" in os.environ:
	os_name = 'Android'

if not os.path.exists("chat_logs"):
    os.makedirs("chat_logs")


# Hide console window on Windows
if os.name == 'nt':
    import ctypes
    ctypes.windll.user32.ShowWindow(ctypes.windll.kernel32.GetConsoleWindow(), 0)

# configure config file
config_filename = "config.ini"
if not os.path.exists(config_filename):
    with open(config_filename, "w") as f:
        f.write("[openai]\n")
        f.write("[anthropic]\n")
        f.write("[custom_server]\n")
        f.write("[app]\n")
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
if not config.has_section("custom_server"):
    config.add_section("custom_server")
    with open(config_filename, "w") as f:
        config.write(f)
if not config.has_section("app"):
    config.add_section("app")
    config.set("app", "dark_mode", "False")
    with open(config_filename, "w") as f:
        config.write(f)

client = OpenAI(api_key=config.get("openai", "api_key", fallback="insert-key"), organization=config.get("openai", "organization", fallback=""))
aclient = AsyncOpenAI(api_key=config.get("openai", "api_key", fallback="insert-key"), organization=config.get("openai", "organization", fallback=""))
anthropic_client = anthropic.Anthropic(api_key=config.get("anthropic", "api_key", fallback=""))
custom_aclient = AsyncOpenAI(base_url = config.get("custom_server", "base_url", fallback=""), api_key=config.get("custom_server", "api_key", fallback="ollama"),)

system_message_default_text = ""
vision_models = ['gpt-4-vision-preview', 'gpt-4-1106-vision-preview', 'gpt-4-turbo', 'gpt-4-turbo-2024-04-09']
openai_models = [
    "gpt-3.5-turbo",
    "gpt-4-turbo",
    "gpt-4",
    "gpt-4-32k",
    "gpt-4-turbo-preview",
    "gpt-4-vision-preview",
    "gpt-4-0125-preview",
    "gpt-4-0613",
    "gpt-4-1106-preview",
    "gpt-4-1106-vision-preview",
    "gpt-3.5-turbo-16k",
    "gpt-3.5-turbo-0125",
    "gpt-3.5-turbo-0301",
    "gpt-3.5-turbo-0613",
    "gpt-3.5-turbo-1106",
    "gpt-3.5-turbo-16k-0613",
]
anthropic_models = ["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307", "claude-2.1", "claude-2.0", "claude-instant-1.2"]
custom_models = [model.strip() for model in config.get("custom_server", "models", fallback="").split(",") if model.strip()]

class MainWindow:
    def __init__(self, root):
        # Initialize the main application window
        self.app = root
        self.app.geometry("800x600")
        self.app.title("Chat Completions GUI")

        self.is_streaming_cancelled = False

        self.popup = None
        self.popup_frame = None

        # Create the main_frame for holding the chat and other widgets
        self.main_frame = ttk.Frame(self.app, padding="10")
        self.main_frame.grid(row=1, column=0, sticky="nsew")

        # System message and model selection
        system_message = tk.StringVar(value=system_message_default_text)
        ttk.Label(self.main_frame, text="System message:").grid(row=0, column=0, sticky="w")
        self.system_message_widget = tk.Text(self.main_frame, wrap=tk.WORD, height=5, width=50, undo=True)
        self.system_message_widget.grid(row=0, column=1, sticky="we", pady=3)
        self.system_message_widget.insert(tk.END, system_message.get())

        last_used_model = config.get("app", "last_used_model", fallback="gpt-4-turbo")
        self.model_var = tk.StringVar(value=last_used_model)
        ttk.Label(self.main_frame, text="Model:").grid(row=0, column=6, sticky="ne")
        self.update_models_dropdown()

        # Add sliders for temperature, max length, and top p
        self.temperature_var = tk.DoubleVar(value=0.7)
        ttk.Label(self.main_frame, text="Temperature:").grid(row=0, column=6, sticky="e")
        self.temperature_scale = ttk.Scale(self.main_frame, variable=self.temperature_var, from_=0, to=1, orient="horizontal")
        self.temperature_scale.grid(row=0, column=7, sticky="w")

        self.max_length_var = tk.IntVar(value=4000)
        ttk.Label(self.main_frame, text="Max Length:").grid(row=0, column=6, sticky="se")
        self.max_length_scale = ttk.Scale(self.main_frame, variable=self.max_length_var, from_=1, to=8000, orient="horizontal")
        self.max_length_scale.grid(row=0, column=7, sticky="sw")

        # Add Entry widgets for temperature and max length
        self.temp_entry_var = tk.StringVar()
        self.temp_entry = ttk.Entry(self.main_frame, textvariable=self.temp_entry_var, width=5)
        self.temp_entry.grid(row=0, column=8, sticky="w")
        self.temp_entry_var.set(self.temperature_var.get())
        self.temperature_var.trace("w", lambda *args: self.temp_entry_var.set(f"{self.temperature_var.get():.2f}"))
        self.temp_entry_var.trace("w", self.on_temp_entry_change)

        self.max_len_entry_var = tk.StringVar()
        self.max_len_entry = ttk.Entry(self.main_frame, textvariable=self.max_len_entry_var, width=5)
        self.max_len_entry.grid(row=0, column=8, sticky="sw")
        self.max_len_entry_var.set(self.max_length_var.get())
        self.max_length_var.trace("w", lambda *args: self.max_len_entry_var.set(self.max_length_var.get()))
        self.max_len_entry_var.trace("w", self.on_max_len_entry_change)

        # Chat frame and scrollbar
        self.chat_history = []
        self.chat_frame = tk.Canvas(self.main_frame, highlightthickness=0)
        self.chat_frame.grid(row=1, column=0, columnspan=9, sticky="nsew")

        self.inner_frame = ttk.Frame(self.chat_frame)
        self.inner_frame.rowconfigure(0, weight=1)
        self.chat_frame.create_window((0, 0), window=self.inner_frame, anchor="nw")

        chat_scroll = ttk.Scrollbar(self.main_frame, orient="vertical", command=self.chat_frame.yview)
        chat_scroll.grid(row=1, column=9, sticky="ns")
        self.chat_frame.configure(yscrollcommand=chat_scroll.set)

        # Add button for chat messages
        self.add_button_row = 1
        self.add_button = ttk.Button(self.inner_frame, text="+", width=2, command=self.add_message_via_button)
        self.add_button_label = ttk.Label(self.inner_frame, text="Add")
        ToolTip(self.add_button, "Add new message")

        # Submit button
        self.submit_button_text = tk.StringVar()  # Create a StringVar variable to control the text of the submit button
        self.submit_button_text.set("Submit")  # Set the initial text of the submit button to "Submit"
        self.submit_button = ttk.Button(self.main_frame, textvariable=self.submit_button_text, command=self.send_request)  # Use textvariable instead of text
        self.submit_button.grid(row=7, column=7, sticky="e")

        # Add a new button for counting tokens (new code)
        self.token_count_button = ttk.Button(self.main_frame, text="Count Tokens", command=self.show_token_count)
        self.token_count_button.grid(row=7, column=0, sticky="w")  # Place it on the bottom left, same row as 'Submit'

        self.add_message("user", "")

        # Configuration frame for API key, Org ID, and buttons
        self.configuration_frame = ttk.Frame(self.app, padding="3")
        self.configuration_frame.grid(row=0, column=0, sticky="new")
        config_row = 0

        # Add a dropdown menu to select a chat log file to load
        self.chat_filename_var = tk.StringVar()
        self.chat_files = sorted(
            [f for f in os.listdir("chat_logs") if os.path.isfile(os.path.join("chat_logs", f)) and f.endswith('.json')],
            key=lambda x: os.path.getmtime(os.path.join("chat_logs", x)),
            reverse=True
        )
        ttk.Label(self.configuration_frame, text="Chat Log:").grid(row=config_row, column=0, sticky="w")
        default_chat_file = "<new-log>"
        self.chat_files.insert(0, default_chat_file)
        self.chat_file_dropdown = ttk.OptionMenu(self.configuration_frame, self.chat_filename_var, default_chat_file, *self.chat_files)
        self.chat_file_dropdown.grid(row=config_row, column=1, sticky="w")

        # Add a button to load the selected chat log
        self.load_button = ttk.Button(self.configuration_frame, text="Load Chat", command=self.load_chat_history)
        self.load_button.grid(row=config_row, column=2, sticky="w")

        # Add a button to save the chat history
        self.save_button = ttk.Button(self.configuration_frame, text="Save Chat", command=self.save_chat_history)
        self.save_button.grid(row=config_row, column=3, sticky="w")

        # Add api configuration variables
        self.apikey_var = tk.StringVar(value=client.api_key)
        self.orgid_var = tk.StringVar(value=client.organization)
        self.anthropic_apikey_var = tk.StringVar(value=anthropic_client.api_key)
        self.custom_baseurl_var = tk.StringVar(value=custom_aclient.base_url)
        self.custom_apikey_var = tk.StringVar(value=custom_aclient.api_key)
        self.custom_models_var = tk.StringVar(value=", ".join(custom_models))

        self.apikey_var.trace("w", self.on_config_changed)
        self.orgid_var.trace("w", self.on_config_changed)
        self.anthropic_apikey_var.trace("w", self.on_config_changed)
        self.custom_baseurl_var.trace("w", self.on_config_changed)
        self.custom_apikey_var.trace("w", self.on_config_changed)
        self.custom_models_var.trace("w", self.on_config_changed)

        # Add image detail dropdown
        self.image_detail_var = tk.StringVar(value="low")
        self.image_detail_dropdown = ttk.OptionMenu(self.main_frame, self.image_detail_var, "low", "low", "high")
        self.update_image_detail_visibility()

        # Update image detail visibility based on selected model
        self.model_var.trace("w", self.update_image_detail_visibility)
        # Create the hamburger menu button and bind it to the show_popup function
        self.hamburger_button = ttk.Button(self.configuration_frame, text="≡", command=self.show_popup)
        self.hamburger_button.grid(row=config_row, column=9, padx=10, pady=10, sticky="w")

        self.default_bg_color = self.get_default_bg_color(self.app)
        # Create styles for light and dark modes
        self.style = ttk.Style(self.app)
        self.style.configure("Dark.TFrame", background="#2c2c2c")
        self.style.configure("Dark.TLabel", background="#2c2c2c", foreground="#ffffff")
        # style.configure("Dark.TButton", background="#2c2c2c", foreground="2c2c2c")
        self.style.configure("Dark.TOptionMenu", background="#2c2c2c", foreground="#ffffff")
        self.style.configure("Dark.TCheckbutton", background="#2c2c2c", foreground="#ffffff")

        self.dark_mode_var = tk.BooleanVar()
        if self.load_dark_mode_state():
            self.dark_mode_var.set(True)
            self.toggle_dark_mode()

        # Add a separator
        ttk.Separator(self.configuration_frame, orient='horizontal').grid(row=config_row+1, column=0, columnspan=10, sticky="we", pady=3)

        # Set the weights for the configuration frame
        self.configuration_frame.columnconfigure(3, weight=1)

        # Configure weights for resizing behavior
        self.app.columnconfigure(0, weight=1)
        self.app.rowconfigure(0, weight=0)
        self.app.rowconfigure(1, weight=1)
        self.main_frame.columnconfigure(1, weight=1)
        self.main_frame.rowconfigure(1, weight=1)

        # Initialize the previous_focused_widget variable
        self.previous_focused_widget = None

        # Bind events
        self.inner_frame.bind("<Configure>", self.configure_scrollregion)
        self.app.bind("<Configure>", self.update_entry_widths)
        self.app.bind_class('Entry', '<FocusOut>', self.update_previous_focused_widget)
        self.app.bind("<Escape>", lambda event: self.show_popup())
        # Bind Command-N to open new windows (Control-N on Windows/Linux)
        # modifier = 'Command' if sys.platform == 'darwin' else 'Control'
        modifier = 'Control'
        self.app.bind_all(f'<{modifier}-n>', self.create_new_window)
        # Add a protocol to handle the close event
        self.app.protocol("WM_DELETE_WINDOW", self.on_close)
        # Start the application main loop
        self.app.mainloop()

    def clear_chat_history(self):
        for row in reversed(range(len(self.chat_history))):
            self.delete_message(row + 1)

        self.chat_history.clear()


    def save_chat_history(self):
        filename = self.chat_filename_var.get()
        chat_data = {
            "system_message": self.system_message_widget.get("1.0", tk.END).strip(),
            "chat_history": [
                {
                    "role": message["role"].get(),
                    "content": message["content_widget"].get("1.0", tk.END).strip()
                }
                for message in self.chat_history
            ]
        }

        if filename == "<new-log>":
            # Get a file name suggestion from the API
            suggested_filename = self.request_file_name()
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
            json.dump(chat_data, f, indent=4)

        self.update_chat_file_dropdown(file_path)

    def count_tokens(self, messages, model):
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

    def get_messages_from_chat_history(self):
        messages = [
            {"role": "system", "content": self.system_message_widget.get("1.0", tk.END).strip()}
        ]
        for message in self.chat_history:
            messages.append(
                {
                    "role": message["role"].get(),
                    "content": message["content_widget"].get("1.0", tk.END).strip()
                }
            )
        return messages

    def request_file_name(self):
        # add to messages a system message informing the AI to create a title
        messages = self.get_messages_from_chat_history()
        messages.append(
            {
                "role": "system",
                "content": "The user is saving this chat log. In your next message, please write only a suggested name for the file. It should be in the format 'file-name-is-separated-by-hyphens', it should be descriptive of the chat you had with the user, and it should be very concise - no more than 4 words (and ideally just 2 or 3). Do not acknowledge this system message with any additional words, please simply write the suggested filename."
            }
        )
        # remove excess messages beyond context window limit for gpt-3.5-turbo
        num_tokens = self.count_tokens(messages, "gpt-3.5-turbo")
        num_messages = len(messages)
        if num_tokens > 4096:
            for i in range(num_messages):
                if i < 0:
                    break
                num_tokens_in_this_message = self.count_tokens([messages[num_messages-i-2]], "gpt-3.5-turbo")
                messages[num_messages-i-2]["content"] = ""
                num_tokens = num_tokens - num_tokens_in_this_message
                if num_tokens <= 4096:
                    break
        # get completion
        response = client.chat.completions.create(model="gpt-3.5-turbo", messages=messages)
        # return the filename
        suggested_filename = response.choices[0].message.content.strip()
        return suggested_filename

    def show_error_popup(self, message):
        error_popup = tk.Toplevel(self.app)
        error_popup.title("Error")
        error_popup.geometry("350x100")

        error_label = ttk.Label(error_popup, text=message, wraplength=300)
        error_label.pack(padx=20, pady=20)

        error_popup.focus_force()
        self.center_popup_over_main_window(error_popup, self.app, 0, -150)

    def show_error_and_open_settings(self, message):
        if self.popup is not None:
            self.popup.focus_force()
        else:
            self.show_popup()
        self.show_error_popup(message)

    def parse_and_create_image_messages(self, content):
        url_pattern = r"(https?://[^\s,\"\{\}]+)"
        parts = re.split(url_pattern, content)

        messages = []
        for text in parts:
            if self.is_image_url(text):
                messages.append({"type": "image_url", "image_url": {"url": text, "detail": self.image_detail_var.get()}})
            elif messages and messages[-1].get("type") == "text":
                messages[-1]["text"] += text
            elif text:
                messages.append({"type": "text", "text": text})
        return {"role": "user", "content": messages}

    def is_url(self, str):
        url_pattern = r"https?://[^\s,\"\{\}]+"
        return re.match(url_pattern, str)

    def is_image_url(self, url):
        if not self.is_url(url):
            return False
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.head(url, timeout=5, headers=headers)
            content_type = response.headers.get('Content-Type', '')
            return 'image' in content_type
        except requests.RequestException as e:
            return False

    def send_request(self):
        # get messages
        messages = self.get_messages_from_chat_history()
        # check if too many tokens
        model_max_context_window = (16384 if ('16k' in self.model_var.get() or 'gpt-3.5-turbo-0125' in self.model_var.get()) else 128000 if (self.model_var.get() in ['gpt-4-1106-preview','gpt-4-0125-preview', 'gpt-4-turbo'] or 'claude' in self.model_var.get()) else 8192 if self.model_var.get().startswith('gpt-4') else 128000)
        num_prompt_tokens = self.count_tokens(messages, self.model_var.get())
        num_completion_tokens = int(self.max_length_var.get())
        if num_prompt_tokens + num_completion_tokens > model_max_context_window:
            self.show_error_popup(f"combined prompt and completion tokens ({num_prompt_tokens} + {num_completion_tokens} = {num_prompt_tokens+num_completion_tokens}) exceeds this model's maximum context window of {model_max_context_window}.")
            return
        # convert messages to image api format, if necessary
        if self.model_var.get() in vision_models:
            # Update the messages to include image data if any image URLs are found in the user's input
            new_messages = []
            for message in messages:
                if message["role"] == "user" and "content" in message:
                    # Check for image URLs and create a single message with a 'content' array
                    message_with_images = self.parse_and_create_image_messages(message["content"])
                    new_messages.append(message_with_images)
                else:
                    # System or assistant messages are added unchanged
                    new_messages.append(message)
            messages = new_messages
        # send request
        def request_thread():
            model_name = self.model_var.get()
            if model_name.startswith("claude"):
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
                        anthropic_messages.insert(i, {"role": "user" if anthropic_messages[i]["role"] == "assistant" else "assistant", "content": "<no message>"})
                if anthropic_messages[-1]["role"] == "assistant":
                    anthropic_messages.append({"role": "user", "content": "<no message>"})
                async def streaming_anthropic_chat_completion():
                    with anthropic_client.messages.stream(
                            model=model_name,
                            max_tokens=min(self.max_length_var.get(), 4000),
                            messages=anthropic_messages,
                            system=system_content.strip(),
                            temperature=self.temperature_var.get()
                        ) as stream:
                        try:
                            for text in stream.text_stream:
                                self.app.after(0, self.add_to_last_message, text)
                                if self.is_streaming_cancelled:
                                    break
                        except Exception as e:
                            error_message = f"An unexpected error occurred: {e}"
                            loop.call_soon_threadsafe(self.show_error_popup, error_message)
                        finally:
                            stream.close()
                    if not self.is_streaming_cancelled:
                        self.app.after(0, self.add_empty_user_message)
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(streaming_anthropic_chat_completion())
            else:
                # Existing streaming code for OpenAI models
                async def streaming_chat_completion():
                    streaming_client = aclient if model_name in openai_models else custom_aclient
                    try:
                        response = streaming_client.chat.completions.create(model=model_name,
                            messages=messages,
                            temperature=self.temperature_var.get(),
                            max_tokens=self.max_length_var.get(),
                            # top_p=1,
                            # frequency_penalty=0,
                            # presence_penalty=0,
                            stream=True)
                    except Exception as e:
                        error_message = f"An error occurred: {e}"
                        loop.call_soon_threadsafe(self.show_error_popup, error_message)
                    
                    if response:
                        try:
                            async for chunk in await response:
                                content = chunk.choices[0].delta.content
                                if content is not None:
                                    self.app.after(0, self.add_to_last_message, content)
                                if self.is_streaming_cancelled:
                                    break
                        except openai.AuthenticationError as e:
                            if "Incorrect API key" in str(e):
                                error_message = "API key is incorrect, please configure it in the settings."
                            elif "No such organization" in str(e):
                                error_message = "Organization not found, please configure it in the settings."
                            loop.call_soon_threadsafe(self.show_error_and_open_settings, error_message)
                        except Exception as e:
                            error_message = f"An unexpected error occurred: {e}"
                            loop.call_soon_threadsafe(self.show_error_popup, error_message)
                        finally:
                            response.close()
                            print("Closed response")
                    if not self.is_streaming_cancelled:
                        self.app.after(0, self.add_empty_user_message)
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(streaming_chat_completion())
        self.is_streaming_cancelled = False
        self.set_submit_button(False)
        Thread(target=request_thread).start()

    def update_chat_file_dropdown(self, new_file_path):
        # Refresh the list of chat files from the directory
        self.chat_files = sorted(
            [f for f in os.listdir("chat_logs") if os.path.isfile(os.path.join("chat_logs", f)) and f.endswith('.json')],
            key=lambda x: os.path.getmtime(os.path.join("chat_logs", x)),
            reverse=True
        )

        new_file_name = os.path.basename(new_file_path)

        # Check if the new file name is already in the list of chat files
        if new_file_name not in self.chat_files:
            self.chat_files.insert(0, new_file_name)  # Insert the file at the beginning if it's not there

        self.chat_filename_var.set(new_file_name)  # Select the newly created log

        # Clear and repopulate the dropdown menu with the refreshed list of files
        menu = self.chat_file_dropdown["menu"]
        menu.delete(0, "end")
        menu.add_command(label="<new-log>", command=lambda value="<new-log>": self.chat_filename_var.set(value))
        for file in self.chat_files:
            menu.add_command(label=file, command=lambda value=file: self.chat_filename_var.set(value))

    def update_models_dropdown(self):
        current_model = self.model_var.get()
        possible_models = [*openai_models, *anthropic_models, *custom_models]
        filtered_models = [model for model in possible_models if model != current_model]
        self.models_dropdown = ttk.OptionMenu(self.main_frame, self.model_var, current_model, current_model, *filtered_models).grid(row=0, column=7, sticky="nw")

    def load_chat_history(self):
        filename = self.chat_filename_var.get()

        if not filename or filename == "<new-log>":
            self.clear_chat_history()
            self.system_message_widget.delete("1.0", tk.END)
            self.system_message_widget.insert(tk.END, system_message_default_text)
            self.add_message("user", "")
            return

        filepath = os.path.join("chat_logs", filename)
        if os.path.exists(filepath) and filepath.endswith('.json'):
            with open(filepath, "r", encoding='utf-8') as f:
                chat_data = json.load(f)

            self.clear_chat_history()

            system_message = chat_data["system_message"]
            self.system_message_widget.delete("1.0", tk.END)
            self.system_message_widget.insert(tk.END, system_message)

            for entry in chat_data["chat_history"]:
                self.add_message(entry["role"], entry["content"])

        self.app.after(100, self.update_height_of_all_messages)


    def update_height_of_all_messages(self):
        for message in self.chat_history:
            self.update_content_height(None, message["content_widget"])

    def add_to_last_message(self, content):
        last_message = self.chat_history[-1]
        if last_message["role"].get() == "assistant":
            last_message["content_widget"].insert(tk.END, content)
            self.update_content_height(None, last_message["content_widget"])
        else:
            self.add_message("assistant", content)

    def cancel_streaming(self):
        self.is_streaming_cancelled = True
        self.set_submit_button(True)

    def add_empty_user_message(self):
        self.add_message("user", "")
        self.set_submit_button(True)

    def update_entry_widths(self, event=None):
        window_width = self.app.winfo_width()
        screen_width = self.app.winfo_screenwidth()
        dpi = self.app.winfo_fpixels('1i')
        if sys.platform == 'darwin':
            scaling_factor = 0.09 * (96/dpi)
        elif os.name == 'posix':
            scaling_factor = 0.08 * (96/dpi)
        else:
            scaling_factor = 0.12 * (96/dpi)
        # Calculate the new width of the Text widgets based on the window width
        new_entry_width = int((window_width - scaling_factor*1000) * scaling_factor)

        for message in self.chat_history:
            message["content_widget"].configure(width=new_entry_width)


    def update_content_height(self, event, content_widget):
        lines = content_widget.get("1.0", "end-1c").split("\n")
        widget_width = int(content_widget["width"])
        wrapped_lines = 0
        for line in lines:
            if line == "":
                wrapped_lines += 1
            else:
                line_length = len(line)
                wrapped_lines += -(-line_length // widget_width)  # Equivalent to math.ceil(line_length / widget_width)
        content_widget.configure(height=wrapped_lines)

    def add_message(self, role="user", content=""):
        message = {
            "role": tk.StringVar(value=role),
            "content": tk.StringVar(value=content)
        }
        self.chat_history.append(message)

        row = len(self.chat_history)
        message["role_button"] = ttk.Button(self.inner_frame, textvariable=message["role"], command=lambda: self.toggle_role(message), width=8)
        message["role_button"].grid(row=row, column=0, sticky="nw")
        message["content_widget"] = tk.Text(self.inner_frame, wrap=tk.WORD, height=1, width=50, undo=True)
        message["content_widget"].grid(row=row, column=1, sticky="we")
        message["content_widget"].insert(tk.END, content)
        message["content_widget"].bind("<KeyRelease>", lambda event, content_widget=message["content_widget"]: self.update_content_height(event, content_widget))
        self.update_content_height(None, message["content_widget"])

        self.add_button_row += 1
        self.align_add_button()

        message["delete_button"] = ttk.Button(self.inner_frame, text="-", width=3, command=lambda: self.delete_message(row))
        message["delete_button"].grid(row=row, column=2, sticky="ne")

        self.chat_frame.yview_moveto(1.5)

    def align_add_button(self):
        self.add_button.grid(row=self.add_button_row, column=0, sticky="e", pady=(5, 0))
        self.add_button_label.grid(row=self.add_button_row, column=1, sticky="sw")

    def delete_message(self, row):
        for widget in self.inner_frame.grid_slaves():
            if int(widget.grid_info()["row"]) == row:
                widget.destroy()

        del self.chat_history[row - 1]

        for i, message in enumerate(self.chat_history[row - 1:], start=row):
            for widget in self.inner_frame.grid_slaves():
                if int(widget.grid_info()["row"]) == i + 1:
                    widget.grid(row=i)

            message["delete_button"].config(command=lambda row=i: self.delete_message(row))

        self.add_button_row -= 1
        self.align_add_button()
        self.cancel_streaming()

    def toggle_role(self, message):
        current_role = message["role"].get()
        if current_role == "user":
            message["role"].set("assistant")
        elif current_role == "assistant":
            message["role"].set("system")
        else:
            message["role"].set("user")

    def configure_scrollregion(self, event):
        self.chat_frame.configure(scrollregion=self.chat_frame.bbox("all"))

    def save_api_key(self):
        global client, aclient, anthropic_client, custom_aclient
        if self.apikey_var.get() != "":
            client = OpenAI(api_key=self.apikey_var.get(), organization=self.orgid_var.get())
            aclient = AsyncOpenAI(api_key=self.apikey_var.get(), organization=self.orgid_var.get())
            config.set("openai", "api_key", client.api_key)
            config.set("openai", "organization", client.organization)
        if self.anthropic_apikey_var.get() != "":
            anthropic_client = anthropic.Anthropic(api_key=self.anthropic_apikey_var.get())
            config.set("anthropic", "api_key", anthropic_client.api_key)
        if self.custom_baseurl_var.get() != "":
            custom_aclient = OpenAI(api_key=self.custom_apikey_var.get(), base_url=self.custom_baseurl_var.get())
            config.set("custom_server", "base_url", str(custom_aclient.base_url))
            config.set("custom_server", "api_key", custom_aclient.api_key)
        if self.custom_models_var.get() != "":
            custom_models.clear()
            custom_models.extend([model.strip() for model in self.custom_models_var.get().split(",") if model.strip()])
            config.set("custom_server", "models", self.custom_models_var.get())
            self.update_models_dropdown()

        with open("config.ini", "w") as config_file:
            config.write(config_file)

    def add_message_via_button(self):
        self.add_message("user" if len(self.chat_history) == 0 or self.chat_history[-1]["role"].get() == "assistant" else "assistant", "")

    def prompt_paste_from_clipboard(self, event, entry):
        # Check if the previously focused widget is the same as the clicked one
        if self.previous_focused_widget != entry:
            clipboard_content = self.app.clipboard_get()
            if messagebox.askyesno("Paste from Clipboard", f"Do you want to paste the following content from the clipboard?\n\n{clipboard_content}"):
                entry.delete(0, tk.END)
                entry.insert(0, clipboard_content)
        self.previous_focused_widget = entry

    def update_previous_focused_widget(self, event):
        self.previous_focused_widget = event.widget

    # Functions for synchronizing slider and entry
    def on_temp_entry_change(self, *args):
        try:
            value = float(self.temp_entry_var.get())
            if 0 <= value <= 1:
                self.temperature_var.set(value)
            else:
                raise ValueError
        except ValueError:
            self.temp_entry_var.set(f"{self.temperature_var.get():.2f}")

    def on_max_len_entry_change(self, *args):
        try:
            value = int(self.max_len_entry_var.get())
            if 1 <= value <= 8000:
               self.max_length_var.set(value)
            else:
                raise ValueError
        except ValueError:
            self.max_len_entry_var.set(self.max_length_var.get())

    def save_dark_mode_state(self):
        config.set("app", "dark_mode", str(self.dark_mode_var.get()))
        with open(config_filename, "w") as f:
            config.write(f)

    def load_dark_mode_state(self):
        return config.getboolean("app", "dark_mode", fallback=False)

    def toggle_dark_mode(self):
        if self.dark_mode_var.get():
            self.app.configure(bg="#2c2c2c")
            self.main_frame.configure(style="Dark.TFrame")
            self.configuration_frame.configure(style="Dark.TFrame")
            self.chat_frame.configure(bg="#2c2c2c") # Change chat_frame background color
            self.inner_frame.configure(style="Dark.TFrame")

            for widget in self.main_frame.winfo_children():
                if isinstance(widget, (ttk.Label, ttk.OptionMenu, ttk.Checkbutton)):
                    widget.configure(style="Dark." + widget.winfo_class())
            for widget in self.configuration_frame.winfo_children():
                if isinstance(widget, (ttk.Label, ttk.OptionMenu, ttk.Checkbutton)):
                    widget.configure(style="Dark." + widget.winfo_class())
            if self.popup_frame is not None:
                self.popup_frame.configure(style="Dark.TFrame")
                for widget in self.popup_frame.winfo_children():
                    if isinstance(widget, (ttk.Label, ttk.OptionMenu, ttk.Checkbutton)):
                        widget.configure(style="Dark." + widget.winfo_class())
        else:
            self.app.configure(bg=self.default_bg_color)
            self.main_frame.configure(style="")
            self.configuration_frame.configure(style="")
            self.chat_frame.configure(bg=self.default_bg_color) # Reset chat_frame background color
            self.inner_frame.configure(style="")

            for widget in self.main_frame.winfo_children():
                if isinstance(widget, (ttk.Label, ttk.Button, ttk.OptionMenu, ttk.Checkbutton, ttk.Scrollbar)):
                    widget.configure(style=widget.winfo_class())
            for widget in self.configuration_frame.winfo_children():
                if isinstance(widget, (ttk.Label, ttk.Button, ttk.OptionMenu, ttk.Checkbutton, ttk.Scrollbar)):
                    widget.configure(style=widget.winfo_class())
            if self.popup_frame is not None:
                self.popup_frame.configure(style="")
                for widget in self.popup_frame.winfo_children():
                    if isinstance(widget, (ttk.Label, ttk.Button, ttk.OptionMenu, ttk.Checkbutton, ttk.Scrollbar)):
                        widget.configure(style=widget.winfo_class())
        self.save_dark_mode_state()

    def get_default_bg_color(self, root):
        # Create a temporary button widget to get the default background color
        temp_button = tk.Button(root)
        default_bg_color = temp_button.cget('bg')
        # Destroy the temporary button
        temp_button.destroy()
        return default_bg_color

    def on_close(self):
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
            "system_message": self.system_message_widget.get("1.0", tk.END).strip(),
            "chat_history": [
                {
                    "role": message["role"].get(),
                    "content": message["content_widget"].get("1.0", tk.END).strip()
                }
                for message in self.chat_history
            ]
        }

        # Save the chat history to the file
        with open(file_path, "w", encoding='utf-8') as f:
            json.dump(chat_data, f, indent=4)

        # Save the last used model
        config.set("app", "last_used_model", self.model_var.get())
        with open("config.ini", "w") as config_file:
            config.write(config_file)

        # Close the application
        self.app.destroy()

    def on_config_changed(self, *args):
        self.save_api_key()

    def on_popup_close(self):
        self.popup.destroy()
        self.popup = None

    def close_popup(self):
        if self.popup is not None:
            self.popup.destroy()
            self.popup = None

    def center_popup_over_main_window(self, popup_window, main_window, x_offset=0, y_offset=0):
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

    def show_popup(self):
        # If the popup already exists, close it and set popup to None
        if self.popup is not None:
            self.popup.destroy()
            self.popup = None
            return

        self.popup = tk.Toplevel(self.app)
        self.popup.title("Settings")
        self.popup_frame = ttk.Frame(self.popup, padding="3")
        self.popup_frame.grid(row=0, column=0, sticky="new")

        # Add API key / Org ID configurations
        ttk.Label(self.popup_frame, text="API Key:").grid(row=0, column=0, sticky="e")
        apikey_entry = ttk.Entry(self.popup_frame, textvariable=self.apikey_var, width=60)
        apikey_entry.grid(row=0, column=1, sticky="e")

        ttk.Label(self.popup_frame, text="Org ID:").grid(row=1, column=0, sticky="e")
        orgid_entry = ttk.Entry(self.popup_frame, textvariable=self.orgid_var, width=60)
        orgid_entry.grid(row=1, column=1, sticky="e")

        # Add Anthropic API key configuration
        ttk.Label(self.popup_frame, text="Anthropic API Key:").grid(row=2, column=0, sticky="e")
        anthropic_apikey_entry = ttk.Entry(self.popup_frame, textvariable=self.anthropic_apikey_var, width=60)
        anthropic_apikey_entry.grid(row=2, column=1, sticky="e")

        # Add Custom Server configuration
        ttk.Separator(self.popup_frame, orient='horizontal').grid(row=3, column=0, columnspan=2, sticky="we", pady=10)
        ttk.Label(self.popup_frame, text="Custom Server Configuration").grid(row=4, column=0, sticky="e")
        ttk.Label(self.popup_frame, text="Base URL:").grid(row=5, column=0, sticky="e")
        custom_baseurl_entry = ttk.Entry(self.popup_frame, textvariable=self.custom_baseurl_var, width=60)
        custom_baseurl_entry.grid(row=5, column=1, sticky="e")
        ttk.Label(self.popup_frame, text="API Key:").grid(row=6, column=0, sticky="e")
        custom_apikey_entry = ttk.Entry(self.popup_frame, textvariable=self.custom_apikey_var, width=60)
        custom_apikey_entry.grid(row=6, column=1, sticky="e")
        ttk.Label(self.popup_frame, text="Models (comma-separated):").grid(row=7, column=0, sticky="e")
        custom_models_entry = ttk.Entry(self.popup_frame, textvariable=self.custom_models_var, width=60)
        custom_models_entry.grid(row=7, column=1, sticky="e")

        # Create a Checkbutton widget for dark mode toggle
        ttk.Separator(self.popup_frame, orient='horizontal').grid(row=8, column=0, columnspan=2, sticky="we", pady=10)
        self.dark_mode_var.set(self.load_dark_mode_state())
        dark_mode_checkbutton = ttk.Checkbutton(self.popup_frame, text="Dark mode", variable=self.dark_mode_var, command=self.toggle_dark_mode)
        dark_mode_checkbutton.grid(row=9, column=0, columnspan=2, padx=10, pady=10, sticky="w")
        # Add a button to close the popup
        close_button = ttk.Button(self.popup_frame, text="Close", command=self.close_popup)
        close_button.grid(row=100, column=0, columnspan=2, pady=10)
        self.toggle_dark_mode()
        # Bind the on_popup_close function to the WM_DELETE_WINDOW protocol
        self.popup.protocol("WM_DELETE_WINDOW", self.on_popup_close)
        # Bind events for api/org clipboard prompts, only in Android
        if(os_name == 'Android'):
            apikey_entry.bind("<Button-1>", lambda event, entry=apikey_entry: self.prompt_paste_from_clipboard(event, entry))
            orgid_entry.bind("<Button-1>", lambda event, entry=orgid_entry: self.prompt_paste_from_clipboard(event, entry))
            apikey_entry.bind("<FocusOut>", self.update_previous_focused_widget)
            orgid_entry.bind("<FocusOut>", self.update_previous_focused_widget)

        # Center the popup over the main window
        self.center_popup_over_main_window(self.popup, self.app)

        self.popup.focus_force()

    def set_submit_button(self, active):
        if active:
            self.submit_button_text.set("Submit")
            self.submit_button.configure(command=self.send_request)
        else:
            self.submit_button_text.set("Cancel")
            self.submit_button.configure(command=self.cancel_streaming)

    def update_image_detail_visibility(self, *args):
        if self.model_var.get() in vision_models:
            self.image_detail_dropdown.grid(row=0, column=8, sticky="ne")
        else:
            self.image_detail_dropdown.grid_remove()

    def show_token_count(self):
        messages = self.get_messages_from_chat_history()
        num_input_tokens = self.count_tokens(messages, self.model_var.get())
        num_output_tokens = self.max_length_var.get()
        total_tokens = num_input_tokens + num_output_tokens
        model = self.model_var.get()
        # Pricing information per 1000 tokens
        pricing_info = {
            "gpt-4-turbo": {"input": 0.01, "output": 0.03},
            "gpt-4-turbo-2024-04-09": {"input": 0.01, "output": 0.03},
            "gpt-4-0125-preview": {"input": 0.01, "output": 0.03},
            "gpt-4-1106-preview": {"input": 0.01, "output": 0.03},
            "gpt-4-1106-vision-preview": {"input": 0.01, "output": 0.03},
            "gpt-4-vision-preview": {"input": 0.01, "output": 0.03},
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-4-0613": {"input": 0.03, "output": 0.06},
            "gpt-4-0314": {"input": 0.03, "output": 0.06},
            "gpt-4-32k": {"input": 0.06, "output": 0.12},
            "gpt-3.5-turbo-0125": {"input": 0.0005, "output": 0.0015},
            "gpt-3.5-turbo": {"input": 0.0010, "output": 0.0020},
            "gpt-3.5-turbo-0613": {"input": 0.0010, "output": 0.0020},
            "gpt-3.5-turbo-0301": {"input": 0.0010, "output": 0.0020},
            "gpt-3.5-turbo-16k": {"input": 0.0010, "output": 0.0020},
            "gpt-3.5-turbo-1106": {"input": 0.0010, "output": 0.0020},
            "gpt-3.5-turbo-instruct": {"input": 0.0015, "output": 0.0020},
            "claude-3-opus-20240229": {"input": 0.015, "output": 0.075},
            "claude-3-sonnet-20240229": {"input": 0.003, "output": 0.015},
            "claude-3-haiku-20240307": {"input": 0.00025, "output": 0.00125},
            "claude-2.1": {"input": 0.008, "output": 0.024},
            "claude-2.0": {"input": 0.008, "output": 0.024},
            "claude-instant-1.2": {"input": 0.0008, "output": 0.0024},
        }


        # Calculate input and output costs for non-vision models
        input_cost = pricing_info[model]["input"] * num_input_tokens / 1000 if model in pricing_info else 0
        output_cost = pricing_info[model]["output"] * num_output_tokens / 1000 if model in pricing_info else 0
        total_cost = input_cost + output_cost
        cost_message = f"Input Cost: ${input_cost:.5f}\nOutput Cost: ${output_cost:.5f}"

        if model in vision_models:
            # Estimation for high detail image cost based on a 1024x1024 image
            # todo: get the actual image sizes for a more accurate estimation
            high_detail_cost_per_image = (170 * 4 + 85) / 1000 * 0.01  # 4 tiles for 1024x1024 + base tokens

            # Count the number of images in the messages
            num_images = 0
            parsed_messages = [self.parse_and_create_image_messages(message.get("content","")) for message in messages]
            for message in parsed_messages:
                for content in message["content"]:
                    if "image_url" in content.get("type", ""):
                        num_images+=1

            # Calculate vision cost if the model is vision preview
            vision_cost = 0
            if self.image_detail_var.get() == "low":
                # Fixed cost for low detail images
                vision_cost_per_image = 0.00085
                vision_cost = vision_cost_per_image * num_images
            else:
                # Estimated cost for high detail images
                vision_cost = high_detail_cost_per_image * num_images
            total_cost = vision_cost
            cost_message += f"\nVision Cost: ${total_cost:.5f} for {num_images} images"

        messagebox.showinfo("Token Count and Cost", f"Number of tokens: {total_tokens} (Input: {num_input_tokens}, Output: {num_output_tokens})\n{cost_message}")

    def create_new_window(self, event):
        # Handle key press event
        if event.state == 0x0004 and event.keysym == 'n':  # 0x0004 is the mask for the Control key on Windows/Linux
            new_root = tk.Toplevel(self.app)
            new_window = MainWindow(new_root)

def main():
    root = tk.Tk()
    app = MainWindow(root)
    root.mainloop()

if __name__ == "__main__":
    main()