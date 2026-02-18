from enum import Enum
import os
import json
import base64
import subprocess
import platform
import time
import openai
from . import utils
import pyautogui

workdir = os.path.dirname(__file__)
screenshot_dir = os.path.join(workdir, "screenshots")

class Operationtype(Enum):
    TERMINAL="terminal"
    CLICK="click"

    def __str__(self):
        return self.value

class Prompt:
    template = """You are an advanced AI agent capable of controlling a computer (system: {system}, screen: {screen}) via terminal commands and mouse clicks.
Your objective is to complete the user's request: "{user_request}"

Current State:
- Last Terminal Output: {last_output}

Instructions:
1. Analyze the user request, the current screen (screenshot), and the last terminal output.
2. Decide the next step. It can be a terminal command or a mouse click.
3. If the task is completed, set "is_finished" to true. 
4. If the user request is empty or the task is done, respond with "is_finished": true.

Output Format (JSON only):
{{
    "operation_type": "terminal" | "click",
    "command": "string", 
    "is_finished": boolean,
    "reasoning": "string"
}}

- For "terminal": "command" should be the shell command.
- For "click": "command" should be "x,y,button,clicks" (e.g., "100,200,left,2"). 
  - "button" can be "left", "right", "middle". Default is "left".
  - "clicks" is the number of clicks. Default is 1.
- If "is_finished" is true, "operation_type" and "command" are ignored.
"""

class TerminalAgent:
    def __init__(self, prompt):
        self.prompt = prompt
        config = utils.get_config()
        terminal_agent_config = config.get("terminal_agent", {})
        self.model_name = terminal_agent_config.get("openai_model_name", "gpt-4o-mini")
        api_key = terminal_agent_config.get("openai_api_key")
        base_url = terminal_agent_config.get("openai_base_url")
        
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        
        self.history = []
        self.last_output = ""
        self.system_info = platform.platform()
        self.screen_info = pyautogui.size()
        self._take_screenshot()

    def _encode_image(self, image_path):
        if not image_path or not os.path.exists(image_path):
            return None
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def chat(self, messages, screenshot_path=None):
        """
        call llm
        
        :param messages: chat history
        :param screenshot_path: current pane screenshot path

        :return: str
        """
        api_messages = []
        for msg in messages:
            api_messages.append(msg.copy()) # Copy to avoid modifying original history
            
        if screenshot_path:
            base64_image = self._encode_image(screenshot_path)
            if base64_image:
                # Attach image to the last user message
                if api_messages and api_messages[-1]["role"] == "user":
                    content = api_messages[-1]["content"]
                    if isinstance(content, str):
                        api_messages[-1]["content"] = [
                            {"type": "text", "text": content},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                        ]
                    elif isinstance(content, list):
                        # Check if image already exists to avoid duplication if we are retrying or something
                        # But here we construct api_messages fresh each time
                        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}})

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=api_messages,
                response_format={"type": "json_object"}
            )
            print("LLM Response:", response)
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error calling LLM: {e}")
            return "{}"

    def get_terminal_operation(self, screenshot_path=None):
        """
        get_terminal_operation
        
        :param screenshot_path: Path to the screenshot file

        :return: (Operationtype, str, bool)
        """
        # Construct system prompt
        system_content = Prompt.template.format(
            system=self.system_info,
            screen=f"Width: {self.screen_info.width}, Height: {self.screen_info.height}",
            user_request=self.prompt,
            last_output=self.last_output if self.last_output else "None"
        )
        
        current_messages = [{"role": "system", "content": system_content}]
        
        # If history is empty, add the initial user request
        if not self.history:
            self.history.append({"role": "user", "content": f"Start task: {self.prompt}"})
            
        current_messages.extend(self.history)

        target_screenshot_path = screenshot_path if screenshot_path else self.latest_screenshot_path
        response_str = self.chat(current_messages, target_screenshot_path)
        
        try:
            response_json = json.loads(response_str)
            op_type_str = response_json.get("operation_type", "").lower()
            command = response_json.get("command", "")
            is_finished = response_json.get("is_finished", False)
            
            op_type = None
            if op_type_str == "terminal":
                op_type = Operationtype.TERMINAL
            elif op_type_str == "click":
                op_type = Operationtype.CLICK
            
            # Add assistant response to history
            self.history.append({"role": "assistant", "content": response_str})
            
            return command, op_type, is_finished
            
        except json.JSONDecodeError:
            print(f"Failed to parse JSON: {response_str}")
            return None, "", False

    def execute_operation(self, operation, operation_type):
        """
        execute_operation
        
        :param operation: command or click action description
        :param operation_type: Operationtype
        """
        output = ""
        if operation_type == Operationtype.TERMINAL:
            print(f"Executing Terminal Command: {operation}")
            try:
                result = subprocess.run(
                    operation, 
                    shell=True, 
                    capture_output=True, 
                    text=True,
                    timeout=60
                )
                output = result.stdout
                if result.stderr:
                    output += "\nError:\n" + result.stderr
            except Exception as e:
                output = f"Error executing command: {e}"
                
        elif operation_type == Operationtype.CLICK:
            print(f"Executing Click: {operation}")
            if pyautogui:
                try:
                    # Parse command: x,y,button,clicks
                    parts = operation.split(",")
                    x = int(parts[0])
                    y = int(parts[1])
                    button = parts[2].strip() if len(parts) > 2 else "left"
                    clicks = int(parts[3]) if len(parts) > 3 else 1
                    
                    pyautogui.click(x=x, y=y, button=button, clicks=clicks)
                    output = f"Clicked at {x}, {y}, button={button}, clicks={clicks}"
                    
                    # Take screenshot after click
                    self._take_screenshot()
                    
                except Exception as e:
                    output = f"Error executing click: {e}"
            else:
                output = "pyautogui not installed, cannot click."
        
        self.last_output = output

        print(f"Operation Output:\n{output}")
        
        # Add result to history
        self.history.append({
            "role": "user",
            "content": f"Operation executed. Output:\n{output}"
        })
    
    def _take_screenshot(self):
        timestamp = int(time.time())
        screenshot_filename = f"screenshot_{timestamp}.png"
        screenshot_path = os.path.join(screenshot_dir, screenshot_filename)
        if not os.path.exists(screenshot_dir):
            os.makedirs(screenshot_dir)
        pyautogui.screenshot(screenshot_path)
        self.latest_screenshot_path = os.path.abspath(screenshot_path)
        print(f"Screenshot saved as {self.latest_screenshot_path}")