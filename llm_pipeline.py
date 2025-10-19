import json
import copy
import logging
import os
from typing import Dict, List, Any, Union, Tuple, Optional
from openai import OpenAI
from anthropic import Anthropic
import re

# Constants
OPENAI_MODEL = "gpt-4.1-nano-2025-04-14"
ANTHROPIC_MODEL = "claude-3-5-haiku-latest"

PROMPT_FILE = os.path.join(os.path.dirname(__file__), "prompts.json")

# Centralized per-task output token cap configuration
LLM_TASK_CONFIG = {
    "generate_response_with_info_subset": {"max_tokens": 500},
    "keypoint_relevance_scoring": {"max_tokens": 100},
    "generalize_keypoints": {"max_tokens": 500},
    "separate_keypoints": {"max_tokens": 500},
}

class LLMClient:
    """Base class for LLM clients"""
    def __init__(self, model: str):
        self.model = model

    def get_response(self, messages: List[Dict[str, str]], max_tokens: Optional[int] = None) -> str:
        raise NotImplementedError

class OpenAIClient(LLMClient):
    def __init__(self, model=OPENAI_MODEL):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model
        logging.info(f"OpenAI Model: {model}")

    def get_response(self, messages: List[Dict[str, str]], max_tokens: Optional[int] = None) -> str:
        if max_tokens is None:
            max_tokens = 500
        if self.model == "gpt-5-mini":
            response = self.client.chat.completions.create(
                model=self.model,              # e.g., "gpt-5" or another chat-capable model
                messages=messages              # list[{"role":"system"|"user"|"assistant", "content": "..."}]
            )
            content = response.choices[0].message.content
            return content

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages, # type: ignore
            temperature=0.0,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content # type: ignore

class AnthropicClient(LLMClient):
    def __init__(self, model=ANTHROPIC_MODEL):
        self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.model = model
        logging.info(f"Anthropic Model: {model}")

    def get_response(self, messages: List[Dict[str, str]], max_tokens: Optional[int] = None) -> str:
        if max_tokens is None:
            max_tokens = 500
        # Extract system message if present, otherwise use empty string
        system_message = ""
        user_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            else:
                user_messages.append(msg)
        # Create messages list in Anthropic format (only user/assistant messages)
        anthropic_messages = [
            {"role": msg["role"], "content": msg["content"]}
            for msg in user_messages
        ]
        # if self.model == "claude-sonnet-4-20250514":
        #     response = self.client.messages.create(
        #         model=self.model,
        #         system=system_message,
        #         messages=anthropic_messages,  # type: ignore
        #         max_tokens=max_tokens,
        #         temperature=0.0,
        #         thinking=
        #             {
        #             "type": "enabled",
        #             "budget_tokens": 1000
        #             }
        #     )
        # else:
        response = self.client.messages.create(
            model=self.model,
            system=system_message,
            messages=anthropic_messages,  # type: ignore
            max_tokens=max_tokens,
            temperature=0.0
        )
        content = response.content[0].text # type: ignore
        return content
    
class PromptLoader:
    def __init__(self, llm_client: LLMClient):
        """Initialize the prompt loader with an LLM client.
        Args:
            llm_client: Instance of OpenAIClient or AnthropicClient
        """
        with open(PROMPT_FILE, 'r') as f:
            self.prompt_templates = json.load(f)
        self.llm = llm_client

    def get_prompt(self, task_name: str, sample: Dict[str, Any]) -> List[Dict[str, str]]:
        """Get the formatted prompt for a given task and sample.
        Args:
            task_name: Name of the task/prompt template to use
            sample: Dictionary containing variables to fill in the prompt
        Returns:
            List of message dictionaries ready to send to the LLM
        """
        if task_name not in self.prompt_templates:
            raise ValueError(f"Task '{task_name}' not found in prompt templates.")
        template = self.prompt_templates[task_name]["message_template"]
        filled_messages = copy.deepcopy(template)
        # Variable substitution
        sample = sample.copy()
        if 'sources' in sample:
            sample['info_subset'] = sample['sources']
        for msg in filled_messages:
            msg["content"] = msg["content"].format(**sample)
        return filled_messages

    def call_llm(self, messages: List[Dict[str, str]], max_tokens: Optional[int] = None) -> str:
        """Call the LLM with the given messages.
        Args:
            messages: List of message dictionaries
        Returns:
            Raw response from the LLM
        """
        return self.llm.get_response(messages, max_tokens=max_tokens)

    def parse_response(self, task_name: str, response: str) -> Union[str, Dict[str, Any], List[float], Tuple[float, str]]:  # type: ignore
        """Parse the LLM response based on the task type.
        Args:
            task_name: Name of the task that generated the response
            response: Raw response from the LLM
        Returns:
            Parsed response in the appropriate format for the task
        """
        if task_name == "keypoint_relevance_scoring": 
            # Check if response indicates no key points
            if "no key points" in response.lower() or "no specific key points" in response.lower():
                return 0.0, "No key points available"

            # If already parsed (tuple), just return it
            if isinstance(response, tuple) and len(response) == 2:
                return response

            # Parse explanation and score (explanation comes first, but may be on same line as score)
            try:
                # Use regex to extract EXPLANATION and SCORE from anywhere in the response
                explanation_match = re.search(r"EXPLANATION:\s*(.*?)(?:\n|SCORE:|$)", response, re.DOTALL)
                score_match = re.search(r"SCORE:\s*([0-9.]+)", response)
                explanation = explanation_match.group(1).strip() if explanation_match else "No explanation provided"
                score = float(score_match.group(1)) if score_match else 0.0
                return score, explanation
            except Exception:
                print(f"Warning: Could not parse relevance score and explanation from response: {response}")
                return 0.0, "Failed to parse response"

        elif task_name == "generate_response_with_info_subset": 
            # Return raw response for simple generation tasks
            return response
            
        elif task_name == "generalize_keypoints": 
            try:
                # Use regex to robustly extract the two sections
                justification_match = re.search(r"REASONING:\s*(.*?)(?=\n*REFINED KEYPOINTS:|$)", response, re.DOTALL | re.IGNORECASE)
                keypoints_match = re.search(r"REFINED KEYPOINTS:\s*(.*)", response, re.DOTALL | re.IGNORECASE)
                
                justification = justification_match.group(1).strip() if justification_match else ""
                key_points = []
                if keypoints_match:
                    keypoints_text = keypoints_match.group(1).strip()
                    if keypoints_text:
                        key_points = [line.strip() for line in keypoints_text.split('\n') if line.strip()]

                return {
                    'keypoints': key_points,
                    'justification': justification
                }
            except Exception as e:
                print(f"Warning: Could not parse keypoints and justification from response: {response}")
                return {'keypoints': [], 'justification': ""}

        elif task_name == "separate_keypoints":
            try:
                # Use regex to robustly extract the two sections
                justification_match = re.search(r"JUSTIFICATION:\s*(.*?)(?=\n*KEYPOINTS:|$)", response, re.DOTALL | re.IGNORECASE)
                keypoints_match = re.search(r"KEY POINTS:\s*(.*)", response, re.DOTALL | re.IGNORECASE)
                
                justification = justification_match.group(1).strip() if justification_match else ""
                key_points = []
                if keypoints_match:
                    keypoints_text = keypoints_match.group(1).strip()
                    if keypoints_text:
                        key_points = [line.strip() for line in keypoints_text.split('\n') if line.strip()]

                return {
                    'keypoints': key_points,
                    'justification': justification
                }
            except Exception as e:
                print(f"Warning: Could not parse keypoints and justification from response: {response}")
                return {'keypoints': [], 'justification': ""}
        
    def run_task(self, task_name: str, sample: Dict[str, Any], max_tokens: Optional[int] = None) -> Any:
        """Run a task with the given sample data.
        Args:
            task_name: Name of the task to run
            sample: Dictionary containing variables for the prompt
        Returns:
            Parsed response in the appropriate format for the task
        """
        messages = self.get_prompt(task_name, sample)

        # Use LLM_TASK_CONFIG for max_tokens if not explicitly provided
        if max_tokens is None:
            from llm_pipeline import LLM_TASK_CONFIG
            max_tokens = LLM_TASK_CONFIG.get(task_name, {}).get("max_tokens", 500)
        response = self.call_llm(messages, max_tokens=max_tokens)

        return self.parse_response(task_name, response)

def create_llm_pipeline(provider: str = "anthropic", **kwargs) -> PromptLoader:
    """Create a PromptLoader instance with the specified LLM provider.
    Args:
        provider: "openai", "anthropic", or "ollama"
        **kwargs: Additional arguments to pass to the LLM client constructor 
    Returns:
        Configured PromptLoader instance
    """
    if provider == "openai":
        client = OpenAIClient(**kwargs)
    elif provider == "anthropic":
        client = AnthropicClient(**kwargs)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")
    return PromptLoader(client)

