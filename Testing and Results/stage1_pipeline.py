import json
import re
import time
from nltk.tokenize import sent_tokenize
from pydantic import BaseModel, Field, ValidationError
from typing import List

# Data Model
class StructuredRequirement(BaseModel):
    actors: List[str] = Field(default_factory=list)
    action: str = ""
    object: str = ""
    constraints: List[str] = Field(default_factory=list)
    missing_info: List[str] = Field(default_factory=list)

class Stage1Structuring:
    def __init__(self, llm_handler):
        self.llm = llm_handler
        self.clean_pattern = re.compile(r'[^a-zA-Z0-9\s\.,;\'\"]')
        self.vague_regex = re.compile(r'\b(user-friendly|fast|efficient|robust|optimal|easy|appropriate|better|clean|soon)\b', re.IGNORECASE)

    def _preprocess(self, text):
        text = text.lower()
        text = self.clean_pattern.sub('', text)
        sentences = sent_tokenize(text)
        return " ".join(sentences)

    def _construct_prompt(self, text):
        #  PROMPT:
        return f
    
        """You are a Requirements Extraction Tool. Extract fields from the text.

        OUTPUT JSON:
        {{
            "actors": ["List of actors"],
            "action": "Main action verb",
            "object": "Target object",
            "constraints": ["List of constraints"],
            "missing_info": ["List CRITICAL gaps only"]
        }}

        RULES:
        1. If the sentence has an Actor, Action, and Object, it is valid. Leave "missing_info" EMPTY.
        2. Only populate "missing_info" if the sentence is VAGUE (e.g. "make it fast") or missing a Subject.
        3. Do NOT ask for extra details like "security" or "frequency" if they are not mentioned.

        EXAMPLES:
        Input: "The System shall maintain a Patient Database."
        Output: {{ 
        "actors": ["System"], 
        "action": "maintain", 
        "object": "Patient Database", 
        "missing_info": [] }}

        Input: "System needs backup."
        Output: {{ 
        "actors": ["System"], 
        "action": "backup", 
        "missing_info": ["What data?", "Frequency?"] }}

        TARGET: "{text}"

        """

    def _call_llm_with_retry(self, prompt):
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                raw_json = self.llm.generate(prompt, temperature=0.1)
                if not raw_json: raise ValueError("Empty response")
                return StructuredRequirement(**raw_json)
            except Exception as e:
                time.sleep(1)
        return StructuredRequirement(missing_info=["Analysis Failed"])

    def _calculate_completeness(self, data: StructuredRequirement):

        # 1. Field Presence (Weight: 0.4)
        fields = [data.actors, data.action, data.object, data.constraints]
        present = sum(1 for f in fields if f)
        score_presence = present / 4.0

        # 2. Vague Term Penalty (Weight: 0.3)
        vague_hits = sum(1 for c in data.constraints if self.vague_regex.search(c))
        # Penalty is capped at 1.0 (if 2 or more vague terms found)
        penalty_vague = min(vague_hits * 0.5, 1.0)

        # 3. Missing Info Penalty (Weight: 0.3)
        # Penalty scales with number of missing items detected
        penalty_missing = min(len(data.missing_info) * 0.5, 1.0)

        # Final Calculation
        score = (0.4 * score_presence) + (0.3 * (1.0 - penalty_vague)) + (0.3 * (1.0 - penalty_missing))
        return round(max(score, 0.0), 2)
    def analyze(self, raw_text):
        clean_text = self._preprocess(raw_text)
        structured = self._call_llm_with_retry(self._construct_prompt(clean_text))
        score = self._calculate_completeness(structured)

        questions = []
        if score < 0.80:
            q_prompt = f"Requirement: '{clean_text}'. Gaps: {structured.missing_info}. Generate 2 clarification questions in JSON: {{'questions': ['Q1', 'Q2']}}"
            q_res = self.llm.generate(q_prompt)
            questions = q_res.get('questions', []) if q_res else ["Please clarify."]

        return {
            "text": clean_text,
            "structured": structured.model_dump(),
            "score": score,
            "questions": questions
        }
    

    