import re

SUPPORTED_LANGUAGES = {"EN", "ID"}

def parse_prompt(prompt_text):
    result = {
        "language": None,
        "genre": None,
        "tempo": None,
        "mood": None,
        "description": "",
        "valid": True,
        "error": None,
    }
    tags = re.findall(r"\[(\w+):\s*([^\]]+)\]", prompt_text)
    tag_dict = {key.upper(): value.strip().upper() for key, value in tags}
    lang = tag_dict.get("LANG")
    if lang is None:
        result["valid"] = False
        result["error"] = "Missing [LANG: ...] tag."
        return result
    if lang not in SUPPORTED_LANGUAGES:
        result["valid"] = False
        result["error"] = "Unsupported language."
        return result
    result["language"] = lang
    result["genre"] = tag_dict.get("GENRE", "DJ")
    try:
        tempo = int(tag_dict.get("TEMPO", "120"))
        result["tempo"] = max(40, min(240, tempo))
    except:
        result["tempo"] = 120
    result["mood"] = tag_dict.get("MOOD", "NEUTRAL")
    bracket_end = 0
    for match in re.finditer(r"\[[^\]]*\]", prompt_text):
        bracket_end = max(bracket_end, match.end())
    result["description"] = prompt_text[bracket_end:].strip()
    return result

def format_prompt(language, genre="DJ", tempo=120, mood="ENERGETIC", description=""):
    if language not in SUPPORTED_LANGUAGES:
        raise ValueError("Unsupported language.")
    prompt = f"[LANG: {language}] [GENRE: {genre.upper()}] [TEMPO: {tempo}] [MOOD: {mood.upper()}]"
    if description:
        prompt += " " + description
    return prompt
