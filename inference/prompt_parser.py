import re

SUPPORTED_LANGUAGES = {"EN", "ID"}

def parse_prompt(prompt_text):
    result = {
        "language": "ID",
        "genre": "DJ",
        "tempo": 120,
        "mood": "VOCAL",
        "description": prompt_text.strip(),
        "valid": True,
        "error": None,
    }
    
    tags = re.findall(r"\[(\w+):\s*([^\]]+)\]", prompt_text)
    if not tags:
        if any(w in prompt_text.upper() for w in ["CREATE", "MAKE", "SONG", "MUSIC"]):
            result["language"] = "EN"
        return result

    tag_dict = {key.upper(): value.strip().upper() for key, value in tags}
    lang = tag_dict.get("LANG")
    if lang in SUPPORTED_LANGUAGES:
        result["language"] = lang
    
    result["genre"] = tag_dict.get("GENRE", "DJ")
    try:
        tempo = int(tag_dict.get("TEMPO", "120"))
        result["tempo"] = max(40, min(240, tempo))
    except:
        result["tempo"] = 120
    
    result["mood"] = tag_dict.get("MOOD", "VOCAL")
    
    bracket_end = 0
    for match in re.finditer(r"\[[^\]]*\]", prompt_text):
        bracket_end = max(bracket_end, match.end())
    result["description"] = prompt_text[bracket_end:].strip()
    
    return result

def format_prompt(language, genre="DJ", tempo=120, mood="VOCAL", description=""):
    prompt = f"[LANG: {language}] [GENRE: {genre.upper()}] [TEMPO: {tempo}] [MOOD: {mood.upper()}]"
    if description:
        prompt += " " + description
    return prompt
