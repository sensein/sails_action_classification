"""
prompts
"""


# QWEN2-VL-7B: Structured, direct, numbered format

QWEN2_ACTIVITY_PROMPT = """Describe the child's activity in this video.

Include these details:
1. Main action being performed (be specific: climbing, eating, reading, playing, etc.)
2. Objects or toys involved (blocks, books, food, playground equipment, presents, etc.)
3. Social context (alone, with parent, with sibling, with others)
4. Setting (kitchen, playground, living room, party, outdoors)

Provide a clear, specific description (1-2 sentences):"""

QWEN2_CONTEXT_PROMPT = """Activity: {activity_description}

Select the category number (1-8) that best matches:

1 = special occasion (birthday, celebration, presents, party)
2 = social interaction (talking, gesturing, dancing, interactive play - NOT repetitive)
3 = motor play (playground, climbing, running, ride-on toys)
4 = daily routine (eating, bathing, washing, dressing)
5 = toy play (blocks, dolls, small toys, puzzles)
6 = social routine (peek-a-boo, songs, tickles, repetitive games)
7 = other (unclear, ambiguous, transitional, location-only)
8 = book share (reading, looking at books)

Category number:"""


# TIMEZERO-7B: Clear, direct question-answer format (Qwen-based)

TIMEZERO_ACTIVITY_PROMPT = """What is happening in this video? 

Describe the child's main activity, including:
- The specific action (climbing, eating, reading, playing, opening, etc.)
- Any objects, toys, or equipment involved
- Who the child is interacting with (if anyone)
- The setting or environment

Provide a clear, detailed description:"""

TIMEZERO_CONTEXT_PROMPT = """Given this activity: {activity_description}

Classify into exactly ONE category by responding with only its number (1-8):

1 - special occasion (celebrations, birthdays, presents, parties)
2 - social interaction (talking, gesturing, dancing, interactive - not repetitive)
3 - motor play (playground, climbing, running, physical activity)
4 - daily routine (eating, bathing, washing, dressing)
5 - toy play (blocks, dolls, small toys, games)
6 - social routine (peek-a-boo, songs, tickles, repetitive games)
7 - other (unclear, ambiguous, transitional)
8 - book share (reading, looking at books)

Number:"""


# SMOLVLM2-500M: Simple, concise format (smaller model needs clarity)

SMOLVLM_ACTIVITY_PROMPT = """Describe what the child is doing in this video.

Be specific about:
- The action (climbing, eating, reading, playing, etc.)
- What objects are used (toys, books, food, playground, etc.)
- Who else is present (alone, with parent, with others)

Description:"""

SMOLVLM_CONTEXT_PROMPT = """Activity: {activity_description}

Pick ONE number (1-8):

1 = special occasion (birthday, presents, party)
2 = social interaction (talking, gesturing, dancing)
3 = motor play (playground, climbing, running)
4 = daily routine (eating, bathing, washing)
5 = toy play (blocks, dolls, toys)
6 = social routine (peek-a-boo, songs, tickles)
7 = other (unclear or transitional)
8 = book share (reading books)

Number:"""


# VIDEOLLAMA2-7B: Question-Answer format (trained on video QA)

VIDEOLLAMA_ACTIVITY_PROMPT = """Question: What activity is the child performing in this video?

Describe in detail:
- The specific action being performed
- Objects or equipment involved
- Social context (who is present)
- The setting

Answer:"""

VIDEOLLAMA_CONTEXT_PROMPT = """Question: The child is doing this activity: {activity_description}

Which category (1-8) does this belong to?

1. special occasion - celebrations, birthdays, presents, parties
2. social interaction - talking, gesturing, dancing (not repetitive games)
3. motor play - playground, climbing, running, physical activity
4. daily routine - eating, bathing, washing, dressing
5. toy play - playing with blocks, dolls, small toys
6. social routine - peek-a-boo, songs, tickles (repetitive games)
7. other - unclear, ambiguous, or transitional
8. book share - reading or looking at books

Answer (number only):"""

# LLAVA-NEXT-7B: Conversational, detailed instruction style

LLAVA_ACTIVITY_PROMPT = """Watch this video carefully and describe the child's main activity in a clear, specific phrase.

Focus on these details:
- What ACTION is the child doing? (climbing, eating, reading, playing with toys, opening presents, etc.)
- Who is WITH the child? (alone, with parent, with sibling, with adult)
- What OBJECTS are involved? (toys, books, food, playground equipment, presents)
- Where is this happening? (kitchen, playground, living room, outdoors, party)

Be SPECIFIC and DESCRIPTIVE - capture what's actually happening:
✓ Good examples: "climbing on playground equipment", "eating birthday cake with family", "adult reading picture book to child", "playing with toy blocks on floor"
✗ Bad examples: "playing" (too vague), "motor play" (category not description), "having fun" (not specific)

Look for visual cues:
- Birthday cakes, candles, presents → celebration actions
- Gestures, eye contact, pointing → social interaction
- Playground, climbing, running → physical actions
- Eating, bathing, washing → daily living actions
- Toys, blocks, dolls → play actions
- Books, pages → reading actions
- Peek-a-boo, songs, tickles → routine actions

Describe the activity in 1-2 sentences:"""

LLAVA_CONTEXT_PROMPT = """The child's activity is: "{activity_description}"

Based on what you saw in the video, classify this into exactly ONE category.
Respond with ONLY the category NUMBER (1-8).

CATEGORIES:

1. SPECIAL OCCASION: Celebrations, birthdays, holidays, parties
   Visual cues: Birthday cakes, candles, presents, gift wrapping, decorations, balloons
   Examples: "blowing out birthday candles", "opening christmas presents"

2. GENERAL SOCIAL COMMUNICATION INTERACTION: Talking, gesturing, dancing, interactive play with others (NOT repetitive routines)
   Visual cues: Eye contact, gesturing, pointing, turn-taking, animated interaction
   Examples: "talking with parent while gesturing", "dancing with sibling", "pointing and showing"

3. MOTOR PLAY: Large physical activities and gross motor movements
   Visual cues: Playground equipment (swings, slides), ride-on toys, running, jumping, climbing
   Examples: "climbing playground structure", "riding toy car", "going down slide"

4. DAILY ROUTINE: Activities of daily living
   Visual cues: Eating, drinking, bathing, washing, dressing, grooming
   Examples: "eating meal at table", "washing hands", "taking bath"

5. TOY PLAY: Playing with toys, games, or small objects
   Visual cues: Blocks, dolls, toy vehicles, stuffed animals, puzzles
   Examples: "playing with wooden blocks", "stacking toys", "playing with doll"

6. SOCIAL ROUTINE: Repetitive social interactions/games
   Visual cues: Peek-a-boo, pat-a-cake, tickle games, singing songs together
   Examples: "playing peek-a-boo with parent", "singing songs with caregiver"

7. OTHER: Ambiguous, transitional, or unclear activities
   Use when: Multiple activities, transitional moments, unclear video, location-only description
   Examples: "walking around house", "at aquarium looking around", "in car during drive"

8. BOOK SHARE: Reading or looking at books
   Visual cues: Books visible, pages turning, reading together
   Examples: "reading book with parent", "looking at picture book"

DECISION RULES:
- Birthday cake OR presents → 1
- Gesturing/pointing to person (not repetitive) → 2
- Playground equipment OR ride-on toy → 3
- Food/eating OR bathing/washing → 4
- Handheld toys/blocks → 5
- Peek-a-boo, songs, tickles → 6
- Books/pages → 8
- Unclear/transitional → 7

Respond with ONLY the number (1-8):"""


MODEL_PROMPTS = {
    'llava-next-7b': {
        'activity': LLAVA_ACTIVITY_PROMPT,
        'context': LLAVA_CONTEXT_PROMPT,
        'style': 'Conversational with detailed examples'
    },
    'qwen2-vl-7b': {
        'activity': QWEN2_ACTIVITY_PROMPT,
        'context': QWEN2_CONTEXT_PROMPT,
        'style': 'Structured numbered format'
    },
    'timezero-7b': {
        'activity': TIMEZERO_ACTIVITY_PROMPT,
        'context': TIMEZERO_CONTEXT_PROMPT,
        'style': 'Direct question-answer (Qwen-based)'
    },
    'smolvlm2-500m': {
        'activity': SMOLVLM_ACTIVITY_PROMPT,
        'context': SMOLVLM_CONTEXT_PROMPT,
        'style': 'Simple and concise (smaller model)'
    },
    'videollama2-7b': {
        'activity': VIDEOLLAMA_ACTIVITY_PROMPT,
        'context': VIDEOLLAMA_CONTEXT_PROMPT,
        'style': 'Video QA format'
    }
}


def get_prompts(model_name=None, version='v1'):
    """
    Get prompts for a specific model
    
    Args:
        model_name: Name of the VLM model (e.g., 'llava-next-7b')
                   If None, defaults to LLaVA prompts for backward compatibility
        version: Prompt version (currently only 'v1' supported)
    
    Returns:
        tuple: (activity_prompt, context_prompt)
    """
    if version != 'v1':
        raise ValueError(f"Only version 'v1' is currently supported, got '{version}'")
    
    # Default to LLaVA if no model specified (backward compatibility)
    if model_name is None:
        model_name = 'llava-next-7b'
    
    if model_name not in MODEL_PROMPTS:
        available = ", ".join(MODEL_PROMPTS.keys())
        raise ValueError(f"Unknown model '{model_name}'. Available: {available}")
    
    prompts = MODEL_PROMPTS[model_name]
    return prompts['activity'], prompts['context']


def list_models():
    """List all models with available prompts"""
    print("\nAvailable Model-Specific Prompts:")
    for model_name, info in MODEL_PROMPTS.items():
        print(f"\n{model_name}: {info['style']}")
    print()


def show_prompts(model_name):
    """Display the prompts for a specific model"""
    if model_name not in MODEL_PROMPTS:
        print(f"Error: Unknown model '{model_name}'")
        list_models()
        return
    
    print(f"Prompts for: {model_name}")
    print(f"Style: {MODEL_PROMPTS[model_name]['style']}")
    
    activity, context = get_prompts(model_name)
    print(activity)
    print(context)
    print()


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        show_prompts(sys.argv[1])
    else:
        list_models()
        print("\nUsage: python prompts.py <model_name>")
        print("Example: python prompts.py llava-next-7b")