"""
Standardized Prompts
Version-controlled prompts for activity prediction and context classification
"""

# Activity Prediction Prompt (Stage 1)
ACTIVITY_PROMPT_V1 = """Watch this video carefully and describe the child's main activity in a clear, specific phrase (5-10 words).

This should be MORE DETAILED than just the category - describe the SPECIFIC ACTIONS happening.

Focus on:
- What ACTION is the child doing? (use specific verbs: climbing, eating, reading, playing, opening, tumbling, tickling, etc.)
- Who is WITH the child? (alone, with parent, with sibling, with family, with adult)
- What OBJECTS are involved? (toys, books, food, playground equipment, presents, etc.)
- Where is this HAPPENING? (kitchen, playground, living room, outdoors, party setting)
- What is the MANNER of interaction? (laughing, engaging, singing, reading to child, etc.)

Be SPECIFIC and DESCRIPTIVE - capture the details of what's actually happening:
✓ GOOD: "gymnastics and tumbling with parent" (motor play with specific action)
✓ GOOD: "adult reading picture book to child on couch" (book share with specific detail)
✓ GOOD: "laughing, tickling, engaging with adult" (social routine with emotional detail)
✓ GOOD: "climbing on playground equipment with parent"
✓ GOOD: "eating birthday cake with family at party"
✓ GOOD: "opening christmas presents with siblings"
✓ GOOD: "playing with toy blocks on floor alone"
✓ GOOD: "playing peekaboo and laughing with parent"
✓ GOOD: "singing songs together with caregiver"
✓ GOOD: "stacking colorful blocks at table"
✓ GOOD: "eating lunch with spoon at high chair"

✗ BAD: "playing" (too vague - what kind of play? with what? with whom?)
✗ BAD: "motor play" (this is the category, not the activity description)
✗ BAD: "having fun" (too general - what are they doing?)
✗ BAD: "at home" (location, not an activity)
✗ BAD: "book" (object only, not describing the action)

Look for visual cues to guide your description:
- Birthday cakes, candles, presents → describe celebration actions (e.g., "blowing candles", "opening presents")
- Conversation gestures, eye contact, pointing → describe social interaction (e.g., "pointing and gesturing to parent")
- Playground, climbing, running, ride-on toys → describe physical actions (e.g., "climbing slide", "tumbling on mat")
- Eating, bathing, washing, dressing → describe daily living actions (e.g., "eating pasta with fork", "washing hands")
- Toys, blocks, dolls, games → describe play actions (e.g., "stacking blocks", "playing with toy car")
- Books, reading, pages → describe reading actions (e.g., "adult reading book to child", "turning pages")
- peek-a-boo, songs, tickles → describe routine actions (e.g., "playing peek-a-boo and laughing", "singing together")

Remember: Describe what you SEE happening, not just the category it belongs to.

Describe the activity:"""


# Context Classification Prompt (Stage 2)
CONTEXT_PROMPT_V1 = """The child's activity is: "{activity_description}"

Based on what you SAW in the video, classify this into exactly ONE category.
Respond with ONLY the category NUMBER (1-8).

CATEGORIES (8 total, visual-only):

1. SPECIAL OCCASION: Celebrations, holidays, birthdays, special events
   Visual cues: Birthday cakes, candles, presents being opened, gift wrapping, decorations, party hats, balloons, holiday decorations
   Setting: Party environments, festive gatherings, celebration contexts
   Examples: "blowing out birthday candles", "opening christmas presents", "unwrapping gifts with family"
   Key indicators: birthday, candles, present, gift, christmas, party, celebration, opening, unwrapping

2. GENERAL SOCIAL COMMUNICATION INTERACTION: Verbal/vocal interaction OR social games with others (NOT repetitive routines)
   Visual cues: Child facing another person, gesturing, pointing, eye contact, turn-taking, interactive play patterns, dancing together
   Body language: Animated gestures, social engagement postures, shared attention, conversation-like interactions
   NOT if primarily focused on toys/objects (use toy play)
   NOT if repetitive routine like peek-a-boo or songs (use social routine - category 6)
   Examples: "talking with parent while gesturing", "dancing with sibling", "pointing and gesturing", "interactive conversation"
   Key indicators: talking, pointing, gesturing, together, eye contact, facing person, interactive, dancing, conversation

3. MOTOR PLAY: Large-scale physical activities and gross motor movements
   Visual cues: Playground equipment (swings, slides, climbing structures), large open spaces, running, jumping, climbing
   Equipment: Ride-on toys (toy cars, scooters, tricycles), balance beams, tunnels, large physical play items
   NOT small handheld toys (use toy play)
   Examples: "climbing playground structure", "riding toy car outdoors", "going down slide", "swinging on swing set"
   Key indicators: playground, climbing, slide, swing, riding, running, jumping, physical, outdoor play equipment

4. DAILY ROUTINE: Activities of daily living (eating, hygiene, dressing, cleaning)
   Visual cues: Kitchen/bathroom settings, food items, plates/bowls, water/bathing, toothbrush, clothing changes
   Actions: Eating motions, drinking, washing, bathing, dressing, grooming
   Examples: "eating meal at table", "washing hands at sink", "taking bath", "brushing teeth"
   Key indicators: eating, drinking, food, meal, bath, washing, brushing, dressing, table, bowl, plate

5. TOY PLAY: Playing with toys, games, or small objects
   Visual cues: Toys visible (blocks, dolls, toy vehicles, stuffed animals, puzzles, toy instruments)
   Focus: Child engaged with handheld/tabletop toys, manipulating small objects
   NOT large ride-on toys (use motor play)
   Examples: "playing with wooden blocks", "stacking toys", "playing with doll", "toy xylophone"
   Key indicators: toy, blocks, doll, stuffed animal, puzzle, playing with small objects, manipulating toys

6. SOCIAL ROUTINE: Repetitive social interactions with another person
   Visual cues: Back-and-forth patterns, repeated actions, anticipation and response patterns
   Activities: Peek-a-boo, pat-a-cake, tickle games, singing songs together, repetitive social games
   Focus: REPETITIVE nature and social engagement (not one-time interactions)
   Examples: "playing peek-a-boo with parent", "singing songs with caregiver", "tickle game with sibling", "pat-a-cake with adult"
   Key indicators: peek-a-boo, tickle, song, singing, repetitive game, pat-a-cake, social game, routine

7. OTHER: Ambiguous, transitional, or activities that don't clearly fit above categories
   Use this when:
   - Multiple different activities happening at once (cannot identify primary activity)
   - Transitional moments (walking between locations, getting ready to start activity)
   - Activity is unclear or video quality too poor to determine
   - Location-focused rather than activity-focused (e.g., "at aquarium", "at beach", "in car")
   - Child is just wandering, observing, or waiting (no clear engaged activity)
   - Activity doesn't match any above categories AND you're genuinely uncertain
   
   Examples: "walking around house", "at aquarium looking around", "in car during drive", "transitioning between rooms"
   Key indicators: unclear, ambiguous, multiple activities, transitional, observing, location-only description
   
   IMPORTANT: Only use "other" if you are GENUINELY uncertain. If the activity reasonably fits ANY of the 6 main categories (1-6, 8), choose that category instead.

8. BOOK SHARE: Engaged with books (reading or looking at books)
   Visual cues: Books visible, pages turning, child looking at book, sitting with book
   Settings: Often on couch/chair, adult may be reading to child, child holding book
   Examples: "reading book with parent", "looking at picture book", "turning pages of book"
   Key indicators: book, reading, pages, story, looking at book

DECISION RULES (use visual evidence):
- See birthday cake OR presents being opened → 1 (special occasion)
- See child gesturing/pointing toward person WITHOUT toys and NOT repetitive → 2 (general social communication interaction)
- See playground equipment OR child on ride-on toy → 3 (motor play)
- See food/eating OR bathroom/washing → 4 (daily routine)
- See handheld toys/blocks → 5 (toy play)
- See repetitive social games (peek-a-boo, songs, tickles) → 6 (social routine)
- See books/pages → 8 (book share)
- Activity is genuinely unclear/ambiguous/transitional → 7 (other)

CRITICAL: The activity description says "{activity_description}". 
- If this describes peek-a-boo, tickles, songs, or repetitive social games → 6 (social routine)
- If this describes a clear location with no specific activity (e.g., "at beach", "at aquarium") → 7 (other)
- If this describes multiple simultaneous activities → 7 (other)
- If this describes transition/wandering → 7 (other)
- If this describes books/reading → 8 (book share)
- Otherwise, choose the MOST SPECIFIC category that fits (1-6)

Respond with ONLY the number (1-8), nothing else:"""


# Prompt versions dictionary (for easy switching)
PROMPT_VERSIONS = {
    'v1': {
        'activity': ACTIVITY_PROMPT_V1,
        'context': CONTEXT_PROMPT_V1,
        'description': 'Original optimized prompts with detailed visual cues (8 categories: special occasion, general social communication, motor play, daily routine, toy play, social routine, other, book share)'
    },
    # 'v2': {...},
    # 'v3': {...},
}


def get_prompts(version='v1'):
    """Get prompts for a specific version"""
    if version not in PROMPT_VERSIONS:
        available = ", ".join(PROMPT_VERSIONS.keys())
        raise ValueError(f"Prompt version '{version}' not found. Available: {available}")
    
    return (
        PROMPT_VERSIONS[version]['activity'],
        PROMPT_VERSIONS[version]['context']
    )


def list_prompt_versions():
    """List all available prompt versions"""
    print("\nAvailable Prompt Versions:")
    print("="*80)
    for version, info in PROMPT_VERSIONS.items():
        print(f"\n{version}: {info['description']}")
    print()
