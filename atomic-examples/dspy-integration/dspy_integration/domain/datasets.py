"""
Datasets for movie genre classification benchmark.

This module contains the training and test datasets used to demonstrate
the differences between DSPy, Atomic Agents, and the combined approach.

Dataset Design:
- Training: 60 examples balanced across 6 genres (10 each)
- Test: 30 challenging examples testing edge cases

The test set is intentionally difficult, including:
- Sarcasm and irony
- Multi-genre signals (primary genre detection)
- Misleading genre keywords
- Subverted expectations
- Subtle/ambiguous signals
- Cultural references
"""

from itertools import zip_longest
from typing import List, TypedDict


class MovieExample(TypedDict):
    """Type definition for a movie review example."""

    review: str
    genre: str


# =============================================================================
# TRAINING DATASET (60 examples, 10 per genre)
# =============================================================================

_ACTION_EXAMPLES: List[MovieExample] = [
    {
        "review": "Non-stop car chases and explosions! The hero single-handedly took down an army.",
        "genre": "action",
    },
    {
        "review": "Martial arts sequences were incredible. The final fight scene was epic!",
        "genre": "action",
    },
    {
        "review": "She trained for 10 years to avenge her family. The fight choreography was poetry in motion.",
        "genre": "action",
    },
    {
        "review": "Bullets flying, buildings exploding, and our hero diving through glass windows. Peak adrenaline.",
        "genre": "action",
    },
    {
        "review": "The heist sequence had me on the edge of my seat. Tension and gunfights galore.",
        "genre": "action",
    },
    {
        "review": "Wow, another chosen one saving the world with a magic sword. Groundbreaking. Still epic though.",
        "genre": "action",
    },
    {
        "review": "This action film broke my heart. The hero's best friend didn't make it.",
        "genre": "action",
    },
    {
        "review": "High-octane from start to finish. The stunt work deserves every award.",
        "genre": "action",
    },
    {
        "review": "A revenge thriller with some of the best choreographed fights I've ever seen.",
        "genre": "action",
    },
    {
        "review": "Explosions, car chases, and a hero who refuses to give up. Classic action fare done right.",
        "genre": "action",
    },
]

_COMEDY_EXAMPLES: List[MovieExample] = [
    {
        "review": "I couldn't stop laughing! The jokes were hilarious and the timing was perfect.",
        "genre": "comedy",
    },
    {
        "review": "Witty dialogue and absurd situations had the whole theater in stitches.",
        "genre": "comedy",
    },
    {
        "review": "The jokes were so bad they were good. I hate that I loved this stupid movie.",
        "genre": "comedy",
    },
    {
        "review": "I cried watching this comedy because I related too much to the sad clown.",
        "genre": "comedy",
    },
    {
        "review": "A romantic comedy set during a zombie apocalypse. The jokes land even when heads don't.",
        "genre": "comedy",
    },
    {
        "review": "Slapstick humor meets clever wordplay. My cheeks hurt from laughing.",
        "genre": "comedy",
    },
    {
        "review": "The funniest movie I've seen all year. Every scene had at least one great gag.",
        "genre": "comedy",
    },
    {
        "review": "Dark comedy at its finest - you'll feel guilty for laughing but won't be able to stop.",
        "genre": "comedy",
    },
    {
        "review": "The comedic timing of the leads is impeccable. Chemistry-driven hilarity.",
        "genre": "comedy",
    },
    {
        "review": "Satirical genius. It skewers modern society while making you snort-laugh.",
        "genre": "comedy",
    },
]

_DRAMA_EXAMPLES: List[MovieExample] = [
    {
        "review": "A heart-wrenching story of loss and redemption. I cried for hours.",
        "genre": "drama",
    },
    {
        "review": "A slow burn exploration of grief and family dysfunction. Beautifully acted.",
        "genre": "drama",
    },
    {
        "review": "Yes there's a spaceship, but this is really about the captain dealing with his father's death.",
        "genre": "drama",
    },
    {
        "review": "It's set in space but it's really a courtroom drama about intergalactic law.",
        "genre": "drama",
    },
    {
        "review": "The performances were raw and honest. A meditation on what it means to be human.",
        "genre": "drama",
    },
    {
        "review": "Devastating. The final scene left me emotionally wrecked for days.",
        "genre": "drama",
    },
    {
        "review": "A character study that unfolds like a novel. Patient storytelling at its best.",
        "genre": "drama",
    },
    {
        "review": "The immigrant experience portrayed with such authenticity and grace.",
        "genre": "drama",
    },
    {
        "review": "Three generations of trauma, finally addressed. Cathartic and powerful.",
        "genre": "drama",
    },
    {
        "review": "Oscar-worthy performances in a story about ordinary people facing extraordinary circumstances.",
        "genre": "drama",
    },
]

_HORROR_EXAMPLES: List[MovieExample] = [
    {
        "review": "Terrifying! I slept with the lights on for a week after watching this.",
        "genre": "horror",
    },
    {
        "review": "Jump scares galore! The monster design was genuinely creepy.",
        "genre": "horror",
    },
    {
        "review": "Zombies attack! But the real horror is the breakdown of society and trust.",
        "genre": "horror",
    },
    {
        "review": "The horror movie made me laugh - those deaths were so creative!",
        "genre": "horror",
    },
    {
        "review": "Psychological terror that gets under your skin. No cheap scares, just dread.",
        "genre": "horror",
    },
    {
        "review": "The creature was nightmare fuel. I'm still seeing it when I close my eyes.",
        "genre": "horror",
    },
    {
        "review": "A haunted house movie that actually delivers. Genuinely unsettling atmosphere.",
        "genre": "horror",
    },
    {
        "review": "Gore-fest with a surprising amount of social commentary. Brutal and smart.",
        "genre": "horror",
    },
    {
        "review": "The slow build of dread was masterful. When it finally hit, I screamed.",
        "genre": "horror",
    },
    {
        "review": "Found footage done right. I had to keep reminding myself it wasn't real.",
        "genre": "horror",
    },
]

_SCIFI_EXAMPLES: List[MovieExample] = [
    {
        "review": "Set in 2150, the space battles and alien technology were mind-blowing.",
        "genre": "sci-fi",
    },
    {
        "review": "Time travel paradoxes and quantum physics made this a thinker.",
        "genre": "sci-fi",
    },
    {
        "review": "The robot fell in love with a human. Surprisingly touching for a sci-fi.",
        "genre": "sci-fi",
    },
    {
        "review": "The sci-fi premise was just an excuse for philosophical debates. Loved every second.",
        "genre": "sci-fi",
    },
    {
        "review": "Cyberpunk aesthetic meets thought-provoking questions about consciousness.",
        "genre": "sci-fi",
    },
    {
        "review": "The worldbuilding is incredible. Every detail of this future feels plausible.",
        "genre": "sci-fi",
    },
    {
        "review": "First contact done differently. The aliens were truly alien, not just humans with makeup.",
        "genre": "sci-fi",
    },
    {
        "review": "Hard sci-fi that doesn't dumb down the science. Refreshingly intelligent.",
        "genre": "sci-fi",
    },
    {
        "review": "Dystopian future that feels uncomfortably close to our present. Chilling and prescient.",
        "genre": "sci-fi",
    },
    {
        "review": "Space exploration with a philosophical bent. What does it mean to be alone in the universe?",
        "genre": "sci-fi",
    },
]

_ROMANCE_EXAMPLES: List[MovieExample] = [
    {
        "review": "The chemistry between the leads was electric. A beautiful love story.",
        "genre": "romance",
    },
    {
        "review": "Swoon-worthy moments and a happily ever after. Pure romantic bliss.",
        "genre": "romance",
    },
    {
        "review": "They met during an alien invasion. The world was ending but love found a way.",
        "genre": "romance",
    },
    {
        "review": "Enemies to lovers done perfectly. The tension was delicious.",
        "genre": "romance",
    },
    {
        "review": "A sweeping love story across decades. Their connection transcended time.",
        "genre": "romance",
    },
    {
        "review": "Second chance romance that made me believe in love again. Tissues required.",
        "genre": "romance",
    },
    {
        "review": "The slow burn was worth the wait. When they finally kissed, I cheered.",
        "genre": "romance",
    },
    {
        "review": "A meet-cute for the ages. Charming leads and witty banter throughout.",
        "genre": "romance",
    },
    {
        "review": "Forbidden love with actual stakes. Their sacrifice at the end broke me.",
        "genre": "romance",
    },
    {
        "review": "Holiday romance that's predictable but perfectly executed. Feel-good viewing.",
        "genre": "romance",
    },
]


# Interleave genres so the stages' 30/40-example prefixes retain all categories.
TRAINING_DATASET: List[MovieExample] = [
    example
    for row in zip_longest(
        _ACTION_EXAMPLES, _COMEDY_EXAMPLES, _DRAMA_EXAMPLES, _HORROR_EXAMPLES, _SCIFI_EXAMPLES, _ROMANCE_EXAMPLES
    )
    for example in row
    if example is not None
]


# =============================================================================
# TEST DATASET (30 challenging examples)
# =============================================================================

# Sarcasm & Irony (5 examples)
_SARCASM_TESTS: List[MovieExample] = [
    {
        "review": "Oh great, another movie where the hero walks away from explosions in slow motion. How original. Still watched it twice though.",
        "genre": "action",
    },
    {
        "review": "Groundbreaking stuff: man punches bad guys, gets the girl, saves the day. Revolutionary cinema. Loved every predictable second.",
        "genre": "action",
    },
    {
        "review": "I laughed so hard I cried. Then I just cried. Then I laughed again. What even was this movie?",
        "genre": "comedy",
    },
    {
        "review": (
            "Wow, they really subverted my expectations by doing exactly what I expected. "
            "The jokes were so obvious they circled back to funny."
        ),
        "genre": "comedy",
    },
    {
        "review": (
            "Another 'scary' movie where the characters make terrible decisions. "
            "At least the kills were creative. Actually terrifying creature design though."
        ),
        "genre": "horror",
    },
]

# Multi-Genre / Primary Genre Detection (6 examples)
_MULTIGENRE_TESTS: List[MovieExample] = [
    {
        "review": "The robot's sacrifice to save humanity made me sob uncontrollably. Beautiful storytelling set against a dystopian future.",
        "genre": "sci-fi",
    },
    {
        "review": "A serial killer falls in love with his next victim, but she's also a serial killer. Bloody and romantic.",
        "genre": "horror",
    },
    {
        "review": "Two detectives solve crimes while slowly falling for each other. The mystery was okay but I shipped them so hard.",
        "genre": "romance",
    },
    {
        "review": (
            "It's technically a war movie but really it's about two soldiers finding love "
            "in the trenches. The battle scenes support the love story."
        ),
        "genre": "romance",
    },
    {
        "review": "Space opera with a love triangle at its core. The laser battles are cool but I'm here for the drama between the three leads.",
        "genre": "sci-fi",
    },
    {
        "review": "Post-apocalyptic survival with a found family. The zombies are almost secondary to the human connections.",
        "genre": "drama",
    },
]

# Misleading Genre Signals (5 examples)
_MISLEADING_TESTS: List[MovieExample] = [
    {
        "review": "My heart was RACING the entire time! The courtroom scenes were absolutely EXPLOSIVE! Justice was served!",
        "genre": "drama",
    },
    {
        "review": "The alien invasion was just a backdrop for the family reconciliation story. Dad finally said he was proud.",
        "genre": "drama",
    },
    {
        "review": "Terrifyingly funny. The ghost just wanted to do stand-up comedy but kept accidentally scaring people.",
        "genre": "comedy",
    },
    {
        "review": "Action-packed emotional journey! By action I mean arguments, and by packed I mean I cried the whole time.",
        "genre": "drama",
    },
    {
        "review": "A thriller where the biggest twist was how much I ended up caring about these characters' relationships.",
        "genre": "romance",
    },
]

# Subverted Expectations (5 examples)
_SUBVERTED_TESTS: List[MovieExample] = [
    {
        "review": "Everyone dies at the end. Like, EVERYONE. But somehow it was the most romantic film I've ever seen.",
        "genre": "romance",
    },
    {
        "review": "The monster wasn't scary at all - it just wanted friends. I cried when they finally accepted it.",
        "genre": "drama",
    },
    {
        "review": "Started as a slasher, ended as a meditation on trauma and healing. The horror serves the character development.",
        "genre": "horror",
    },
    {
        "review": "What seemed like a rom-com setup became a profound exploration of self-love and independence. She didn't need him after all.",
        "genre": "drama",
    },
    {
        "review": "The funniest parts were unintentional. This action movie's dialogue is so bad it's become a comedy classic in my friend group.",
        "genre": "action",
    },
]

# Subtle / Ambiguous (5 examples)
_SUBTLE_TESTS: List[MovieExample] = [
    {
        "review": "Set in 2087, but really it's about loneliness. The AI companion understood him better than any human ever did.",
        "genre": "sci-fi",
    },
    {
        "review": "Quiet film about two people sharing a meal. Nothing happens and everything happens. Deeply moving.",
        "genre": "drama",
    },
    {
        "review": "The laughs come from pain, the pain comes from truth. A comedy that understands sadness intimately.",
        "genre": "comedy",
    },
    {
        "review": "Is it a horror movie if the monster is capitalism? Genuinely unsettling corporate satire.",
        "genre": "horror",
    },
    {
        "review": "They never say 'I love you' but every frame screams it. Visual storytelling at its most romantic.",
        "genre": "romance",
    },
]

# Cultural Context / Specific References (4 examples)
_CULTURAL_TESTS: List[MovieExample] = [
    {
        "review": "John Wick energy but make it about a retired chef defending his restaurant. Knife fights choreographed like ballet.",
        "genre": "action",
    },
    {
        "review": "Hereditary meets Little Miss Sunshine. Family dysfunction with supernatural undertones played for dark laughs.",
        "genre": "comedy",
    },
    {
        "review": "Blade Runner questions wrapped in a Her-style relationship. What is real, and does it matter?",
        "genre": "sci-fi",
    },
    {
        "review": "Pride and Prejudice but in space. The Darcy character is an alien prince and it absolutely works.",
        "genre": "romance",
    },
]

# Combine all test examples
TEST_DATASET: List[MovieExample] = (
    _SARCASM_TESTS + _MULTIGENRE_TESTS + _MISLEADING_TESTS + _SUBVERTED_TESTS + _SUBTLE_TESTS + _CULTURAL_TESTS
)
