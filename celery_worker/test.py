import redis
from celery import Celery
import time
import os
from itertools import product
import uuid
import json
from rouge_score import rouge_scorer
import logging
import requests
import random

CELERY_BROKER_URL = os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0")
CELERY_RESULT_BACKEND = os.environ.get("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")

celery = Celery("tasks", broker=CELERY_BROKER_URL, backend=CELERY_RESULT_BACKEND)
r = redis.Redis(host='localhost', port=6379, decode_responses=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RYE_TEXT_FILE = os.path.join(BASE_DIR, "Rye.txt")
TRANSCRIPT_TEXT_FILE = os.path.join(BASE_DIR, "transcript_translated.txt")

DATA_URL = "http://ai.rndl.ru:5017/api/data"

params = {
    "chunk_size":[1500, 2000, 5500, 6000, 6500, 7000, 7500, 8000], 
    "chunk_overlap": [100, 200, 300], 
    "temp_chunk": [0.3, 0.5], 
    "temp_final": [0.5, 0.7]
    }

combinations = list(product(
    params["chunk_size"],
    params['chunk_overlap'],
    params['temp_chunk'],
    params["temp_final"]
))

#combinations = random.sample(combinations, 20) # complete random 20 tests
 
with open(RYE_TEXT_FILE, "r", encoding="utf-8") as file:
    RYE_TEXT = file.read()

with open(TRANSCRIPT_TEXT_FILE, "r", encoding="utf-8") as file:
    TRANSCRIPT_TEXT = file.read()

def run_eval(data):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    reference = """# The Catcher in the Rye - Comprehensive Analysis

I've reviewed the iconic novel "The Catcher in the Rye" by J.D. Salinger. What follows is a detailed analysis that preserves the integrity and nuance of the original text.

## Factual Summary

"The Catcher in the Rye" follows 16-year-old Holden Caulfield over approximately three days after he has been expelled from Pencey Prep School for academic failure. The narrative begins with Holden at Pencey, watching a football game from a distance rather than attending it. He visits his history teacher Mr. Spencer, who lectures him about his academic failures. Back at the dormitory, Holden interacts with his neighbors Ackley and his roommate Stradlater. When Stradlater goes on a date with Jane Gallagher, a girl Holden cares deeply about, Holden becomes agitated and gets into a physical fight with Stradlater upon his return.

Deciding to leave Pencey immediately rather than wait until Wednesday, Holden packs his bags and takes a late-night train to New York City. He checks into the Edmont Hotel, where he witnesses "perverts" from his window. Lonely, he calls Faith Cavendish for companionship but is rejected. He goes to the Lavender Room at his hotel, where he dances with three tourists from Seattle but feels depressed by the encounter.

The next day, Holden arranges a date with Sally Hayes, meets two nuns at breakfast, and buys a record "Little Shirley Beans" for his sister Phoebe. He attends a play with Sally, attempts to discuss running away together, insults her when she rejects his plan, and then meets his former classmate Carl Luce for drinks. After getting drunk alone, Holden visits Central Park to see the ducks in the lagoon, then sneaks into his family's apartment to visit Phoebe. She becomes upset upon learning he's been expelled, but they dance and talk until their parents return home, prompting Holden to leave.

He visits his former teacher Mr. Antolini, who offers advice and a place to sleep. Holden is disturbed when he wakes to find Mr. Antolini patting his head, interpreting it as a homosexual advance. He leaves and spends the night in Grand Central Station. The next day, feeling ill and distraught, Holden decides to leave society and hitchhike west. Before leaving, he arranges to meet Phoebe to say goodbye. Instead of accepting his departure, she appears with a suitcase, wanting to go with him. Holden refuses but agrees not to leave. The novel ends with them at the Central Park carousel, with Holden watching Phoebe ride in the rain while experiencing a moment of genuine happiness.

In the final chapter, Holden reveals he is narrating from some form of institution where he's recovering, but doesn't share details about what happened after the carousel scene.

## Thematic Analysis

### Authenticity vs. Phoniness
Throughout the novel, Holden categorizes people and their behaviors as either authentic or "phony." His preoccupation with phoniness—superficiality, pretension, and hypocrisy—drives his cynicism and alienation. He sees phoniness in his schoolmates, teachers, adults in nightclubs, and nearly all social institutions. His resistance to phoniness represents his broader resistance to adult conformity and compromise.

### Innocence and Its Protection
Holden idealizes childhood innocence, symbolized by his desire to be "the catcher in the rye," saving children from falling off a cliff into adulthood. His deep affection for Phoebe, his memories of Allie, and his concern over the "Fuck you" graffiti in elementary schools all reflect his wish to preserve innocence against corruption.

### Alienation and Isolation
Holden's consistent alienation stems from his inability to connect with others on his terms. He repeatedly attempts human connection (calling old friends, arranging dates, talking to strangers) but sabotages these interactions through cynicism or inappropriate behavior, reinforcing his isolation.

### Grief and Trauma
Unresolved grief over his brother Allie's death underlies much of Holden's emotional distress. His breakdown is precipitated by accumulated losses: Allie's death, academic failure, disconnection from family, and his inability to protect innocence in a world he sees as corrupt.

### The Complexity of Growing Up
The novel portrays adolescence as a painful transition filled with confusion, contradiction, and loss. Holden's resistance to maturity, alongside his inadvertent steps toward it, depicts growing up as both inevitable and traumatic.

## Symbolic and Metaphorical Interpretations

### The Red Hunting Hat
Holden's red hunting hat serves as a multifaceted symbol of his individuality, vulnerability, and connection to childhood. He puts it on when he wants to isolate himself or when he feels most authentic, and gives it to Phoebe when he feels protective of her. The hat's unusual appearance mirrors Holden's non-conformity, while its bright red color connects symbolically to Allie's red hair.

### The Ducks in Central Park
Holden's recurring question about where the ducks go during winter represents his deeper concerns about displacement and change. Just as he wonders about the ducks' survival in changing circumstances, he grapples with his own place in a world transitioning from childhood to adulthood.

### The Museum of Natural History
The museum symbolizes Holden's desire for stasis and preservation. He values how everything in the museum stays exactly the same while the observer changes, reflecting his wish to halt the inevitable passage of time and the corruption of innocence.

### The Carousel
The carousel represents the cyclical nature of life and the balance between change and constancy. While the carousel goes around and around (constant change), it also plays the same music it did in Holden's childhood (permanence). Phoebe's willingness to reach for the gold ring symbolizes healthy risk-taking, contrasting with Holden's fear of change.

### "The Catcher in the Rye" Image
Holden's misinterpretation of the Burns poem creates the central metaphor of the novel—his fantasy of standing at the edge of a cliff, catching children before they fall into adulthood. This powerful image articulates his desire to protect innocence from the corruption he associates with maturity.

## Pivotal Narrative Components

1. **Holden's Fight with Stradlater** - This confrontation catalyzes Holden's departure from Pencey and begins his physical and psychological journey. The fight represents Holden's protective feelings toward Jane and his resentment of Stradlater's casual sexuality and social success.

2. **The Encounter with Sunny, the Prostitute** - This interaction reveals Holden's complicated relationship with sexuality. Despite arranging for a prostitute, he only wants to talk, showing his loneliness, naivety, and the gap between his desires and actions. The subsequent confrontation with Maurice demonstrates how ill-equipped Holden is to navigate adult situations.

3. **The Museum of Natural History Sequence** - Holden's reflection on the museum reveals his desire for a world where "certain things they should stay the way they are." The discovery of the "Fuck you" graffiti in this environment of preserved innocence represents his central fear—the inevitable corruption of purity.

4. **The Mr. Antolini Episode** - This encounter represents Holden's final attempt to seek guidance from adults before his breakdown. Mr. Antolini offers the most substantive advice in the novel, but the ambiguous incident of physical contact undermines this potential connection, reinforcing Holden's isolation.

5. **The Carousel Scene** - This climactic scene brings resolution to Holden's emotional journey. Watching Phoebe reach for the gold ring while accepting the risk of falling marks Holden's first moment of genuine happiness and his acceptance that children must be allowed to grow up, despite the risks.

## Character Relationships and Interactions

### Holden and Family Members

**Phoebe Caulfield** - Holden's 10-year-old sister represents the innocence he seeks to protect. Their relationship is the most genuine in the novel, with Phoebe serving as both his critic and his salvation. She confronts him about his inability to like anything, forces him to articulate what he wants from life, and ultimately prevents his flight from responsibility.

**Allie Caulfield** - Though deceased, Holden's younger brother remains a powerful presence in his mind. Holden idealizes Allie, describing him as exceptionally intelligent and kind. Allie's death from leukemia represents the senseless loss of innocence that haunts Holden, and his unresolved grief contributes to his breakdown.

**D.B. Caulfield** - Holden's relationship with his older brother is ambivalent. He admires D.B.'s writing talent but criticizes him for "prostituting himself" in Hollywood. This relationship reflects Holden's complex attitude toward talent and authenticity.

**Parents** - Though rarely directly present, Holden's parents loom in his consciousness. He avoids them yet seeks connection, as shown when he sneaks home to see Phoebe. His mother's nervousness following Allie's death suggests a family struggling with unresolved grief.

### School Relationships

**Mr. Spencer** - Holden's history teacher represents well-meaning but ineffective adult authority. Their interaction establishes Holden's pattern of respecting individuals while rejecting their conventional values.

**Stradlater** - As Holden's roommate and foil, Stradlater embodies conventional masculinity—athletic, sexually experienced, and concerned with appearance. Their conflict over Jane Gallagher reveals Holden's sexual jealousy and protective instincts.

**Ackley** - This socially awkward, unhygienic neighbor mirrors aspects of Holden's isolation while also serving as an object of his condescension, revealing Holden's own social prejudices despite his critique of others'.

**Mr. Antolini** - As Holden's former English teacher, he offers the most substantial guidance, advising Holden about education and finding purpose. The ambiguous physical incident undermines this potential mentorship, representing Holden's final disillusionment with adult guidance.

### Other Significant Relationships

**Jane Gallagher** - Though never appearing directly, Jane represents a genuine connection from Holden's past. His protective feelings toward her and detailed memories of their interactions contrast with his superficial encounters throughout the novel.

**Sally Hayes** - Their failed date illustrates Holden's inability to communicate his genuine feelings and his tendency to sabotage potential connections. Sally represents the conventional social world Holden claims to despise yet seeks to engage with.

**The Nuns** - Holden's breakfast conversation with the nuns represents one of his few positive social interactions. Their simplicity, authenticity, and dedication to service temporarily break through his cynicism.

## Complete Plot Progression

1. **At Pencey Prep** - Holden watches the football game from afar, visits Mr. Spencer who lectures him about his academic failure, returns to his dorm where he interacts with Ackley and Stradlater, fights with Stradlater after his date with Jane, and impulsively decides to leave school early.

2. **Departure and Travel** - Holden sells his typewriter to Freddie Woodruff, says goodbye to the dormitory, takes a late train to New York, and converses with a classmate's mother (Mrs. Morrow) on the journey, lying about his identity.

3. **First Night in New York** - He checks into the Edmont Hotel, observes "perverts" from his window, contacts Faith Cavendish unsuccessfully, visits the Lavender Room where he dances with three tourists, then moves to Ernie's jazz club where he meets his brother's ex-girlfriend Lillian Simmons.

4. **Encounter with Sunny** - Returning to the hotel, Holden is offered a prostitute by Maurice the elevator operator. When Sunny arrives, Holden only wants to talk. After paying her, he's confronted by Maurice who demands more money and punches him.

5. **Second Day in New York** - Holden calls Sally Hayes and arranges a date, meets the nuns at breakfast, buys the record for Phoebe, attends a play with Sally, proposes they run away together, insults her after rejection, meets Carl Luce who advises him to see a psychoanalyst, and gets drunk alone.

6. **Central Park and Family Visit** - Holden checks the lagoon for ducks, sneaks into his family's apartment, wakes Phoebe and tells her he's been expelled, listens to her criticisms, dances with her, and escapes when his parents return home.

7. **Visit to Mr. Antolini** - Holden visits his former teacher who offers advice and a place to sleep. He wakes to find Mr. Antolini patting his head, interprets this as a sexual advance, and leaves in distress.

8. **Final Day and Crisis** - After spending the night at Grand Central Station, Holden wanders the city, growing increasingly distressed. He decides to hitchhike west but wants to see Phoebe first. He leaves a note at her school to meet him at the Museum of Natural History.

9. **Resolution at the Carousel** - Phoebe arrives with a suitcase, wanting to join Holden. He refuses, causing her distress. They walk to the carousel where Phoebe rides while Holden watches in the rain, experiencing a moment of genuine happiness as he accepts she must take risks to grow.

10. **Epilogue** - Holden reveals he is narrating from some kind of institution or recovery facility. He mentions getting sick afterward but doesn't provide details about his treatment or future, only mentioning that people ask if he'll "apply himself" when he returns to school.

## Narrative Techniques

### Perspective and Voice
The novel employs first-person narration through Holden's distinctive voice, characterized by repetition, hyperbole, digressions, and adolescent slang ("phony," "goddam," "kills me"). This perspective creates both intimacy and unreliability, as readers experience events exclusively through Holden's biased viewpoint. His narration often contradicts itself, revealing his confused state of mind.

### Structure
The narrative uses a frame structure—Holden tells his story from some form of institution after the events have concluded. Within this frame, the novel follows a compressed timeframe of about three days, with frequent flashbacks to earlier experiences. Though the narrative seems to meander, it follows Holden's psychological rather than logical progression.

### Temporal Techniques
Salinger employs numerous flashbacks (Holden's memories of Jane, Allie, and childhood experiences) that interrupt the linear narrative but provide crucial context for understanding Holden's psychology. The present-tense narration creates immediacy despite the past-tense events.

### Symbolism and Motifs
The novel is rich with recurring symbols (the red hunting hat, the ducks, the museum, the carousel) and motifs (falling, phoniness, loneliness) that create thematic coherence beneath the seemingly chaotic surface of Holden's experiences.

## Linguistic Analysis

### Syntax
Holden's narration features fragmented sentences, frequent digressions, and parenthetical qualifications ("if you want to know the truth") that mirror adolescent speech patterns and his scattered mental state. His syntax becomes more disjointed during moments of emotional distress, reflecting his psychological deterioration.

### Diction
The novel's distinctive diction includes adolescent slang, repetition of key phrases ("phony," "killed me," "depressed me"), and Holden's characteristic hyperbole. His vocabulary shifts between sophisticated literary references and crude adolescent expressions, reflecting his liminal position between childhood and adulthood.

### Tone
The narrative tone blends cynicism, vulnerability, and nostalgia. While Holden maintains a surface layer of world-weary sarcasm, moments of genuine emotion repeatedly break through, creating a complex tonal landscape that mirrors his contradictory character.

### Literary Techniques
Salinger employs several key techniques:
- **Irony** - Holden criticizes "phonies" while often being insincere himself
- **Contrast** - Juxtaposition between Holden's cynical observations and his sentimental memories
- **Repetition** - Recurring phrases and obsessions that create thematic consistency
- **Imagery** - Vivid sensory details that establish mood, particularly in descriptions of New York City
- **Understatement** - Used to mute emotional moments that would otherwise be overwhelming, particularly relating to Allie's death

Through these techniques, Salinger creates a psychologically complex portrait of adolescent alienation that continues to resonate with readers despite its specific mid-20th century cultural context.
"""
    scores = scorer.score(reference, data['summary'])
    rouge_l_f1 = scores['rougeL'].fmeasure
    
    return rouge_l_f1


def test_params(combinations, text_name):
    for combination in combinations:
        chunk_size, chunk_overlap, temp_chunk, temp_final = combination

        task_id = str(uuid.uuid4())

        params_dict = {
            "chunk_size": chunk_size,
            "overlap": chunk_overlap / chunk_size,
            "temp_chunk": temp_chunk,
            "temp_final": temp_final,
            "max_tokens_chunk": 1500,
            "max_tokens_final": 5000
        }

        if text_name == "Rye":
            r.set(f"summarize:{task_id}:text", RYE_TEXT)
            params_dict["Rye"] = True
        else:
            r.set(f"summarize:{task_id}:text", TRANSCRIPT_TEXT)
            params_dict["Rye"] = False         
        r.set(f"summarize:{task_id}:params", json.dumps(params_dict))

        result = celery.send_task("tasks.process_document", args=[task_id])

        summary_output = result.get()
        summary_dict = json.loads(summary_output)
        f1_score = run_eval(summary_dict)
        summary_dict["f1_score"] = f1_score

        updated_final_output = json.dumps(summary_dict, indent=2)

        try:
            response = requests.post(
                DATA_URL,
                headers={"Content-Type": "application/json"},
                data=updated_final_output,
            )
            response.raise_for_status()
            logging.info(f"[UPLOAD] Successfully sent summary to {DATA_URL}. Response: {response.text}")
        except Exception as e:
            logging.info(f"Error uploading data: {e}")

        time.sleep(5)

if __name__ == "__main__":
    test_params(combinations)
