# -*- coding: utf-8 -*-
"""Planner category prompts for MT-Bench's 8 categories, in the exact template of
the MT-Bench-101 prompts (category sentence, one invented example conversation,
an evidence plan whose LLM[...] instructions quote what the Worker needs, the
closing rating plan, and the fixed closing line). Writes mtbench_planner_categories.json."""
import json, os

BASE = os.path.dirname(os.path.abspath(__file__))
TAIL = "\n\nBegin! Describe your plans with rich details.Plans should always be generated in the same template with the above example.\n\n"

CATS = {
"writing": ("Writing", "to compose or revise texts such as essays, emails, stories, or posts while following the user's content and format constraints across turns",
[("Write a four-line poem about autumn leaves for a children's newsletter.",
  "Red and gold leaves drift and play,\nDancing down on a windy day.\nCrunch them, jump them, toss them high,\nAutumn waves a bright goodbye."),
 ("Now rewrite it so that every line starts with the letter B.",
  "Bright red and gold leaves drift and play,\nBlowing down on a windy day.\nBounce on them, toss them up so high,\nBye says autumn, waving goodbye.")],
[("Check the content and format constraints of the first poem (Turn 1).",
  "Check whether the following poem has exactly four lines, is about autumn leaves, and uses simple language suitable for children: \"Red and gold leaves drift and play, / Dancing down on a windy day. / Crunch them, jump them, toss them high, / Autumn waves a bright goodbye.\""),
 ("Check the rewriting constraint (Turn 2).",
  "Check whether every line of the following poem starts with the letter B, and list any line that does not: \"Bright red and gold leaves drift and play, / Blowing down on a windy day. / Bounce on them, toss them up so high, / Bye says autumn, waving goodbye.\""),
 ("Check that the rewrite kept the original content (Turn 2).",
  "Compare the original poem \"Red and gold leaves drift and play, / Dancing down on a windy day. / Crunch them, jump them, toss them high, / Autumn waves a bright goodbye.\" with the rewrite \"Bright red and gold leaves drift and play, / Blowing down on a windy day. / Bounce on them, toss them up so high, / Bye says autumn, waving goodbye.\" and judge whether the rewrite keeps the meaning, rhythm, and child-friendly tone.")],
"constraint satisfaction, writing quality, and consistency across turns"),

"roleplay": ("Roleplay", "to stay in an assigned character or persona and respond in that voice while following the user's scenario across turns",
[("Pretend you are a medieval blacksmith. A traveler asks what you make. Answer in character.",
  "Aye, traveler! I forge horseshoes, nails, and the odd sword for the lord's guard. Iron from the hills, fire from good oak charcoal. What brings ye to my anvil?"),
 ("Stay in character. The traveler asks what you think of the new king.",
  "The new king? Hah, his taxes weigh heavier than my hammer. But his soldiers need blades, so my forge stays hot. I'll speak no more of it where the guards might hear.")],
[("Evaluate whether the first reply fits the persona (Turn 1).",
  "Assess whether the following reply sounds like a medieval blacksmith in voice, vocabulary, and content, and note any modern or out-of-character elements: \"Aye, traveler! I forge horseshoes, nails, and the odd sword for the lord's guard. Iron from the hills, fire from good oak charcoal. What brings ye to my anvil?\""),
 ("Evaluate whether the second reply stays in character (Turn 2).",
  "Assess whether the following reply stays in a medieval blacksmith persona while answering a traveler's question about the new king, and whether its attitude is plausible for a commoner of that era: \"The new king? Hah, his taxes weigh heavier than my hammer. But his soldiers need blades, so my forge stays hot. I'll speak no more of it where the guards might hear.\""),
 ("Check persona consistency across turns (Turns 1-2).",
  "A blacksmith character first says he forges horseshoes, nails, and swords for the lord's guard, and later says the king's soldiers need blades so his forge stays hot. Check whether these two replies are consistent in persona, tone, and details, and point out any contradiction.")],
"persona consistency, engagement, and coherence across turns"),

"reasoning": ("Reasoning", "to solve logic and commonsense reasoning problems and justify the answer",
[("Tom is taller than Ann. Ann is taller than Raj. Who is the shortest?",
  "Raj is the shortest, because Tom is taller than Ann and Ann is taller than Raj."),
 ("Now add that Mia is shorter than Raj. Who is the shortest now, and who is the second tallest?",
  "Mia is now the shortest. The order from tallest is Tom, Ann, Raj, Mia, so Ann is the second tallest.")],
[("Verify the first deduction (Turn 1). (Tom > Ann > Raj).",
  "Logic Check: if Tom is taller than Ann and Ann is taller than Raj, determine who is the shortest, and based on this result, validate the answer \"Raj is the shortest\"."),
 ("Verify the updated ordering (Turn 2). (Tom > Ann > Raj > Mia).",
  "Logic Check: if Tom is taller than Ann, Ann is taller than Raj, and Mia is shorter than Raj, determine the full height order, the shortest person, and the second tallest, and based on this result, validate the answer \"Mia is the shortest and Ann is the second tallest\"."),
 ("Check the justifications (Turns 1-2).",
  "Given #E1 and #E2, assess whether the explanations \"because Tom is taller than Ann and Ann is taller than Raj\" and \"The order from tallest is Tom, Ann, Raj, Mia\" correctly and sufficiently justify the conclusions.")],
"logical correctness, quality of justification, and use of earlier information"),

"math": ("Math", "to solve mathematical problems with correct computation and clear steps",
[("A shirt costs $40 and is discounted by 25%. What is the sale price?",
  "25% of $40 is $10, so the sale price is $30."),
 ("If 8% sales tax is then added to the sale price, what is the final price?",
  "8% of $30 is $2.40, so the final price is $32.40.")],
[("Verify the discount computation (Turn 1). (40 - 0.25 x 40 = 30).",
  "Compute 40 - 0.25 x 40, and based on this result, validate the claimed sale price of $30."),
 ("Verify the tax computation (Turn 2). (30 x 1.08 = 32.40).",
  "Compute 30 x 1.08, and based on this result, validate the claimed final price of $32.40."),
 ("Check the use of the previous result (Turn 2).",
  "A shirt costs $40, is discounted by 25%, and then 8% sales tax is added. Check whether the tax should be applied to the discounted price or the original price, and whether computing 8% of $30 is the correct step.")],
"mathematical accuracy, correct use of the previous result, and clarity"),

"coding": ("Coding", "to write, explain, or fix program code that correctly meets the user's specification",
[("Write a Python function that returns the second largest number in a list.",
  "def second_largest(nums):\n    nums = sorted(set(nums))\n    return nums[-2]"),
 ("What happens if the list has fewer than two distinct values? Fix it.",
  "It raises an IndexError. Fixed version:\ndef second_largest(nums):\n    nums = sorted(set(nums))\n    if len(nums) < 2:\n        return None\n    return nums[-2]")],
[("Verify the correctness of the first function (Turn 1).",
  "Check whether a Python function that removes duplicates with set(), sorts the result in ascending order, and returns the second-to-last element correctly returns the second largest number in a list, including lists with duplicate values."),
 ("Verify the claimed failure mode (Turn 2).",
  "Determine what a Python function that removes duplicates with set(), sorts the result, and returns the second-to-last element does when the list has fewer than two distinct values, and check whether the claim that it raises an IndexError is correct."),
 ("Verify the fix (Turn 2).",
  "Check whether adding a guard that returns None when the de-duplicated sorted list has fewer than two elements, before returning its second-to-last element, handles lists with fewer than two distinct values and still returns correct results for other lists.")],
"code correctness, handling of edge cases, and responsiveness to the follow-up"),

"extraction": ("Extraction", "to extract specified information from a given text and present it in the requested format",
[("From the text below, extract the company name, the year it was founded, and the city of its headquarters as JSON. Text: \"Lumora, a maker of solar lamps founded in 2014, moved its headquarters from Austin to Denver in 2020.\"",
  "{\"company\": \"Lumora\", \"founded\": 2014, \"headquarters\": \"Denver\"}"),
 ("Add a field for the previous headquarters and return it as YAML instead.",
  "company: Lumora\nfounded: 2014\nheadquarters: Denver\nprevious_headquarters: Austin")],
[("Verify the extracted values (Turn 1).",
  "From the text \"Lumora, a maker of solar lamps founded in 2014, moved its headquarters from Austin to Denver in 2020.\", extract the company name, the founding year, and the current headquarters city, and compare them with the answer {\"company\": \"Lumora\", \"founded\": 2014, \"headquarters\": \"Denver\"}."),
 ("Verify the added field and the format change (Turn 2).",
  "Check whether \"company: Lumora / founded: 2014 / headquarters: Denver / previous_headquarters: Austin\" (one field per line) is valid YAML, and whether the previous headquarters in the text \"Lumora ... moved its headquarters from Austin to Denver in 2020\" is Austin."),
 ("Check consistency across turns (Turns 1-2).",
  "A first answer returned the JSON {\"company\": \"Lumora\", \"founded\": 2014, \"headquarters\": \"Denver\"}; after being asked to add the previous headquarters and switch to YAML, the second answer returned company: Lumora, founded: 2014, headquarters: Denver, previous_headquarters: Austin. Check whether all earlier fields were kept, only the requested field was added, and the format was switched as requested.")],
"extraction accuracy, format compliance, and consistency across turns"),

"stem": ("STEM", "to explain knowledge in science, technology, engineering, and mathematics accurately and clearly",
[("Why is the sky blue?",
  "Sunlight is scattered by molecules in the air, and shorter blue wavelengths scatter much more than longer red ones (Rayleigh scattering), so blue light reaches our eyes from all directions."),
 ("Then why are sunsets red?",
  "At sunset, light travels through much more atmosphere, so most blue light is scattered out of the direct path before it reaches you, leaving mostly red and orange light.")],
[("Verify the first explanation (Turn 1).",
  "Verify the scientific accuracy of this explanation of why the sky is blue, and note any errors or important omissions: \"Sunlight is scattered by molecules in the air, and shorter blue wavelengths scatter much more than longer red ones (Rayleigh scattering), so blue light reaches our eyes from all directions.\""),
 ("Verify the second explanation (Turn 2).",
  "Verify the scientific accuracy of this explanation of why sunsets are red, and note any errors or important omissions: \"At sunset, light travels through much more atmosphere, so most blue light is scattered out of the direct path before it reaches you, leaving mostly red and orange light.\""),
 ("Check coherence across turns (Turns 1-2).",
  "Assess whether an explanation of red sunsets based on blue light being scattered out of a longer atmospheric path builds correctly on an earlier explanation of the blue sky based on Rayleigh scattering, and whether both are clear for a general reader.")],
"scientific accuracy, clarity, and coherence across turns"),

"humanities": ("Humanities", "to discuss topics in history, economics, philosophy, social science, and the arts with accurate knowledge and balanced reasoning",
[("Briefly explain what caused the fall of the Western Roman Empire.",
  "Historians point to several causes: political instability and frequent changes of emperor, economic strain and heavy taxation, reliance on mercenary armies, and pressure from migrating groups such as the Goths. In 476 the Germanic leader Odoacer deposed the last western emperor."),
 ("Which of these causes do most historians consider the most important?",
  "There is no single agreed answer. Some historians stress internal decline, while others, such as Peter Heather, emphasize external pressure from migrating peoples. Most see the fall as the result of several causes working together.")],
[("Verify the historical accuracy of the first answer (Turn 1).",
  "Check the historical accuracy of this summary of the causes of the fall of the Western Roman Empire, including the claim that Odoacer deposed the last western emperor in 476: \"political instability and frequent changes of emperor, economic strain and heavy taxation, reliance on mercenary armies, and pressure from migrating groups such as the Goths\"."),
 ("Evaluate how the second answer represents scholarly debate (Turn 2).",
  "Assess whether the following answer fairly represents scholarly debate on the main cause of the fall of the Western Roman Empire, and whether attributing an emphasis on external pressure to Peter Heather is accurate: \"There is no single agreed answer. Some historians stress internal decline, while others, such as Peter Heather, emphasize external pressure from migrating peoples. Most see the fall as the result of several causes working together.\""),
 ("Check balance and consistency across turns (Turns 1-2).",
  "An assistant first listed several causes for the fall of the Western Roman Empire and then, when asked which cause historians consider most important, said there is no single agreed answer and that most see several causes working together. Assess whether the second answer is consistent with the first and gives a balanced, well-supported view.")],
"historical and factual accuracy, balance, and coherence across turns"),
}


def build(name, desc, conv, plan, rate):
    s = f"\nThe conversation here is about {name}, which is {desc}.\nHere is an example of a conversation and evaluation plan.\n\n<|The Start of Conversation|>\n"
    s += "\n\n".join(f"### User: {u}\n### Assistant A: {a}" for u, a in conv)
    s += "\n<|The End of Conversation|>\n\n<|The Start of Evaluation Plan|>\n"
    s += "\n\n".join(f"Plan: {p}\n#E{i} = LLM[{x}]" for i, (p, x) in enumerate(plan, 1))
    evs = [f"#E{i}" for i in range(1, len(plan) + 1)]
    s += f"\n\nPlan: Rate the answer of Assistant A based on {rate}, given {', '.join(evs[:-1])}, and {evs[-1]}.\n<|The End of Evaluation Plan|>"
    return s + TAIL


out = {k: build(*v) for k, v in CATS.items()}
json.dump(out, open(os.path.join(BASE, "mtbench_planner_categories.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print({k: len(v) for k, v in out.items()})
