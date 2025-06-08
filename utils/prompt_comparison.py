import os
import requests
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import json


# Define a question to experiment with
question = "What foods should be avoided by patients with gout?"

# Example for one-shot and few-shot prompting
example_q = "What are the symptoms of gout?"
example_a = "Gout symptoms include sudden severe pain, swelling, redness, and tenderness in joints, often the big toe."

# Examples for few-shot prompting
examples = [
    ("What are the symptoms of gout?",
     "Gout symptoms include sudden severe pain, swelling, redness, and tenderness in joints, often the big toe."),
    ("How is gout diagnosed?",
     "Gout is diagnosed through physical examination, medical history, blood tests for uric acid levels, and joint fluid analysis to look for urate crystals.")
]

# TODO: Create prompting templates
# Zero-shot template (just the question)
zero_shot_template = "Question: {question}\nAnswer:"

# One-shot template (one example + the question)
one_shot_template = """Question: {example_q}
Answer: {example_a}

Question: {question}
Answer:"""

# Few-shot template (multiple examples + the question)
few_shot_template = """Question: {examples[0][0]}
Answer: {examples[0][1]}

Question: {examples[1][0]}
Answer: {examples[1][1]}

Question: {question}
Answer:"""

# TODO: Format the templates with your question and examples
zero_shot_prompt = zero_shot_template.format(question=question)
one_shot_prompt = one_shot_template.format(example_q=example_q, example_a=example_a, question=question)
# For few-shot, you'll need to format it with the examples list
few_shot_prompt = few_shot_template.format(examples=examples, question=question)

def get_llm_response(prompt, model_name="google/flan-t5-small", api_key=None):
    """Get a response from the LLM based on the prompt"""
    # TODO: Implement the get_llm_response function
    # Set up the API URL and headers
    API_URL = f"https://api-inference.huggingface.co/models/{model_name}"

    api_key = "hf_XXXXXXXXXXXXXXXXX"

    # try api
    if api_key: 
        try:
            headers = {"Authorization": f"Bearer {api_key}"}
            # Create a payload with the prompt
            payload = {"inputs": prompt}

            # Send the payload to the API
            response = requests.post(API_URL, 
                                     headers=headers, 
                                     json=payload, 
                                     timeout=10)
            # Extract and return the generated text from the response
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and "generated_text" in result[0]:
                    return result[0]["generated_text"]
                if isinstance(result, dict) and "generated_text" in result:
                    return result["generated_text"]
        except Exception as e:
            print(f"\nError: {str(e)}")
        except:
            pass
    
    # local tokenizer fallback
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=64)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        return f"[Local Error: {str(e)}]"
    

# List of healthcare questions to test
questions = [
    "What foods should be avoided by patients with gout?",
    "What medications are commonly prescribed for gout?",
    "How can gout flares be prevented?",
    "Is gout related to diet?",
    "Can gout be cured permanently?"
]

# TODO: Compare the different prompting strategies on these questions
# For each question:
# - Create prompts using each strategy
# - Get responses from the LLM
# - Store the results

results = {}

for question in questions:
    zero_shot_prompt = zero_shot_template.format(question=question)
    zero_shot_response = get_llm_response(zero_shot_prompt)

    one_shot_prompt = one_shot_template.format(example_q=example_q, example_a=example_a, question=question)
    one_shot_response = get_llm_response(one_shot_prompt)

    few_shot_prompt = few_shot_template.format(examples=examples, question=question)
    few_shot_response = get_llm_response(few_shot_prompt)

    results[question] = {
    "zero_shot": zero_shot_response,
    "one_shot": one_shot_response,
    "few_shot": few_shot_response
    }
    

def score_response(response, keywords):
    """Score a response based on the presence of expected keywords"""
    # TODO: Implement the score_response function
    # Example implementation:
    response = response.lower()
    found_keywords = 0
    for keyword in keywords:
        if keyword.lower() in response:
            found_keywords += 1
    return found_keywords / len(keywords) if keywords else 0

# Expected keywords for each question
expected_keywords = {
    "What foods should be avoided by patients with gout?": 
        ["purine", "red meat", "seafood", "alcohol", "beer", "organ meats"],
    "What medications are commonly prescribed for gout?": 
        ["nsaids", "colchicine", "allopurinol", "febuxostat", "probenecid", "corticosteroids"],
    "How can gout flares be prevented?": 
        ["medication", "diet", "weight", "alcohol", "water", "exercise"],
    "Is gout related to diet?": 
        ["yes", "purine", "food", "alcohol", "seafood", "meat"],
    "Can gout be cured permanently?": 
        ["manage", "treatment", "lifestyle", "medication", "chronic"]
}

# TODO: Score the responses and calculate average scores for each strategy
# Determine which strategy performs best overall

for question, response in results.items():
    keywords = expected_keywords.get(question, [])
    response["zero_shot_score"] = score_response(response["zero_shot"], keywords)
    response["one_shot_score"] = score_response(response["one_shot"], keywords)
    response["few_shot_score"] = score_response(response["few_shot"], keywords)

n = len(results)
avg_zero_shot = sum(entry["zero_shot_score"] for entry in results.values()) / n
avg_one_shot = sum(entry["one_shot_score"] for entry in results.values()) / n
avg_few_shot = sum(entry["few_shot_score"] for entry in results.values()) / n

avg_scores = {
    "zero_shot": avg_zero_shot,
    "one_shot": avg_one_shot,
    "few_shot": avg_few_shot
}


# save results
os.makedirs("results/part_3", exist_ok=True)

with open("results/part_3/prompting_results.txt", "w") as f:
    f.write("# Prompt Engineering Results\n\n")
    for question, answers in results.items():
        f.write(f"## Question: {question}\n\n")
        f.write("### Zero-shot response:\n")
        f.write(f"{answers['zero_shot']}\n\n")
        f.write("### One-shot response:\n")
        f.write(f"{answers['one_shot']}\n\n")
        f.write("### Few-shot response:\n")
        f.write(f"{answers['few_shot']}\n\n")
        f.write("-" * 50 + "\n\n")

    f.write("## Scores\n\n")
    f.write("question,zero_shot,one_shot,few_shot\n")
    for question, answers in results.items():
        qid = question.lower().replace("?", "").replace(" ", "_")[:20]
        zs = f"{answers['zero_shot_score']:.2f}"
        os = f"{answers['one_shot_score']:.2f}"
        fs = f"{answers['few_shot_score']:.2f}"
        f.write(f"{qid},{zs},{os},{fs}\n")
    f.write(f"\naverage,{avg_zero_shot:.2f}, {avg_one_shot:.2f}, {avg_few_shot:.2f}\n")
    f.write(f"best_method,{max(avg_scores, key=avg_scores.get)}\n")
    f.write("```\n")