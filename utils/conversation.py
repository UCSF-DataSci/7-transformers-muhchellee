import requests
import argparse
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def get_response(prompt, history=None, model_name="google/flan-t5-small", api_key=None, history_length=3):
    """
    Get a response from the model using conversation history
    
    Args:
        prompt: The current user prompt
        history: List of previous (prompt, response) tuples
        model_name: Name of the model to use
        api_key: API key for authentication
        history_length: Number of previous exchanges to include in context
        
    Returns:
        The model's response
    """
    # TODO: Implement the contextual response function
    # Initialize history if None
    history = list(history) if history is not None else []

    for i in range(len(history)):
        item = history[i]
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            history[i] = (str(item[0]), str(item[1]))
        elif isinstance(item, str):
            history[i] = (str(item), "")
        else:
            history[i] = ("[Invalid history]", "")

    format = "\n".join(
        f"User: {user}\nAssistant: {assistant}"
        for user, assistant in history[-history_length:]
    )

    # TODO: Format a prompt that includes previous exchanges
    full_prompt = f"{format}\nUser: {prompt}\nAssistant:" if format else f"User: {prompt}\nAssistant:"
    payload = {"inputs": full_prompt}

    # Get a response from the API
    if api_key:
        try:
            response = requests.post(
                f"https://api-inference.huggingface.co/models/{model_name}",
                headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
                json=payload,
                timeout=10
                )
            if response.status_code == 200:
                result = response.json()
                # Return the response
                if isinstance(result, list) and "generated_text" in result[0]:
                    return response.json()[0]['generated_text']
                if isinstance(result, dict) and "generated_text" in result:
                    return result["generated_text"]
        except Exception as e:
            return f"Request failed: {str(e)}"
        except:
            pass
    
    # local tokenizer fallback
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        inputs = tokenizer(full_prompt, return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=150)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        return f"[Local Error: {str(e)}]"
        
    

def run_chat(model_name="google/flan-t5-small", api_key=None, history_length=3):
    """Run an interactive chat session with context"""
    print("Welcome to the Contextual LLM Chat! Type 'exit' to quit.")
    
    # Initialize conversation history
    history = []
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
            
        # TODO: Get response using conversation history
        response = get_response(
            prompt=user_input,
            history=history,
            model_name=model_name,
            api_key=api_key,
            history_length=history_length
        )

        # Update history
        history.append((user_input, response))

        # Print the response
        print(f"\nAssistant: {response}")
        

def main():
    parser = argparse.ArgumentParser(description="Chat with an LLM using conversation history")
    # TODO: Add arguments to the parser
    parser.add_argument(
        "--model",
        type=str,
        default="google/flan-t5-small",
        help="Model name (default: google/flan-t5-small)"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Hugging Face API key"
    )
    parser.add_argument(
        "--history-length",
        type=int,
        default=3,
        help="Number of previous exchanges in chat history"
    )
    args = parser.parse_args()
    
    # TODO: Run the chat function with parsed arguments
    run_chat(args.model, args.api_key, args.history_length)
    
if __name__ == "__main__":
    main()