# utils/one_off_chat.py

import requests
import argparse
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from dotenv import load_dotenv

load_dotenv()

def get_response(prompt, model_name="google/flan-t5-small", api_key=None):
    """
    Get a response from the model
    
    Args:
        prompt: The prompt to send to the model
        model_name: Name of the model to use 
        api_key: API key for authentication (optional for some models)
        
    Returns:
        The model's response
    """
    # TODO: Implement the get_response function

    # Set up the API URL and headers
    API_URL = f"https://api-inference.huggingface.co/models/{model_name}"

    # try api
    if api_key: 
        try:
            headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
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
        outputs = model.generate(**inputs, max_new_tokens=100)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        return f"[Local Error: {str(e)}]"



def run_chat(model_name="google/flan-t5-small", api_key=None):
    """Run an interactive chat session"""
    print("Welcome to the Simple LLM Chat! Type 'exit' to quit.")
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
            
        # TODO: Get response from the model

        try:
            response = get_response(user_input, model_name, api_key)
            if response:
                # Print the response
                print(f"\nAssistant: {response}")
            else:
                print("\nError: Could not get a response")
                
        except Exception as e:
            print(f"\nError: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description="Chat with an LLM")
    
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

    # TODO: Run the chat function with parsed arguments
    args = parser.parse_args()
    
    # Start the chat session
    run_chat(model_name=args.model, api_key=args.api_key)

if __name__ == "__main__":
    main()