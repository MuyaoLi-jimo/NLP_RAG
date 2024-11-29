from openai import OpenAI

def generate_qa(data,model_id ,api_key, api_base, temperature=0.7,max_tokens=4096):
    
    api_key = data.get("api_key",api_key)
    api_base = data.get("api_base",api_base)
    client = OpenAI(api_key=api_key, base_url=api_base)
    model = data.get("model_id",model_id)
    
    temperature = data.get("temperature",temperature)
    max_tokens = data.get("max_tokens",max_tokens)

    response = None
    try:
        response = client.chat.completions.create(
            model=model,
            messages=data["messages"],
            temperature=temperature,
            max_tokens= max_tokens,
        )
        content = response.choices[0].message.content
    except Exception as e:
        print(response)
        raise Exception(e)
    token = {
        "input" : response.usage.prompt_tokens,
        "output" : response.usage.completion_tokens,
    }
    return content,token


def create_system_message(system_prompt):
    message = None
    message = {
            "role": "system",
            "content": system_prompt,
        }
    return message

def create_assistant_message(assistant_prompt):
    message = {
        "role": "assistant",
        "content": f"{assistant_prompt}\n",
    }
    return message

def create_user_message(user_prompt:str=""):

    message = {
        "role": "user",
        "content": [{
            "type": "text",
            "text": f"{user_prompt}\n"
        }],
    }
    return message
