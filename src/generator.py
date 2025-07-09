import requests

def load_local_llm():
    return None  # Not needed for Ollama

def generate_answer(prompt: str, model: str = "gemma:2b") -> str:
    print(f"\n🧾 Prompt Sent to Ollama ({model}):\n{prompt[:300]}...\n")
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "num_predict": 100
                }
            },
            timeout=30
        )
        response.raise_for_status()
        result = response.json()
        return result.get("response", "⚠️ No response from model.").strip()
    except requests.exceptions.Timeout:
        return f"⚠️ {model} timed out. Try a lighter model like gemma:2b."
    except requests.exceptions.RequestException as e:
        return f"⚠️ Failed to get response from {model}: {e}"
