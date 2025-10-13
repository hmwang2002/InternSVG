from openai import OpenAI
import base64
import time


def build_client(base_url: str, api_key: str, timeout: int = 20):
    return OpenAI(
        base_url=base_url,
        api_key=api_key,
        timeout=timeout  # seconds
    )

def encode_image(image_path: str):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    
def encode_video(video_path):
    with open(video_path, "rb") as video_file:
        return base64.b64encode(video_file.read()).decode('utf-8')
    
def chat_gpt(client, model:str, text:str, image_path=None, retry=2, temperature=0.2):
    for attempt in range(retry):
        try:
            if image_path is not None:
                base64_image = encode_image(image_path)
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": text},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}"
                                    }
                                }
                            ]
                        }
                    ],
                    temperature=temperature,
                )
            else:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": text}
                            ]
                        }
                    ],
                    temperature=temperature,
                )
            return response.choices[0].message.content
        except Exception as e:
            if attempt == retry - 1:
                print(f"[QA‑ERROR] {image_path}: {e}", flush=True)
                return None
            time.sleep(2)

def chat(client, model:str, text:str, image_path=None, retry=2, temperature=0.2, max_tokens=16000):
    for attempt in range(retry):
        try:
            if image_path is not None:
                base64_image = encode_image(image_path)
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": text},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}"
                                    }
                                }
                            ]
                        }
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            else:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": text}
                            ]
                        }
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            return response.choices[0].message.content
        except Exception as e:
            if attempt == retry - 1:
                print(f"[QA‑ERROR] {image_path}: {e}", flush=True)
                return None
            time.sleep(2)
            
def chat_video(client, model:str, text:str, video_path, retry=2, temperature=0.2, max_tokens=8000):
    for attempt in range(retry):
        try:
            if video_path is not None:
                base64_video = encode_video(video_path)
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "user", 
                            "content": [
                                {"type": "text", "text": text}, 
                                {"type": "image_url", "image_url": f"data:video/mp4;base64,{base64_video}"}
                            ]
                        }
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content
            else:
                print(f"[QA‑ERROR] {video_path}: video_path is None", flush=True)
                return None
        except Exception as e:
            if attempt == retry - 1:
                print(f"[QA‑ERROR] {video_path}: {e}", flush=True)
                return None
            time.sleep(2)
            
def chat_video_gemini_http(api_key: str, base_url: str, model: str, text: str,
                           video_path: str, retry: int = 3, timeout: int = 200):
    import base64, pathlib, time, requests
    b64 = base64.b64encode(pathlib.Path(video_path).read_bytes()).decode("utf-8")
    url = base_url.rstrip("/") + f"/models/{model}:generateContent"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    user_text = (text or "Describe the video.") + "\n\nOnly respond with plain text."

    payload = {
        "contents": [{
            "role": "user",
            "parts": [
                {"text": user_text},
                {"inlineData": {"mimeType": "video/mp4", "data": b64}}
            ]
        }],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 1024,
            "response_mime_type": "text/plain"
        }
    }

    last = None
    for i in range(retry):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=timeout)
            if r.status_code == 200:
                data = r.json()
                if "candidates" in data:
                    parts = data["candidates"][0]["content"]["parts"]
                    return "".join(p.get("text", "") for p in parts if "text" in p)
                if "output_text" in data:
                    return data["output_text"]
                return str(data)
            last = f"{r.status_code} {r.text}"
        except Exception as e:
            last = str(e)
        time.sleep(1.5 * (i + 1))
    print(f"[QA-ERROR] {video_path}: {last}", flush=True)
    return None

