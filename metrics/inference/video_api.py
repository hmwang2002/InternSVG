import cv2
import numpy as np
import base64
import time

from openai import OpenAI

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
    
def sample_frames_from_video(video_path, num_samples=8):
    video = cv2.VideoCapture(video_path)
    total = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        video.release()
        return []
    
    indices = np.unique(np.round(np.linspace(0, total - 1, num_samples, dtype=int)))
    
    frames = []
    for idx in indices:
        video.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = video.read()
        if ret:
            ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            if not ok:
                continue
            b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
            frames.append(f"data:image/jpeg;base64,{b64}")
    
    video.release()
    return frames

def build_multimodal_content(images_b64, instruction):
    content = []
    content.append({"type": "text", "text": instruction})

    for url in images_b64:
        content.append({
            "type": "image_url",
            "image_url": {"url": url}
        })
    return content

def chat_gpt(client, model:str, text:str, video_path=None, num_samples=8, retry=2, temperature=0.2):
    for attempt in range(retry):
        try:
            if video_path is not None:
                images_b64 = sample_frames_from_video(video_path, num_samples)
                content = build_multimodal_content(images_b64, text)
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "user",
                            "content": content
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
                print(f"[QA‑ERROR] {video_path}: {e}", flush=True)
                return None
            time.sleep(2)

def chat(client, model:str, text:str, video_path=None, num_samples=8, retry=2, temperature=0.2, max_tokens=16000):
    for attempt in range(retry):
        try:
            if video_path is not None:
                images_b64 = sample_frames_from_video(video_path, num_samples)
                content = build_multimodal_content(images_b64, text)
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "user",
                            "content": content
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
                print(f"[QA‑ERROR] {video_path}: {e}", flush=True)
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
            "response_mime_type": "text/plain"   # 关键：强制文本
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


if __name__ == "__main__":
    client = build_client(base_url="BASE_URL", api_key="API_KEY", timeout=200)
    res = chat(client=client, model="gpt-4o", text="describe the video.", video_path="PATH_TO_VIDEO", num_samples=8)
    print(res)