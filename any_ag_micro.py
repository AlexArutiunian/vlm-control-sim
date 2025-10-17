# =========================
# Imports
# =========================
import time
import math
import json
import ast
import re
from typing import List, Dict, Any
from pathlib import Path
import json
import math
import numpy as np
import imageio.v2 as imageio

import numpy as np
import yaml
import torch
import tkinter as tk
import csv
import uuid
from pathlib import Path

import mujoco as mj
from mujoco.glfw import glfw

from tkinter import filedialog

from llm_providers import build_llm
import settings
# === Voice agent deps ===
import threading, io, os, requests
try:
    import sounddevice as sd
    import soundfile as sf
except Exception:
    sd = None
    sf = None
# === OpenAI Realtime (WS) ===
import os, json, base64, time, threading
import numpy as np
import sounddevice as sd
import websocket  # пакет websocket-client
import settings
from queue import SimpleQueue
from queue import Empty

def process_action_queue():
    # забираем все накопившиеся задания, исполняем по одному
    while True:
        try:
            actions = ACTION_QUEUE.get_nowait()  # не блокируемся
        except Empty:
            break
        if actions:
            try:
                with EXEC_LOCK:
                    print(f"[EXEC] run {len(actions)} actions from queue")
                    move_joints_by_name(actions)  # БЛОКИРУЕТ главный цикл на время движения
            except Exception as e:
                print(f"[EXEC] error: {e}")


ACTION_QUEUE = SimpleQueue()

def enqueue_actions(actions):
    if actions:
        ACTION_QUEUE.put(actions)

EXEC_LOCK = threading.Lock()  # если ещё не добавляли

# =========================
# Imports (extended)
# =========================
# ... existing import ...
from demo_player import load_commands_csv, load_actions_json  # Add functions from demo_player.py
# =========================
# === LLM ДИАЛОГ В ФОНЕ ===
# =========================
import io
import threading
import numpy as np
import sounddevice as sd
import soundfile as sf
import requests
import openai

# Ключ: берём из settings/окружения, при наличии — из secrets_keys (необязательно)
try:
    from secrets_keys import OPENAI_KEY as _OPENAI_KEY_FALLBACK
except Exception:
    _OPENAI_KEY_FALLBACK = None

_OPENAI_KEY = getattr(settings, "OPENAI_API_KEY",
                      os.getenv("OPENAI_API_KEY", _OPENAI_KEY_FALLBACK or ""))

if not _OPENAI_KEY:
    print("[LLM] ВНИМАНИЕ: ключ OpenAI не задан (settings.OPENAI_API_KEY / env / secrets_keys.OPENAI_KEY)")

client = openai.OpenAI(api_key=_OPENAI_KEY)

# --------- Тихий ввод с микрофона (копия из твоего файла) ---------
def _silence_alsa():
    try:
        from ctypes import cdll, CFUNCTYPE, c_char_p, c_int
        asound = cdll.LoadLibrary("libasound.so")
        CBFUNC = CFUNCTYPE(None, c_char_p, c_int, c_char_p, c_int, c_char_p)
        def _noop(file, line, func, err, fmt):  # подавление спама ALSA
            return
        asound.snd_lib_error_set_handler(CBFUNC(_noop))
    except Exception:
        pass
_silence_alsa()

_SR = 16000
_CH = 1
_DTYPE = "int16"

def record_push_to_talk() -> bytes | None:
    """
    Нажми Enter — старт записи, ещё раз Enter — стоп.
    Возвращает WAV-байты.
    Если вместо Enter введена команда, вернёт None (обработается ниже).
    """
    print("\nНажмите Enter — старт записи, затем Enter — стоп. Для команд введите /help.")
    line = input()
    if line.strip().startswith("/"):
        return None

    buf = bytearray()
    overflow = False
    def cb(indata, frames, time_, status):
        nonlocal overflow
        if status.input_overflow:
            overflow = True
        buf.extend(indata)

    with sd.RawInputStream(samplerate=_SR, channels=_CH, dtype=_DTYPE, blocksize=0, callback=cb):
        print("Запись... (Enter — стоп)")
        input()

    if not buf:
        return b""
    arr = np.frombuffer(bytes(buf), dtype=_DTYPE)
    if _CH > 1:
        arr = arr.reshape(-1, _CH)
    bio = io.BytesIO()
    with sf.SoundFile(bio, mode="w", samplerate=_SR, channels=_CH, format="WAV", subtype="PCM_16") as f:
        f.write(arr)
    if overflow:
        print("[WARN] Обнаружен overflow входного буфера.")
    return bio.getvalue()

def transcribe_wav_bytes(wav_bytes: bytes, api_key: str) -> str:
    """STT: сначала gpt-4o-mini-transcribe, потом whisper-1 (копия из твоего файла)."""
    if not api_key:
        raise RuntimeError("OPENAI_KEY пуст.")
    url = "https://api.openai.com/v1/audio/transcriptions"
    headers = {"Authorization": f"Bearer {api_key}"}
    for model in ("gpt-4o-mini-transcribe", "whisper-1"):
        try:
            r = requests.post(
                url,
                headers=headers,
                data={"model": model, "language": "ru"},
                files={"file": ("speech.wav", wav_bytes, "audio/wav")},
                timeout=60,
            )
            if r.ok:
                return (r.json().get("text") or "").strip()
            else:
                print(f"[STT] {model}: {r.status_code} {r.text}")
        except Exception as e:
            print(f"[STT] {model} ошибка: {e}")
    return ""

# --------- Диалоговый клиент с вычленением JSON-действий ---------
class RobotDialogLLM:
    """
    Держит историю диалога. На каждую реплику ассистента:
      - пытается вытащить первый валидный JSON (массив/объект) через _extract_first_json()
      - парсит в list[dict]
      - enqueue_actions(actions) -> их выполнит главный цикл через process_action_queue()
    """
    def __init__(self, model: str = None):
        self.model = model or getattr(settings, "OPENAI_MODEL", "gpt-4o")
        self.history = []
        self._ensure_system()

    def _ensure_system(self):
        if self.history and self.history[0].get("role") == "system":
            return
        system_prompt = (
           "Отвечай кратко и по делу.  "
            "Если уместно, включай ТОЛЬКО JSON с массивом действий суставов. "
            "Если пользователь хочет просто поговорить - отвечай естественно и вежливо без JSON."
            "Допустимые форматы действий: "
            "{\"name\":\"<joint>\",\"angle\":<deg>,\"duration\":<sec?>,\"robot\":\"r1_\"|\"r2_\"}, "
            "{\"frame\":[{\"name\":\"<joint>\",\"angle\":<deg>},...],\"duration\":<sec?>,\"robot\":\"...\"}, "
            "{\"name\":\"walk\",\"num\":<int>,\"dir_deg\":<float>,\"spd\":<float>,\"robot\":\"...\"}, "
            "{\"name\":\"turn\",\"deg\":<float>,\"spd_deg_s\":<float?>,\"robot\":\"...\"}. "
            "Параллельность/повторы: для одновременных действий — {\"parallel\":[...]}; для повторов — {\"repeat\":[...],\"times\":<int>}. "
            "Понимание суставов (основные): "
            "«плечо» → shoulder_pitch/roll/yaw_joint, "
            "«локоть» → elbow_pitch/roll_joint, elbow_pitch = 90 соответствует выпрямленной руке "
            "«кисть» → wrist_pitch/yaw_joint, "
            "«бедро» → hip_pitch/roll/yaw_joint, "
            "«колено» → knee_joint, "
            "«щиколотка» → ankle_pitch/roll_joint, "
            "«туловище» → torso_joint. "
            "Имя сустава всегда с префиксом r1_ или r2_ по роботу. "
            "Например: r1_right_elbow_pitch_joint — правый локоть робота r1."

        )
        self.history = [{"role": "system", "content": system_prompt}]

    def _send_and_get(self, user_text: str) -> str:
        self.history.append({"role": "user", "content": user_text})
        # Сначала пробуем с max_completion_tokens (новые модели), при ошибке — без лимита
        try:
            resp = client.chat.completions.create(
                model=self.model,
                messages=self.history,
                max_completion_tokens=800
            )
        except Exception as e:
            # повтор без лимита — совместимо со старыми/нестандартными билдами
            resp = client.chat.completions.create(
                model=self.model,
                messages=self.history
            )
        reply = resp.choices[0].message.content or ""
        self.history.append({"role": "assistant", "content": reply})
        return reply


    def _try_extract_and_enqueue(self, text: str) -> bool:
        try:
            js = _extract_first_json(text)      # твоя функция очистки/детекта первого JSON-блока
            data = json.loads(js)
            if isinstance(data, dict):
                data = [data]
            if isinstance(data, list):
                enqueue_actions(data)           # твоя очередь; выполнит move_joints_by_name()
                print(f"[DIALOG] Взял {len(data)} действий из ответа LLM и отправил в очередь.")
                return True
        except Exception:
            pass
        return False

    def handle_text(self, user_text: str):
        reply = self._send_and_get(user_text)
        print(f"Ассистент: {reply}")
        self._try_extract_and_enqueue(reply)

    def loop_blocking(self):
        """
        Бесконечная терминальная сессия: push-to-talk, команды /q /history /clear /t.
        Не блокирует главный цикл — запускается в отдельном потоке.
        """
        print("Голосовой режим. Команды: /q — выход, /history — история, /clear — очистить, /t — ввод текстом, /help — справка.")
        self._ensure_system()
        while True:
            try:
                wav = record_push_to_talk()
                if wav is None:
                    cmd = input("\nКоманда: ").strip().lower()
                    if cmd in ("/q", "/quit", "quit", "exit"):
                        print("[DIALOG] Завершение диалогового потока.")
                        break
                    if cmd in ("/history", "/h"):
                        self._print_history(); continue
                    if cmd in ("/clear", "/c"):
                        self._ensure_system(); print("История очищена"); continue
                    if cmd in ("/t",):
                        txt = input("Текст: ").strip()
                        if txt:
                            self.handle_text(txt)
                        continue
                    if cmd in ("/help",):
                        print("Команды: /q — выход, /history — история, /clear — очистить, /t — ввод текстом.")
                        continue
                    print("Неизвестная команда. Введите /help для списка.")
                    continue

                if not wav:
                    print("Пустая запись.")
                    continue

                text = transcribe_wav_bytes(wav, _OPENAI_KEY)
                if not text:
                    print("Ничего не распознано.")
                    continue

                print(f"Вы (STT): {text}")
                reply = self._send_and_get(text)
                print(f"Ассистент: {reply}")
                self._try_extract_and_enqueue(reply)

            except KeyboardInterrupt:
                print("\n[DIALOG] Прерывание пользователем")
                break
            except Exception as e:
                print(f"[DIALOG] Ошибка: {e}")

    def _print_history(self):
        if len(self.history) <= 1:
            print("История пуста")
            return
        print("\n" + "="*60)
        print("История диалога:")
        print("="*60)
        for i, msg in enumerate(self.history[1:], 1):
            role = "Вы" if msg["role"] == "user" else "Ассистент"
            print(f"{i}. {role}: {msg['content']}")
        print("="*60)

# Глобальные хендлы фонового диалога
_DIALOG_THREAD = None
_DIALOG = None

def start_llm_dialog_thread():
    """Запустить диалог с LLM в отдельном потоке (один раз)."""
    global _DIALOG_THREAD, _DIALOG
    if _DIALOG_THREAD and _DIALOG_THREAD.is_alive():
        print("[DIALOG] Уже запущен.")
        return
    _DIALOG = RobotDialogLLM()
    _DIALOG_THREAD = threading.Thread(target=_DIALOG.loop_blocking, daemon=True)
    _DIALOG_THREAD.start()
    print("[DIALOG] Фоновый диалог запущен. Окно MuJoCo продолжает работать.")


# --- Few-shot via cosine similarity (new) ---
from sim_phrases import load_csv as load_cmds_csv, compute_similarities, DEFAULT_MODEL  # noqa
class LLMRealtimeWSAgent:
    """
    Потоковая транскрипция через OpenAI Realtime (WebSocket) + немедленное исполнение команд.
    Без локального VAD: используем серверный turn detection + частые commit'ы.
    """
    def __init__(self, samplerate=16000, chunk_ms=40, commit_ms=250):
        self.samplerate = samplerate
        self.chunk = int(samplerate * chunk_ms / 1000)
        self.commit_ms = commit_ms / 1000.0
        self.api_key = getattr(settings, "OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
        self.model = getattr(settings, "OPENAI_REALTIME_MODEL", "gpt-4o-mini-realtime-preview")
        self.ws = None
        self.running = False
        self.tx_thread = None
        self._last_commit = 0.0
        self._buf_text = ""

    # ---------- WS callbacks ----------
    def _on_open(self, ws):
        # Настройка сессии: PCM16 вход, серверный VAD, входная транскрипция RU, и не говорим вслух.
        cfg = {
            "type": "session.update",
            "session": {
                "instructions": "Ты интерпретируешь голос оператора в текст команды. Ничего не озвучивай.",
                "input_audio_format": {"type": "pcm16", "sample_rate_hz": self.samplerate},
                "turn_detection": {"type": "server_vad", "silence_duration_ms": 200},
                "input_audio_transcription": {"model": "gpt-4o-mini-transcribe", "language": "ru"}
            }
        }
        ws.send(json.dumps(cfg))
        # Запуск микрофона
        self.tx_thread = threading.Thread(target=self._mic_loop, daemon=True)
        self.tx_thread.start()
        print("[RT] Подключено. Говорите...")

    def _on_message(self, ws, message):
        # Ловим дельты транскрипции входа и финалы
        try:
            msg = json.loads(message)
        except Exception:
            return
        t = msg.get("type", "")
        # дельты текста входной транскрипции (названия событий могут меняться, читаем по полям)
        if ("transcript" in msg) and (t.endswith(".delta") or "input_audio_transcription" in t):
            self._buf_text += msg.get("transcript", "")

        # конец сегмента: speech_stopped / completed / done — исполняем
        if ("speech_stopped" in t) or ("completed" in t) or ("done" in t):
            text = (self._buf_text or msg.get("transcript", "")).strip()
            self._buf_text = ""
            if not text:
                return
            print(f"[RT USER] {text}")
            try:
                actions = parse_command(
                    text,
                    rag_k=getattr(settings, "RAG_K", 3),
                    rag_csv=getattr(settings, "RAG_CSV", "1000.csv"),
                    rag_json=getattr(settings, "RAG_JSON", "data_action"),
                )
                if actions:
                    with EXEC_LOCK:
                        enqueue_actions(actions)
            except Exception as e:
                print(f"[RT] Ошибка исполнения: {e}")

    def _on_error(self, ws, err):
        print(f"[RT] Ошибка WS: {err}")

    def _on_close(self, ws, *args):
        print("[RT] Соединение закрыто.")

    # ---------- MIC loop ----------
    def _mic_loop(self):
        self._last_commit = time.time()
        with sd.RawInputStream(samplerate=self.samplerate, channels=1, dtype="int16", blocksize=self.chunk) as stream:
            while self.running:
                try:
                    data, _ = stream.read(self.chunk)
                except Exception:
                    break
                b = bytes(data)
                if not b:
                    continue
                audio_b64 = base64.b64encode(b).decode("ascii")
                try:
                    self.ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": audio_b64}))
                except Exception:
                    break
                now = time.time()
                # Частые коммиты буфера — сервер начнёт отдавать транскрипцию очень быстро
                if (now - self._last_commit) >= self.commit_ms:
                    try:
                        self.ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
                    except Exception:
                        break
                    self._last_commit = now

    # ---------- Public API ----------
    def start(self):
        if self.running:
            return
        if not self.api_key:
            print("[RT] OPENAI_API_KEY пуст.")
            return
        self.running = True
        headers = [
            f"Authorization: Bearer {self.api_key}",
            "OpenAI-Beta: realtime=v1",
        ]
        url = f"wss://api.openai.com/v1/realtime?model={self.model}"
        self.ws = websocket.WebSocketApp(
            url,
            header=headers,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )
        threading.Thread(target=self.ws.run_forever, daemon=True).start()

    def stop(self):
        if not self.running:
            return
        self.running = False
        try:
            if self.ws:
                self.ws.close()
        except Exception:
            pass

class LLMMicAgent:
    """
    Push-to-talk агент:
    - start(): начать запись микрофона (держим ПРОБЕЛ)
    - stop_and_submit(): остановить запись и асинхронно: STT -> parse_command -> move_joints_by_name
    """
    def __init__(self, samplerate: int = 16000, channels: int = 1):
        self.samplerate = samplerate
        self.channels = channels
        self.dtype = "int16"
        self.frames = []
        self.stream = None
        self.recording = False
        self.api_key = getattr(settings, "OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))

    def _cb(self, indata, frames, time_, status):
        if status:
            print("[MIC]", status)
        self.frames.append(indata.copy())

    def start(self):
        if sd is None or sf is None:
            print("[MIC] Нет sounddevice/soundfile. Установите: pip install sounddevice soundfile")
            return
        if self.recording:
            return
        self.frames = []
        self.stream = sd.InputStream(
            samplerate=self.samplerate, channels=self.channels, dtype=self.dtype, callback=self._cb
        )
        self.stream.start()
        self.recording = True
        print("[MIC] Говорите… Держите ПРОБЕЛ. Отпустите для отправки.")

    def stop_and_submit(self):
        if not self.recording:
            return
        try:
            self.stream.stop()
            self.stream.close()
        except Exception:
            pass
        self.recording = False
        if not self.frames:
            print("[MIC] Пустая запись, пропускаю.")
            return

        # Собираем WAV в память
        bio = io.BytesIO()
        with sf.SoundFile(
            bio, mode="w", samplerate=self.samplerate, channels=self.channels, format="WAV", subtype="PCM_16"
        ) as f:
            for chunk in self.frames:
                f.write(chunk)
        wav_bytes = bio.getvalue()

        threading.Thread(target=self._worker, args=(wav_bytes,), daemon=True).start()

    def _stt(self, wav_bytes: bytes) -> str:
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY пуст — укажите в settings.py или переменной окружения.")
        url = "https://api.openai.com/v1/audio/transcriptions"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        # Быстрый стек: mini-transcribe -> whisper-1
        for model in ("gpt-4o-mini-transcribe", "whisper-1"):
            try:
                r = requests.post(
                    url,
                    headers=headers,
                    data={"model": model, "language": "ru"},
                    files={"file": ("speech.wav", wav_bytes, "audio/wav")},
                    timeout=45,
                )
                if r.status_code == 200:
                    return (r.json().get("text") or "").strip()
                else:
                    print(f"[MIC] STT {model}: {r.status_code} {r.text}")
            except Exception as e:
                print(f"[MIC] STT ошибка ({model}): {e}")
        return ""

    def _worker(self, wav_bytes: bytes):
        try:
            text = self._stt(wav_bytes)
        except Exception as e:
            print(f"[MIC] STT ошибка: {e}")
            return
        if not text:
            print("[MIC] Не распознано.")
            return

        print(f"[MIC → TEXT] {text!r}")
        try:
            actions = parse_command(
                text,
                rag_k=getattr(settings, "RAG_K", 3),
                rag_csv=getattr(settings, "RAG_CSV", "1000.csv"),
                rag_json=getattr(settings, "RAG_JSON", "data_action"),
            )
            print("[VOICE ACTIONS]", actions)
            enqueue_actions(actions)
        except Exception as e:
            print(f"[MIC] Ошибка выполнения команды: {e}")


# --- RAG block builder (text few-shots) ---
def _fewshot_block_text(
    user_cmd: str,
    k: int = 5,
    csv_path: str = "1000.csv",
    json_dir: str | Path = "data_action",
) -> str:
    """
    Returns a multiline text with k examples:
        Example #i
        Command: "..."
    Output JSON:
    [ ... ]
    If there are no examples, returns an empty string.
        """
    examples = _load_top_sim_examples(
        user_cmd=user_cmd,
        csv_path=csv_path,
        json_dir=json_dir,
        k=k,
        model_name=DEFAULT_MODEL,
    )
    if not examples:
        return ""

    lines = []
    lines.append("\n### Few-shot examples (follow the output format EXACTLY):")
    for i, (cmd_text, ex_json) in enumerate(examples, 1):
        try:
            if isinstance(ex_json, dict):
                ex_json = [ex_json]
            ex_json_str = json.dumps(ex_json, ensure_ascii=False, indent=2)
        except Exception:
            continue
        lines.append(f"\n# Example {i}")
        lines.append(f'Command: "{cmd_text}"')
        lines.append("Output JSON:")
        lines.append(ex_json_str)
    lines.append("\n### END examples\n")
    return "\n".join(lines)


def _load_top_sim_examples(
    user_cmd: str,
    csv_path: str = "1000.csv",
    json_dir: str | Path = "data_action",
    k: int = 5,
    model_name: str = DEFAULT_MODEL,
) -> list[tuple[str, dict]]:
    """
    Returns a list of pairs (command_text, example_json) up to length k,
    sorted by descending cosine similarity to user_cmd.
    Skips examples without a JSON file.
    Safely returns [] on any error (missing model/files, etc.).
    """
    try:
        df = load_cmds_csv(csv_path)
        ranked = compute_similarities(user_cmd, df, model_name)
    except Exception as e:
        print(f"[few-shot] skip (similarity failed): {e}")
        return []

    out: list[tuple[str, dict]] = []
    json_dir = Path(json_dir)
    for _, row in ranked.head(k).iterrows():
        cid = str(row["id"]).strip()
        cmd_text = str(row["command"]).strip()
        jf = json_dir / f"{cid}.json"
        if not jf.exists():
            continue
        try:
            with jf.open("r", encoding="utf-8") as f:
                ex = json.load(f)
            # normalize to list (as expected by your parser)
            if isinstance(ex, dict):
                ex = [ex]
            if not isinstance(ex, list):
                continue
        except Exception as e:
            print(f"[few-shot] bad json for id={cid}: {e}")
            continue
        out.append((cmd_text, ex))
    return out


def run_walk_blocking(num: int, dir_deg: float, spd: float):
    """Start walking and, without leaving the function, wait for the steps to finish,
    performing the same logic as in the main loop."""
    global steps_done, steps_needed, counter, target_dof_pos, old_sin_phase
    global action, obs, cmd, default_angles
    global m, d, opt, cam, scene, context, window

    start_new_steps(num, dir_deg, spd)

    while steps_done < steps_needed and not glfw.window_should_close(window):
        # === same block as in main loop ===
        na = 12  # forced to 10 in main
        tau = pd_control(
            target_dof_pos,
            d.qpos[7:7 + na],
            kps,
            np.zeros_like(kds),
            d.qvel[6:6 + na],
            kds
        )
        d.ctrl[:na] = tau

        mj.mj_step(m, d)
        counter += 1

        if counter % control_decimation == 0:
            qj = d.qpos[7:7 + na]
            dqj = d.qvel[6:6 + na]
            quat = d.qpos[3:7]
            omega = d.qvel[3:6]

            qj_scaled = (qj - default_angles) * dof_pos_scale
            dqj_scaled = dqj * dof_vel_scale
            grav = get_gravity_orientation(quat)
            omega_scaled = omega * ang_vel_scale

            period = GAIT_PERIOD
            time_in_sim = counter * simulation_dt
            phase = (time_in_sim % period) / period
            sin_phase = math.sin(2 * math.pi * phase)
            cos_phase = math.cos(2 * math.pi * phase)

            obs[:3] = omega_scaled
            obs[3:6] = grav
            obs[6:9] = cmd * cmd_scale
            obs[9:9 + na] = qj_scaled
            obs[9 + na: 9 + 2 * na] = dqj_scaled
            obs[9 + 2 * na: 9 + 3 * na] = action
            obs[9 + 3 * na: 9 + 3 * na + 2] = np.array([sin_phase, cos_phase])
            action[:] = POL[ACTIVE].act(obs)
            target_dof_pos = action * action_scale + default_angles

            if sin_phase >= 0 and old_sin_phase < 0:
                steps_done += 1
                print(f"Got step #{steps_done} / {steps_needed}")
                if steps_done >= steps_needed:
                    cmd[:] = 0.0
                    target_dof_pos = default_angles.copy()
            old_sin_phase = sin_phase

        # Render (as in main)
        width, height = glfw.get_framebuffer_size(window)
        viewport = mj.MjrRect(0, 0, width, height)
        mj.mjv_updateScene(m, d, opt, None, cam, mj.mjtCatBit.mjCAT_ALL, scene)
        mj.mjr_render(viewport, scene, context)
        glfw.swap_buffers(window)
        glfw.poll_events()


def _wrap_to_pi(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def _quat_yaw(qw: float, qx: float, qy: float, qz: float) -> float:
    # yaw around +Z (MuJoCo: Z-up)
    return math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))


def run_turn_blocking(angle_deg: float, spd_deg_s: float = 45.0):
    """
    Smoothly rotate around the body Z axis by the given angle (in degrees).
    Positive — counterclockwise when viewed from above (standard +Z).
    Blocking, like run_walk_blocking: keep running PD/gait until the turn completes.
    """
    global m, d, policy, counter, cmd, action, target_dof_pos
    global default_angles, obs, old_sin_phase
    # turn speed (rad/s) with the correct sign
    direction = 1.0 if angle_deg >= 0 else -1.0
    yaw_rate = direction * math.radians(abs(spd_deg_s))
    target_abs = math.radians(abs(angle_deg))

    # zero linear speed, set angular speed
    cmd[:] = [0.0, 0.0, yaw_rate]

    # initial yaw and accumulator with wraparound across -pi..pi
    qw, qx, qy, qz = d.qpos[3:7]
    last_yaw = _quat_yaw(qw, qx, qy, qz)
    acc = 0.0

    while acc < target_abs and not glfw.window_should_close(window):
        # === same as in your main loop ===
        na = 12  # fixed to 10 in your script
        tau = pd_control(
            target_dof_pos,
            d.qpos[7:7 + na],
            kps,
            np.zeros_like(kds),
            d.qvel[6:6 + na],
            kds
        )
        d.ctrl[:na] = tau

        mj.mj_step(m, d)
        counter += 1

        if counter % control_decimation == 0:
            qj = d.qpos[7:7 + na]
            dqj = d.qvel[6:6 + na]
            quat = d.qpos[3:7]
            omega = d.qvel[3:6]

            qj_scaled = (qj - default_angles) * dof_pos_scale
            dqj_scaled = dqj * dof_vel_scale
            grav = get_gravity_orientation(quat)
            omega_scaled = omega * ang_vel_scale

            period = GAIT_PERIOD
            time_in_sim = counter * simulation_dt
            phase = (time_in_sim % period) / period
            sin_phase = math.sin(2 * math.pi * phase)
            cos_phase = math.cos(2 * math.pi * phase)

            obs[:3] = omega_scaled
            obs[3:6] = grav
            obs[6:9] = cmd * cmd_scale
            obs[9:9 + na] = qj_scaled
            obs[9 + na: 9 + 2 * na] = dqj_scaled
            obs[9 + 2 * na: 9 + 3 * na] = action
            obs[9 + 3 * na: 9 + 3 * na + 2] = np.array([sin_phase, cos_phase])
            action[:] = POL[ACTIVE].act(obs)
            target_dof_pos = action * action_scale + default_angles

        # render
        width, height = glfw.get_framebuffer_size(window)
        viewport = mj.MjrRect(0, 0, width, height)
        mj.mjv_updateScene(m, d, opt, None, cam, mj.mjtCatBit.mjCAT_ALL, scene)
        mj.mjr_render(viewport, scene, context)
        glfw.swap_buffers(window)
        glfw.poll_events()

        # update accumulated rotation
        qw, qx, qy, qz = d.qpos[3:7]
        yaw_now = _quat_yaw(qw, qx, qy, qz)
        inc = _wrap_to_pi(yaw_now - last_yaw)
        acc += abs(inc)
        last_yaw = yaw_now

    # stop rotation
    cmd[:] = 0.0


# =========================
# Global variables (extended)
# =========================
# ... existing globals ...
demo_pause_between = 3  # Pause between demos (configurable)
_initialized = False
# --- NEW: review dialog (Confirm / Repeat / Reject + similarity) ---
def _review_demo_dialog(cid: str, cmd_text: str) -> tuple[str, int]:
    """
    Returns (decision, similarity), where:
      decision ∈ {'confirm','repeat','reject'}
      similarity ∈ [0..100] integer from the slider
    Blocking modal Tk window.
    """
    result = {"val": None, "sim": 50}  # default similarity = 50
    root = tk.Tk()
    root.title("Demo Review")
    root.geometry("560x200")
    root.attributes("-topmost", True)

    frm = tk.Frame(root, padx=12, pady=12)
    frm.pack(fill="both", expand=True)

    tk.Label(frm, text=f"id: {cid}", font=("TkDefaultFont", 11, "bold")).pack(anchor="w")
    tk.Label(frm, text=f"command: {cmd_text}", wraplength=520, justify="left").pack(anchor="w", pady=(6, 10))

    # Similarity slider
    sfrm = tk.Frame(frm)
    sfrm.pack(fill="x", pady=(0, 10))
    tk.Label(sfrm, text="How similar did it look? (0–100)").pack(anchor="w")
    sim_scale = tk.Scale(sfrm, from_=0, to=100, orient="horizontal", length=420)
    sim_scale.set(50)
    sim_scale.pack(anchor="w")

    btns = tk.Frame(frm)
    btns.pack(fill="x")

    def _set(v):
        result["val"] = v
        result["sim"] = int(sim_scale.get())
        root.destroy()

    tk.Button(btns, text="✅ Confirm", width=16, command=lambda: _set("confirm")).pack(side="left", padx=4)
    tk.Button(btns, text="🔁 Repeat",  width=16, command=lambda: _set("repeat")).pack(side="left", padx=4)
    tk.Button(btns, text="❌ Reject",   width=12, command=lambda: _set("reject")).pack(side="right", padx=4)

    root.mainloop()
    # default to reject if window closed without choice
    return (result["val"] or "reject", int(result["sim"]))


# --- NEW: log rejected samples ---
def _append_reject(
    reject_csv_path: str | Path,
    cid: str,
    cmd_text: str,
    similarity: int | float = None,
    reason: str = "manual_reject"
) -> None:
    p = Path(reject_csv_path)
    exists = p.exists()
    with p.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
       
        if not exists:
            w.writerow(["id", "command", "similarity", "reason", "ts"])
        w.writerow([
            cid,
            cmd_text,
            (None if similarity is None else int(similarity)),
            reason,
            time.strftime("%Y-%m-%d %H:%M:%S"),
        ])
def _resolve_joint_name(jname: str) -> str | None:
    """Подбирает имя сустава с учётом суффикса _joint и префиксов r1_/r2_."""
    if jname in joint_map:
        return jname
    # пробуем без суффикса "_joint"
    if jname.endswith("_joint"):
        base = jname[:-6]
        if base in joint_map:
            return base
    # если в JSON пришёл без префикса, а в сцене он есть
    for pref in ("r1_", "r2_"):
        cand = pref + jname
        if cand in joint_map:
            return cand
        if jname.endswith("_joint"):
            base = jname[:-6]
            cand2 = pref + base
            if cand2 in joint_map:
                return cand2
    return None


# =========================
# Demo function (UPDATED)
# =========================
def run_demo(csv_path="1000.csv", json_dir="data_action", reject_csv: str | Path = "reject.csv"):
    """
    Plays actions from json_dir based on csv_path.
    After each action, show a dialog: Confirm / Repeat / Reject + Similarity (0–100).
    'Repeat' resets to zero pose and replays until Confirm or Reject.
    'Reject' appends the sample and similarity to reject_csv.
    """
    ensure_initialized()
    commands = load_commands_csv(csv_path)
    if not commands:
        print("[INFO] Nothing to play. No commands found.")
        return

    print(f"[INFO] Starting demo: {len(commands)} commands")
    for row in commands:
        cid = row["id"]
        cmd_text = row["command"]
        actions = load_actions_json(json_dir, cid)
        if actions is None:
            print(f"[SKIP] id={cid} — no JSON found, command='{cmd_text}'")
            continue

        print("=" * 80)
        print(f"id: {cid}")
        print(f"command: {cmd_text}")

        try:
            move_joints_by_name(actions)
            print(f"[OK] executed id={cid}\n")
        except Exception as e:
            print(f"[ERR] execution failed for id={cid}: {e}\n")

        while True:
            decision, similarity = _review_demo_dialog(cid, cmd_text)
            if decision == "confirm":
                break
            elif decision == "reject":
                _append_reject(reject_csv_path=reject_csv,
               cid=cid,
               cmd_text=cmd_text,
               similarity=similarity,
               reason="manual_reject")

                print(f"[REJECT] id={cid} sim={similarity} -> {reject_csv}")
                break
            else:  # repeat
                do_reset()
                try:
                    move_joints_by_name(actions)
                except Exception as e:
                    print(f"[ERR] repeat failed for id={cid}: {e}")

        if demo_pause_between > 0:
            time.sleep(demo_pause_between)

    print("[DONE] Demo finished.")



m = d = policy = None
window = cam = opt = scene = context = None
joint_map = {}
joint_index_map = {}
INIT_QPOS = None
simulation_dt = control_decimation = None
kps = kds = default_angles = None
ang_vel_scale = dof_pos_scale = dof_vel_scale = action_scale = None
cmd_scale = None
num_actions = num_obs = None
movement = {}

R = {} 
POL = {}

# =========================
# Saving helpers
# =========================

def _ensure_list_of_dicts(actions: Any) -> list[dict]:
    """Normalize actions to a list of objects."""
    if actions is None:
        return []
    if isinstance(actions, dict):
        return [actions]
    if isinstance(actions, list):
        # accept list of lists (old frame format); still save as-is
        return actions
    raise ValueError("Actions must be list or dict")


def save_actions_json(
    actions: Any,
    command_id: str | None = None,
    command_text: str | None = None,
    out_dir: str | Path = "captured",
    filename: str | None = None
) -> Path:
    """
    Save an array of actions to JSON in the captured/ directory.
    Additionally, append a record to data_captured.csv (id, command, json_file).
    """
    actions_norm = _ensure_list_of_dicts(actions)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not filename:
        filename = f"capture_{int(time.time())}_{uuid.uuid4().hex[:8]}.json"
    p = out_dir / filename

    with p.open("w", encoding="utf-8") as f:
        json.dump(actions_norm, f, ensure_ascii=False, indent=2)

    print(f"[SAVE] {p}")

    # write to csv (id, command, json_file)
    log_capture_to_csv(command_id, command_text, filename)

    return p


def log_capture_to_csv(
    command_id: str | None,
    command_text: str | None,
    json_filename: str,
    csv_path: str | Path = "data_captured.csv"
) -> None:
    """
    Append a record about the saved json to data_captured.csv.
    Format: id,command,json_file
    """
    csv_path = Path(csv_path)
    file_exists = csv_path.exists()

    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["id", "command", "json_file"])
        writer.writerow([
            command_id or "",
            command_text or "",
            json_filename
        ])


def save_actions_json_by_id(actions: Any, command_id: str, out_dir: str | Path = "data_action") -> Path:
    """
    Save an array of actions to file data_action/<id>.json (overwriting existing file).
    """
    actions_norm = _ensure_list_of_dicts(actions)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{command_id}.json"
    with p.open("w", encoding="utf-8") as f:
        json.dump(actions_norm, f, ensure_ascii=False, indent=2)
    print(f"[SAVE-ID] {p}")
    return p


def generate_from_csv_and_play(
    csv_path: str | Path = "1000.csv",
    out_dir: str | Path = "data_action",
    start_index: int = 0,
    max_count: int | None = None,
    interval_sec: float = 0.0,
    preview_only: bool = False,
) -> None:
    """
    Read a CSV (columns: id, command), for each row:
      - generate actions via parse_command(command),
      - play them on the robot (move_joints_by_name),
      - save to data_action/<id>.json (if preview_only=False).

    Params:
      start_index  — which row to start from (0-based)
      max_count    — how many rows to process (None = until end)
      interval_sec — pause between examples (may be 0)
      preview_only — if True, show but do NOT save by id
    """
    ensure_initialized()

    p = Path(csv_path)
    if not p.exists():
        print(f"[ERR] CSV not found: {p.resolve()}")
        return

    rows: list[dict] = []
    with p.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            print("[ERR] CSV has no header")
            return
        fmap = {k.lower().strip(): k for k in reader.fieldnames}
        id_col = fmap.get("id")
        cmd_col = fmap.get("command") or fmap.get("cmd")
        if not id_col or not cmd_col:
            print("[ERR] CSV must contain columns: id, command")
            return
        for row in reader:
            cid = str(row.get(id_col, "")).strip()
            cmd = str(row.get(cmd_col, "")).strip()
            if cid and cmd:
                rows.append({"id": cid, "command": cmd})

    if start_index < 0 or start_index >= len(rows):
        print(f"[INFO] start_index out of range (0..{len(rows)-1})")
        return

    end_index = len(rows) if max_count is None else min(len(rows), start_index + max_count)

    print(f"[INFO] Generating {end_index - start_index} samples from {p.resolve()}")
    for i in range(start_index, end_index):
        cid = rows[i]["id"]
        cmd_text = rows[i]["command"]
        print("=" * 80)
        print(f"id: {cid}")
        print(f"command: {cmd_text}")
        print("Thinking...")
        try:
            actions = parse_command(cmd_text)  # LLM → list of action objects
        except Exception as e:
            print(f"[ERR] parse failed for id={cid}: {e}")
            continue

        try:
            # Play on the robot
            move_joints_by_name(actions)
        except Exception as e:
            print(f"[ERR] play failed for id={cid}: {e}")
            # even if playing failed, save anyway (if desired)
            if not preview_only:
                save_actions_json_by_id(actions, cid, out_dir=out_dir)
            continue

        # Save <id>.json
        if not preview_only:
            save_actions_json_by_id(actions, cid, out_dir=out_dir)
            log_capture_to_csv(cid, cmd_text, p.name)
        # Pause between examples
        if interval_sec > 0:
            time.sleep(interval_sec)

    print("[DONE] CSV generation completed.")

def _deg(x: float) -> float:
    return float(np.degrees(x))

def _axis_letter(ax: np.ndarray) -> str:
    if np.linalg.norm(ax) < 1e-8:
        return "-"
    return "XYZ"[int(np.argmax(np.abs(ax)))]

def _format_joint_range(j: int) -> str:
    jtype = m.jnt_type[j]
    limited = bool(m.jnt_limited[j])
    rng = m.jnt_range[j]
    if jtype == mj.mjtJoint.mjJNT_HINGE:
        return f"{np.degrees(rng[0]):7.2f}..{np.degrees(rng[1]):7.2f} deg" if limited else "unlimited"
    if jtype == mj.mjtJoint.mjJNT_SLIDE:
        return f"{rng[0]:7.3f}..{rng[1]:7.3f} m" if limited else "unlimited"
    if jtype == mj.mjtJoint.mjJNT_BALL:
        return "quat (no range)"
    return "free (6-DoF)"

def _current_angle_str(j: int) -> str:
    jtype = m.jnt_type[j]
    adr = m.jnt_qposadr[j]
    if jtype == mj.mjtJoint.mjJNT_HINGE:
        return f"{_deg(d.qpos[adr]):.2f} deg"
    if jtype == mj.mjtJoint.mjJNT_SLIDE:
        return f"{d.qpos[adr]:.4f} m"
    return "-"  # BALL/FREE

def build_current_pose_tables() -> str:
  
    if m is None or d is None:
        return "[WARN] current pose unavailable: model/data not initialized"

    lines = []
    lines.append("Joints overview (current):")
    lines.append(f"{'id':>3} {'name':<28} {'range':>21} {'axis':>4} {'curr':>10}")
    lines.append("-" * 70)
    for j in range(m.njnt):
        name = mj.mj_id2name(m, mj.mjtObj.mjOBJ_JOINT, j) or f"joint_{j}"
        rng = _format_joint_range(j)
        jtype = m.jnt_type[j]
        ax = _axis_letter(m.jnt_axis[j]) if jtype in (mj.mjtJoint.mjJNT_HINGE, mj.mjtJoint.mjJNT_SLIDE) else "-"
        curr = _current_angle_str(j)
        lines.append(f"{j:3d} {name:<28} {rng:>21} {ax:>4} {curr:>10}")

    return "\n".join(lines)

# ======= OBJECTS / DISTANCES / MINIMAP =======
OBJECT_PREFIXES = ("box_", "cube_", "target_", "obj_")  # расширяй по своим названиям
OBJECTS: dict[str, dict] = {}      # name -> {"type": "geom", "pos": np.ndarray(3)}
DIST: dict[str, dict[str, float]] = {}  # robot -> {object -> distance}
MINIMAP: dict[str, Any] = {}       # {"center": np.ndarray(3), "cell": float, "grid": list[list[list[str]]]}
MINIMAP_CELL = 1.0                 # 1м клетка
MINIMAP_SIZE = 3                   # 3x3

def _geom_pos(name: str) -> np.ndarray | None:
    try:
        gid = mj.mj_name2id(m, mj.mjtObj.mjOBJ_GEOM, name)
        if gid < 0: 
            return None
        return d.geom_xpos[gid].copy()
    except Exception:
        return None

def get_robot_world_pos(p: str) -> np.ndarray:
    """xyz центра базового фри-джойнта робота p ('r1_', 'r2_', ...)."""
    mp = R[p]
    return d.qpos[mp["base_qpos_adr"] : mp["base_qpos_adr"]+3].copy()

def discover_objects(prefixes: tuple[str, ...] = OBJECT_PREFIXES) -> dict[str, dict]:
    """Сканирует все геомы и берёт те, чьи имена начинаются с prefixes."""
    out = {}
    for i in range(m.ngeom):
        nm = mj.mj_id2name(m, mj.mjtObj.mjOBJ_GEOM, i) or ""
        for pref in prefixes:
            if nm.startswith(pref):
                pos = d.geom_xpos[i].copy()
                out[nm] = {"type": "geom", "pos": pos}
                break
    return out

def update_world_index() -> None:
    """Обнови список объектов и матрицу расстояний."""
    global OBJECTS, DIST
    OBJECTS = discover_objects()
    DIST = {p: {} for p in (R.keys())}
    for p in DIST.keys():
        rp = get_robot_world_pos(p)
        for nm, meta in OBJECTS.items():
            op = meta["pos"]
            # горизонтальная дистанция XY (для навигации так удобнее); хочешь — поставь 3D
            dist = float(np.linalg.norm(op[:2] - rp[:2]))
            DIST[p][nm] = dist

def build_minimap(center: np.ndarray, size: int = MINIMAP_SIZE, cell: float = MINIMAP_CELL) -> dict:
    """
    Возвращает структуру с 3×3 клетками и списками объектов по клеткам.
    Центр — (x,y) в мире, сетка осевая.
    """
    half = size // 2
    grid = [[[] for _ in range(size)] for __ in range(size)]
    cx, cy = center[:2]
    for nm, meta in OBJECTS.items():
        x, y = meta["pos"][:2]
        ix = int(np.floor((x - (cx - half*cell)) / cell))
        iy = int(np.floor((y - (cy - half*cell)) / cell))
        # координаты в пределах 0..size-1
        if 0 <= ix < size and 0 <= iy < size:
            grid[size-1-iy][ix].append(nm)  # инвертируем y, чтобы «верх» был сверху
    return {"center": center.copy(), "cell": cell, "size": size, "grid": grid}

def render_minimap_to_text(mm: dict) -> str:
    """ASCII-миникарта: в клетке показываем количество объектов (или первый символ названия)."""
    lines = []
    lines.append(f"MiniMap {mm['size']}x{mm['size']} (cell={mm['cell']}m), center=({mm['center'][0]:.2f},{mm['center'][1]:.2f})")
    for row in mm["grid"]:
        cells = []
        for objs in row:
            if not objs:
                cells.append(" . ")
            else:
                # покажем количество (или, если хочешь, objs[0][:1])
                n = len(objs)
                cells.append(f"{n:2d} ")
        lines.append("".join(cells))
    return "\n".join(lines)

def _append_minimap_to_prompt(txt: str, user_cmd: str | None = None) -> str:
    """
    Appends the live scene context to the prompt:
      • ASCII minimap centered at ACTIVE robot
      • distances table robot→object
      • EN instruction to pick r1_/r2_ prefix by proximity
    Falls back silently if scene not initialized.
    """
    try:
        # Refresh objects & distances
        update_world_index()
        # Same center as in the 'M' hotkey: ACTIVE robot
        center_robot = ACTIVE if ('ACTIVE' in globals() and ACTIVE in R) else next(iter(R.keys()))
        mm = build_minimap(center=get_robot_world_pos(center_robot))
        mm_txt = render_minimap_to_text(mm)
        dist_txt = distances_summary_text()

        # Important instruction (EN) for the LLM:
        instr = ("Instruction: Choose the joint-name prefix 'r1_' or 'r2_' "
                 "based on which robot is closer to the object referenced in the Command."
                 "Not forget to add the field 'robot' to the json answer"
                 "In this task (to grab the object) you have to base on the example strict (repeat actions from its)")

        block = "\n".join([
            "\n### Scene context",
            mm_txt,
            "",
            dist_txt,
            "",
            instr
        ])
        return (txt or "") + "\n" + block + "\n"
    except Exception as e:
        print(f"[WARN] can't build minimap block for prompt: {e}")
        return txt


def distances_summary_text() -> str:
    """Короткая табличка расстояний (сортировано по объектам для каждого робота)."""
    if not DIST:
        return "(no distances yet)"
    lines = ["[Distances robot→object (XY, m)]"]
    for p in sorted(DIST.keys()):
        lines.append(f"{p}:")
        for nm, val in sorted(DIST[p].items(), key=lambda kv: kv[1]):
            lines.append(f"  - {nm}: {val:.2f}")
    return "\n".join(lines)

def guess_target_object_from_text(text: str) -> str | None:
    """Пробуем найти в тексте имя известного объекта (простое совпадение по подстроке, без регистра)."""
    t = text.lower()
    # точное совпадение по известным именам
    by_name = [nm for nm in OBJECTS.keys() if nm.lower() in t]
    if by_name:
        # если несколько — возьмём самое длинное (обычно более специфичное)
        return sorted(by_name, key=len, reverse=True)[0]
    # fallback: по префиксам (например 'box', 'cube', 'target')
    for pref in ("box", "cube", "target", "obj"):
        if pref in t:
            # выберем ближайший к активному роботу
            p0 = ACTIVE if ACTIVE in DIST else next(iter(DIST.keys()))
            cands = [nm for nm in OBJECTS.keys() if nm.lower().startswith(pref)]
            if cands:
                return min(cands, key=lambda nm: DIST[p0].get(nm, float("inf")))
    return None

def pick_best_robot_for_object(obj_name: str) -> str | None:
    if obj_name not in OBJECTS or not DIST:
        return None
    best = None
    best_val = float("inf")
    for p, table in DIST.items():
        v = table.get(obj_name, float("inf"))
        if v < best_val:
            best_val, best = v, p
    return best


def _append_current_pose_tables_to_prompt(txt: str) -> str:
 
    try:
        block = build_current_pose_tables()
        print(block)
        return (txt or "") + "\n\n" + block + "\n"
    except Exception as e:
        print(f"[WARN] can't build current pose tables: {e}")
        return txt

def _normalize_robots(sel=None) -> list[str]:
    """
    sel: None|'both'|'r1'|'r2'|'r1_'|'r2_'|['r1_','r2_']|...
  
    """
    if sel is None:
        return [ACTIVE] if ACTIVE in R else list(R.keys())
    if isinstance(sel, str):
        s = sel.strip().lower()
        if s in ("both", "all", "r12", "r1+r2"):
            return [p for p in ("r1_","r2_") if p in R]
        if not s.endswith("_"):
            s += "_"
        return [s] if s in R else []
    # список
    out = []
    for x in sel:
        xs = str(x).strip().lower()
        if not xs.endswith("_"):
            xs += "_"
        if xs in R:
            out.append(xs)
    return out

# ==== ADD (глобальные настройки) ====
IDLE_AFTER = 5   # сек тишины до «заморозки»
IDLE_SLEEP = 0.05  # интервал ожидания в паузе
# ==== ADD (утилиты рядом с _one_tick) ====
def _is_idle() -> bool:
    if manual_override:
        return False
    # нет активных таймеров ходьбы/поворота
    if any(WALK_UNTIL.get(p, 0.0) or TURN_TGT.get(p, 0.0) > 0.0 for p in robots):
        return False
    # нет задач поз
    if any(ACTIVE_TASK.get(p) or TASKS.get(p) for p in robots):
        return False
    # нет управляющих команд
    if any(np.any(cmd_vec.get(p, np.zeros(3))) for p in robots):
        return False
    return True

def _maybe_freeze() -> bool:
    """Возвращает True, если заморозили кадр и шагать физикой не надо."""
    global last_activity_time
    if not _is_idle():
        last_activity_time = time.time()
        return False
    if time.time() - last_activity_time < IDLE_AFTER:
        return False

    # стоп торки и скорости — «застывшее» состояние
    d.ctrl[:] = 0
    d.qvel[:] = 0
    d.qacc[:] = 0
    d.qfrc_applied[:] = 0
    d.xfrc_applied[:] = 0

    # только перерисовка + ждём события, физику не шагаем
    width, height = glfw.get_framebuffer_size(window)
    viewport = mj.MjrRect(0, 0, width, height)
    mj.mjv_updateScene(m, d, opt, None, cam, mj.mjtCatBit.mjCAT_ALL, scene)
    mj.mjr_render(viewport, scene, context)
    glfw.swap_buffers(window)
    glfw.wait_events_timeout(IDLE_SLEEP)
    return True

def ensure_initialized():
    global _initialized, m, d, policy, INIT_QPOS
    global simulation_dt, control_decimation, kps, kds, default_angles
    global ang_vel_scale, dof_pos_scale, dof_vel_scale, action_scale, cmd_scale
    global num_actions, num_obs
    global joint_map, joint_index_map, ALLOWED_JOINTS
    global window, cam, opt, scene, context
    global R  
    global POL 

    if _initialized:
        return

    # ---- file paths (as in __main__) ----
   # ---- file paths ----
    name_ = "h1_2"
    config_file = f"unitree_rl_gym/deploy/deploy_mujoco/configs/{name_}.yaml"
    policy_path  = f"unitree_rl_gym/deploy/pre_train/{name_}/motion.pt"
    xml_path     = f"unitree_rl_gym/resources/robots/{name_}/scene_twins_obj.xml"

    # ---- load config ----
    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    simulation_dt      = config["simulation_dt"]
    control_decimation = config["control_decimation"]
    kps = np.array(config["kps"], dtype=np.float32)
    kds = np.array(config["kds"], dtype=np.float32)
    default_angles = np.array(config["default_angles"], dtype=np.float32)
    ang_vel_scale  = config["ang_vel_scale"]
    dof_pos_scale  = config["dof_pos_scale"]
    dof_vel_scale  = config["dof_vel_scale"]
    action_scale   = config["action_scale"]
    cmd_scale      = np.array(config["cmd_scale"], dtype=np.float32)
    num_actions    = config["num_actions"]
    num_obs        = config["num_obs"]


    m = mj.MjModel.from_xml_path(xml_path)
    d = mj.MjData(m)
    m.opt.timestep = simulation_dt

    # ---- helpers ----
    def _jid(name: str) -> int:
        return mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, name)

    def _aid(name: str) -> int:
        return mj.mj_name2id(m, mj.mjtObj.mjOBJ_ACTUATOR, name)

    # FIX: 12 DOF ног с коленями и правильным порядком
    JOINTS = [
        "left_hip_yaw_joint", "left_hip_pitch_joint", "left_hip_roll_joint",
        "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
        "right_hip_yaw_joint", "right_hip_pitch_joint", "right_hip_roll_joint",
        "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    ]


    def build_robot_maps(prefix: str):
    
        base_candidates = []
        for j in range(m.njnt):
            if m.jnt_type[j] == mj.mjtJoint.mjJNT_FREE:
                nm = mj.mj_id2name(m, mj.mjtObj.mjOBJ_JOINT, j) or ""
                if nm.startswith(prefix):
                    base_candidates.append(j)
        assert base_candidates, f"FREE joint for prefix '{prefix}' not found"
        jid_base = base_candidates[0]

        base_qpos_adr = m.jnt_qposadr[jid_base]
        base_qvel_adr = m.jnt_dofadr[jid_base]

        jids      = np.array([_jid(prefix + j) for j in JOINTS], dtype=int)
        qpos_adrs = np.array([m.jnt_qposadr[j] for j in jids], dtype=int)
        dof_adrs  = np.array([m.jnt_dofadr[j]  for j in jids], dtype=int)
        act_ids   = np.array([_aid(prefix + j) for j in JOINTS], dtype=int) 

        return {
            "base_qpos_adr": base_qpos_adr,
            "base_qvel_adr": base_qvel_adr,
            "qpos_adrs": qpos_adrs,
            "dof_adrs":  dof_adrs,
            "act_ids":   act_ids,
        }


    R = {
        "r1_": build_robot_maps("r1_"),
        "r2_": build_robot_maps("r2_"),
    }
    _build_upper_hold_maps() 


    

    m.opt.timestep = simulation_dt

    # ---- allowed joints map ----
    ALLOWED_JOINTS = build_allowed_joints_from_model(m)

    # ---- IMPORTANT: joint_map first, then joint_index_map ----
    joint_map = {}
    for j in range(m.njnt):
        name = mj.mj_id2name(m, mj.mjtObj.mjOBJ_JOINT, j) or f"joint_{j}"
        joint_map[name] = m.jnt_qposadr[j]

    PREFERRED_FRAME_ORDER = [
        "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow",
        "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow",
    ]
    joint_index_map = build_joint_index_map(joint_map, PREFERRED_FRAME_ORDER)

    # ---- reset and store initial pose ----
    mj.mj_resetData(m, d)
    mj.mj_forward(m, d)
    INIT_QPOS = d.qpos.copy()

    # ---- policy ----
       
    POLICY_PATHS = {
        "r1_": "unitree_rl_gym/deploy/pre_train/h1_2/motion.pt",
        "r2_": "unitree_rl_gym/deploy/pre_train/h1_2/motion.pt",  # или свой .pt
    }
    POL = {p: PolicyWrapper(POLICY_PATHS[p]) for p in ("r1_","r2_") if p in R}

    # ---- use the already created viewer window ----
    if window is None:
        raise RuntimeError("GLFW window is not initialized. Run from main viewer first.")
    # ensure context is active (just in case)
    try:
        cur_ctx = glfw.get_current_context()
    except Exception:
        cur_ctx = None
    if not cur_ctx:
        glfw.make_context_current(window)

    # cam/opt/scene/context are created in __main__; if missing — create here
    if cam is None:
        cam = mj.MjvCamera()
        cam.azimuth = 180
        cam.elevation = -15
        cam.distance = 4.0
        cam.lookat = np.array([0.0, 0.0, 0.8])
    if opt is None:
        opt = mj.MjvOption()
    if scene is None:
        scene = mj.MjvScene(m, maxgeom=10000)
    if context is None:
        context = mj.MjrContext(m, mj.mjtFontScale.mjFONTSCALE_100)

    _initialized = True
    print("[whole_body] initialized for dataset player (using main viewer window)")


# =========================
# Global constants & aliases
# =========================

def build_allowed_joints_from_model(m: mj.MjModel) -> set[str]:
    """
    Collect allowed joint names from the model.
    We include HINGE and SLIDE joints and exclude FREE/BALL joints.
    """
    allowed = set()
    for j in range(m.njnt):
        jtype = m.jnt_type[j]
        if jtype in (mj.mjtJoint.mjJNT_HINGE, mj.mjtJoint.mjJNT_SLIDE):
            name = mj.mj_id2name(m, mj.mjtObj.mjOBJ_JOINT, j) or f"joint_{j}"
            allowed.add(name)
    return allowed


def build_joint_index_map(joint_map: Dict[str, int],
                          preferred_order: List[str] | None = None) -> Dict[int, str]:
    if preferred_order is None:
        preferred_order = [
            "left_shoulder_pitch","left_shoulder_roll","left_shoulder_yaw","left_elbow_pitch","left_elbow_roll",
            "right_shoulder_pitch","right_shoulder_roll","right_shoulder_yaw","right_elbow_pitch","right_elbow_roll",
        ]

    names, missing = [], []
    for base in preferred_order:
        if base in joint_map:
            names.append(base); continue
        found = None
        for pref in ("r1_","r2_"):
            for cand in (f"{pref}{base}_joint", f"{pref}{base}"):
                if cand in joint_map:
                    found = cand; break
            if found: break
        if found:
            names.append(found)
        else:
            missing.append(base)

    if missing:
        print("Warning: missing joints for frame control:", ", ".join(missing))
    return {i: n for i, n in enumerate(names)}


# Simple aliases the LLM might produce
JOINT_ALIASES = {
    "right_arm": "right_shoulder_pitch",
    "left_arm": "left_shoulder_pitch",
    "right_hand": "right_elbow",
    "left_hand": "left_elbow",
}

# Cache for parsed commands
command_cache: Dict[str, Any] = {}

# Initialize LLM once
LLM = build_llm()


# =========================
# LLM helpers
# =========================

def _llm_chat(messages: List[Dict[str, str]], max_tokens: int | None = None, temperature: float | None = None) -> str:
    """Single entry point into the LLM with default settings from `settings`."""
    return LLM.chat(
        messages,
        # max_tokens=max_tokens if max_tokens is not None else settings.LLM_MAX_TOKENS,
        temperature=temperature if temperature is not None else settings.LLM_TEMPERATURE,
    )


def _vlm_chat(user_text: str, image_path: str, max_tokens: int | None = None, temperature: float | None = None) -> str:
    return LLM.chat_vision(
        user_text,
        image_path,
        # max_tokens=max_tokens if max_tokens is not None else settings.LLM_MAX_TOKENS,
        temperature=temperature if temperature is not None else settings.LLM_TEMPERATURE,
    )


def open_cmd_with_image_window():
    root = tk.Tk()
    root.title("Command + Image")
    root.geometry("640x240")

    state = {"path": None}

    frm = tk.Frame(root)
    frm.pack(fill="both", expand=True, padx=10, pady=10)

    tk.Label(frm, text="Natural-language command:").pack(anchor="w")
    txt = tk.Text(frm, width=80, height=6)
    txt.pack(fill="both", expand=True, pady=(0, 8))

    path_lbl = tk.Label(frm, text="No image selected")
    path_lbl.pack(anchor="w")

    def choose_file():
        path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.gif *.webp"), ("All files", "*.*")]
        )
        if path:
            state["path"] = path
            path_lbl.config(text=f"Image: {path}")

    btns = tk.Frame(frm)
    btns.pack(fill="x")
    tk.Button(btns, text="Choose image…", command=choose_file).pack(side="left")

    result = {"text": None, "image_path": None}

    def apply_and_close():
        t = txt.get("1.0", "end").strip()
        if not t or not state["path"]:
            path_lbl.config(text="Need both: command text AND an image file.")
            return
        result["text"] = t
        result["image_path"] = state["path"]
        root.destroy()

    def cancel():
        root.destroy()

    tk.Button(btns, text="Apply", command=apply_and_close).pack(side="right", padx=6)
    tk.Button(btns, text="Cancel", command=cancel).pack(side="right")

    root.mainloop()
    return result if result["text"] and result["image_path"] else None


def _read_system_prompt() -> str:
    """Load the base system prompt from settings.PROMPT_PATH."""
    with open(settings.PROMPT_PATH, "r", encoding="utf-8") as f:
        return f.read()


# =========================
# JSON cleaning & parsing
# =========================

def _clean_json_block(s: str) -> str:
    """
    Remove provider markers, code fences, and single-line comments; return raw JSON-ish text.
    Does not parse — only cleans.
    """
    t = s.strip()

    # 1) Remove provider-style markers like <|start|>assistant<|...|>
    t = re.sub(r"<\|.*?\|>", "", t)

    # 2) Remove Markdown code fences: ```json ... ``` or ```
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.IGNORECASE | re.DOTALL)

    # 3) Remove single-line comments starting with // ...
    t = re.sub(r"//.*?(?=\n|$)", "", t)

    return t.strip()


def _extract_first_json(s: str) -> str:
    """
    Try to extract the first valid JSON object/array from text.
    Returns a JSON string or raises ValueError.
    """
    txt = _clean_json_block(s)

    # Quick path: the whole string is JSON
    try:
        json.loads(txt)
        return txt
    except Exception:
        pass

    # Otherwise, scan for {...} or [...] candidates and test them
    candidates = re.findall(r"(\{.*\}|\[.*\])", txt, flags=re.DOTALL)
    for cand in candidates:
        cand_stripped = cand.strip()
        try:
            json.loads(cand_stripped)
            return cand_stripped
        except Exception:
            continue

    # Last resort: literal_eval for cases with single quotes
    try:
        obj = ast.literal_eval(txt)
        if isinstance(obj, (list, dict)):
            return json.dumps(obj, ensure_ascii=False)
    except Exception:
        pass

    raise ValueError(f"Could not extract JSON from: {s}")

# ===== Upper-body zero-hold (для обоих роботов) =====
HOLD_UPPER_ZERO = True
UPPER_BASENAMES = [
    # ===== LEFT ARM =====
    "left_shoulder_pitch","left_shoulder_roll","left_shoulder_yaw",
    "left_elbow_pitch","left_elbow_roll",
    "left_wrist_pitch","left_wrist_yaw",
    # Left hand — thumb + fingers
    "L_thumb_proximal_yaw","L_thumb_proximal_pitch","L_thumb_intermediate","L_thumb_distal",
    "L_index_proximal","L_index_intermediate",
    "L_middle_proximal","L_middle_intermediate",
    "L_ring_proximal","L_ring_intermediate",
    "L_pinky_proximal","L_pinky_intermediate",

    # ===== RIGHT ARM =====
    "right_shoulder_pitch","right_shoulder_roll","right_shoulder_yaw",
    "right_elbow_pitch","right_elbow_roll",
    "right_wrist_pitch","right_wrist_yaw",
    # Right hand — thumb + fingers
    "R_thumb_proximal_yaw","R_thumb_proximal_pitch","R_thumb_intermediate","R_thumb_distal",
    "R_index_proximal","R_index_intermediate",
    "R_middle_proximal","R_middle_intermediate",
    "R_ring_proximal","R_ring_intermediate",
    "R_pinky_proximal","R_pinky_intermediate",
]

HOLD_UPPER_QPOS: dict[str, np.ndarray] = {}   # по роботу -> индексы qpos
HOLD_UPPER_DOF:  dict[str, np.ndarray] = {}   # по роботу -> индексы dof (для qvel=0)

def _build_upper_hold_maps():
    """Собираем индексы верхних суставов (qpos и dof) по каждому роботу r1_/r2_."""
    global HOLD_UPPER_QPOS, HOLD_UPPER_DOF
    HOLD_UPPER_QPOS = {}
    HOLD_UPPER_DOF  = {}

    def _jid_by_candidates(cands: list[str]) -> int | None:
        for nm in cands:
            try:
                j = mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, nm)
                if j >= 0:
                    return j
            except Exception:
                pass
        return None

    for p in R.keys():  # "r1_", "r2_"
        qpos_idx = []
        dof_idx  = []
        for base in UPPER_BASENAMES:
            # пробуем оба варианта имён: с "_joint" и без, с нужным префиксом
            cands = [f"{p}{base}_joint", f"{p}{base}"]
            j = _jid_by_candidates(cands)
            if j is None:
                continue
            qpos_idx.append(int(m.jnt_qposadr[j]))
            dof_idx.append(int(m.jnt_dofadr[j]))
        if qpos_idx:
            HOLD_UPPER_QPOS[p] = np.array(sorted(set(qpos_idx)), dtype=int)
            HOLD_UPPER_DOF[p]  = np.array(sorted(set(dof_idx)),  dtype=int)
        else:
            HOLD_UPPER_QPOS[p] = np.array([], dtype=int)
            HOLD_UPPER_DOF[p]  = np.array([], dtype=int)
    print(f"[HOLD] upper joints collected:",
          {p: (len(HOLD_UPPER_QPOS[p]), len(HOLD_UPPER_DOF[p])) for p in R.keys()})

# =========================
# Natural-language command parsing
# =========================

def parse_command(command: str, rag_k: int = 5, rag_csv: str = "1000.csv", rag_json: str = "data_action"):
    """
    Convert a natural-language user command into a normalized list of JSON actions.
    New features:
      • Inject few-shot from top-5 similar examples (cosine similarity) with gold JSON answers.
      • Carefully extract the first valid JSON from the model's response.
      • Soft normalization of action fields (type, numeric values, joint names).

    Environment requirements (optional — safe fallbacks are used if absent):
      _read_system_prompt() -> str
      _llm_chat(messages: list[dict], max_tokens: int, temperature: float) -> str
      _extract_first_json(text: str) -> str
      settings.LLM_TEMPERATURE : float
      ALLOWED_JOINTS : set[str]
      JOINT_ALIASES : dict[str, str]
      command_cache : dict[str, Any]
      _load_top_sim_examples(user_cmd, csv_path, json_dir, k, model_name) -> list[tuple[str, list|dict]]
      DEFAULT_MODEL : str
    """
    import json
    from pathlib import Path

    # ---------- safe defaults/globals ----------
    try:
        temp = float(getattr(settings, "LLM_TEMPERATURE", 0.2))
    except Exception:
        temp = 0.2

    # Command cache (if not defined above)
    global command_cache
    if "command_cache" not in globals() or not isinstance(globals().get("command_cache"), dict):
        command_cache = {}

    # Normative lists (use module-level ones if available)
    ALLOWED = set(globals().get("ALLOWED_JOINTS", []))
    ALIASES = dict(globals().get("JOINT_ALIASES", {}))

    def _read_prompt_safe() -> str:
        try:
            return str(_read_system_prompt()).strip()
        except Exception:
            # minimal system prompt by default
            return (
                "You are a strict JSON planner. "
                "Given a human 'Command', respond with JSON ONLY: a list of actions, "
                "each action being an object. No prose. Use unicode, keep keys in snake_case."
            )

    def _few_shots(command_text: str, k: int = 5, csv_path="1000.csv", json_dir="data_action") -> list[tuple[str, list]]:
        """
        Get pairs (example_command, gold_JSON_as_list).
        Safely returns [] if something is missing.
        """
        try:
            examples = _load_top_sim_examples(
                user_cmd=command_text,
                csv_path=csv_path,
                json_dir=json_dir,
                k=k,
                model_name=globals().get("DEFAULT_MODEL", None),
            )
        except Exception:
            examples = []

        # normalize JSON to list
        out: list[tuple[str, list]] = []
        for ex_cmd, ex_json in examples:
            if isinstance(ex_json, dict):
                ex_json = [ex_json]
            if isinstance(ex_json, list):
                out.append((str(ex_cmd), ex_json))
        return out

    def _normalize_joint_name(name: str) -> str:
        """Normalize joint name via aliases and validate if lists are provided."""
        j = str(name).strip()
        if ALLOWED:
            if j not in ALLOWED and j in ALIASES:
                j = ALIASES[j]
            if j not in ALLOWED:
                # if a control list is present and the name is unknown — raise
                raise ValueError(f"Unknown joint name '{name}'. Allowed: {sorted(ALLOWED)}")
        return j

    def _normalize_action(action: dict) -> dict:
        """
        Soft-normalize a single action:
          • keys → snake_case/lower
          • strings — strip whitespace
          • numbers — cast to float/int where appropriate
          • fields 'joint'/'joints' validated via ALLOWED/ALIASES (if provided)
        """
        if not isinstance(action, dict):
            raise ValueError("Each action must be an object.")

        def _snake(s: str) -> str:
            return str(s).strip().replace(" ", "_").replace("-", "_").lower()

        # 1) keys to snake_case + copy
        norm = {}
        for k, v in action.items():
            norm[_snake(k)] = v

        # 2) action type (help model keep type/action)
        if "type" not in norm and "action" in norm:
            norm["type"] = norm.pop("action")

        # 3) string fields — trim
        for k, v in list(norm.items()):
            if isinstance(v, str):
                norm[k] = v.strip()

        # 4) numeric conversions (softly)
        def _maybe_num(x):
            try:
                if isinstance(x, bool):
                    return x
                if isinstance(x, (int, float)):
                    return x
                if isinstance(x, str) and x:
                    # int if no dot; else float
                    if x.isdigit() or (x[0] in "+-" and x[1:].isdigit()):
                        return int(x)
                    return float(x)
            except Exception:
                pass
            return x

        for k, v in list(norm.items()):
            if k in {"angle", "speed", "duration", "x", "y", "z", "roll", "pitch", "yaw", "value"}:
                norm[k] = _maybe_num(v)

        # 5) joint normalization
        if "joint" in norm:
            norm["joint"] = _normalize_joint_name(norm["joint"])
        if "joints" in norm and isinstance(norm["joints"], list):
            norm["joints"] = [_normalize_joint_name(j) for j in norm["joints"]]

        return norm

    # ---------- quick cache ----------
    if command in command_cache:
        return command_cache[command]

    # ---------- build messages for the LLM ----------
  
    base_prompt = _read_prompt_safe()
    base_prompt = _append_current_pose_tables_to_prompt(base_prompt)  # уже было
    base_prompt = _append_minimap_to_prompt(base_prompt, user_cmd=command)  # <<< ДОБАВИТЬ
    messages = [{"role": "system", "content": base_prompt}]
    few_shots = _few_shots(command, k=rag_k, csv_path=rag_csv, json_dir=rag_json)
    show_examples = True

    if show_examples:
        print("by commands examples:")
        if few_shots:
            for i, (ex_cmd, _) in enumerate(few_shots, 1):
                print(f"{i}) {ex_cmd}")
        else:
            # fallback for printing only
            try:
                df = load_cmds_csv(args.csv)
                ranked = compute_similarities(command, csv_path=args.csv, json_dir=args.json_dir)
                for i, (_, row) in enumerate(ranked.head(5).iterrows(), 1):
                    print(f"{i}) {row['command']}  [no json]")
            except Exception as e:
                print(f"  [none] ({e})")

    if few_shots:
        messages.append({
            "role": "system",
            "content": ("""
Below are guidelines: pairs of “command example → target JSON”. Strictly follow the response format from the examples. Reply with JSON ONLY."""
                        ),
        })
        for ex_cmd, ex_json in few_shots:
            messages.append({"role": "user", "content": f'Command: "{ex_cmd}". Output JSON only.'})
            try:
                messages.append({"role": "assistant", "content": json.dumps(ex_json, ensure_ascii=False)})
            except Exception:
                # if serialization fails — skip that particular example
                continue

    # final user request
    messages.append({"role": "user", "content": f'Command: "{command}". Output JSON only.'})

    # ---------- model call ----------
    try:
        print("\n=== PROMPT DUMP ===")
        for i, m in enumerate(messages, 1):
            role = m.get("role")
            content = m.get("content")
            print(f"[{i}] {role}:\n{content}\n")
        print("=== END PROMPT ===\n")
        raw = _llm_chat(messages, max_tokens=2048, temperature=temp)
    except Exception as e:
        raise RuntimeError(f"LLM call failed: {e}")

    # ---------- extract JSON ----------
    try:
        json_text = _extract_first_json(raw)
    except Exception:
        # if no special extractor — naive attempt to find the first JSON block
        import re
        m = re.search(r"(\[.*\]|\{.*\})", raw, flags=re.S)
        json_text = m.group(1) if m else raw.strip()

    try:
        parsed = json.loads(json_text)
    except Exception:
        # developer diagnostics
        print("[parse_command] Raw LLM response:\n", raw)
        raise

    # ensure list
    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list):
        raise ValueError("Expected JSON array of actions or a single JSON object.")
    print(parsed)

    # ---------- normalize the set of actions ----------
    normalized: list[dict] = []
    for item in parsed:
        norm = _normalize_action(item)
        normalized.append(norm)

    # ---------- cache and return ----------
    command_cache[command] = normalized
    return normalized

# --- Policy wrapper: статeless/состояние + мягкий reset ---
class PolicyWrapper:
    def __init__(self, model_or_path: str | torch.jit.ScriptModule):
        if isinstance(model_or_path, str):
            self.path = model_or_path
            self.model = torch.jit.load(self.path)
        else:
            self.path = None
            self.model = model_or_path
        self.state = None  # на случай, если модель возвращает (act, new_state)

    def reset(self, hard: bool = False):
        """Сброс внутреннего состояния контроллера. При hard=True — перезагрузка .pt."""
        self.state = None
        # если у модели есть .reset() — попробуем вызвать
        if hasattr(self.model, "reset") and callable(getattr(self.model, "reset")):
            try:
                self.model.reset()
            except Exception:
                pass
        if hard and self.path:
            try:
                self.model = torch.jit.load(self.path)
            except Exception:
                pass

    def act(self, obs_np: np.ndarray) -> np.ndarray:
        """Вызов политики. Поддержка выходов вида act ИЛИ (act, new_state)."""
        x = torch.from_numpy(obs_np).unsqueeze(0)  # (1, num_obs)
        out = self.model(x)
        if isinstance(out, (tuple, list)) and len(out) >= 2:
            act_t, new_state = out[0], out[1]
            # держим state, если вдруг модель рекуррентная
            try:
                self.state = new_state
            except Exception:
                pass
        else:
            act_t = out
        return act_t.detach().numpy().squeeze()


def parse_command_with_image(command: str, image_path: str):
    """
    Same as parse_command(), but with VLM.
    Supports 'frame', single joints and 'walk'.
    """
    cache_key = f"{command}||{image_path}"
    if cache_key in command_cache:
        return command_cache[cache_key]

        # parse_command_with_image()
    user_text = _build_prompt_with_extra(
        command,
        rag_k=getattr(settings, "RAG_K", 3),
        rag_csv=getattr(settings, "RAG_CSV", "rag_best.csv"),
        rag_json=getattr(settings, "RAG_JSON", "best"),
    )
    raw = _vlm_chat(user_text, image_path)

    json_text = _extract_first_json(raw)
    try:
        parsed = json.loads(json_text)
    except Exception:
        print("[EXCEPTION CAUGHT] Original response:", raw)
        raise

    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list):
        raise ValueError("Expected a JSON array with actions.")

    normalized: list[dict] = []

    def _normalize_joint_name(jname: str) -> str:
        jname = str(jname).strip()
        if jname not in ALLOWED_JOINTS and jname in JOINT_ALIASES:
            jname = JOINT_ALIASES[jname]
        if jname not in ALLOWED_JOINTS:
            raise ValueError(f"Unknown joint name '{jname}'. Allowed: {sorted(ALLOWED_JOINTS)}")
        return jname

    for i, item in enumerate(parsed):
        if not isinstance(item, dict):
            raise ValueError(f"Action[{i}] must be an object, got {type(item)}")

        if "frame" in item:
            frame = item["frame"]
            if not isinstance(frame, list) or not frame:
                raise ValueError(f"Action[{i}].frame must be a non-empty list")
            norm_frame: list[dict] = []
            for j, sub in enumerate(frame):
                if not isinstance(sub, dict):
                    raise ValueError(f"Action[{i}].frame[{j}] must be an object")
                if "name" not in sub:
                    raise ValueError(f"Action[{i}].frame[{j}] missing 'name'")
                if "angle" not in sub:
                    raise ValueError(f"Action[{i}].frame[{j}] missing 'angle'")
                jname = _normalize_joint_name(sub["name"])
                angle = float(sub["angle"])
                norm_frame.append({"name": jname, "angle": angle})
            duration = float(item.get("duration", 0.3))
            normalized.append({"frame": norm_frame, "duration": duration})
            continue

        if item.get("name") == "walk":
            required_walk = {"num", "dir_deg", "spd"}
            missing = required_walk - set(item.keys())
            if missing:
                raise ValueError(f"'walk' action must include {required_walk}. Missing: {missing}. Got: {item}")
            normalized.append({
                "name": "walk",
                "num": int(item["num"]),
                "dir_deg": float(item["dir_deg"]),
                "spd": float(item["spd"]),
            })
            continue

        if "name" not in item:
            raise ValueError(f"Action at index {i} missing 'name'.")
        if "angle" not in item:
            raise ValueError(f"Action at index {i} missing 'angle'.")

        jname = _normalize_joint_name(item["name"])
        angle = float(item["angle"])
        duration = float(item.get("duration", 0.3))
        normalized.append({"name": jname, "angle": angle, "duration": duration})

    command_cache[cache_key] = normalized
    return normalized


# =========================
# Small GUI helper
# =========================

def open_paste_window():
    """Open a modal window, return parsed list (from JSON or Python literal) or None."""
    root = tk.Tk()
    root.title("Paste motion array")

    txt = tk.Text(root, width=80, height=18)
    txt.pack(fill="both", expand=True, padx=10, pady=10)

    result = {"data": None}

    def apply_and_close():
        raw = txt.get("1.0", "end").strip()
        if not raw:
            root.destroy()
            return
        # Try JSON first, then Python literal
        try:
            data = json.loads(raw)
        except Exception:
            try:
                data = ast.literal_eval(raw)
            except Exception as e:
                print("Parse error:", e)
                return  # keep the window open for user to fix
        if isinstance(data, dict):
            data = [data]  # single object -> list
        result["data"] = data
        root.destroy()

    def cancel():
        root.destroy()

    btns = tk.Frame(root)
    btns.pack(fill="x", padx=10, pady=(0, 10))
    tk.Button(btns, text="Apply", command=apply_and_close).pack(side="right", padx=4)
    tk.Button(btns, text="Cancel", command=cancel).pack(side="right", padx=4)

    # Blocking modal run
    root.mainloop()
    return result["data"]

def _try_vlm_multi(prompt_text: str, image_paths: list[Path]):

    try:
        func = globals().get("_vlm_chat_multi", None)
        if callable(func):
            return func(prompt_text, [str(p) for p in image_paths])
    except Exception as e:
        print(f"[INFO] _vlm_chat_multi does not work: {e}")
    return None


def make_mosaic(image_paths: list[Path], cols: int = 3, pad: int = 8, out_path: str = "captures/views_mosaic.png") -> Path:
   
    assert image_paths, "at least one picture"
    imgs = [imageio.imread(str(p)) for p in image_paths]
    h, w = imgs[0].shape[:2]

    n = len(imgs)
    rows = math.ceil(n / cols)
    H = rows * h + (rows - 1) * pad
    W = cols * w + (cols - 1) * pad

    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    for idx, img in enumerate(imgs):
        r = idx // cols
        c = idx % cols
        y = r * (h + pad)
        x = c * (w + pad)
        canvas[y:y+h, x:x+w, :3] = img[:, :, :3]

    outp = Path(out_path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(outp, canvas)
    print(f"[MOSAIC] saved -> {outp}")
    return outp


def vlm_analyze_views_single_call(prompt_text: str, image_paths: list[Path], out_json: str = "captures/vlm_batch.json", mosaic_cols: int = 3) -> dict:

    # try 1: naive multi-input
    resp = _try_vlm_multi(prompt_text, image_paths)
    meta = {"mode": "multi_api", "images": [str(p) for p in image_paths]}
    if resp is None:
        # try 2: fallback by mosaic
        mosaic_path = make_mosaic(image_paths, cols=mosaic_cols, pad=8, out_path="captures/views_mosaic.png")
        try:
            resp = _vlm_chat(prompt_text, str(mosaic_path))
            meta = {"mode": "mosaic", "mosaic_path": str(mosaic_path), "images": [str(p) for p in image_paths]}
        except Exception as e:
            resp = f"[VLM error: {e}]"
            meta["error"] = str(e)

    result = {"prompt": prompt_text, "response": resp, **meta}
    try:
        outp = Path(out_json)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with outp.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"[SAVE] batch VLM -> {outp}")
    except Exception as e:
        print(f"[WARN] can't save batch JSON: {e}")
    return result

# =========================
# Multi-view capture helpers
# =========================

def _render_and_read_pixels(width: int, height: int) -> np.ndarray:

    global m, d, opt, cam, scene, context
    viewport = mj.MjrRect(0, 0, width, height)
   
    mj.mjv_updateScene(m, d, opt, None, cam, mj.mjtCatBit.mjCAT_ALL, scene)
    mj.mjr_render(viewport, scene, context)

    rgb = np.empty((height, width, 3), dtype=np.uint8)
    depth = np.empty((height, width), dtype=np.float32)
    mj.mjr_readPixels(rgb, depth, viewport, context)

    return np.flipud(rgb)


def capture_views(
    out_dir: str = "captures",
    prefix: str = "view",
    num: int = 6,
    azimuth_span_deg: float = 360.0,
    elevation_deg: float | None = None,
    distance: float | None = None,
    size: tuple[int, int] | None = None,
) -> list[Path]:

    global window, cam

 
    if window is None:
        raise RuntimeError("GLFW window is not initialized")
    try:
        cur_ctx = glfw.get_current_context()
        if not cur_ctx:
            glfw.make_context_current(window)
    except Exception:
        glfw.make_context_current(window)

    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)

    cam_backup = mj.MjvCamera()
    cam_backup.azimuth = cam.azimuth
    cam_backup.elevation = cam.elevation
    cam_backup.distance = cam.distance
    cam_backup.lookat = cam.lookat.copy()

    if size is None:
        fb_w, fb_h = glfw.get_framebuffer_size(window)
    else:
        fb_w, fb_h = int(size[0]), int(size[1])

    if elevation_deg is not None:
        cam.elevation = float(elevation_deg)
    if distance is not None:
        cam.distance = float(distance)

    start_az = float(cam.azimuth)
    step = float(azimuth_span_deg) / max(1, num)

    saved: list[Path] = []
    try:
        for i in range(num):
            cam.azimuth = start_az + i * step
            rgb = _render_and_read_pixels(fb_w, fb_h)
            path = outp / f"{prefix}_{i:03d}.png"
            imageio.imwrite(path, rgb)
            print(f"[CAPTURE] {path}")
            saved.append(path)

            glfw.swap_buffers(window)
            glfw.poll_events()
    finally:
        cam.azimuth = cam_backup.azimuth
        cam.elevation = cam_backup.elevation
        cam.distance = cam_backup.distance
        cam.lookat = cam_backup.lookat

    return saved


def capture_and_query_vlm_batch():

    try:
        user_extra = input("What's object interaction?").strip()
        prompt_text = _build_prompt_with_extra(
            user_extra,
            rag_k=getattr(settings, "RAG_K", 3),
            rag_csv=getattr(settings, "RAG_CSV", "rag_best.csv"),
            rag_json=getattr(settings, "RAG_JSON", "best"),
        )

        n = input("How much views [6]? ").strip()
        n = int(n) if n else 6
        out_dir = input("Path to save [captures]: ").strip() or "captures"
        prefix = input("Prefics [view]: ").strip() or "view"
        elev = input("Fix elevation (deg) [Enter - current]: ").strip()
        elev = float(elev) if elev else None
        dist = input("FIx distance [Enter — current]: ").strip()
        dist = float(dist) if dist else None
        cols = input("How much rows [3]? ").strip()
        cols = int(cols) if cols else 3
    except Exception as e:
        print(f"Bad input: {e}")
        return

    paths = capture_views(
        out_dir=out_dir,
        prefix=prefix,
        num=n,
        azimuth_span_deg=360.0,
        elevation_deg=elev,
        distance=dist,
        size=None,
    )
    if not paths:
        print("No frames to VLM.")
        return

    vlm_analyze_views_single_call(
        prompt_text, 
        paths, 
        out_json=str(Path(out_dir) / "vlm_batch.json"), 
        mosaic_cols=cols
    )
def _build_prompt_with_extra(user_extra: str,
                             rag_k: int = 5,
                             rag_csv: str = "1000.csv",
                             rag_json: str | Path = "data_action") -> str:
    """
    Collects the final text prompt for VLM:
        • basic prompt.txt (+ current pose),
        • a few-shot block of CSV/JSON (RAG),
    • your user_extra text as 'Command: "<...>"',
    • strict instructions to respond ONLY with JSON.
        """
    base_prompt = ""
    try:
        with open("prompt.txt", "r", encoding="utf-8") as f:
            base_prompt = f.read().strip()
    except Exception as e:
        print(f"[WARN] prompt.txt not found: {e}")

    base_prompt = _append_current_pose_tables_to_prompt(base_prompt)
    base_prompt = _append_minimap_to_prompt(base_prompt, user_cmd=user_extra)  # <<< ДОБАВИТЬ

    # RAG few-shots
    fewshot_txt = _fewshot_block_text(
        user_cmd=user_extra,
        k=rag_k,
        csv_path=rag_csv,
        json_dir=rag_json,
    )

    # final text
    out = []
    out.append(base_prompt)
    if fewshot_txt:
        out.append(fewshot_txt)
    out.append(f'Command: "{user_extra}"')
    out.append("Reply with JSON ONLY (array or object); NO prose.")
    return "\n\n".join(out).strip()

def _parse_actions_from_text(raw_text: str) -> list[dict]:
   
    js = _extract_first_json(raw_text)
    data = json.loads(js)
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        raise ValueError("At least 1 element is expected")
    return data


def capture_and_execute_vlm_batch(default_views: int = 6, default_cols: int = 3) -> None:
 

    user_extra = input("Object action to prompt_twins.txt (emp): ").strip()
    prompt_text = _build_prompt_with_extra(
        user_extra,
        rag_k=getattr(settings, "RAG_K", 3),
        rag_csv=getattr(settings, "RAG_CSV", "rag_best.csv"),
        rag_json=getattr(settings, "RAG_JSON", "best"),
    )



    try:
        n_raw = input(f"How much views [{default_views}]? ").strip()
        n = int(n_raw) if n_raw else default_views
    except Exception:
        n = default_views

    try:
        cols_raw = input(f"Columns in mosaic [{default_cols}]? ").strip()
        cols = int(cols_raw) if cols_raw else default_cols
    except Exception:
        cols = default_cols

    out_dir = f"robot_captures/vlm_{time.strftime('%Y-%m-%d_%H-%M-%S')}"

    prefix = "view"
    elev = None
    dist = None

 
    paths = capture_views(
        out_dir=out_dir,
        prefix=prefix,
        num=n,
        azimuth_span_deg=360.0,
        elevation_deg=elev,
        distance=dist,
        size=None,
    )
    if not paths:
        print("[VLM] No frames.")
        return
    print(prompt_text)
    print("Thinking...") 

   
    result = vlm_analyze_views_single_call(
        prompt_text,
        paths,
        out_json=str(Path(out_dir) / "vlm_batch.json"),
        mosaic_cols=cols
    )
    resp_text = result.get("response", "")

   
    try:
        actions = _parse_actions_from_text(resp_text)
    except Exception as e:
        print(f"[ERR] Cant json read from VLM: {e}")
        print("=== RAW RESPONSE START ===")
        print(resp_text)
        print("=== RAW RESPONSE END ===")
        return

    print("[ACTIONS]", actions)
   
    move_joints_by_name(actions)

  
    # ans = input("Save to JSON? (y/N): ").strip().lower()
    # if ans in ("y", "yes"):
    #    
    #     save_actions_json(actions, command_text=f"[VLM batch] extra='{user_extra}'", out_dir="data_action", filename=None)

# =========================
# Physics / control helpers
# =========================

def get_gravity_orientation(quaternion):
    """Compute an approximate gravity orientation vector from a quaternion."""
    qw, qx, qy, qz = quaternion
    gravity_orientation = np.zeros(3)
    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)
    return gravity_orientation


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """PD control law."""
    return (target_q - q) * kp + (target_dq - dq) * kd

def do_reset():
    """Hard reset = как при старте программы: точное восстановление spawn-состояния."""
    global steps_needed, steps_done, counter, old_sin_phase, short_extra_done, last_activity_time
    global cmd, cmd_vec, action, target, robots
    global WALK_UNTIL, TURN_UNTIL, TURN_ACC, TURN_LAST, TURN_TGT
    global ACTIVE_TASK, TASKS

    # 1) Полный сброс физики и возврат ровно к INIT_QPOS (из XML)
    mj.mj_resetData(m, d)
    d.qpos[:] = INIT_QPOS        # <<< ключевое — НИЧЕГО не перетирать после этого
    d.qvel[:] = 0
    d.qacc[:] = 0
    d.qfrc_applied[:] = 0
    d.xfrc_applied[:] = 0
    d.ctrl[:] = 0
    mj.mj_forward(m, d)

    # 2) Служебные счётчики/фазы
    steps_needed = 0
    steps_done = 0
    counter = 0
    old_sin_phase = 0.0
    short_extra_done = False
    last_activity_time = time.time()

    # 3) Команды/политики на ноль, цели как на старте (== текущим суставам)
    try:
        cmd[:] = 0.0
    except Exception:
        pass

    for p in robots:
        WALK_GOAL[p]  = 0
        WALK_COUNT[p] = 0
        cmd_vec[p][:] = 0.0
        action[p][:] = 0.0
        # как в __main__: цель = текущему положению суставов (без рывка на default_angles)
        target[p] = d.qpos[R[p]["qpos_adrs"]].copy()

    # 4) Сброс таймеров ходьбы/поворота
    if 'WALK_UNTIL' in globals():
        for p in robots: WALK_UNTIL[p] = 0.0
    if 'TURN_UNTIL' in globals():
        for p in robots: TURN_UNTIL[p] = 0.0
    if 'TURN_TGT' in globals():
        for p in robots: TURN_TGT[p] = 0.0
    if 'TURN_ACC' in globals():
        for p in robots: TURN_ACC[p] = 0.0
    if 'TURN_LAST' in globals():
        # пересчитать yaw в текущем (spawn) состоянии
        for p in robots:
            mp = R[p]
            qw, qx, qy, qz = d.qpos[mp["base_qpos_adr"]+3 : mp["base_qpos_adr"]+7]
            TURN_LAST[p] = math.atan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))

    # 5) Очистить очереди поз
    if 'TASKS' in globals() and 'ACTIVE_TASK' in globals():
        for p in robots:
            TASKS[p].clear()
            ACTIVE_TASK[p] = None

            # --- ВАЖНО: сброс контроллера и входов политики ---
    for p in robots:
        # полностью «чистые» входы в политику
        try:
            obs[p][:] = 0.0
            action[p][:] = 0.0
            cmd_vec[p][:] = 0.0
            target[p] = d.qpos[R[p]["qpos_adrs"]].copy()
        except Exception:
            pass
        # мягкий сброс политики (без перезагрузки файла)
        if p in POL and hasattr(POL[p], "reset"):
            POL[p].reset(hard=True)

    # (по желанию) выровнять фазы подсчёта шагов/поворота
    for p in robots:
        if 'TURN_ACC' in globals(): TURN_ACC[p] = 0.0
        if 'TURN_TGT' in globals(): TURN_TGT[p] = 0.0
        if 'WALK_UNTIL' in globals(): WALK_UNTIL[p] = 0.0
        if 'TURN_UNTIL' in globals(): TURN_UNTIL[p] = 0.0
        if 'old_sin' in globals(): old_sin[p] = 0.0


    print("Reset done: restored exact spawn state from INIT_QPOS (same as fresh start).")



# =========================
# Motion helpers
# =========================

def _make_pose_task(frame: list[dict], duration: float):
    # frame: [{'name': joint, 'angle': deg}, ...]
    start = {}
    target = {}
    for j in frame:
        jname = j['name']
        if jname not in joint_map:
            continue
        start[jname]  = float(d.qpos[joint_map[jname]])
        target[jname] = math.radians(float(j['angle']))
    t0 = time.time()
    t1 = t0 + max(0.01, float(duration))
    return {'type':'pose','t0':t0,'t1':t1,'start':start,'target':target}

def schedule_action(p: str, item: dict):
  
    if item.get('name') in ('turn','rotate'):
        ang = float(item.get('deg') or item.get('angle') or item.get('yaw_deg') or 0.0)
        spd = float(item.get('spd_deg_s') or item.get('speed_deg_s') or 45.0)
        start_turn_for(p, ang, spd)
        return
    if item.get('name') == 'walk':
        start_walk_for(p, int(item['num']), float(item['dir_deg']), float(item['spd']))
        return
   
    if 'frame' in item:
        dur = float(item.get('duration', 0.3))
        TASKS[p].append(_make_pose_task(item['frame'], dur))
        return
def _zero_dyn():
    d.qvel[:] = 0.0
    d.qacc[:] = 0.0
    d.qfrc_applied[:] = 0.0
    d.xfrc_applied[:] = 0.0

def move_joints_by_name(joints_list, duration_per_frame=0.3, fps=60):
    """
    Execute motion in one of the formats:
      A) Frames of angles [[8 angles], ...] — as before (for 8 fixed joints).
      B) List of action objects:
         - {"name": "<joint>", "angle": <deg>, "duration": <sec optional>}  # single joint
         - {"frame": [{"name": "<joint>", "angle": <deg>}, ...], "duration": <sec optional>}  # SIMULTANEOUS
         - {"name": "walk", "num": <int>, "dir_deg": <float>, "spd": <float>}
    """
    global last_activity_time
    last_activity_time = time.time()

    if not joints_list:
        return

    if isinstance(joints_list, list) and joints_list and isinstance(joints_list[0], dict):
        wants_async = any(it.get('async') for it in joints_list if isinstance(it, dict))
       # We are NOT enabling asynchronous mode just because of the presence of the field 'robot'
        if wants_async:
            for it in joints_list:
                sel = _normalize_robots(it.get('robot') or it.get('robots')) or [ACTIVE]
                for p in sel:
                    schedule_action(p, it)
            return


    # ===== helpers =====
    def render_once():
        width, height = glfw.get_framebuffer_size(window)
        viewport = mj.MjrRect(0, 0, width, height)
        mj.mjv_updateScene(m, d, opt, None, cam, mj.mjtCatBit.mjCAT_ALL, scene)
        mj.mjr_render(viewport, scene, context)
        glfw.swap_buffers(window)
        glfw.poll_events()

    def smoothstep(t: float) -> float:
        # t in [0,1] -> smooth S-curve
        return t * t * (3 - 2 * t)

    fps_dt = 1.0 / fps

    # ===== Variant A: array of frames with 8 angles =====
    if isinstance(joints_list[0], list):
        print(f"Executing motion with {len(joints_list)} frames (8-angle frames)")
        for frame in joints_list:
            if len(frame) != 8:
                print(f"Invalid frame: {frame}. Skipping.")
                continue

            target_angles = {}
            for joint_idx, angle_deg in enumerate(frame):
                joint_name = joint_index_map.get(joint_idx)
                if joint_name:
                    target_angles[joint_name] = math.radians(angle_deg)

            # store starting angles
            start_angles = {name: d.qpos[joint_map[name]] for name in target_angles.keys()}

            # spread time uniformly per frame
            target_total_duration = 4.0
            if len(joints_list) > 20:
                target_total_duration = 4.0 * len(joints_list) / 15
            per_frame_duration = target_total_duration / len(joints_list)
            n_steps = max(1, int(per_frame_duration * fps))

            for step in range(n_steps):
                t = smoothstep((step + 1) / n_steps)
                for jname, target_rad in target_angles.items():
                    start_rad = start_angles[jname]
                    d.qpos[joint_map[jname]] = (1 - t) * start_rad + t * target_rad
                mj.mj_forward(m, d)
                _zero_dyn()
                render_once()
                time.sleep(fps_dt)
        return

    # ===== Variant B: list of objects (incl. simultaneous frames via "frame") =====
    for item in joints_list:
        # --- NEW: repeat block ---
        if isinstance(item, dict) and "repeat" in item and "times" in item:
            subactions = item["repeat"]
            n_times = int(item["times"])
            if not isinstance(subactions, list) or n_times <= 0:
                continue
            for _ in range(n_times):
                move_joints_by_name(subactions, duration_per_frame, fps)
            continue


        if isinstance(item, dict) and "parallel" in item and isinstance(item["parallel"], list):
           
            started_turn = set()
            started_walk = set()
            for sub in item["parallel"]:
                sel = _normalize_robots(sub.get('robot') or sub.get('robots')) or [ACTIVE]
                if sub.get("name") in ("turn", "rotate"):
                    ang = float(sub.get('deg') or sub.get('angle') or sub.get('yaw_deg') or 0.0)
                    spd = float(sub.get('spd_deg_s') or sub.get('speed_deg_s') or 45.0)
                    for p in sel:
                        start_turn_for(p, ang, spd)
                        started_turn.add(p)
                elif sub.get("name") == "walk":
                    for p in sel:
                        start_walk_for(p, int(sub["num"]), float(sub["dir_deg"]), float(sub["spd"]))
                        started_walk.add(p)
                else:
                
                    move_joints_by_name([sub], duration_per_frame, fps)
            if started_turn:
                _wait_turns_done(sorted(started_turn))
            if started_walk:
                _wait_walks_done(sorted(started_walk))
            continue


        if isinstance(item, dict) and item.get("name") == "walk":
            sel = _normalize_robots(item.get("robot")) or [ACTIVE]
            started = []
            for p in sel:
                start_walk_for(p, int(item["num"]), float(item["dir_deg"]), float(item["spd"]))
                started.append(p)
            _wait_walks_done(started)
            continue

        if isinstance(item, dict) and item.get("name") in ("turn","rotate"):
            sel = _normalize_robots(item.get("robot")) or [ACTIVE]
            ang = float(item.get("deg") or item.get("angle") or item.get("yaw_deg") or 0.0)
            spd = float(item.get("spd_deg_s") or item.get("speed_deg_s") or 45.0)
            started = []
            for p in sel:
                start_turn_for(p, ang, spd)
                started.append(p)
            _wait_turns_done(started)
            continue




        # Frame with several joints — SIMULTANEOUS
        if isinstance(item, dict) and "frame" in item:
            joints = item["frame"]
            if not isinstance(joints, list) or not joints:
                print(f"Empty or invalid 'frame' in item: {item}")
                continue

            # filter valid joints and prepare target angles
            target_angles = {}
            for j in joints:
                raw = j.get("name")
                jname = _resolve_joint_name(str(raw))
                if not jname:
                    print(f"[skip] joint name not found in model: {raw}")
                    continue
                target_angles[jname] = math.radians(float(j["angle"]))


            if not target_angles:
                continue

            duration = float(item.get("duration", duration_per_frame))
            n_steps = max(1, int(duration * fps))

            # starting angles
            start_angles = {name: d.qpos[joint_map[name]] for name in target_angles.keys()}

            for step in range(n_steps):
                t = smoothstep((step + 1) / n_steps)
                for jname, target_rad in target_angles.items():
                    start_rad = start_angles[jname]
                    d.qpos[joint_map[jname]] = (1 - t) * start_rad + t * target_rad
                mj.mj_forward(m, d)
                _zero_dyn()
                render_once()
                time.sleep(fps_dt)
            continue

        # Regular single joint — backward compatible with previous format
        if isinstance(item, dict) and "name" in item and "angle" in item:
            jname = item["name"]
            if jname not in joint_map:
                print(f"Joint '{jname}' not found.")
                continue

            target_angle_rad = math.radians(float(item["angle"]))
            qpos_idx = joint_map[jname]
            start_rad = d.qpos[qpos_idx]

            duration = float(item.get("duration", duration_per_frame))
            n_steps = max(1, int(duration * fps))

            for step in range(n_steps):
                t = smoothstep((step + 1) / n_steps)
                d.qpos[qpos_idx] = (1 - t) * start_rad + t * target_angle_rad
                mj.mj_forward(m, d)
                _zero_dyn()
                render_once()
                time.sleep(fps_dt)
            continue

        print(f"[WARN] Unknown item format: {item}")

manual_override = False 

def _one_tick(render=True):
    """One simulation step + updating tasks/policies/states for ALL robots."""
    if _maybe_freeze():
        return
    global counter

    now = time.time()


    for p in robots:
        if ACTIVE_TASK[p] is None and TASKS[p]:
            ACTIVE_TASK[p] = TASKS[p].pop(0)
        task = ACTIVE_TASK[p]
        if task and task['type'] == 'pose':
            t0, t1 = task['t0'], task['t1']
            if t1 <= t0:
                alpha = 1.0
            else:
                u = (now - t0) / (t1 - t0)
                u = max(0.0, min(1.0, u))
                alpha = u*u*(3-2*u)
            for jname, q1 in task['target'].items():
                idx = joint_map.get(jname)
                if idx is not None:
                    q0 = task['start'][jname]
                    d.qpos[idx] = (1.0 - alpha) * q0 + alpha * q1
            d.qvel[:] = 0.0
            d.qacc[:] = 0.0
            d.qfrc_applied[:] = 0.0
            d.xfrc_applied[:] = 0.0
            if now >= t1:
                ACTIVE_TASK[p] = None
        # --- ДЕРЖИМ ВЕРХ В НУЛЕ, ЕСЛИ НЕТ АКТИВНОЙ ПОЗЫ У РОБОТА ---
    if HOLD_UPPER_ZERO:
        for p in robots:
            if ACTIVE_TASK[p] is not None:
                continue
            dof_idx = HOLD_UPPER_DOF.get(p)
            if dof_idx is not None and dof_idx.size:
                d.qvel[dof_idx] = 0.0   
            
    mj.mj_forward(m, d)

    # 2) PD joints
    for p in robots:
        mp = R[p]

        # Глушим PD/политику, если ручной режим или сейчас идёт pose-таска
        if manual_override or (ACTIVE_TASK[p] and ACTIVE_TASK[p]['type'] == 'pose'):
            d.ctrl[mp["act_ids"]] = 0.0
            # target выравниваем по текущей позе, чтобы не было рывка после выхода
            target[p] = d.qpos[mp["qpos_adrs"]].copy()
            continue

        qj  = d.qpos[mp["qpos_adrs"]]
        dqj = d.qvel[mp["dof_adrs"]]
        tau = pd_control(target[p], qj, kps, np.zeros_like(kds), dqj, kds)
        d.ctrl[mp["act_ids"]] = tau
    # 3) Step phys
    mj.mj_step(m, d)
    counter += 1

  # 4) Timers/flags of walking/turning on time
    now = time.time()
    for p in robots:
        if WALK_UNTIL[p] and now >= WALK_UNTIL[p]:
            WALK_UNTIL[p] = 0.0
            cmd_vec[p][:] = 0.0
        if TURN_UNTIL[p] and now >= TURN_UNTIL[p]:
            TURN_UNTIL[p] = 0.0
            cmd_vec[p][:] = 0.0
#5) Policy/observation update + Rotation angle accumulation
    if counter % control_decimation == 0:
        def _wrap_to_pi(a): return (a + math.pi) % (2*math.pi) - math.pi
        def _yaw(qw,qx,qy,qz): return math.atan2(2*(qw*qz+qx*qy), 1-2*(qy*qy+qz*qz))

        na = 12
        for p in robots:
            mp   = R[p]
            quat = d.qpos[mp["base_qpos_adr"]+3 : mp["base_qpos_adr"]+7]
            omega= d.qvel[mp["base_qvel_adr"]+3 : mp["base_qvel_adr"]+6]
            qj   = d.qpos[mp["qpos_adrs"]]
            dqj  = d.qvel[mp["dof_adrs"]]

          
            if TURN_TGT[p] > 0.0:
                inc = _wrap_to_pi(_yaw(*quat) - TURN_LAST[p])
                TURN_ACC[p] += abs(inc)
                TURN_LAST[p] = _yaw(*quat)
                if TURN_ACC[p] >= TURN_TGT[p]:
                    cmd_vec[p][:] = 0.0
                    TURN_TGT[p] = 0.0
                    TURN_ACC[p] = 0.0

            qj_scaled  = (qj - default_angles) * dof_pos_scale
            dqj_scaled = dqj * dof_vel_scale
            grav       = get_gravity_orientation(quat)
            omega_scaled = omega * ang_vel_scale

            period = GAIT_PERIOD
            time_in_sim = counter * simulation_dt
            phase = (time_in_sim % period) / period
            sin_phase = math.sin(2*math.pi*phase); cos_phase = math.cos(2*math.pi*phase)
            # Завершение строго по числу шагов (нолепереход sin: - → +)
            if WALK_GOAL[p] > 0 and (sin_phase >= 0.0 and old_sin[p] < 0.0):
                WALK_COUNT[p] += 1
                if WALK_COUNT[p] >= WALK_GOAL[p]:
                    cmd_vec[p][:] = 0.0          # стоп команда ходьбы
                    WALK_GOAL[p]  = 0
                    WALK_COUNT[p] = 0
                    WALK_UNTIL[p] = 0.0          # на всякий случай
            # обновляем следящий синус
            old_sin[p] = sin_phase
            o = obs[p]
            o[:3] = omega_scaled
            o[3:6] = grav
            o[6:9] = cmd_vec[p] * cmd_scale
            o[9:9+na] = qj_scaled
            o[9+na:9+2*na] = dqj_scaled
            o[9+2*na:9+3*na] = action[p]
            o[9+3*na:9+3*na+2] = np.array([sin_phase, cos_phase])
            if manual_override or (ACTIVE_TASK[p] and ACTIVE_TASK[p]['type'] == 'pose'):
                continue
            act = POL[p].act(o)
            action[p][:] = act
            target[p]    = act * action_scale + default_angles

    # 6) Render
    if render:
        width, height = glfw.get_framebuffer_size(window)
        viewport = mj.MjrRect(0, 0, width, height)
        mj.mjv_updateScene(m, d, opt, None, cam, mj.mjtCatBit.mjCAT_ALL, scene)
        mj.mjr_render(viewport, scene, context)
        glfw.swap_buffers(window)
        glfw.poll_events()


def _wait_turns_done(sel, timeout=None):
    t0 = time.time()
    while True:
       
        if all(TURN_TGT.get(p, 0.0) == 0.0 for p in sel):
            return
        if timeout and (time.time() - t0) > timeout:
            print("[WARN] wait_turns timeout"); return
        _one_tick(render=True)  

def _wait_walks_done(sel, timeout=None):
    t0 = time.time()
    while True:
        now = time.time()
        if all(WALK_GOAL.get(p, 0) == 0 for p in sel):
            return
        if timeout and (now - t0) > timeout:
            print("[WARN] wait_walks timeout"); return
        _one_tick(render=True)




def open_joint_slider_window():
    """
    Live mode: sliders -> immediately in d.qpos. PD/policy are muted by the manual_override flag.
    Initialize with the current angles once when the window is opened.
    """
    import tkinter as tk
    from tkinter import ttk

    global manual_override

    R = {
        "left_shoulder_pitch": (-164.44, 164.44),
        "left_shoulder_roll":  (-19.48, 178.19),
        "left_shoulder_yaw":   (-74.48, 254.97),
        "left_elbow":          (-71.62, 149.54),
        "right_shoulder_pitch": (-164.44, 164.44),
        "right_shoulder_roll":  (-178.19, 19.48),
        "right_shoulder_yaw":   (-254.97, 74.48),
        "right_elbow":          (-71.62, 149.54),
    }
    needed = [
        "left_shoulder_pitch","left_shoulder_roll","left_shoulder_yaw","left_elbow",
        "right_shoulder_pitch","right_shoulder_roll","right_shoulder_yaw","right_elbow",
    ]

    def clamp(x, lo, hi): return max(lo, min(hi, x))
    def curr_deg(name: str) -> float:
        idx = joint_map.get(name)
        return 0.0 if idx is None else float(np.degrees(d.qpos[idx]))

    # --- окно ---
    win = tk.Toplevel()
    win.title("H1 — sliders (Live)")
    win.geometry("640x560")
    win.attributes("-topmost", True)

    top = ttk.Frame(win, padding=8); top.pack(fill="x")
    ttk.Label(top, text="Animation lenght is not used (live)").pack(anchor="w")

    middle = ttk.Frame(win); middle.pack(expand=True, fill="both")

    controls: dict[str, dict] = {}

    def make_arm_block(parent, prefix: str, title: str):
        frm = ttk.LabelFrame(parent, text=title, padding=8)
        frm.pack(side="left", expand=True, fill="both", padx=8, pady=8)
        def add_slider(jkey: str, label_text: str):
            full = f"{prefix}_{jkey}" if jkey != "elbow" else f"{prefix}_elbow"
            lo, hi = R[full]
            var = tk.DoubleVar(value=0.0)
            ttk.Label(frm, text=f"{label_text} ({lo:.0f}..{hi:.0f})").pack(anchor="w")
            scale = ttk.Scale(frm, orient="horizontal", from_=lo, to=hi, variable=var)
            scale.pack(fill="x")
            lbl = ttk.Label(frm, text="0.0°"); lbl.pack(anchor="e", pady=(0,6))
            def on_move(_=None):
                lbl.config(text=f"{var.get():.1f}°")
            scale.configure(command=on_move)
            controls[full] = {"var": var, "label": lbl, "range": (lo, hi)}
        add_slider("shoulder_pitch", "Shoulder Pitch (−forwart/+backward)")
        add_slider("shoulder_roll",  "Shoulder Roll (outward/inward)")
        add_slider("shoulder_yaw",   "Shoulder Yaw (rotate)")
        add_slider("elbow",          "Elbow (0≈bend, +=out bend)")
    make_arm_block(middle, "left",  "Left arm")
    make_arm_block(middle, "right", "Right arm")

    bottom = ttk.Frame(win, padding=8); bottom.pack(fill="x")
    def zero_all():
        for name,c in controls.items():
            lo, hi = c["range"]
            c["var"].set(0.0 if lo <= 0.0 <= hi else lo)
    ttk.Button(bottom, text="All 0°", command=zero_all).pack(side="left")
    ttk.Button(bottom, text="Close", command=win.destroy).pack(side="right")


    for name in needed:
        if name in controls and name in joint_map:
            val = clamp(curr_deg(name), *controls[name]["range"])
            controls[name]["var"].set(val)
            controls[name]["label"].config(text=f"{val:.1f}°")

   
    manual_override = True

    def tick_apply():
        if not str(win.winfo_exists()): 
            return
        try:
          
            for name, c in controls.items():
                if name not in joint_map: 
                    continue
                val = clamp(c["var"].get(), *c["range"])
                d.qpos[joint_map[name]] = math.radians(val)
            mj.mj_forward(m, d)
        except Exception as e:
            print(f"[GUI] live apply error: {e}")
        finally:
            win.after(30, tick_apply)  # ~33 Гц

    win.after(30, tick_apply)

    def on_close():
        
        global manual_override
        manual_override = False
        win.destroy()

    win.protocol("WM_DELETE_WINDOW", on_close)
    win.focus_force()
    win.grab_set()
    win.wait_window()

def start_turn_for(p: str, angle_deg: float, spd_deg_s: float = 45.0):
    global last_activity_time
    last_activity_time = time.time()
    direction = 1.0 if angle_deg >= 0 else -1.0
    yaw_rate = direction * math.radians(abs(spd_deg_s))


    cmd_vec[p][:] = [0.0, 0.0, yaw_rate]

 
    mp = R[p]
    qw, qx, qy, qz = d.qpos[mp["base_qpos_adr"]+3 : mp["base_qpos_adr"]+7]
    def _yaw(qw,qx,qy,qz):
        return math.atan2(2*(qw*qz+qx*qy), 1-2*(qy*qy+qz*qz))
    TURN_LAST[p] = _yaw(qw,qx,qy,qz)
    TURN_ACC[p]  = 0.0
    TURN_TGT[p]  = abs(math.radians(angle_deg))



def start_walk_for(p: str, num: int, dir_deg: float, spd: float):
    global last_activity_time
    last_activity_time = time.time()
    dir_rad = math.radians(dir_deg)
    vx = spd * math.cos(dir_rad); vy = spd * math.sin(dir_rad)
    cmd_vec[p][:] = [vx, vy, 0.0]

    WALK_GOAL[p]  = max(1, int(num))
    WALK_COUNT[p] = 0
    WALK_UNTIL[p] = 0.0  # таймер больше не нужен; можно оставить как fail-safe, напр. *2*GAIT_PERIOD



def start_new_steps_for(p: str, num: int, dir_deg: float, speed: float):
    global last_activity_time
    last_activity_time = time.time()
    dir_rad = math.radians(dir_deg)
    vx = speed * math.cos(dir_rad)
    vy = speed * math.sin(dir_rad)
    cmd_vec[p][:] = [vx, vy, 0.0]

def run_walk_blocking_for(p: str, num: int, dir_deg: float, spd: float):
    start_walk_for(p, num, dir_deg, spd)
    _wait_walks_done([p])

def run_turn_blocking_for(p: str, angle_deg: float, spd_deg_s: float = 45.0):
    start_turn_for(p, angle_deg, spd_deg_s)
    _wait_turns_done([p])


def start_new_steps(num, dir_deg, speed, is_extra=False):
    """
    Start a new walking sequence:
      - num: number of steps
      - dir_deg: direction in degrees (0° = +X, 90° = +Y)
      - speed: linear speed (m/s)
      - is_extra: internal flag used when auto-balancing
    """
    global last_activity_time, steps_needed, steps_done, cmd, old_sin_phase, short_extra_done
    last_activity_time = time.time()
    if not is_extra:
        short_extra_done = False

    steps_needed = num
    steps_done = 0
    old_sin_phase = 0.0

    dir_rad = math.radians(dir_deg)
    vx = speed * math.cos(dir_rad)
    vy = speed * math.sin(dir_rad)
    cmd[:] = [vx, vy, 0.0]
    print(f"Starting {steps_needed} steps for direction = {dir_deg}°, speed = {speed} m/s")


def _vlm_describe_image(image_path: str) -> str:
    """Ask the VLM to briefly describe the image for safe context."""
    try:
        return LLM.chat_vision_describe(image_path)
    except Exception as e:
        return f"[vision description failed: {e}]"


# =========================
# GLFW input callbacks
# =========================
ACTIVE = "r1_" 
def key_callback(window_, key, scancode, action, mods):
    pressed = (action == glfw.PRESS or action == glfw.REPEAT)
    global ACTIVE
    if action == glfw.RELEASE:
        pressed = False

    # WASD
    if key == glfw.KEY_W:
        movement["forward"] = pressed
    elif key == glfw.KEY_S:
        movement["backward"] = pressed
    elif key == glfw.KEY_A:
        movement["left"] = pressed
    elif key == glfw.KEY_D:
        movement["right"] = pressed

    # Up/Down
    elif key == glfw.KEY_UP:
        movement["rise"] = pressed
    elif key == glfw.KEY_DOWN:
        movement["fall"] = pressed
        # ПРОБЕЛ — push-to-talk: PRESS -> start, RELEASE -> stop+submit
    if key == glfw.KEY_SPACE:
        if action == glfw.PRESS:
            try:
                VOICE.start()
            except Exception as e:
                print(f"[MIC] start(): {e}")
        elif action == glfw.RELEASE:
            try:
                VOICE.stop_and_submit()
            except Exception as e:
                print(f"[MIC] stop(): {e}")
        return

    # ESC
    elif key == glfw.KEY_ESCAPE and pressed:
        glfw.set_window_should_close(window_, True)



    # в key_callback(...)
    if key == glfw.KEY_V and action == glfw.PRESS:
        start_llm_dialog_thread()
        return


        
    elif key == glfw.KEY_J and pressed:
        open_joint_slider_window()    
    

    # 'O' — example placeholder: do nothing (empty list)
    elif key == glfw.KEY_O and pressed:
        list_unit = []
        move_joints_by_name(list_unit)

    elif key == glfw.KEY_M and pressed:
        try:
            update_world_index()
            center = get_robot_world_pos(ACTIVE)
            mm = build_minimap(center=center)
            print(render_minimap_to_text(mm))
            print(distances_summary_text())
        except Exception as e:
            print(f"[MiniMap] {e}")
    

    elif key == glfw.KEY_T and pressed:
        try:
            ang = float(input("Turn angle (deg, +L/-R): "))
            spd = input("Speed (deg/s) [45]: ").strip()
            spd = float(spd) if spd else 45.0
        except Exception:
            print("Invalid input."); return
        which = input("Which robot? [r1/r2/both, Enter=ACTIVE] ").strip().lower()
        sel = _normalize_robots(which) if which else [ACTIVE]
        for p in sel:
            start_turn_for(p, ang, spd)



    # 'R' — hard reset
    elif key == glfw.KEY_R and pressed:
        do_reset()

    # 'P' — paste motions via modal window (frames or action objects)
    elif key == glfw.KEY_P and pressed:
        cmd_list = open_paste_window()   # blocking modal window
        if cmd_list:
            move_joints_by_name(cmd_list)
            # Ask about saving
            # ans = input("Save this motion to JSON? (y/N): ").strip().lower()
            # if ans in ("y", "yes"):
            #     save_actions_json(cmd_list, out_dir="data_action", filename=None)

    # 'L' — natural language command -> parsed JSON actions -> execute
    elif key == glfw.KEY_L and pressed:
        user_cmd = input("Enter the command in natural language: ")
        print("Thinking...")
        parsed_moves = parse_command(
            user_cmd,
            rag_k=settings.RAG_K,
            rag_csv=settings.RAG_CSV,
            rag_json=settings.RAG_JSON
        )
        print(parsed_moves)
        move_joints_by_name(parsed_moves)
        # Ask about saving
        ans = input("Save this motion to JSON? (y/N): ").strip().lower()
        if ans in ("y", "yes"):
            save_actions_json(parsed_moves, command_text=user_cmd, out_dir="data_action", filename=None)

    elif key == glfw.KEY_C and pressed:
        try:
            num = int(input("Number of steps (0 to cancel)? "))
        except Exception:
            print("Invalid number.")
            return

        if num <= 0:
            print("0 steps. Robot will stay.")
            return

        try:
            dir_deg = float(input("Direction (degrees, 0° = +X, 90° = +Y): "))
            spd = float(input("Speed (m/s): "))
        except Exception:
            print("Invalid direction/speed.")
            return

        which = input("Which robot? [r1/r2/both, Enter = current ACTIVE] ").strip().lower()
        sel = _normalize_robots(which) if which else [ACTIVE]   


        if len(sel) == 1:
            ACTIVE = sel[0]

        for p in sel:
           
            start_walk_for(p, num, dir_deg, spd)

    elif key == glfw.KEY_G and pressed:
        try:
            path = input("CSV path [1000.csv]: ").strip() or "1000.csv"
            start_index = input("Start index (0-based) [0]: ").strip()
            start_index = int(start_index) if start_index else 0
            max_count = input("How many rows to process (blank=all): ").strip()
            max_count = int(max_count) if max_count else None
            interval = input("Interval between samples, seconds [0.0]: ").strip()
            interval = float(interval) if interval else 0.0
            preview = input("Preview only (y/N)? ").strip().lower() in ("y", "yes")
        except Exception as e:
            print(f"[ERR] invalid input: {e}")
            return

        generate_from_csv_and_play(
            csv_path=path,
            out_dir="data_action",
            start_index=start_index,
            max_count=max_count,
            interval_sec=interval,
            preview_only=preview,
        )

    # =========================
    # Key callback (Q) — UPDATED
    # =========================
    elif key == glfw.KEY_Q and pressed:
        print("Starting demo...")
        csv_ = input("Enter CSV path for demo: ")
        jsons_ = input("Enter JSON directory for demo: ")
        reject_ = input("Reject CSV [reject.csv]: ").strip() or "reject.csv"
        run_demo(csv_path=csv_, json_dir=jsons_, reject_csv=reject_)


    elif key == glfw.KEY_F and pressed:
        data = open_cmd_with_image_window()
        if data:
            user_cmd = data["text"]
            image_path = data["image_path"]
            brief = _vlm_describe_image(image_path=image_path)
            print(brief)
            print("[/Vision brief]\n")

            print("Thinking with vision...")
            parsed_moves = parse_command_with_image(user_cmd, image_path)
            print(parsed_moves)
            move_joints_by_name(parsed_moves)


def mouse_button_callback(window_, button, action, mods):
    global is_dragging, last_cursor_pos
    if button == glfw.MOUSE_BUTTON_LEFT:
        if action == glfw.PRESS:
            is_dragging = True
            last_cursor_pos = glfw.get_cursor_pos(window_)
        elif action == glfw.RELEASE:
            is_dragging = False


def cursor_pos_callback(window_, xpos, ypos):
    global is_dragging, last_cursor_pos, cam
    if is_dragging:
        x0, y0 = last_cursor_pos
        dx = xpos - x0
        dy = ypos - y0
        sensitivity = 0.4
        cam.azimuth -= sensitivity * dx
        cam.elevation -= sensitivity * dy
        last_cursor_pos = (xpos, ypos)


def scroll_callback(window_, x_offset, y_offset):
    global cam
    zoom_factor = 1.05
    if y_offset < 0:
        cam.distance *= zoom_factor
    else:
        cam.distance /= zoom_factor
    cam.distance = max(0.05, min(100.0, cam.distance))


# =========================
# Main
# =========================
import argparse
if __name__ == "__main__":
    last_activity_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("--rag-k", type=int, default=5, help="number of examples to few-shot")
    parser.add_argument("--rag-csv", default="1000.csv", help="CSV with commands")
    parser.add_argument("--rag-json", default="data_action", help="dir with JSON")
    args = parser.parse_args()

    settings.RAG_K = args.rag_k
    settings.RAG_CSV = args.rag_csv
    settings.RAG_JSON = args.rag_json



    # File paths
    name_ = "h1_2"
    config_file = f"unitree_rl_gym/deploy/deploy_mujoco/configs/{name_}.yaml"
    policy_path = f"unitree_rl_gym/deploy/pre_train/{name_}/motion.pt"
    xml_path = f"unitree_rl_gym/resources/robots/{name_}/scene_twins_obj.xml" # with objects
    #xml_path = f"unitree_rl_gym/resources/robots/{name_}/scene_zero.xml"

    # Load configuration
    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]
        kps = np.array(config["kps"], dtype=np.float32)
        kds = np.array(config["kds"], dtype=np.float32)
        default_angles = np.array(config["default_angles"], dtype=np.float32)

        ang_vel_scale = config["ang_vel_scale"]
        dof_pos_scale = config["dof_pos_scale"]
        dof_vel_scale = config["dof_vel_scale"]
        action_scale = config["action_scale"]
        cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

        num_actions = config["num_actions"]
        num_obs = config["num_obs"]

    # Initialize MuJoCo
    m = mj.MjModel.from_xml_path(xml_path)
    d = mj.MjData(m)
    m.opt.timestep = simulation_dt

    # (1) Build joint name -> qpos index map + compact overview
    joint_map = {}

    # Build dynamic ALLOWED_JOINTS from the model (hinge/slide only)
    ALLOWED_JOINTS = build_allowed_joints_from_model(m)
    print(f"\nAllowed joints for commands ({len(ALLOWED_JOINTS)}):")
    print(", ".join(sorted(ALLOWED_JOINTS)))
    print()

    # Build dynamic frame joint order (index -> name)
    PREFERRED_FRAME_ORDER = [
        "left_shoulder_pitch",
        "left_shoulder_roll",
        "left_shoulder_yaw",
        "left_elbow",
        "right_shoulder_pitch",
        "right_shoulder_roll",
        "right_shoulder_yaw",
        "right_elbow",
    ]

    for j in range(m.njnt):
        name = mj.mj_id2name(m, mj.mjtObj.mjOBJ_JOINT, j) or f"joint_{j}"
        joint_map[name] = m.jnt_qposadr[j]
    joint_index_map = build_joint_index_map(joint_map, PREFERRED_FRAME_ORDER)
    print(f"Frame joint order ({len(joint_index_map)}):")
    print(", ".join(joint_index_map[i] for i in range(len(joint_index_map))))
    print()

    # Reset to defaults and save initial base pose (free joint) from XML
    mj.mj_resetData(m, d)
    mj.mj_forward(m, d)
    INIT_QPOS = d.qpos.copy()

    def axis_letter(ax: np.ndarray) -> str:
        """Return X/Y/Z based on the largest component of the joint axis direction."""
        if np.linalg.norm(ax) < 1e-8:
            return "-"
        return "XYZ"[int(np.argmax(np.abs(ax)))]

    print("\nJoints overview (compact):")
    print(f"{'id':>3} {'name':<28} {'range':>21} {'axis':>4}")
    print("-" * 60)

    for j in range(m.njnt):
        name = mj.mj_id2name(m, mj.mjtObj.mjOBJ_JOINT, j) or f"joint_{j}"
        joint_map[name] = m.jnt_qposadr[j]

        jtype = m.jnt_type[j]
        limited = bool(m.jnt_limited[j])
        rng = m.jnt_range[j]

        # Range string
        if jtype == mj.mjtJoint.mjJNT_HINGE:
            if limited:
                rng_str = f"{np.degrees(rng[0]):7.2f}..{np.degrees(rng[1]):7.2f} deg"
            else:
                rng_str = "unlimited"
        elif jtype == mj.mjtJoint.mjJNT_SLIDE:
            rng_str = f"{rng[0]:7.3f}..{rng[1]:7.3f} m" if limited else "unlimited"
        elif jtype == mj.mjtJoint.mjJNT_BALL:
            rng_str = "quat (no range)"
        else:  # free
            rng_str = "free (6-DoF)"

        # Axis letter
        if jtype in (mj.mjtJoint.mjJNT_HINGE, mj.mjtJoint.mjJNT_SLIDE):
            ax_letter = axis_letter(m.jnt_axis[j])
        else:
            ax_letter = "-"

        print(f"{j:3d} {name:<28} {rng_str:>21} {ax_letter:>4}")
    print()

    # Load policy
  

    # Create GLFW window
    glfw.init()
    width, height = 1200, 900
    window = glfw.create_window(width, height, "MuJoCo Manual Viewer", None, None)
    glfw.make_context_current(window)

    # Rendering structures
    cam = mj.MjvCamera()
    opt = mj.MjvOption()
    scene = mj.MjvScene(m, maxgeom=10000)
    context = mj.MjrContext(m, mj.mjtFontScale.mjFONTSCALE_100)

    # Camera params
    cam.azimuth = 180
    cam.elevation = -15
    cam.distance = 4.0
    cam.lookat = np.array([0.0, 0.0, 0.8])

    # Camera keys (WASD + Up/Down)
    camera_speed = 0.2
    movement = {"forward": False, "backward": False, "left": False, "right": False, "rise": False, "fall": False}

    VOICE = LLMMicAgent()
    RTWS  = LLMRealtimeWSAgent()
    glfw.set_key_callback(window, key_callback)

    

    is_dragging = False
    last_cursor_pos = (0, 0)
    glfw.set_mouse_button_callback(window, mouse_button_callback)
    glfw.set_cursor_pos_callback(window, cursor_pos_callback)
    glfw.set_scroll_callback(window, scroll_callback)

    def _jid(name: str) -> int:
        return mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, name)

    def _aid(name: str) -> int:
        return mj.mj_name2id(m, mj.mjtObj.mjOBJ_ACTUATOR, name)

    # FIX: 12 DOF ног с коленями и правильным порядком
    JOINTS = [
        "left_hip_yaw_joint", "left_hip_pitch_joint", "left_hip_roll_joint",
        "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
        "right_hip_yaw_joint", "right_hip_pitch_joint", "right_hip_roll_joint",
        "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    ]


    def build_robot_maps(prefix: str):
      
        base_candidates = []
        for j in range(m.njnt):
            if m.jnt_type[j] == mj.mjtJoint.mjJNT_FREE:
                nm = mj.mj_id2name(m, mj.mjtObj.mjOBJ_JOINT, j) or ""
                if nm.startswith(prefix):
                    base_candidates.append(j)
        assert base_candidates, f"FREE joint for prefix '{prefix}' not found"
        jid_base = base_candidates[0]

        base_qpos_adr = m.jnt_qposadr[jid_base]
        base_qvel_adr = m.jnt_dofadr[jid_base]

        jids      = np.array([_jid(prefix + j) for j in JOINTS], dtype=int)
        qpos_adrs = np.array([m.jnt_qposadr[j] for j in jids], dtype=int)
        dof_adrs  = np.array([m.jnt_dofadr[j]  for j in jids], dtype=int)
        act_ids   = np.array([_aid(prefix + j) for j in JOINTS], dtype=int)

        return {
            "base_qpos_adr": base_qpos_adr,
            "base_qvel_adr": base_qvel_adr,
            "qpos_adrs": qpos_adrs,
            "dof_adrs":  dof_adrs,
            "act_ids":   act_ids,
        }

    R = {
        "r1_": build_robot_maps("r1_"),
        "r2_": build_robot_maps("r2_"),
    }
    _build_upper_hold_maps() 
    
    POLICY_PATHS = {
        "r1_": "unitree_rl_gym/deploy/pre_train/h1_2/motion.pt",
        "r2_": "unitree_rl_gym/deploy/pre_train/h1_2/motion.pt",  # или свой .pt
    }
    POL = {p: PolicyWrapper(POLICY_PATHS[p]) for p in ("r1_","r2_") if p in R}


    # =========================
    # Walking / balance control state
    # =========================
    steps_needed = 0
    steps_done = 0
    cmd = np.zeros(3, dtype=np.float32)

    robots = [p for p in ("r1_","r2_") if p in R]
    if not robots:
        robots = ["r1_"]

    na = 12
    action   = {p: np.zeros(na, np.float32) for p in robots}
    target   = {p: d.qpos[R[p]["qpos_adrs"]].copy() for p in robots}  
    obs      = {p: np.zeros(num_obs, np.float32) for p in robots}

    cmd_vec  = {p: np.zeros(3, np.float32) for p in robots}
    old_sin  = {p: 0.0 for p in robots}

    GAIT_PERIOD = 0.8
    WALK_UNTIL = {p: 0.0 for p in robots}
    TURN_UNTIL = {p: 0.0 for p in robots}

    TURN_ACC   = {p: 0.0 for p in robots}  
    TURN_LAST  = {p: 0.0 for p in robots} 
    TURN_TGT   = {p: 0.0 for p in robots}   

    WALK_GOAL  = {p: 0 for p in robots}   # сколько шагов осталось
    WALK_COUNT = {p: 0 for p in robots}   # сколько сделано
    TASKS       = {p: [] for p in robots}  
    ACTIVE_TASK = {p: None for p in robots} 


    short_extra_done = False
    last_dir_deg = 0.0

    counter = 0

    # =========================
    # Main simulation loop
    # =========================
    while not glfw.window_should_close(window):
        process_action_queue()
        azimuth_rad = math.radians(cam.azimuth)
        right_vec = np.array([math.sin(azimuth_rad), -math.cos(azimuth_rad), 0])
        forward_vec = np.array([right_vec[1], -right_vec[0], 0])

        if _maybe_freeze():
            continue

        if movement["forward"]:
            cam.lookat -= camera_speed * forward_vec
        if movement["backward"]:
            cam.lookat += camera_speed * forward_vec
        if movement["left"]:
            cam.lookat -= camera_speed * right_vec
        if movement["right"]:
            cam.lookat += camera_speed * right_vec
        if movement["rise"]:
            cam.lookat[2] += camera_speed
        if movement["fall"]:
            cam.lookat[2] -= camera_speed

        # NOTE: This overrides num_actions each loop as in the original script.
        # Keep as-is unless your model/config demands otherwise.
        num_actions = 12

        for p in robots:
            mp = R[p]
            qj  = d.qpos[mp["qpos_adrs"]]
            dqj = d.qvel[mp["dof_adrs"]]                
            tau = pd_control(target[p], qj, kps, np.zeros_like(kds), dqj, kds)
            d.ctrl[mp["act_ids"]] = tau                   


        mj.mj_step(m, d)
        counter += 1

        now = time.time()
        for p in robots:
            if WALK_UNTIL[p] and now >= WALK_UNTIL[p]:
                WALK_UNTIL[p] = 0.0
                cmd_vec[p][:] = 0.0
            if TURN_UNTIL[p] and now >= TURN_UNTIL[p]:        # <<< new
                TURN_UNTIL[p] = 0.0
                cmd_vec[p][:] = 0.0


        now = time.time()
       
        for p in robots:
          
            if ACTIVE_TASK[p] is None and TASKS[p]:
                ACTIVE_TASK[p] = TASKS[p].pop(0)

            task = ACTIVE_TASK[p]
            if task and task['type'] == 'pose':
                t0, t1 = task['t0'], task['t1']
             
                if t1 <= t0:
                    alpha = 1.0
                else:
                    u = (now - t0) / (t1 - t0)
                 
                    u = max(0.0, min(1.0, u))
                    alpha = u*u*(3-2*u)

                for jname, q1 in task['target'].items():
                    idx = joint_map.get(jname)
                    if idx is None: 
                        continue
                    q0 = task['start'][jname]
                    d.qpos[idx] = (1.0 - alpha) * q0 + alpha * q1

            
                if now >= t1:
                    ACTIVE_TASK[p] = None
        # --- ДЕРЖИМ ВЕРХ В НУЛЕ, ЕСЛИ НЕТ АКТИВНОЙ ПОЗЫ У РОБОТА ---
        if HOLD_UPPER_ZERO:
            for p in robots:
                if ACTIVE_TASK[p] is not None:
                    continue
                dof_idx = HOLD_UPPER_DOF.get(p)
                if dof_idx is not None and dof_idx.size:
                    d.qvel[dof_idx] = 0.0
        mj.mj_forward(m, d)
        



        if counter % control_decimation == 0:
            for p in robots:
                mp = R[p]
              
                xyz = d.qpos[mp["base_qpos_adr"] : mp["base_qpos_adr"]+3]
                quat = d.qpos[mp["base_qpos_adr"]+3 : mp["base_qpos_adr"]+7]
             
                omega = d.qvel[mp["base_qvel_adr"]+3 : mp["base_qvel_adr"]+6]


                qj  = d.qpos[mp["qpos_adrs"]]
                dqj = d.qvel[mp["dof_adrs"]]

                qj_scaled  = (qj - default_angles) * dof_pos_scale
                dqj_scaled = dqj * dof_vel_scale
                grav = get_gravity_orientation(quat)
                omega_scaled = omega * ang_vel_scale

                def _wrap_to_pi(a): return (a + math.pi) % (2*math.pi) - math.pi
                def _yaw(qw,qx,qy,qz):
                    return math.atan2(2*(qw*qz+qx*qy), 1-2*(qy*qy+qz*qz))

                if TURN_TGT[p] > 0.0:
                    yw_now  = _yaw(*quat)
                    inc     = _wrap_to_pi(yw_now - TURN_LAST[p])
                    TURN_ACC[p] += abs(inc)
                    TURN_LAST[p] = yw_now

                    if TURN_ACC[p] >= TURN_TGT[p]:
                       
                        cmd_vec[p][:] = 0.0
                        TURN_TGT[p] = 0.0
                        TURN_ACC[p] = 0.0
                    

                period = GAIT_PERIOD
                time_in_sim = counter * simulation_dt
                phase = (time_in_sim % period) / period
                sin_phase = math.sin(2*math.pi*phase); cos_phase = math.cos(2*math.pi*phase)
                                # Завершение строго по числу шагов (нолепереход sin: - → +)
                if WALK_GOAL[p] > 0 and (sin_phase >= 0.0 and old_sin[p] < 0.0):
                    WALK_COUNT[p] += 1
                    if WALK_COUNT[p] >= WALK_GOAL[p]:
                        cmd_vec[p][:] = 0.0          # стоп команда ходьбы
                        WALK_GOAL[p]  = 0
                        WALK_COUNT[p] = 0
                        WALK_UNTIL[p] = 0.0          # на всякий случай
                # обновляем следящий синус
                old_sin[p] = sin_phase

                o = obs[p]
                o[:3] = omega_scaled
                o[3:6] = grav
                o[6:9] = cmd_vec[p] * cmd_scale
                o[9:9+na] = qj_scaled
                o[9+na:9+2*na] = dqj_scaled
                o[9+2*na:9+3*na] = action[p]
                o[9+3*na:9+3*na+2] = np.array([sin_phase, cos_phase])

                act = POL[p].act(o)
                action[p][:] = act
                target[p] = act * action_scale + default_angles

              
                old_sin[p] = sin_phase


      

        # Render
        width, height = glfw.get_framebuffer_size(window)
        viewport = mj.MjrRect(0, 0, width, height)
        mj.mjv_updateScene(m, d, opt, None, cam, mj.mjtCatBit.mjCAT_ALL, scene)
        mj.mjr_render(viewport, scene, context)

        glfw.swap_buffers(window)
        glfw.poll_events()

    glfw.terminate()
    print("Window closed, exiting.")
