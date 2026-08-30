# -*- coding: utf-8 -*-
"""
全局语音输入工具
按住 Ctrl+Space 说话，松开后自动把识别出的文字粘贴到当前光标位置。

资源占用设计（空闲时近乎为零）：
- 全程事件驱动：键盘钩子 + 主线程挂起等待，无任何轮询，空闲时 CPU 占用 0%
- 只在录音期间创建音频流，回调每 100ms 追加一次数据，开销极小
- 识别使用 int8 量化的 faster-whisper 模型（CTranslate2 推理），内存约为原版的 1/4
"""

import os

# 模型下载走国内镜像，必须放在导入 faster_whisper 之前
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import threading
import time

import numpy as np
import sounddevice as sd
import keyboard
import pyperclip
from faster_whisper import WhisperModel

# ---------------- 可按需修改的配置 ----------------
MODEL_SIZE = "base"    # 模型档位: tiny(最省内存最快) | base(推荐) | small(更准但更慢)
LANGUAGE = "zh"        # 识别语言；改为 None 可自动检测中英文
COMPUTE_TYPE = "int8"  # int8 量化，CPU 推理最省内存
SAMPLE_RATE = 16000    # Whisper 固定采样率
HOTKEY_MOD = "ctrl"    # 组合键修饰键
HOTKEY_KEY = "space"   # 组合键主键，与修饰键组合即 Ctrl+Space
MIN_SECONDS = 0.3      # 短于该时长的录音直接忽略（防误触）
# --------------------------------------------------

model = None


class Recorder:
    """按需启停的录音器：不录音时不存在音频流，不占用任何资源"""

    def __init__(self):
        self._frames = []
        self._stream = None
        self._lock = threading.Lock()

    def start(self):
        self._frames = []

        def _on_block(indata, frames, t, status):
            with self._lock:
                self._frames.append(indata.copy())

        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=1600,  # 100ms 回调一次，降低线程唤醒频率
            callback=_on_block,
        )
        self._stream.start()

    def stop(self):
        stream, self._stream = self._stream, None
        if stream is not None:
            stream.stop()
            stream.close()
        with self._lock:
            if not self._frames:
                return None
            return np.concatenate(self._frames).ravel()


recorder = Recorder()
_state = {"recording": False, "busy": False}
_state_lock = threading.Lock()


def transcribe_and_paste(audio):
    try:
        segments, _ = model.transcribe(
            audio,
            language=LANGUAGE,
            beam_size=1,
            vad_filter=True,
            initial_prompt="以下是普通话的句子，请用简体中文输出。",
        )
        text = "".join(seg.text for seg in segments).strip()
        if text:
            print(f"[结果] {text}")
            pyperclip.copy(text)
            time.sleep(0.05)  # 等剪贴板就绪
            keyboard.send("ctrl+v")  # 粘贴到当前光标处
        else:
            print("[提示] 未识别到语音")
    except Exception as exc:
        print(f"[错误] 识别失败: {exc}")
    finally:
        with _state_lock:
            _state["busy"] = False


def on_key_event(event):
    audio = None
    try:
        with _state_lock:
            if _state["busy"]:
                return
            if event.event_type == "down":
                if (event.name == HOTKEY_KEY
                        and keyboard.is_pressed(HOTKEY_MOD)
                        and not _state["recording"]):
                    _state["recording"] = True
                    recorder.start()
                    print("● 录音中…（松开 Ctrl+Space 结束）")
                return
            # 松开事件：结束录音并转后台识别
            if _state["recording"] and event.name in (HOTKEY_KEY, HOTKEY_MOD):
                _state["recording"] = False
                audio = recorder.stop()
    except Exception as exc:
        print(f"[错误] 录音异常: {exc}")
        return

    if audio is None:
        return
    if len(audio) < int(SAMPLE_RATE * MIN_SECONDS):
        print("[提示] 录音太短，已忽略")
        return
    with _state_lock:
        _state["busy"] = True
    print("○ 识别中…")
    threading.Thread(target=transcribe_and_paste, args=(audio,), daemon=True).start()


def main():
    global model
    print(f"[启动] 正在加载模型 {MODEL_SIZE}（首次运行需联网下载）…")
    model = WhisperModel(MODEL_SIZE, device="cpu", compute_type=COMPUTE_TYPE)
    print("[启动] 模型就绪")
    keyboard.hook(on_key_event)
    print("[启动] 全局热键已生效：按住 Ctrl+Space 说话，松开后文字自动粘贴到光标处")
    print("[启动] 直接关闭本窗口即可退出")
    keyboard.wait()  # 挂起主线程，空闲时 0% CPU


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[退出] 再见")
