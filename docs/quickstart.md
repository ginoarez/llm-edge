# Quick Start (colibri-inspired)

1. Detect hardware:
   ```powershell
   python start_here.py --list
   ```
2. Get a model (example):
   ```powershell
   ollama pull hf.co/mradermacher/DeepSeek-V4-Pro-Qwen3.5-4B-UNCENSORED-heretic-GGUF:Q4_K_M
   ```
3. Benchmark:
   ```powershell
   python benchmark.py --model <model> --runs 3 --num-ctx 4096
   ```
4. Serve:
   ```powershell
   uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```
5. Dashboard: http://localhost:8000/dashboard
