# Which File Should I Read?

**Choose based on what you need:**

---

## 🎯 I Just Want to Run It

👉 **[COPY_PASTE_COMMANDS.sh](COPY_PASTE_COMMANDS.sh)**

Copy and paste commands from this file. No explanations, just commands.

---

## 🚀 Quick Start (3 Commands)

👉 **[README_SIMPLE.md](README_SIMPLE.md)**

Absolute minimum to get running. 3 commands, then you're done.

---

## 📖 Complete Setup Guide

👉 **[START_HERE.md](START_HERE.md)**

Detailed step-by-step with explanations. Best for first-time setup.

---

## 🔧 I'm Having Problems

👉 **[START_HERE.md](START_HERE.md)** - Section "Troubleshooting"

Or **[COPY_PASTE_COMMANDS.sh](COPY_PASTE_COMMANDS.sh)** - Section 7

---

## 🤔 I Want to Understand How It Works

👉 **[README.md](README.md)** - Full technical documentation

Or **[README_VLM.md](README_VLM.md)** - VLM architecture details

---

## 📋 Installation Checklist

👉 **[INSTALLATION_CHECKLIST.md](INSTALLATION_CHECKLIST.md)**

Step-by-step checklist format.

---

## 🎬 I Already Set It Up, Just Need to Run Again

```bash
cd /home/nick/Project_dir/neuro-nav-vlm
source /home/nick/Project_dir/neuro-nav/.venv/bin/activate
source /home/nick/Project_dir/neuro-nav/use-cuda-126.sh
export PYTHONPATH=/home/nick/Project_dir/neuro-nav-vlm:$PYTHONPATH
./run_everything.sh
```

---

## 📁 File Overview

| File | Purpose | Length |
|------|---------|--------|
| **COPY_PASTE_COMMANDS.sh** | Commands only, no explanation | 5 min |
| **README_SIMPLE.md** | Ultra-quick start guide | 2 min |
| **START_HERE.md** | Complete setup guide | 10 min |
| **run_everything.sh** | Automated pipeline runner | - |
| **README.md** | Full technical docs | 20 min |
| **INSTALLATION_CHECKLIST.md** | Installation checklist | 10 min |
| **SETUP_GUIDE.md** | Detailed setup instructions | 15 min |
| **QWEN_ONLY_SETUP.md** | Qwen2-VL architecture info | 10 min |

---

## 🆘 Emergency: Something Broke

**Problem: "ModuleNotFoundError"**
```bash
cd /home/nick/Project_dir/neuro-nav-vlm
pip install -e .
export PYTHONPATH=/home/nick/Project_dir/neuro-nav-vlm:$PYTHONPATH
```

**Problem: "Scene map not found"**
```bash
find /home/nick/Project_dir/neuro-nav/data/Replica/room0/exps/r_* -name "*.pkl.gz"
# See START_HERE.md Step 5
```

**Problem: "CUDA out of memory"**
```bash
pkill -9 python
# Then restart with fewer frames or close other GPU programs
```

---

## 💡 Recommendation

1. **First time?** → Read **START_HERE.md**
2. **Already know what you're doing?** → Use **COPY_PASTE_COMMANDS.sh**
3. **Quick refresh?** → Use **README_SIMPLE.md**
4. **Something broke?** → Check troubleshooting in **START_HERE.md**

---

**Ready?** → Open **[START_HERE.md](START_HERE.md)** 📖


