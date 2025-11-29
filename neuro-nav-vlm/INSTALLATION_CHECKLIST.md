# VLM Installation Checklist

Follow this checklist to ensure everything is set up correctly.

## ☑️ Pre-Installation

- [ ] Linux system with NVIDIA GPU
- [ ] GPU has 8GB+ VRAM (check with `nvidia-smi`)
- [ ] CUDA 12.6+ installed
- [ ] Python 3.8+ available
- [ ] 10GB+ free disk space
- [ ] Internet connection (for model downloads)

## ☑️ Environment Setup

- [ ] Navigate to directory: `cd /home/nick/Project_dir/neuro-nav-vlm`
- [ ] Activate environment: `source /home/nick/Project_dir/neuro-nav/.venv/bin/activate`
- [ ] Load CUDA: `source /home/nick/Project_dir/neuro-nav/use-cuda-126.sh`
- [ ] Verify Python: `python --version` (should be 3.8+)
- [ ] Verify CUDA: `python -c "import torch; print(torch.cuda.is_available())"` (should be True)

## ☑️ Dependencies

- [ ] Install VLM requirements: `pip install -r requirements_vlm.txt`
- [ ] Verify imports: `python test_vlm_setup.py`
- [ ] Check for errors in import test

## ☑️ Data Setup

- [ ] Run data linker: `./setup_data_link.sh`
- [ ] Verify link: `ls -la data` (should point to neuro-nav/data)
- [ ] Check data exists: `ls data/outputs/` (should show timestamp directories)

## ☑️ Model Download

- [ ] Run download script: `python download_models.py`
- [ ] Select option 5 (recommended models)
- [ ] Wait for Florence-2-large (~770MB)
- [ ] Wait for Qwen2-VL-2B (~4GB)
- [ ] Verify cache: `ls ~/.cache/huggingface/hub/` (should show model directories)

## ☑️ Verification

- [ ] Run full test: `python test_vlm_setup.py`
- [ ] Answer 'y' to inference tests
- [ ] All tests pass ✓

## ☑️ Scene Map Preparation

Choose one:

**Option A: Use existing scene map**
- [ ] Find existing map: `find data/outputs -name "scene_map_cfslam.pkl.gz"`
- [ ] Note the path for later

**Option B: Create new scene map**
- [ ] Go to neuro-nav: `cd /home/nick/Project_dir/neuro-nav`
- [ ] Activate environment
- [ ] Run SLAM: `python conceptgraph/slam/rerun_realtime_mapping.py --config-name=rerun_simple_test`
- [ ] Wait for completion (~5-10 minutes)
- [ ] Find output: `find data/outputs -name "scene_map_cfslam.pkl.gz" | tail -1`

## ☑️ Run VLM Pipeline

- [ ] Return to neuro-nav-vlm: `cd /home/nick/Project_dir/neuro-nav-vlm`
- [ ] Run pipeline: `./run_vlm_pipeline.sh`
- [ ] OR specify path: `./run_vlm_pipeline.sh data/outputs/[timestamp]`
- [ ] Wait for Step 1: Extract captions (~5-10 min)
- [ ] Wait for Step 2: Refine captions (~3-5 min)
- [ ] Wait for Step 3: Build scene graph (~5-10 min)
- [ ] Wait for Step 4: Generate JSON (~1 min)
- [ ] Check for success message ✓

## ☑️ Verify Results

- [ ] Check scene graph exists: `ls data/outputs/*/room0/scene_graph.json`
- [ ] View sample: `cat data/outputs/*/room0/scene_graph.json | jq '.[0:2]'`
- [ ] Check debug images: `ls data/outputs/*/room0/cfslam_captions_florence_debug/`
- [ ] View captions: `cat data/outputs/*/room0/cfslam_florence_captions.json | head -50`

## ☑️ Test Querying

- [ ] Run query script: `python query_vlm_scene.py`
- [ ] Try sample query: "Where is the chair?"
- [ ] Verify response makes sense
- [ ] Try another query: "What objects are in the room?"
- [ ] Exit with 'quit'

## ☑️ Optional: Compare with Original

- [ ] Run original pipeline on same scene (in neuro-nav directory)
- [ ] Compare processing times
- [ ] Compare caption quality
- [ ] Compare scene graph structure
- [ ] Note differences

## 🎉 Success Criteria

All of the following should be true:

✅ All imports work without errors  
✅ Models downloaded successfully  
✅ Pipeline completes all 4 steps  
✅ scene_graph.json created  
✅ Query script responds to questions  
✅ No CUDA out of memory errors  

## 🐛 Common Issues

| Issue | Solution |
|-------|----------|
| "No module named 'torch'" | Activate environment first |
| "CUDA out of memory" | Use smaller models or reduce --max-detections-per-object |
| "No scene map found" | Run SLAM pipeline first |
| "qwen-vl-utils not found" | `pip install qwen-vl-utils` |
| Model download hangs | Check internet, check disk space |
| Import errors | `pip install -r requirements_vlm.txt --force-reinstall` |

## 📖 Next Steps

After completing this checklist:

1. Read [README.md](README.md) for usage examples
2. Try different queries with `query_vlm_scene.py`
3. Experiment with different VLM models
4. Integrate with your robot code
5. Compare results with original neuro-nav

## 💡 Tips

- **Save time:** Download models overnight
- **Save memory:** Close other GPU applications before running
- **Save money:** Everything runs locally, no API costs
- **Save results:** Each run creates timestamped output directory
- **Debug:** Check debug images in `cfslam_captions_florence_debug/`

---

**Need help?** → [SETUP_GUIDE.md](SETUP_GUIDE.md)  
**Quick reference?** → [QUICKSTART.md](QUICKSTART.md)  
**Technical details?** → [README_VLM.md](README_VLM.md)

