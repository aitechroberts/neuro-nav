# 🚀 Quick Start Cheat Sheet

## Start Working (30 seconds)

```bash
cd ~/projects/neuro-nav
git checkout jesse-dev              # Your working branch with all fixes
source .venv/bin/activate
source ./use-cuda-126.sh
python conceptgraph/slam/rerun_realtime_mapping.py --config-name=rerun_simple_test
```

## Common Commands

```bash
# Different scenes
scene_id: room0      # Living room
scene_id: office0    # Office
scene_id: room1      # Bedroom

# Process more frames
end: 100             # Default: 30
stride: 1            # Default: 3 (every 3rd frame)

# More/less objects
mask_conf_threshold: 0.5   # Higher = fewer objects
obj_min_detections: 3      # Require more sightings
```

## File Locations

- **Configs**: `conceptgraph/hydra_configs/`
- **Data**: `~/projects/neuro-nav/data/Replica/`
- **Code**: `conceptgraph/slam/` and `conceptgraph/utils/`
- **Output**: `data/Replica/{scene}/exps/`

## Stop Everything

```bash
# Kill running processes
pkill -f realtime_mapping.py

# Or just close the Rerun window
```

## View Saved Recording

```bash
rerun ~/projects/neuro-nav/data/Replica/room0/exps/*/rerun*.rrd
```

---
See **JesseReadMe.md** for full documentation and research ideas!

