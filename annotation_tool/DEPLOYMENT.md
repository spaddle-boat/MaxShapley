# MaxShapley Annotation Tool - Deployment Guide

## Overview

This is a Streamlit-based web application for annotating source relevance in multi-hop QA datasets (HotpotQA, MS MARCO, MuSiQue).

## Prerequisites

- Python 3.8+
- pip

## Installation

### 1. Install Dependencies

```bash
cd MaxShapley
pip install streamlit filelock pandas extra-streamlit-components
```

Or install from requirements:
```bash
pip install -r annotation_tool/requirements.txt
```

### 2. Verify Data Files

Ensure these files exist:
- `data/samples/hotpotqa_100.json`
- `data/samples/msmarco_100.json`
- `data/samples/musique_100.json`
- `data/annotations/*.jsonl` (will be created automatically if not present)

## Starting the Service

### Development Mode (Local)

```bash
cd MaxShapley
streamlit run annotation_tool/app.py --server.port 8501
```

Access at: http://localhost:8501

### Production Mode (Server)

#### Option 1: Direct Run (Background)

```bash
cd MaxShapley
nohup streamlit run annotation_tool/app.py \
    --server.port 8501 \
    --server.headless true \
    --server.address 0.0.0.0 \
    > logs/streamlit.log 2>&1 &

echo $! > streamlit.pid
```

#### Option 2: Using Screen/Tmux

```bash
# Using screen
screen -S annotation
streamlit run annotation_tool/app.py --server.port 8501 --server.headless true --server.address 0.0.0.0
# Detach with Ctrl+A, D

# Reattach later
screen -r annotation
```

#### Option 3: Systemd Service (Recommended for Production)

Create `/etc/systemd/system/maxshapley-annotation.service`:

```ini
[Unit]
Description=MaxShapley Annotation Tool
After=network.target

[Service]
Type=simple
User=YOUR_USER
WorkingDirectory=/path/to/MaxShapley
ExecStart=/usr/bin/python3 -m streamlit run annotation_tool/app.py --server.port 8501 --server.headless true --server.address 0.0.0.0
Restart=always
RestartSec=10
Environment=STREAMLIT_GATHER_USAGE_STATS=false

[Install]
WantedBy=multi-user.target
```

Then:
```bash
sudo systemctl daemon-reload
sudo systemctl enable maxshapley-annotation
sudo systemctl start maxshapley-annotation
```

## Stopping the Service

### If running in foreground
Press `Ctrl+C`

### If running in background with PID file
```bash
kill $(cat streamlit.pid)
```

### If using systemd
```bash
sudo systemctl stop maxshapley-annotation
```

### Find and kill by process
```bash
pkill -f "streamlit run annotation_tool/app.py"
```

## Service Management Commands

### Check Status
```bash
# Systemd
sudo systemctl status maxshapley-annotation

# Or check if process is running
pgrep -f "streamlit run annotation_tool"

# Check which port is in use
lsof -i :8501
```

### View Logs
```bash
# Systemd
sudo journalctl -u maxshapley-annotation -f

# Or if using log file
tail -f logs/streamlit.log
```

### Restart Service
```bash
# Systemd
sudo systemctl restart maxshapley-annotation

# Or manual
pkill -f "streamlit run annotation_tool/app.py"
sleep 2
streamlit run annotation_tool/app.py --server.port 8501 --server.headless true &
```

## Configuration

### Authentication
- Default password: `shapley123888`
- To change: Edit `AUTH_PASSWORD` in `annotation_tool/app.py`

### Port
- Default: 8501
- Change with `--server.port XXXX`

### Network Access
- Local only: `--server.address localhost`
- All interfaces: `--server.address 0.0.0.0`

## Data Backup & Maintenance

### Backup Annotations
```bash
# Create timestamped backup
cp -r data/annotations data/annotations_backup_$(date +%Y%m%d_%H%M%S)

# Or tar archive
tar -czvf annotations_backup_$(date +%Y%m%d).tar.gz data/annotations/
```

### Data Files Structure
```
data/
├── annotations/           # User annotations (JSONL format)
│   ├── hotpotqa_annotations.jsonl
│   ├── msmarco_annotations.jsonl
│   └── musique_annotations.jsonl
├── samples/               # 100-sample datasets for annotation
│   ├── hotpotqa_100.json
│   ├── msmarco_100.json
│   └── musique_100.json
└── *_annotated_subset.json  # Original 30 annotated samples
```

### Lock Files
Lock files (`*.lock`) are created for concurrent access safety. They can be safely deleted when the service is stopped:
```bash
rm data/annotations/*.lock
```

## Troubleshooting

### Port Already in Use
```bash
# Find process using port
lsof -i :8501

# Kill it
kill -9 <PID>
```

### Cookie/Session Issues
- Clear browser cookies for the site
- Or restart the service

### Module Import Errors
```bash
# Ensure you're in the right directory
cd MaxShapley

# Check Python path
python3 -c "import sys; print(sys.path)"

# Reinstall dependencies
pip install --upgrade streamlit filelock pandas extra-streamlit-components
```

### Annotations Not Saving
- Check file permissions: `ls -la data/annotations/`
- Ensure write access: `chmod 755 data/annotations/`
- Check disk space: `df -h`

## Reverse Proxy Setup (Optional)

### Nginx Configuration
```nginx
server {
    listen 80;
    server_name annotation.yourdomain.com;

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 86400;
    }
}
```

### With HTTPS (Let's Encrypt)
```bash
sudo certbot --nginx -d annotation.yourdomain.com
```

## Health Check

Simple health check endpoint:
```bash
curl -s http://localhost:8501/_stcore/health
# Should return: ok
```

For monitoring scripts:
```bash
#!/bin/bash
if ! curl -s http://localhost:8501/_stcore/health | grep -q "ok"; then
    echo "Service down, restarting..."
    sudo systemctl restart maxshapley-annotation
fi
```
