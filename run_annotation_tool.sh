#!/bin/bash
# Startup script for MaxShapley Annotation Tool

echo "Starting MaxShapley Annotation Tool..."
echo ""

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null
then
    echo "Error: streamlit is not installed or not in PATH"
    echo "Please install with: pip install streamlit>=1.30.0"
    exit 1
fi

# Check if data has been prepared
if [ ! -d "data/samples" ] || [ ! -f "data/samples/hotpotqa_100.json" ]; then
    echo "Warning: Sample data not found!"
    echo "Running data preparation script..."
    echo ""
    cd annotation_tool
    python3 data_preparation.py
    cd ..
    echo ""
fi

# Run Streamlit app
echo "Launching annotation tool in your browser..."
echo "Press Ctrl+C to stop the server"
echo ""
streamlit run annotation_tool/app.py
