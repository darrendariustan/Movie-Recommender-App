#!/bin/bash

# Create streamlit config directory
mkdir -p ~/.streamlit/

# Create streamlit config file
cat > ~/.streamlit/config.toml << EOF
[server]
port = $PORT
address = "0.0.0.0"
headless = true
enableCORS = false
enableXsrfProtection = false

[browser]
gatherUsageStats = false
EOF

# Start the application
streamlit run app.py --server.port $PORT --server.address 0.0.0.0 