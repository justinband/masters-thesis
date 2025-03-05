# #!/bin/bash
# cp ./simulator_cli.py /usr/local/bin/simulator
# chmod +x /usr/local/bin/simulator


#!/bin/bash

# Define the source file and destination
SOURCE_SCRIPT_NAME="simulator_cli.py"
DEST_SCRIPT_NAME="simulator"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SOURCE_PATH="$SCRIPT_DIR/$SOURCE_SCRIPT_NAME"
echo $SOURCE_PATH
DEST_PATH="/usr/local/bin/$DEST_SCRIPT_NAME"
echo $DEST_PATH

# Ensure the script exists
if [[ ! -f "$SOURCE_PATH" ]]; then
    echo "Error: $SCRIPT_NAME not found in $(pwd)"
    exit 1
fi

# Copy the script
cp "$SOURCE_PATH" "$DEST_PATH"

# Make it executable
chmod +x "$DEST_PATH"

# Confirm success
echo "$SCRIPT_NAME has been installed to /usr/local/bin and is now executable."
