#!/bin/bash
# Install (or remove) the local schedule that keeps citations.yml current.
#
# Google Scholar blocks GitHub Actions runners, so the refresh has to happen
# from a home network. This sets up a launchd job that runs
# bin/refresh_citations.sh on Mon/Wed/Fri at 10:00 local time. launchd runs a
# missed job when the machine next wakes, so a closed laptop only delays it.
#
# Usage:
#   bin/install_citation_schedule.sh            # install
#   bin/install_citation_schedule.sh --uninstall
#   bin/install_citation_schedule.sh --status

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LABEL="io.github.rishav1.citations"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
VENV="$HOME/.venvs/al-folio-citations"
LOG_DIR="$HOME/Library/Logs"

case "${1:-}" in
  --uninstall)
    launchctl unload "$PLIST" 2>/dev/null || true
    rm -f "$PLIST"
    echo "Removed $LABEL. The venv at $VENV was left in place."
    exit 0
    ;;
  --status)
    launchctl list | grep "$LABEL" || echo "$LABEL is not loaded."
    echo "--- last 20 log lines ---"
    tail -n 20 "$LOG_DIR/$LABEL.log" 2>/dev/null || echo "(no log yet)"
    exit 0
    ;;
esac

echo "Creating interpreter at $VENV..."
python3 -m venv "$VENV"
"$VENV/bin/pip" install --quiet --upgrade pip
"$VENV/bin/pip" install --quiet pyyaml

echo "Writing $PLIST..."
mkdir -p "$(dirname "$PLIST")" "$LOG_DIR"
cat > "$PLIST" <<PLIST_EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>$LABEL</string>
    <key>ProgramArguments</key>
    <array>
        <string>/bin/bash</string>
        <string>$REPO_DIR/bin/refresh_citations.sh</string>
    </array>
    <key>EnvironmentVariables</key>
    <dict>
        <key>CITATIONS_PYTHON</key>
        <string>$VENV/bin/python</string>
        <key>PATH</key>
        <string>/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
    </dict>
    <key>StartCalendarInterval</key>
    <array>
        <dict><key>Weekday</key><integer>1</integer><key>Hour</key><integer>10</integer><key>Minute</key><integer>0</integer></dict>
        <dict><key>Weekday</key><integer>3</integer><key>Hour</key><integer>10</integer><key>Minute</key><integer>0</integer></dict>
        <dict><key>Weekday</key><integer>5</integer><key>Hour</key><integer>10</integer><key>Minute</key><integer>0</integer></dict>
    </array>
    <key>StandardOutPath</key>
    <string>$LOG_DIR/$LABEL.log</string>
    <key>StandardErrorPath</key>
    <string>$LOG_DIR/$LABEL.log</string>
    <key>RunAtLoad</key>
    <false/>
</dict>
</plist>
PLIST_EOF

launchctl unload "$PLIST" 2>/dev/null || true
launchctl load "$PLIST"

echo
echo "Installed. Runs Mon/Wed/Fri at 10:00 local time."
echo "  Logs:      $LOG_DIR/$LABEL.log"
echo "  Status:    bin/install_citation_schedule.sh --status"
echo "  Run now:   launchctl start $LABEL"
echo "  Remove:    bin/install_citation_schedule.sh --uninstall"
