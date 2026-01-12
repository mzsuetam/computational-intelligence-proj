## Using `tmux` for Jupyter Lab Sessions

`tmux` is a terminal multiplexer that allows you to run persistent terminal sessions, which is especially useful for running Jupyter Lab on remote servers. Here’s a quick guide:

0. **Enable lingering for your user (if not already enabled):**
    ```bash
    loginctl enable-linger $(whoami)
    ```

1. **Start a new tmux session and launch Jupyter Lab:**
    ```bash
    tmux new -s deeplearning
    jupyter lab --no-browser --port=8888
    ```
    - This creates a new tmux session named `deeplearning` and starts Jupyter Lab.  
    - You can safely disconnect from the server or close your terminal; the session will keep running.

2. **Detach from the tmux session:**  
    - Press `Ctrl+b`, then `d` to detach and return to your normal shell.

3. **Reattach to your tmux session later:**
    ```bash
    tmux attach -t deeplearning
    ```
    - If you forget the session name, list all sessions with `tmux ls` and attach using the session number or name.

4. **Kill the tmux session when done:**
    ```bash
    tmux kill-session -t deeplearning
    ```
    - This will terminate the session and any processes running inside it.

**Summary:**  
- Use `tmux` to keep your Jupyter Lab running even if you disconnect.
- Detach and reattach as needed.
- Kill the session when finished to free resources.