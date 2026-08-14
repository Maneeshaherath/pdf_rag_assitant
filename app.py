import os

os.environ.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "none")

from ui import main


if __name__ == "__main__":
    main()
