import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from rov.control_server import main


if __name__ == "__main__":
    main()
