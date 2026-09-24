"""ให้ tests import แพ็กเกจ pipeline ได้ (เพิ่มโฟลเดอร์โปรเจกต์เข้า sys.path)"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
