#!/usr/bin/env python3
import sys
try:
    import pytest
except Exception as e:
    print('pytest not available:', e)
    sys.exit(2)

sys.exit(pytest.main(['-q']))
