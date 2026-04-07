---
name: test
description: Run the SmartSplit test suite and verify everything passes. Use when code has been modified or before committing.
---

# Test SmartSplit

Run the full test suite and report the results:

```bash
source .venv/bin/activate && python -m pytest tests/ -v
```

Then verify SmartSplit starts without errors:

```bash
source .venv/bin/activate && timeout 3 python -m smartsplit 2>&1 || true
```

**Expected:** All tests pass, SmartSplit starts and shows "Initialized providers: [...]".

If tests fail:
1. Read the error output carefully
2. Fix the failing test or the code it tests
3. Re-run until all tests pass
4. Do NOT skip or delete failing tests
