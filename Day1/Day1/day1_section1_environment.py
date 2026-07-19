"""
=============================================================================
Applied GenAI Engineering Program
Day 1 · Section 1: Verify Your Environment
=============================================================================

WHAT THIS FILE DOES:
  Checks three things on your machine before we write any AI code:
    1. Is your Python version 3.10 or higher?
    2. Are you inside a virtual environment?
    3. Are the required packages installed?

WHY THIS MATTERS:
  Every professional project starts by confirming the environment.
  If your Python version is wrong or a package is missing, nothing
  we build later will work — and the errors will be confusing.
  Better to catch it now in 30 seconds than debug it for 30 minutes.

HOW TO RUN:
  1. Open your terminal
  2. Activate your virtual environment:
       Mac/Linux:  source genai_env/bin/activate
       Windows:    genai_env\Scripts\activate
  3. Run:  python day1_section1_environment.py
=============================================================================
"""

# ─── IMPORTS ────────────────────────────────────────────────────────────────
# Each import brings in a "toolbox" of pre-written code we can use.
# Python comes with many built-in toolboxes — these are called the
# "standard library". You don't need to install them.

import sys              # sys = "system" — gives us info about the Python
                        # installation itself (version, path, etc.)

import importlib.util   # importlib.util lets us check if a package is
                        # installed WITHOUT actually importing it.
                        # This is faster and safer than try/except ImportError.


# ─── SECTION START ──────────────────────────────────────────────────────────
print("=" * 60)
print("  Section 1: Verifying your environment")
print("=" * 60)
print()


# ─── CHECK 1: Python Version ───────────────────────────────────────────────

def check_python_version() -> bool:
    """
    Confirms Python 3.10 or higher is running.

    Why 3.10+?
    - Modern AI libraries (LangChain, Pydantic v2) require it
    - Structural pattern matching (match/case) was added in 3.10
    - Running 3.8 or 3.9 will cause confusing errors later in Month 2

    The -> bool at the end is called a TYPE HINT.
    It tells anyone reading this code: "this function returns True or False."
    We'll learn type hints properly later in the course.
    For now, just notice it's there.
    """
    # sys.version_info is an object with attributes: major, minor, micro
    # For Python 3.11.5:  major=3, minor=11, micro=5
    major = sys.version_info.major   # e.g. 3
    minor = sys.version_info.minor   # e.g. 11

    # f-strings (f"...") let you put variables inside curly braces {}
    # You learned these in Day 0
    print(f"  Python version:  {major}.{minor}.{sys.version_info.micro}")

    # sys.executable tells you exactly WHERE this Python lives on your disk
    # Useful for confirming you're running the right Python
    print(f"  Python location: {sys.executable}")

    # Check: major must be 3, and minor must be 10 or higher
    if major == 3 and minor >= 10:
        print("  ✅ Python version is good.\n")
        return True     # True = check passed
    else:
        print("  ❌ Python 3.10 or higher is required.")
        print("     Visit https://www.python.org/downloads/ to upgrade.\n")
        return False    # False = check failed


# ─── CHECK 2: Virtual Environment ──────────────────────────────────────────

def check_virtual_environment() -> bool:
    """
    Detects whether we are running inside a virtual environment.

    HOW PYTHON SIGNALS "I am in a venv":
      sys.prefix       = the active Python environment path
      sys.base_prefix  = the original system Python path
      If they are DIFFERENT → we are inside a virtual environment.
      If they are the SAME  → we are using the system Python (not good).

    ANALOGY:
    Think of the venv as your own private desk at work.
    Your packages don't touch anyone else's and theirs don't touch yours.
    Without a venv, everyone shares one desk and overwrites each other's stuff.
    """
    # != means "not equal to"
    # If prefix and base_prefix are different, we're in a venv
    in_venv = sys.prefix != sys.base_prefix

    if in_venv:
        # sys.prefix shows the path to our active virtual environment
        print(f"  ✅ Virtual environment is active: {sys.prefix}\n")
    else:
        print("  ⚠️  NOT inside a virtual environment.")
        print("     Run: source genai_env/bin/activate  (Mac/Linux)")
        # The r before the string means "raw string" — it treats backslashes
        # as literal characters instead of escape sequences (important for
        # Windows paths which use backslashes)
        print(r"          genai_env\Scripts\activate       (Windows)" + "\n")

    return in_venv


# ─── CHECK 3: Required Packages ────────────────────────────────────────────

def check_packages() -> bool:
    """
    Checks that the two libraries we need for today's session are installed.

    importlib.util.find_spec("package_name") looks for the package without
    importing it. Returns an object if found, or None if not found.
    This is better than 'import openai' because:
      - It doesn't run any of the package's code
      - It's faster
      - It won't cause side effects
    """
    # A dictionary: package_name → description of what it does
    # We'll loop through this to check each one
    required = {
        "openai":     "The official OpenAI Python SDK — talks to GPT models",
        "dotenv":     "Loads .env files into environment variables",
    }

    all_good = True  # Assume everything is fine; flip to False if anything fails

    # .items() gives us both the key AND value from the dictionary
    for package, description in required.items():
        # find_spec returns an object if the package exists, None if not
        spec = importlib.util.find_spec(package)

        if spec:
            # {package:<12} means: print the package name, padded to 12
            # characters wide, left-aligned. This makes the output line up
            # neatly in a column.
            print(f"  ✅ {package:<12} — {description}")
        else:
            print(f"  ❌ {package:<12} — NOT INSTALLED")
            # Special case: the package you import as 'dotenv' is actually
            # installed with 'pip install python-dotenv' (not 'pip install dotenv')
            # .replace() handles this edge case
            print(f"     Fix: pip install {package.replace('dotenv','python-dotenv')}")
            all_good = False

    print()  # blank line for readability
    return all_good


# ─── RUN ALL THREE CHECKS ──────────────────────────────────────────────────
# We call each function and store its result (True or False) in a variable.
# Then we check if ALL three passed.

python_ok   = check_python_version()
venv_ok     = check_virtual_environment()
packages_ok = check_packages()

# 'and' means ALL conditions must be True
if python_ok and venv_ok and packages_ok:
    print("✅ Environment verified. Everything looks good.")
    print("   You're ready for Section 2.\n")
else:
    print("⚠️  Some checks failed. Fix the issues above before continuing.\n")
