import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import streamlit as st
from youtube_rag.main import load_app

if __name__ == "__main__":
    load_app()