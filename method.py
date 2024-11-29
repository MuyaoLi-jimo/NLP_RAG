"""
API for different methods
"""

from rag_prepare import RAG_SYSTEM

def plain():
    """use plain prompt"""
    pass

def cot():
    """use Chain-of-Thought"""
    
def icl(k:int=5):
    """use few shot learning"""
    assert k<=5
    
def rag():
    pass
