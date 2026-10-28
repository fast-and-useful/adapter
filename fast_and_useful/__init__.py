"""Fast and Useful: adapt a released distilled text-to-image generator so that its images train better classifiers.

A low-rank update on the student's first prediction is fitted to the state its unguided base reaches from the same
noise; at generation the update runs for that one prediction, and the released student finishes the image. See
README.md, and https://arxiv.org/abs/2610.32512 for the method and its evaluation.
"""
__version__ = "0.1.0"


def __getattr__(name):   # `from fast_and_useful import AdaptedGenerator` without importing diffusers up front
    if name == "AdaptedGenerator":
        from .generate import AdaptedGenerator
        return AdaptedGenerator
    raise AttributeError(name)
