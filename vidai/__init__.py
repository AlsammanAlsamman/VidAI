"""VidAI: a Claude video-editing plugin. Anchors + tools + custom models so Claude can edit videos."""
from .anchors import STATS, AnchorFile, Event, Segment, Series, list_stats, select_stats
from .brief import QUESTIONS, Brief
from .edit import (ApplyModel, Audio, Chapter, Cut, EditPlan, Image, Shape, Subtitle, Text, Zoom,
                   new_plan)

__version__ = "0.3.0"

__all__ = ["STATS", "AnchorFile", "Event", "Segment", "Series", "list_stats", "select_stats", "QUESTIONS",
           "Brief", "ApplyModel", "Audio", "Chapter", "Cut", "EditPlan", "Image", "Shape", "Subtitle", "Text",
           "Zoom", "new_plan", "__version__"]
