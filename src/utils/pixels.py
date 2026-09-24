"""How many channels an image file's pixels really carry.

VOC's ``<depth>`` means "channels a reader must allocate", which is not the
same thing as ``len(image.getbands())``: a palette PNG reports a single band
(``P``) but every reader expands it to three, so declaring 1 would be wrong
for the consumer that matters.

This lives in its own leaf module because two callers that must agree need
it and neither can import the other: ``AnnotationService`` writes the value
into the XML, ``cleanup.common_rules`` checks it back. They disagreed once
already -- the writer used ``len(path.suffix)``, so every JPEG the app saved
declared ``<depth>4</depth>``, and a validator reading the real channel count
could only report the file as broken.
"""
from __future__ import annotations

#: Modes whose bands do not match the channels a reader materialises.
_COLOR_MODES = {"P": 3, "PA": 4}


def channel_count(image) -> int:
    """Channels ``image`` carries once decoded. Takes a PIL image."""
    override = _COLOR_MODES.get(image.mode)
    if override is not None:
        return override
    return len(image.getbands())
