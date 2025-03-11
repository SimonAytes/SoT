# Warning for when an image is passed, but the multimodal format was not specified
MULTIMODAL_MISALIGNMENT = (
    "Image data was passed, but the format is set to `llm`. "
    "Resulting context will not include image data. "
    "Please change format to `vlm` to include image data."
)

NO_IMAGE = (
    "Format was specified as `vlm` but no image data was passed."
    "Resulting multimodal context will not include image data. "
)