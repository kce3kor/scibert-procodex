import pickle
from scibert.utils.logger import logger


def pickle_serializer(object=None, path=None, mode="save") -> bool:
    """Pickle serializer to save or load objects to/from checkpoints

    Args:
        object (_type_, optional): Object to save. Defaults to None.
        path (_type_, optional): Path to save to/load from. Defaults to None.
        mode (str, optional): mode of serializer:  save or load. Defaults to "save".

    Raises:
        Exception: FileNotFound error while trying to load a non existing object
        Exception: Serializer mode not found, except load/save

    Returns:
        bool: status of serialization
    """
    if mode == "save":
        logger.info(f"Saving object at {path}.")
        with open(path, "wb") as handle:
            pickle.dump(object, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return True

    elif mode == "load":
        try:
            logger.info(f"Loading object from {path}")
            with open(path, "rb") as handle:
                object = pickle.load(handle)
            return object
        except FileNotFoundError:
            raise Exception(f"File not found at '{path}'")
    else:
        raise Exception("Serializer mode not found")


if __name__ == "__main__":
    pass
