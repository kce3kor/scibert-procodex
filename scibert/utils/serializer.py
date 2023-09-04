import pickle
from scibert.utils.logger import logger


def pickle_serializer(object=None, path=None, mode="save"):
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
