class ProjectBaseError(Exception):
    pass


class DataLoadError(ProjectBaseError):
    pass


class ModelNotFoundError(ProjectBaseError):
    pass


class ClassificationError(ProjectBaseError):
    pass


class PolicyViolationError(ProjectBaseError):
    pass
