from typing import Text


def get_module(package_path: Text = None, class_name: Text = None, **kwargs):
    """
    Dynamically load the specified class.

    :param package_path: Path to the package to load
    :param class_name: Name of the class within the package
    :param args: arguments to pass when creating the object
    :return: the instantiated class object
    """
    package_path = kwargs.get("package") if not package_path else package_path
    class_name = kwargs.get("class") if not class_name else class_name
    if not class_name:
        class_name = kwargs.get("class")

    module = __import__(package_path, fromlist=[class_name])
    klass = getattr(module, class_name)
    return klass
