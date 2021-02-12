def init_sys_path() -> None:
    # stdlib
    import os
    import sys

    depth = 1
    path = os.path.dirname(os.path.realpath(__file__))
    for d in range(depth):
        path = os.path.join(path, os.pardir)
    proj_dir = os.path.abspath(path)
    print(proj_dir)
    sys.path.append(os.path.join(proj_dir, "init"))
    # third party
    import initpath

    initpath.platform_init_path(proj_dir)
