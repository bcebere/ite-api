def init_sys_path() -> None:
    # stdlib
    import os
    import sys

    proj_dir = os.path.abspath(
        os.path.join(
            os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir),
            os.pardir,
        )
    )
    sys.path.append(os.path.join(proj_dir, "init"))
    # third party
    import initpath

    initpath.platform_init_path(proj_dir)
